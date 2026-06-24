"""
QLoRA Fine-tuning Script for Qwen 2.5 (0.5B)
=============================================
Target dataset  : Upinder Singh – A History of Ancient and Early Medieval India (PDF)
Model           : Qwen/Qwen2.5-0.5B-Instruct
Quantisation    : 4-bit NF4 (QLoRA via bitsandbytes)
Adapter         : LoRA (PEFT) – full attention + FFN modules
Prompt format   : Qwen ChatML (<|im_start|>/<|im_end|>)
Device          : CUDA (preferred) → CPU fallback
"""

import os
import sys
import time
import warnings
import argparse
import torch
from pathlib import Path

# ── UTF-8 on Windows terminal ──────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"]       = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from data_prep import prepare_dataset

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Data ──────────────────────────────────────────────────────────────────
    # Set this to the absolute path of your Upinder Singh PDF
    DATA_PATH  = r"C:\Users\comed\Desktop\A History of ancient an early mideaval India by Upinder sing.pdf"
    OUTPUT_DIR = "./qlora_history"

    # ── QLoRA (4-bit quantisation) ────────────────────────────────────────────
    LOAD_IN_4BIT        = True
    BNB_4BIT_COMPUTE    = torch.bfloat16   # bfloat16 stable on Ampere (RTX 30xx)
    BNB_4BIT_QUANT_TYPE = "nf4"            # Normal Float 4 – best quality
    USE_NESTED_QUANT    = True             # Double quantisation → more memory savings

    # ── LoRA Adapter ──────────────────────────────────────────────────────────
    LORA_R         = 16        # rank
    LORA_ALPHA     = 32        # scaling (2 × rank is standard)
    LORA_DROPOUT   = 0.05
    LORA_BIAS      = "none"
    LORA_TASK_TYPE = TaskType.CAUSAL_LM

    # Target all projection layers in Qwen 2.5 (attention + FFN)
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # ── Training (tuned for ~4 GB VRAM) ──────────────────────────────────────
    NUM_EPOCHS       = 3            # 3 epochs is solid for domain adaptation
    BATCH_SIZE       = 2
    GRAD_ACCUM_STEPS = 8            # effective batch = 16
    LEARNING_RATE    = 2e-4
    MAX_SEQ_LENGTH   = 512          # longer context captures more history text
    WARMUP_RATIO     = 0.05
    LR_SCHEDULER     = "cosine"
    WEIGHT_DECAY     = 0.01
    MAX_GRAD_NORM    = 0.3
    SAVE_STEPS       = 100
    LOG_STEPS        = 10
    VAL_SPLIT        = 0.1
    SEED             = 42

    # ── Precision (RTX 30xx / Ampere → BF16 supported) ────────────────────────
    FP16 = False   # disabled – BNB uses bfloat16 internally; FP16 scaler crashes
    BF16 = True    # safe on Ampere, matches BNB compute dtype


# ──────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[CUDA] {gpu}  ({vram:.1f} GB VRAM)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[MPS] Apple Metal GPU detected")
        return "mps"
    else:
        print("[WARN] No GPU found – running on CPU (slow!)")
        return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# MODEL + TOKENIZER
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config, device: str):
    print(f"\n[MODEL] Loading: {cfg.MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    # Qwen uses <|endoftext|> as EOS; ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantisation config (CUDA only)
    bnb_config = None
    if device == "cuda" and cfg.LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=cfg.BNB_4BIT_COMPUTE,
            bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=cfg.USE_NESTED_QUANT,
        )
        print("   [OK] 4-bit NF4 quantisation enabled (QLoRA)")
    else:
        print("   [WARN] 4-bit quantisation requires CUDA – skipping")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"    if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",   # avoid flash-attn dependency on Windows
    )

    if device != "cuda":
        model = model.to(device)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    total = sum(p.numel() for p in model.parameters())
    print(f"   [OK] Model loaded ({total / 1e6:.1f}M parameters)")

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# LORA SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_lora(model, cfg: Config):
    print("\n[LORA] Attaching LoRA adapter...")
    print(f"   Target modules : {cfg.LORA_TARGET_MODULES}")
    print(f"   Rank (r)       : {cfg.LORA_R}")
    print(f"   Alpha          : {cfg.LORA_ALPHA}")
    print(f"   Dropout        : {cfg.LORA_DROPOUT}")

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        bias=cfg.LORA_BIAS,
        task_type=cfg.LORA_TASK_TYPE,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total
    print(f"   Trainable      : {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({pct:.2f}%)")
    print("   [OK] LoRA adapter attached")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING ARGUMENTS
# ──────────────────────────────────────────────────────────────────────────────

def build_training_args(cfg: Config, device: str) -> SFTConfig:
    use_fp16 = cfg.FP16 and device == "cuda"
    use_bf16 = cfg.BF16 and device == "cuda"

    return SFTConfig(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        learning_rate=cfg.LEARNING_RATE,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        warmup_ratio=cfg.WARMUP_RATIO,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=cfg.LOG_STEPS,
        save_steps=cfg.SAVE_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=cfg.SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=(device == "cuda"),
        max_length=cfg.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        # Pack short sequences together to maximise GPU utilisation
        packing=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    print("=" * 65)
    print("  QLoRA Fine-tuning | Qwen 2.5 (0.5B) | Upinder Singh History")
    print("=" * 65)

    # 1. Device
    device = detect_device()

    # 2. Data
    print(f"\n[DATA] Source: {cfg.DATA_PATH}")
    train_ds, val_ds = prepare_dataset(
        cfg.DATA_PATH,
        val_split=cfg.VAL_SPLIT,
        seed=cfg.SEED,
        model_name=cfg.MODEL_NAME,
    )

    # 3. Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, device)

    # 4. LoRA
    model = setup_lora(model, cfg)

    # 5. Training args
    training_args = build_training_args(cfg, device)

    # 6. Print training summary
    print(f"\n[TRAIN] Starting fine-tuning...")
    print(f"   Model       : {cfg.MODEL_NAME}")
    print(f"   Epochs      : {cfg.NUM_EPOCHS}")
    print(f"   Batch size  : {cfg.BATCH_SIZE} x {cfg.GRAD_ACCUM_STEPS} accum = "
          f"{cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS} effective")
    print(f"   Max seq len : {cfg.MAX_SEQ_LENGTH}")
    print(f"   Output dir  : {cfg.OUTPUT_DIR}")
    print(f"   Train size  : {len(train_ds)} samples")
    print(f"   Val size    : {len(val_ds)} samples")
    print()

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # 8. Train!
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\n[DONE] Training complete in {elapsed / 60:.1f} minutes")

    # 9. Save LoRA adapter
    adapter_path = os.path.join(cfg.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[SAVE] LoRA adapter → {adapter_path}")

    # 10. Merge weights into a standalone model
    print("\n[MERGE] Merging LoRA weights into base model...")
    try:
        merged_model = trainer.model.merge_and_unload()
        merged_path  = os.path.join(cfg.OUTPUT_DIR, "merged_model")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"[SAVE] Merged model → {merged_path}")
    except Exception as e:
        print(f"[WARN] Merge skipped (use adapter for inference): {e}")

    print("\n" + "=" * 65)
    print("  Fine-tuning complete!")
    print(f"  Adapter  : {adapter_path}")
    print("  Run inference:  python inference.py --interactive")
    print("=" * 65)
    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning – Qwen 2.5 0.5B on Upinder Singh PDF"
    )
    p.add_argument("--model",    default=None, help="HuggingFace model ID")
    p.add_argument("--data",     default=None, help="Path to PDF or CSV")
    p.add_argument("--output",   default=None, help="Output directory")
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--batch",    type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--max-len",  type=int,   default=None)
    p.add_argument("--lora-r",   type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()

    if args.model:   cfg.MODEL_NAME     = args.model
    if args.data:    cfg.DATA_PATH      = args.data
    if args.output:  cfg.OUTPUT_DIR     = args.output
    if args.epochs:  cfg.NUM_EPOCHS     = args.epochs
    if args.batch:   cfg.BATCH_SIZE     = args.batch
    if args.lr:      cfg.LEARNING_RATE  = args.lr
    if args.max_len: cfg.MAX_SEQ_LENGTH = args.max_len
    if args.lora_r:  cfg.LORA_R        = args.lora_r

    train(cfg)
