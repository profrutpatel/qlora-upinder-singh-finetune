# ============================================================
# QLoRA Fine-tuning — Google Colab Quick Start
# Run this notebook on Colab with GPU (T4 is free!)
# Runtime → Change runtime type → GPU
# ============================================================

# ── Cell 1: Install dependencies ────────────────────────────
# !pip install -q torch transformers peft bitsandbytes accelerate trl datasets pandas pypdf

# ── Cell 2: Complete QLoRA pipeline (all-in-one) ─────────────
"""
Paste this into a Colab cell and run it.
It includes sample data, training, and inference in one go.
"""

COLAB_NOTEBOOK = '''
# Install
!pip install -q transformers peft bitsandbytes accelerate trl datasets

import os, torch, pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ── 1. Configuration ──────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR    = "/content/qlora_output"
NUM_EPOCHS    = 3
BATCH_SIZE    = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 512
LORA_R        = 16
LORA_ALPHA    = 32

# ── 2. Sample training data ───────────────────────────────────
data = [
    {"instruction": "Explain what machine learning is", "input": "",
     "output": "Machine learning enables systems to learn from data without being explicitly programmed."},
    {"instruction": "What is Python used for?", "input": "",
     "output": "Python is used for web development, data science, AI, automation, and scientific computing."},
    {"instruction": "Give me 3 tips for better sleep", "input": "",
     "output": "1. Maintain a consistent schedule.\\n2. Avoid screens before bed.\\n3. Keep bedroom cool and dark."},
    {"instruction": "What is a neural network?", "input": "",
     "output": "A neural network is a computational model inspired by the brain, consisting of layered nodes that learn patterns from data."},
    {"instruction": "Explain recursion", "input": "",
     "output": "Recursion is when a function calls itself, breaking a problem into smaller sub-problems until a base case is reached."},
    {"instruction": "What is Docker?", "input": "",
     "output": "Docker containerizes applications and their dependencies so they run consistently across any environment."},
    {"instruction": "What is the Pythagorean theorem?", "input": "",
     "output": "a² + b² = c², where c is the hypotenuse of a right triangle."},
    {"instruction": "Translate to Spanish: Hello, how are you?", "input": "",
     "output": "Hola, ¿cómo estás?"},
    {"instruction": "How does photosynthesis work?", "input": "",
     "output": "Plants convert sunlight, water, and CO2 into glucose and oxygen using chlorophyll in chloroplasts."},
    {"instruction": "What is cloud computing?", "input": "",
     "output": "Cloud computing delivers computing services over the internet on a pay-as-you-go basis, replacing on-premise hardware."},
]

def format_prompt(ex):
    if ex["input"].strip():
        text = f"### Instruction:\\n{ex[\'instruction\']}\\n\\n### Input:\\n{ex[\'input\']}\\n\\n### Response:\\n{ex[\'output\']}"
    else:
        text = f"### Instruction:\\n{ex[\'instruction\']}\\n\\n### Response:\\n{ex[\'output\']}"
    return {"text": text}

dataset = Dataset.from_list(data).map(format_prompt)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# ── 3. Load quantized model ───────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# ── 4. LoRA adapter ────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 5. Train ──────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=4,
    optim="paged_adamw_8bit", learning_rate=LEARNING_RATE,
    fp16=True, logging_steps=5, save_steps=50,
    eval_strategy="epoch", max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text", report_to="none",
)
trainer = SFTTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    processing_class=tokenizer,
)
trainer.train()

# ── 6. Save ───────────────────────────────────────────────────
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print("✅ Adapter saved!")

# ── 7. Inference ──────────────────────────────────────────────
def ask(instruction, inp=""):
    if inp:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{inp}\\n\\n### Response:\\n"
    else:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(ask("Explain what machine learning is"))
print(ask("Give me 3 tips for better sleep"))
'''

print("Copy the COLAB_NOTEBOOK string above into Google Colab cells.")
print("Make sure to enable GPU: Runtime → Change runtime type → T4 GPU")
