# QLoRA Fine-tuning: Qwen 2.5 (0.5B) on Upinder Singh's History of Ancient India

Fine-tune **Qwen/Qwen2.5-0.5B-Instruct** with **4-bit QLoRA** on  
*A History of Ancient and Early Medieval India* by Upinder Singh.

---

## Architecture

```
Qwen 2.5 0.5B (frozen, 4-bit NF4)
        +
LoRA Adapter (trainable ~1-2% of params)
    └── q_proj, k_proj, v_proj, o_proj
    └── gate_proj, up_proj, down_proj
        ↓
Fine-tuned on Upinder Singh PDF → ChatML prompts
        ↓
Merged standalone model (optional)
```

---

## Quick Start

### 1. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run fine-tuning

```powershell
python run_finetune.py
```

Or with custom settings:

```powershell
python run_finetune.py --epochs 5 --max-len 512 --lora-r 32
```

### 3. Run inference after training

```powershell
# Interactive Q&A session
python inference.py --interactive

# Quick demo
python inference.py
```

---

## Files

| File | Purpose |
|------|---------|
| `finetune.py` | Main training script (QLoRA + PEFT + TRL) |
| `data_prep.py` | PDF → instruction-following samples |
| `run_finetune.py` | Convenience launcher (validates PDF path) |
| `inference.py` | Load adapter and generate responses |
| `requirements.txt` | Python dependencies |

---

## Key Configuration (finetune.py → Config)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Downloaded from HuggingFace |
| `LOAD_IN_4BIT` | `True` | NF4 quantisation (QLoRA) |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | Scaling = 2× rank |
| `NUM_EPOCHS` | `3` | Increase for more specialisation |
| `BATCH_SIZE` | `2` | Per-device (effective = 16 with grad_accum=8) |
| `MAX_SEQ_LENGTH` | `512` | Tokens per sample |
| `LEARNING_RATE` | `2e-4` | Cosine schedule |
| `BF16` | `True` | Ampere GPUs (RTX 30xx+) |

---

## Data Pipeline

The PDF is processed as follows:

```
PDF pages
    ↓
pypdf text extraction + cleaning (removes headers/footers/hyphenation)
    ↓
Sentence-aware chunking (~250 words, 2-sentence overlap)
    ↓
8 history-specific instruction templates (round-robin)
    ↓
Qwen ChatML format (<|im_start|> / <|im_end|>)
    ↓
90% Train / 10% Validation split
```

### Sample formatted prompt

```
<|im_start|>system
You are a knowledgeable assistant specialised in ancient and early medieval Indian history, trained on Upinder Singh's scholarship.<|im_end|>
<|im_start|>user
Summarise the following excerpt from Upinder Singh's history of ancient India in 2–3 sentences:

The Mauryan Empire, founded by Chandragupta Maurya in c. 321 BCE...
<|im_end|>
<|im_start|>assistant
The Mauryan Empire was one of the largest empires of ancient India...<|im_end|>
```

---

## Outputs

After training, two directories are created inside `qlora_history/`:

| Path | Contents |
|------|---------|
| `qlora_history/final_adapter/` | LoRA adapter weights (small, ~20-50 MB) |
| `qlora_history/merged_model/` | Full merged model (ready for deployment) |

---

## GPU Requirements

| Configuration | Min VRAM |
|---------------|----------|
| 4-bit QLoRA (default) | ~3.5 GB |
| Full FP16 (no QLoRA) | ~2 GB inference / ~6 GB training |

Your RTX 3050 (4 GB) is sufficient for the default QLoRA config.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 1 or `MAX_SEQ_LENGTH` to 256 |
| `FileNotFoundError` for PDF | Update `DATA_PATH` in `Config` or pass `--data` flag |
| `bitsandbytes` error | Run: `pip install bitsandbytes --upgrade` |
| Slow training (no CUDA) | Training runs on CPU; expect 10-50× slower |
