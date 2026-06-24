# 📚 QLoRA Fine-Tuning Guide — Fine-tune Any LLM on Any PDF

A simple, reusable toolkit to fine-tune a small language model (Qwen 2.5 0.5B)
on **any PDF book or document** using QLoRA — without needing a powerful GPU.

---

## 🖥️ Requirements

| What | Details |
|------|---------|
| GPU | NVIDIA GPU with **4 GB+ VRAM** (RTX 3050 or better) |
| RAM | 8 GB minimum |
| Python | 3.10 or 3.11 |
| Disk | ~5 GB free (for model weights) |

---

## ⚙️ One-Time Setup (Do This First)

Open a terminal inside this folder (`qlora-finetune`) and run:

```
setup.bat
```

This creates a virtual environment and installs all required packages automatically.

---

## 🚀 Fine-tune on Your PDF — 3 Steps

### Step 1 — Open `finetune.py` and change two lines

Find the `Config` class at the top of [`finetune.py`](finetune.py) and update:

```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # ← keep this (or change model)

    DATA_PATH  = r"C:\path\to\your\book.pdf"      # ← PUT YOUR PDF PATH HERE
    OUTPUT_DIR = "./my_finetuned_model"            # ← WHERE TO SAVE THE MODEL
```

> **Tip:** Right-click your PDF → "Copy as path" → paste it replacing the path above.

---

### Step 2 — Run fine-tuning

```
venv\Scripts\activate
python run_finetune.py
```

Or pass arguments directly:

```
python run_finetune.py --data "C:\Users\You\Desktop\mybook.pdf" --epochs 3
```

That's it! Training will take **30–90 minutes** depending on your PDF size and GPU.

---

### Step 3 — Test your model

```
python inference.py --interactive
```

Then type any question related to your PDF content.

---

## 📂 What Each File Does

| File | What it does |
|------|-------------|
| `finetune.py` | **Main training script** — edit `DATA_PATH` here |
| `run_finetune.py` | Convenience launcher — validates PDF then runs finetune.py |
| `data_prep.py` | Reads PDF, splits into chunks, creates Q&A training samples |
| `inference.py` | Chat with your fine-tuned model after training |
| `push_to_hub.py` | Upload your model to HuggingFace |
| `requirements.txt` | All Python packages needed |
| `setup.bat` | One-click setup script |

---

## 🎛️ Tuning for Different PDF Sizes

Open `finetune.py` and adjust the `Config` class:

| Your PDF | Recommended Settings |
|----------|---------------------|
| Small (< 100 pages) | `NUM_EPOCHS = 5`, `LORA_R = 8` |
| Medium (100–500 pages) | `NUM_EPOCHS = 3`, `LORA_R = 16` ← **default** |
| Large (500+ pages) | `NUM_EPOCHS = 2`, `LORA_R = 16` |
| Out of memory error | Reduce `BATCH_SIZE = 1` and `MAX_SEQ_LENGTH = 256` |

---

## 🌐 Upload to HuggingFace (Optional)

1. Go to https://huggingface.co/settings/tokens → create a **Write** token
2. Run:

```
python push_to_hub.py --token hf_YOUR_TOKEN --repo my-model-name --upload-merged
```

Your model will appear at: `https://huggingface.co/YOUR_USERNAME/my-model-name`

---

## 🔄 Change the Base Model

You can swap Qwen 2.5 for another model by changing one line in `finetune.py`:

```python
# Other good small models to try:
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"          # default (recommended)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"          # better quality, needs ~6 GB VRAM
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"  # smallest & fastest
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"     # very capable, needs 6 GB VRAM
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Set `BATCH_SIZE = 1` and `MAX_SEQ_LENGTH = 256` in Config |
| `FileNotFoundError` for PDF | Check your `DATA_PATH` — use raw string `r"C:\..."` |
| `ModuleNotFoundError` | Run `setup.bat` again or `pip install -r requirements.txt` |
| Model generates gibberish | Train for more epochs or use a larger `LORA_R` (try 32) |
| Training is very slow | Normal on CPU — use a CUDA GPU for best results |

---

## 📋 Quick Reference Card

```
# Setup (once)
setup.bat

# Fine-tune on your PDF
python run_finetune.py --data "path\to\your.pdf" --epochs 3

# Chat with your model
python inference.py --interactive

# Push to HuggingFace
python push_to_hub.py --token hf_xxx --repo my-model --upload-merged
```

---

*Built with: HuggingFace Transformers · PEFT · TRL · bitsandbytes · QLoRA*
