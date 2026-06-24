# QLoRA Fine-tuning — Qwen 2.5 (0.5B) on Upinder Singh's History

Fine-tune **Qwen/Qwen2.5-0.5B-Instruct** with 4-bit QLoRA on any PDF.
Trained on *A History of Ancient and Early Medieval India* by Upinder Singh (1,094 pages).

## 🤗 Fine-tuned Model on HuggingFace
👉 **[https://huggingface.co/Rut-ai/qwen2.5-upinder-singh-history](https://huggingface.co/Rut-ai/qwen2.5-upinder-singh-history)**

> Model weights are hosted on HuggingFace (too large for GitHub).

## Quick Start
```bash
pip install -r requirements.txt
python run_finetune.py --data your_book.pdf
python inference.py --interactive
```

## Training Results
| Metric | Value |
|--------|-------|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | 1,094 pages, 429,110 words |
| Training samples | 1,759 |
| Method | QLoRA (4-bit NF4) + LoRA r=16 α=32 |
| Trainable params | 8.80M / 323.9M (2.72%) |
| Train loss | 2.316 |
| Eval loss | 2.306 |
| Token accuracy | 54.4% |
| GPU | RTX 3050 Laptop 4GB |
| Training time | 36.2 minutes |

## Files
| File | Purpose |
|------|---------|
| `finetune.py` | Main training script |
| `data_prep.py` | PDF → training samples |
| `run_finetune.py` | Launcher |
| `inference.py` | Chat with model |
| `push_to_hub.py` | Upload to HuggingFace |
| `HOW_TO_USE.md` | Step-by-step guide |
