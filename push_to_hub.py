"""
Push fine-tuned Qwen 2.5 (0.5B) model to HuggingFace Hub
==========================================================
Usage:
    python push_to_hub.py --token hf_xxxx
    python push_to_hub.py --token hf_xxxx --repo qwen2.5-upinder-singh-history --upload-merged
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL CARD  (written to the repo README.md)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: apache-2.0
tags:
- indian-history
- ancient-india
- medieval-india
- qlora
- peft
- lora
- qwen2.5
- fine-tuned
- history
- upinder-singh
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
---

# Qwen 2.5 (0.5B) — Ancient & Early Medieval Indian History

A domain-adapted version of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
fine-tuned on **Upinder Singh's _A History of Ancient and Early Medieval India_** (1,094 pages, 429,110 words)
using **QLoRA** (4-bit NF4 quantisation + LoRA adapters).

The model can explain historical events, summarise passages, answer questions,
and continue text about ancient India — from the Indus Valley Civilisation
through the Vedic period, Mauryan and Gupta empires, up to early medieval India.

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA) |
| Training data | Upinder Singh — *A History of Ancient and Early Medieval India* (PDF, 44.9 MB) |
| Training pages | 1,094 pages |
| Training words | 429,110 |
| Training samples | 1,759 instruction-following samples |
| Validation samples | 196 |
| Epochs | 3 |
| Effective batch size | 16 (batch 2 × grad_accum 8) |
| Learning rate | 2e-4 (cosine decay) |
| Max sequence length | 512 tokens |
| LoRA rank / alpha | 16 / 32 |
| Trainable parameters | 8.80M / 323.9M (2.72%) |
| Final train loss | 2.316 |
| Final eval loss | 2.306 |
| Token accuracy | 54.4% |
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| Training time | 36.2 minutes |

---

## Quick Start

### Option A — Use the merged model (no base model needed)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

def ask(instruction: str, context: str = "") -> str:
    system = (
        "You are a knowledgeable assistant specialised in ancient and early "
        "medieval Indian history, trained on Upinder Singh's scholarship."
    )
    user = instruction
    if context:
        user += f"\\n\\n{{context}}"

    messages = [
        {{"role": "system", "content": system}},
        {{"role": "user",   "content": user}},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# Example
print(ask("Who was Ashoka and why is he important in Indian history?"))
print(ask("Describe the Indus Valley Civilisation."))
print(ask("Summarise the Gupta period in Indian history."))
```

### Option B — Load LoRA adapter on base model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base  = "Qwen/Qwen2.5-0.5B-Instruct"
lora  = "{repo_id}"   # same repo hosts the adapter

tokenizer = AutoTokenizer.from_pretrained(lora)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, lora)
model.eval()
```

---

## Training Data Format

Each sample uses **Qwen ChatML** format:

```
<|im_start|>system
You are a knowledgeable assistant specialised in ancient and early medieval
Indian history, trained on Upinder Singh's scholarship.<|im_end|>
<|im_start|>user
Summarise the following excerpt from Upinder Singh's history of ancient India:

[~150-word passage from the book]<|im_end|>
<|im_start|>assistant
[Continuation / summary / explanation]<|im_end|>
```

Eight instruction templates were used in round-robin:
- Continue the following passage...
- Summarise the following excerpt...
- Explain the historical significance...
- Answer this question based on the text...
- Elaborate on the following historical account...
- Teaching-style explanation...
- Key historical insights...
- Describe the events/developments...

---

## Topics Covered

The training data spans the full book, including:

- 🏺 **Indus Valley / Harappan Civilisation** (c. 2600–1900 BCE)
- 📜 **Vedic Period** — Rigveda, Later Vedic texts, early society
- 👑 **Mahajanapadas & Rise of Magadha**
- 🦁 **Mauryan Empire** — Chandragupta, Ashoka, Dhamma
- 🛕 **Post-Mauryan Period** — Kushanas, Satavahanas
- ✨ **Gupta Empire** — art, science, literature, administration
- 🕌 **Early Medieval India** — Rajputs, Pallavas, Chalukyas, Rashtrakutas
- 📿 **Religious developments** — Buddhism, Jainism, Bhakti, Vedic religion

---

## Limitations

- Small 0.5B model — may hallucinate specific dates or names on pure factual Q&A
- Best suited for text continuation, passage explanation, and historical summarisation
- For high-accuracy factual recall, consider pairing with RAG over the source PDF

## License

Apache 2.0 — inherits from Qwen/Qwen2.5-0.5B-Instruct base model.
"""


# ──────────────────────────────────────────────────────────────────────────────
# PUSH FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def push_model(token: str, repo_name: str, model_path: str, private: bool) -> str:
    from huggingface_hub import HfApi, login

    # 1. Login
    print(f"\n[HF] Logging in to HuggingFace...")
    login(token=token)
    api = HfApi()

    user     = api.whoami()
    username = user["name"]
    print(f"   [OK] Logged in as: {username}")

    repo_id = f"{username}/{repo_name}"
    print(f"   [OK] Target repo : https://huggingface.co/{repo_id}")

    # 2. Create repo
    print(f"\n[HF] Creating repository '{repo_id}'...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    visibility = "private" if private else "public"
    print(f"   [OK] Repo ready ({visibility})")

    # 3. Upload files
    print(f"\n[HF] Uploading model files from: {model_path}")
    size_mb = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()) / (1024**2)
    print(f"   Upload size : ~{size_mb:.0f} MB  (may take a few minutes...)")

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload QLoRA fine-tuned Qwen 2.5 — Upinder Singh Indian History",
        ignore_patterns=["*.pyc", "__pycache__", "*.tmp", "*.log"],
    )

    print(f"\n[DONE] Model files uploaded!")
    print(f"   URL: https://huggingface.co/{repo_id}")
    return repo_id


def push_model_card(token: str, repo_id: str):
    """Upload a rich model card (README.md) to the repo."""
    from huggingface_hub import HfApi

    print(f"\n[HF] Uploading model card...")
    card = MODEL_CARD_TEMPLATE.replace("{repo_id}", repo_id)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add detailed model card",
    )
    print(f"   [OK] Model card uploaded")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push QLoRA fine-tuned Qwen 2.5 model to HuggingFace Hub"
    )
    parser.add_argument(
        "--token", required=True,
        help="HuggingFace API token (https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--repo", default="qwen2.5-upinder-singh-history",
        help="Repository name (default: qwen2.5-upinder-singh-history)"
    )
    parser.add_argument(
        "--upload-merged", action="store_true",
        help="Upload the full merged model (standalone, no base model needed; ~1 GB)"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make the repository private"
    )
    args = parser.parse_args()

    model_path = (
        "./qlora_history/merged_model"
        if args.upload_merged
        else "./qlora_history/final_adapter"
    )
    model_type = "Full merged model" if args.upload_merged else "LoRA adapter"

    print("=" * 60)
    print("  Pushing to HuggingFace Hub")
    print("=" * 60)
    print(f"  Model type : {model_type}")
    print(f"  Model path : {model_path}")
    print(f"  Repo name  : {args.repo}")
    print(f"  Visibility : {'private' if args.private else 'public'}")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"\n[ERROR] Model path not found: {model_path}")
        print("        Run fine-tuning first:  python run_finetune.py")
        sys.exit(1)

    repo_id = push_model(args.token, args.repo, model_path, args.private)
    push_model_card(args.token, repo_id)

    print("\n" + "=" * 60)
    print("  PUBLISHED SUCCESSFULLY!")
    print(f"  https://huggingface.co/{repo_id}")
    print("=" * 60)
