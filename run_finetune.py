"""
Quick launcher for fine-tuning Qwen 2.5 (0.5B) on the Upinder Singh PDF.

Usage:
    python run_finetune.py                          # uses defaults
    python run_finetune.py --epochs 5               # more epochs
    python run_finetune.py --data path/to/book.pdf  # custom PDF path
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Default PDF path (update if your file is elsewhere) ──────────────────────
DEFAULT_PDF = r"C:\Users\comed\Desktop\A History of ancient an early mideaval India by Upinder sing.pdf"

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


def check_pdf(path: str) -> bool:
    if not Path(path).exists():
        print(f"\n[ERROR] PDF not found: {path}")
        print("        Please check the file path and try again.")
        return False
    size_mb = Path(path).stat().st_size / (1024 ** 2)
    print(f"[OK] PDF found: {path}  ({size_mb:.1f} MB)")
    return True


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run QLoRA fine-tuning on Upinder Singh PDF")
    p.add_argument("--data",    default=DEFAULT_PDF, help="Path to PDF")
    p.add_argument("--output",  default="./qlora_history", help="Output directory")
    p.add_argument("--epochs",  type=int,   default=3)
    p.add_argument("--batch",   type=int,   default=2)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--max-len", type=int,   default=512)
    p.add_argument("--lora-r",  type=int,   default=16)
    args = p.parse_args()

    print("=" * 65)
    print("  QLoRA Fine-tuning Launcher")
    print("  Model  : Qwen/Qwen2.5-0.5B-Instruct")
    print(f"  Data   : {args.data}")
    print(f"  Output : {args.output}")
    print("=" * 65)

    if not check_pdf(args.data):
        sys.exit(1)

    cmd = [
        sys.executable, str(ROOT / "finetune.py"),
        "--data",    args.data,
        "--output",  args.output,
        "--epochs",  str(args.epochs),
        "--batch",   str(args.batch),
        "--lr",      str(args.lr),
        "--max-len", str(args.max_len),
        "--lora-r",  str(args.lora_r),
    ]

    print(f"\n[RUN] {' '.join(cmd)}\n")
    os.chdir(ROOT)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
