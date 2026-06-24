"""
Data preparation utilities for QLoRA fine-tuning.
Supports loading from CSV and PDF files.

Upinder Singh History Book Edition
------------------------------------
- Smart sentence-aware chunking (avoids splitting mid-sentence)
- Rich Q&A instruction templates tailored to ancient/medieval Indian history
- Qwen 2.5 ChatML format prompts for best instruction-following quality
"""

import re
import random
import pandas as pd
from pathlib import Path
from datasets import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# INSTRUCTION TEMPLATES  (history-specific)
# ──────────────────────────────────────────────────────────────────────────────

HISTORY_QA_TEMPLATES = [
    # Continuation / passage-completion
    ("Continue the following passage from Upinder Singh's 'A History of Ancient and Early Medieval India':",
     "{context}",
     "{completion}"),

    # Summarisation
    ("Summarise the following excerpt from Upinder Singh's history of ancient India in 2–3 sentences:",
     "{context}",
     "{completion}"),

    # Explain
    ("Explain the historical significance of the following passage about ancient or early medieval India:",
     "{context}",
     "{completion}"),

    # Context question
    ("Answer this question based on the following text from Upinder Singh's history book:\n"
     "Question: What does this passage describe?",
     "{context}",
     "{completion}"),

    # Elaborate
    ("Elaborate on the following historical account from ancient India:",
     "{context}",
     "{completion}"),

    # Teaching style
    ("You are a history teacher. Using the passage below as context, write a brief educational explanation:",
     "{context}",
     "{completion}"),

    # Insight
    ("What are the key historical insights in the following passage from Upinder Singh's book?",
     "{context}",
     "{completion}"),

    # Timeline
    ("Describe the events or developments mentioned in the following passage about ancient Indian history:",
     "{context}",
     "{completion}"),
]


# ──────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str,
                  instruction_col: str = "instruction",
                  input_col: str = "input",
                  output_col: str = "output") -> Dataset:
    """
    Load training data from a CSV file.

    Expected columns:
        instruction : The task instruction
        input       : Optional context/input (can be empty)
        output      : The expected model response
    """
    print(f"[DATA] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in [instruction_col, output_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. Available: {list(df.columns)}"
            )

    if input_col not in df.columns:
        df[input_col] = ""

    df = df.dropna(subset=[instruction_col, output_col])
    df[input_col] = df[input_col].fillna("")

    for col in [instruction_col, input_col, output_col]:
        df[col] = df[col].astype(str).str.strip()

    print(f"   [OK] Loaded {len(df)} samples from CSV")
    return Dataset.from_pandas(df[[instruction_col, input_col, output_col]])


# ──────────────────────────────────────────────────────────────────────────────
# PDF LOADER  (sentence-aware chunking)
# ──────────────────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Remove PDF artefacts (headers, footers, hyphenation, extra spaces)."""
    # Remove page numbers (lone digits on a line)
    raw = re.sub(r"^\s*\d+\s*$", "", raw, flags=re.MULTILINE)
    # Re-join hyphenated line-breaks
    raw = re.sub(r"-\n(\S)", r"\1", raw)
    # Collapse multiple blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    # Collapse internal whitespace
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def _sentence_chunk(text: str, target_words: int = 250, overlap_sentences: int = 2) -> list:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Args:
        text          : Input text
        target_words  : Approximate words per chunk
        overlap_sentences : Number of sentences to overlap between chunks

    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    current_sentences = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        current_sentences.append(sentence)
        current_words += word_count

        if current_words >= target_words:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text.strip()) > 100:
                chunks.append(chunk_text)
            # Overlap: keep last N sentences for next chunk
            current_sentences = current_sentences[-overlap_sentences:]
            current_words = sum(len(s.split()) for s in current_sentences)

    # Flush remainder
    if current_sentences and current_words > 50:
        chunks.append(" ".join(current_sentences))

    return chunks


def load_from_pdf(pdf_path: str,
                  chunk_size: int = 250,
                  overlap: int = 2,
                  seed: int = 42) -> Dataset:
    """
    Load and chunk text from the Upinder Singh PDF for fine-tuning.

    Creates rich Q&A instruction-following samples using history-specific
    templates. Uses sentence-aware chunking so each sample is coherent.

    Args:
        pdf_path   : Path to the PDF file
        chunk_size : Target words per chunk
        overlap    : Overlap sentences between chunks
        seed       : Random seed for template shuffling
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    random.seed(seed)

    print(f"[DATA] Loading PDF: {pdf_path}")
    reader    = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    full_text = _clean_text(full_text)
    total_words = len(full_text.split())
    print(f"   Total pages : {len(reader.pages)}")
    print(f"   Total words : {total_words:,}")

    # Sentence-aware chunks
    chunks = _sentence_chunk(full_text, target_words=chunk_size, overlap_sentences=overlap)
    print(f"   Text chunks : {len(chunks)} (target={chunk_size}w, overlap={overlap} sentences)")

    # Build instruction-following samples
    samples = []
    template_order = list(range(len(HISTORY_QA_TEMPLATES)))

    for idx, chunk in enumerate(chunks):
        words = chunk.split()
        if len(words) < 60:          # skip very short chunks
            continue

        # Split chunk: first ~60% = context, last ~40% = completion
        split_idx  = int(len(words) * 0.60)
        context    = " ".join(words[:split_idx])
        completion = " ".join(words[split_idx:])

        if len(completion.split()) < 20:   # completion too short → skip
            continue

        # Pick a template in round-robin order (deterministic)
        tmpl_idx = idx % len(HISTORY_QA_TEMPLATES)
        instruction_tmpl, input_tmpl, output_tmpl = HISTORY_QA_TEMPLATES[tmpl_idx]

        samples.append({
            "instruction": instruction_tmpl,
            "input":       input_tmpl.format(context=context),
            "output":      output_tmpl.format(completion=completion),
        })

    print(f"   [OK] Created {len(samples)} training samples from PDF")
    return Dataset.from_list(samples)


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT FORMATTING  (Qwen 2.5 ChatML format)
# ──────────────────────────────────────────────────────────────────────────────

def format_prompt_qwen(example: dict) -> dict:
    """
    Format a training example using Qwen 2.5 ChatML format.

    <|im_start|>system
    You are a knowledgeable assistant specialised in ancient and early medieval Indian history.
    <|im_end|>
    <|im_start|>user
    {instruction}

    {input}
    <|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input",        "").strip()
    output      = example.get("output",       "").strip()

    system_msg = (
        "You are a knowledgeable assistant specialised in ancient and early "
        "medieval Indian history, trained on Upinder Singh's scholarship."
    )

    user_content = instruction
    if inp:
        user_content += f"\n\n{inp}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )

    return {"text": prompt}


def format_prompt_alpaca(example: dict) -> dict:
    """Fallback Alpaca-style prompt (used for non-Qwen models)."""
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input",        "").strip()
    output      = example.get("output",       "").strip()

    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": prompt}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def prepare_dataset(data_path: str,
                    val_split: float = 0.1,
                    seed: int = 42,
                    model_name: str = "qwen") -> tuple:
    """
    Load, format, and split a CSV or PDF dataset.

    Args:
        data_path  : Path to .csv or .pdf file
        val_split  : Fraction of data for validation
        seed       : Random seed
        model_name : Model family ('qwen' uses ChatML format, others use Alpaca)

    Returns:
        (train_dataset, val_dataset)
    """
    ext = Path(data_path).suffix.lower()

    if ext == ".csv":
        dataset = load_from_csv(data_path)
    elif ext == ".pdf":
        dataset = load_from_pdf(data_path, seed=seed)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .pdf")

    # Choose prompt format based on model family
    if "qwen" in model_name.lower():
        fmt_fn = format_prompt_qwen
    else:
        fmt_fn = format_prompt_alpaca

    dataset = dataset.map(fmt_fn)

    split    = dataset.train_test_split(test_size=val_split, seed=seed)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"\n[DATA] Dataset split:")
    print(f"   Train : {len(train_ds)} samples")
    print(f"   Val   : {len(val_ds)} samples")
    print(f"\n[SAMPLE] First formatted prompt (truncated):")
    print("-" * 60)
    print(train_ds[0]["text"][:400] + "...")
    print("-" * 60)

    return train_ds, val_ds


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    train, val = prepare_dataset(path, model_name="qwen")
    print(f"\n[OK] Data ready: {len(train) + len(val)} total samples")
