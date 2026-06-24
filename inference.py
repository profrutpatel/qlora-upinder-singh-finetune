"""
Inference script for the QLoRA fine-tuned model.
Loads the saved LoRA adapter and generates responses.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_finetuned_model(base_model_name: str, adapter_path: str, device: str = None):
    """
    Load the fine-tuned model with LoRA adapter.
    
    Args:
        base_model_name: Original HuggingFace model name
        adapter_path: Path to the saved LoRA adapter directory
        device: 'cuda', 'mps', or 'cpu'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📦 Loading base model: {base_model_name}")
    print(f"📎 Loading LoRA adapter: {adapter_path}")
    print(f"💻 Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    if device != "cuda":
        model = model.to(device)
    
    print("✅ Fine-tuned model loaded successfully\n")
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    device: str = "cpu",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate a response for a given instruction.
    
    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        instruction: Task instruction
        input_text: Optional additional input context
        device: Compute device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling probability
        do_sample: Whether to sample (True) or greedy decode (False)
    
    Returns:
        Generated response string
    """
    # Format prompt in Alpaca style (matching training format)
    if input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode only the generated tokens (not the input prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Clean up response (stop at next "###" if model over-generates)
    if "###" in response:
        response = response.split("###")[0]
    
    return response.strip()


def interactive_chat(model, tokenizer, device: str):
    """Run an interactive Q&A session with the fine-tuned model."""
    print("\n" + "=" * 60)
    print("  💬 Interactive Chat with Fine-tuned Model")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        instruction = input("📝 Instruction: ").strip()
        
        if instruction.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        
        if not instruction:
            continue
        
        inp = input("📋 Input (optional, press Enter to skip): ").strip()
        
        print("\n🤖 Generating response...")
        response = generate_response(
            model, tokenizer, instruction, inp, device=device
        )
        
        print(f"\n💡 Response:\n{response}")
        print("\n" + "-" * 60 + "\n")


def batch_inference(model, tokenizer, device: str, examples: list) -> list:
    """Run batch inference on a list of examples."""
    results = []
    for i, ex in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] Processing: {ex['instruction'][:60]}...")
        response = generate_response(
            model, tokenizer,
            ex.get("instruction", ""),
            ex.get("input", ""),
            device=device,
        )
        results.append({**ex, "response": response})
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with QLoRA fine-tuned model")
    parser.add_argument("--base-model",   default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model name on HuggingFace")
    parser.add_argument("--adapter-path", default="./qlora_history/final_adapter",
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--merged-path",  default=None,
                        help="Path to merged model (if available, skips adapter loading)")
    parser.add_argument("--interactive",  action="store_true",
                        help="Run interactive chat mode")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.merged_path:
        # Load fully merged model (no adapter needed)
        print(f"📦 Loading merged model from: {args.merged_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.merged_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
    else:
        model, tokenizer, device = load_finetuned_model(
            args.base_model, args.adapter_path, device
        )
    
    if args.interactive:
        interactive_chat(model, tokenizer, device)
    else:
        # Demo batch inference – history Q&A prompts
        demo_examples = [
            {
                "instruction": "Who was Ashoka and why is he significant in Indian history?",
                "input": "",
            },
            {
                "instruction": "Describe the social structure of the Mauryan Empire.",
                "input": "",
            },
            {
                "instruction": "Explain the historical significance of the following passage about ancient India:",
                "input": (
                    "The Indus Valley Civilisation, also known as the Harappan civilisation, "
                    "flourished between c. 2600 and 1900 BCE and was one of the earliest urban "
                    "cultures in the ancient world, with cities like Mohenjo-daro and Harappa."
                ),
            },
            {
                "instruction": "Summarise the religious developments in early medieval India.",
                "input": "",
            },
        ]

        print("\n[TEST] Running demo inference on history Q&A prompts...\n")
        results = batch_inference(model, tokenizer, device, demo_examples)

        print("\n" + "=" * 60)
        print("  Inference Results – Upinder Singh Fine-tuned Model")
        print("=" * 60)
        for r in results:
            print(f"\nInstruction : {r['instruction']}")
            if r['input']:
                print(f"Input       : {r['input'][:120]}...")
            print(f"Response    : {r['response']}")
            print("-" * 60)
