from __future__ import annotations

"""Run text generation from a local merged Hugging Face model."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = """### Instruction:
Summarise the mortgage product and highlight the main lending terms.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included.

### Response:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation from a local merged model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/exports/base-run-merged",
        help="Path to the merged Hugging Face model folder.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to send to the model.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a text file containing the prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--repetition-penalty", type=float, default=1.08, help="Penalty for repeated text.")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads if model files are missing locally.",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt or DEFAULT_PROMPT


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    prompt = read_prompt(args)
    local_files_only = not args.allow_downloads
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_length": input_length + args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
    print(completion)


if __name__ == "__main__":
    main()
