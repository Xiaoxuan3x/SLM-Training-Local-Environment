from __future__ import annotations

"""Run text generation from a local merged Hugging Face model."""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a UK mortgage assistant. Only answer questions about UK mortgage "
    "products, interest rates, LTV, fees, eligibility, and related topics. "
    "For any other question, respond that the question is outside your knowledge scope."
)

DEFAULT_PROMPT = """### System:
You are a UK mortgage assistant. Only answer questions about UK mortgage products, interest rates, LTV, fees, eligibility, and related topics. For any other question, respond that the question is outside your knowledge scope.

### Instruction:
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
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="System prompt prepended to every prompt. Pass empty string to disable.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum number of generated tokens before generation is truncated.",
    )
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
    if args.system_prompt and (args.prompt or args.prompt_file):
        prompt = f"### System:\n{args.system_prompt.strip()}\n\n{prompt}"
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
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }
    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        t0 = time.perf_counter()
        outputs = model.generate(**inputs, **generate_kwargs)
        elapsed = time.perf_counter() - t0

    generated_ids = outputs.sequences[0]
    new_token_count = generated_ids.shape[-1] - inputs["input_ids"].shape[-1]
    stopped_by_max_tokens = (
        new_token_count >= args.max_new_tokens and generated_ids[-1].item() != tokenizer.eos_token_id
    )

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
    print(completion)
    print(f"\n[timing] {elapsed:.2f}s | {new_token_count} tokens | {new_token_count / elapsed:.1f} tok/s", file=sys.stderr)
    if stopped_by_max_tokens:
        print(
            f"\n[warning] Generation reached the max token limit ({args.max_new_tokens}) before EOS. "
            "Increase --max-new-tokens for longer answers.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
