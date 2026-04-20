from __future__ import annotations

"""Export a LoRA adapter as a merged standalone Hugging Face model."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_slm import load_config


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_adapter_base_model(adapter_dir: str) -> str | None:
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    with adapter_config_path.open("r", encoding="utf-8") as fh:
        adapter_config = json.load(fh)
    return adapter_config.get("base_model_name_or_path")


def has_saved_tokenizer(adapter_dir: str) -> bool:
    adapter_path = Path(adapter_dir)
    return (adapter_path / "tokenizer_config.json").exists() or (adapter_path / "tokenizer.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a saved LoRA adapter into its base model")
    parser.add_argument("--config", type=str, default="configs/default_training.yaml", help="Path to the training config")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="Path to the saved LoRA adapter. Defaults to output_dir from the config.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="artifacts/exports/base-run-merged",
        help="Directory where the merged standalone model will be saved.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to merge into. Defaults to adapter_config.json, then model_name from the config.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help="Precision to load and save the merged model with.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files; do not attempt downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    adapter_dir = args.adapter_dir or cfg.output_dir
    adapter_path = Path(adapter_dir)
    export_path = Path(args.export_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    adapter_base_model = get_adapter_base_model(adapter_dir)
    base_model_name = args.base_model or adapter_base_model or cfg.model_name
    tokenizer_source = adapter_dir if has_saved_tokenizer(adapter_dir) else base_model_name

    print(f"Config: {args.config}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Tokenizer: {tokenizer_source}")
    print(f"Export dir: {args.export_dir}")
    print(f"Dtype: {args.dtype}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=DTYPES[args.dtype],
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    adapter_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=args.local_files_only,
    )

    merged_model = adapter_model.merge_and_unload()
    export_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(export_path, safe_serialization=True)
    tokenizer.save_pretrained(export_path)

    print(f"Saved merged standalone model to {export_path}")


if __name__ == "__main__":
    main()
