from __future__ import annotations

"""Compare eval loss for a base model and the same model with a LoRA adapter."""

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from train_slm import Config, load_config, prepare_dataset


def get_quantization_config(cfg: Config) -> BitsAndBytesConfig | None:
    quantization = cfg.quantization
    if not quantization.get("load_in_4bit"):
        return None

    torch_dtype = torch.bfloat16 if quantization.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch_dtype,
    )


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


def make_tokenizer(source: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str, cfg: Config) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=get_quantization_config(cfg),
    )


def get_eval_batch_size(cfg: Config, batch_size: int | None) -> int:
    return batch_size or cfg.training.get("micro_batch_size", 1)


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Any,
    batch_size: int,
) -> float:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    device = get_input_device(model)

    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            current_batch_size = batch["input_ids"].shape[0]
            total_loss += float(outputs.loss.detach().cpu()) * current_batch_size
            total_examples += current_batch_size

    if total_examples == 0:
        raise ValueError("Evaluation dataset is empty")
    return total_loss / total_examples


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def perplexity(eval_loss: float) -> float:
    return math.exp(eval_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base-model eval loss with LoRA-adapter eval loss")
    parser.add_argument("--config", type=str, default="configs/default_training.yaml", help="Path to the training config")
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="Path to the saved LoRA adapter. Defaults to output_dir from the config.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to evaluate. Defaults to adapter_config.json, then model_name from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/eval-loss-comparison",
        help="Temporary output directory used by Hugging Face Trainer during evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    adapter_dir = args.adapter_dir or cfg.output_dir
    adapter_base_model = get_adapter_base_model(adapter_dir)
    base_model_name = args.base_model or adapter_base_model or cfg.model_name

    if not Path(adapter_dir).exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    if adapter_base_model and adapter_base_model != base_model_name:
        print(f"Warning: adapter was trained from {adapter_base_model}, but evaluating base model {base_model_name}")

    tokenizer_source = adapter_dir if has_saved_tokenizer(adapter_dir) else base_model_name
    tokenizer = make_tokenizer(tokenizer_source)

    # Use the same deterministic train/eval split as training.
    eval_cfg = Config(raw=copy.deepcopy(cfg.raw))
    _, eval_dataset = prepare_dataset(eval_cfg, tokenizer)
    eval_batch_size = get_eval_batch_size(cfg, args.batch_size)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Tokenizer: {tokenizer_source}")
    print(f"Eval examples: {len(eval_dataset)}")
    print(f"Eval batch size: {eval_batch_size}")
    print()

    base_model = load_base_model(base_model_name, cfg)
    base_loss = evaluate_model(base_model, tokenizer, eval_dataset, eval_batch_size)
    cleanup_model(base_model)

    adapter_base = load_base_model(base_model_name, cfg)
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_dir)
    adapter_loss = evaluate_model(adapter_model, tokenizer, eval_dataset, eval_batch_size)
    cleanup_model(adapter_model)

    delta = adapter_loss - base_loss
    print("Eval loss comparison")
    print(f"Base model eval_loss:       {base_loss:.4f} | perplexity: {perplexity(base_loss):.2f}")
    print(f"LoRA adapter eval_loss:     {adapter_loss:.4f} | perplexity: {perplexity(adapter_loss):.2f}")
    print(f"Delta adapter - base:       {delta:.4f}")
    print()

    if delta < 0:
        print("Result: the LoRA adapter improved eval loss on this eval split.")
    elif delta > 0:
        print("Result: the LoRA adapter made eval loss worse on this eval split.")
    else:
        print("Result: both models have the same eval loss on this eval split.")


if __name__ == "__main__":
    main()
