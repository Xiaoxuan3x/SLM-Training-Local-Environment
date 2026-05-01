from __future__ import annotations

"""Simple entry point to fine-tune a small language model"""
"""This train script focuses on the most practical techniques:
LoRA for efficient adaptation,
domain-specific fine-tuning, and
4-bit quantization for deployment efficiency."""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def model_name(self) -> str:
        return self.raw["model_name"]

    @property
    def dataset_path(self) -> str:
        return self.raw["dataset"]["path"]

    @property
    def field_names(self) -> Dict[str, str]:
        return self.raw["dataset"]["field_names"]

    @property
    def max_samples(self) -> int | None:
        return self.raw["dataset"].get("max_samples")

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def lora(self) -> Dict[str, Any]:
        return self.raw.get("lora", {})

    @property
    def quantization(self) -> Dict[str, Any]:
        return self.raw.get("quantization", {})

    @property
    def max_seq_length(self) -> int:
        return self.raw.get("max_seq_length", 512)

    @property
    def output_dir(self) -> str:
        return self.raw["output_dir"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def system_prompt(self) -> str | None:
        return self.raw.get("system_prompt")

    @property
    def boundary_data_path(self) -> str | None:
        return self.raw.get("boundary_data", {}).get("path")

    @property
    def boundary_ratio(self) -> float:
        return float(self.raw.get("boundary_data", {}).get("ratio", 0.2))

    @property
    def boundary_repeat(self) -> int:
        return int(self.raw.get("boundary_data", {}).get("repeat", 1))


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data)


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(example: Dict[str, Any], tokenizer: AutoTokenizer, fields: Dict[str, str], max_length: int, system_prompt: str | None = None) -> Dict[str, Any]:
    instruction = example.get(fields["instruction"], "").strip()
    input_text = example.get(fields.get("input", ""), "").strip()
    output = example.get(fields["output"], "").strip()

    user_content = instruction
    if input_text:
        user_content += "\n\n" + input_text

    messages: list[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    # Prefix up to (and including) the assistant turn opener — used only for masking.
    prefix_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_tensors=None,
    )

    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": output}],
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    # Mask all prefix tokens so loss is only computed on the response.
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    labels = labels[: len(full_ids)]

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def prepare_dataset(cfg: Config, tokenizer: AutoTokenizer):
    from datasets import concatenate_datasets

    dataset = load_dataset("json", data_files=cfg.dataset_path)["train"]
    if cfg.max_samples:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

    if cfg.boundary_data_path:
        boundary = load_dataset("json", data_files=cfg.boundary_data_path)["train"]
        n_boundary = min(int(len(dataset) * cfg.boundary_ratio), len(boundary))
        if n_boundary > 0:
            boundary = boundary.shuffle(seed=cfg.seed).select(range(n_boundary))
            repeated = concatenate_datasets([boundary] * cfg.boundary_repeat)
            dataset = concatenate_datasets([dataset, repeated])

    dataset = dataset.shuffle(seed=cfg.seed)
    split = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    preprocess = lambda example: tokenize(example, tokenizer, cfg.field_names, cfg.max_seq_length, cfg.system_prompt)  # noqa: E731
    tokenized_train = split["train"].map(preprocess, remove_columns=split["train"].column_names)
    tokenized_eval = split["test"].map(preprocess, remove_columns=split["test"].column_names)
    return tokenized_train, tokenized_eval


def load_model(cfg: Config, device: torch.device):
    quantization = cfg.quantization
    bnb_config = None
    torch_dtype = torch.bfloat16 if quantization.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    if quantization.get("load_in_4bit"):
        if device.type != "cuda":
            raise ValueError("4-bit quantization in this training script requires CUDA. Set quantization.load_in_4bit to false for MPS or CPU training.")
        # Quantisation: load the base model with 4-bit weights via bitsandbytes to
        # reduce VRAM usage for local experimentation.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=bnb_config,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    # LoRA / efficient fine-tuning: inject small trainable adapters instead of
    # updating all base weights so experiments remain lightweight.
    lora_cfg = LoraConfig(
        r=cfg.lora.get("r", 8),
        lora_alpha=cfg.lora.get("alpha", 16),
        target_modules=cfg.lora.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=cfg.lora.get("dropout", 0.05),
        bias=cfg.lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model


def make_tokenizer(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def train(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = get_training_device()
    if device.type == "mps":
        print("Training on Apple Metal (mps).")
    elif device.type == "cuda":
        print("Training on CUDA GPU.")
    else:
        print("Training on CPU.")
    tokenizer = make_tokenizer(cfg)
    train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)
    model = load_model(cfg, device)
    fp16 = bool(cfg.training.get("fp16", True) and device.type == "cuda")
    bf16 = bool(cfg.training.get("bf16", False) and device.type == "cuda")

    # Dynamic padding: pad each batch to its longest sequence instead of always
    # padding to max_seq_length, which wastes compute on short samples.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training["epochs"],
        per_device_train_batch_size=cfg.training["micro_batch_size"],
        gradient_accumulation_steps=max(1, cfg.training["batch_size"] // cfg.training["micro_batch_size"]),
        learning_rate=cfg.training["learning_rate"],
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        logging_steps=cfg.training.get("logging_steps", 10),
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=cfg.training.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=cfg.training.get("save_steps", 100),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        dataloader_pin_memory=device.type == "cuda",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a small language model with LoRA")
    parser.add_argument("--config", type=str, default="configs/default_training.yaml", help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
