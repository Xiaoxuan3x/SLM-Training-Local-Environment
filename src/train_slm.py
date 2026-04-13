"""Simple entry point to fine-tune a small language model"""
"""This train script focuses on the most practical techniques: LoRA for efficient adaptation, domain-specific fine-tuning, and 4-bit quantization for deployment efficiency."""
from __future__ import annotations

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
    DataCollatorForLanguageModeling,
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


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data)


def build_prompt(example: Dict[str, Any], fields: Dict[str, str]) -> str:
    instruction = example.get(fields["instruction"], "").strip()
    input_text = example.get(fields.get("input", ""), "").strip()
    output = example.get(fields["output"], "").strip()
    prompt = "### Instruction:\n" + instruction + "\n\n"
    if input_text:
        prompt += "### Input:\n" + input_text + "\n\n"
    prompt += "### Response:\n" + output
    return prompt


def tokenize(example: Dict[str, Any], tokenizer: AutoTokenizer, fields: Dict[str, str], max_length: int) -> Dict[str, Any]:
    prompt = build_prompt(example, fields)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(cfg: Config, tokenizer: AutoTokenizer):
    # Domain-specific fine-tuning: load the JSONL data the user curated so the
    # model learns the target terminology/workflows instead of generic corpora.
    dataset = load_dataset("json", data_files=cfg.dataset_path)["train"]
    if cfg.max_samples:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
    dataset = dataset.shuffle(seed=cfg.seed)
    split = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    preprocess = lambda example: tokenize(example, tokenizer, cfg.field_names, cfg.max_seq_length)  # noqa: E731
    tokenized_train = split["train"].map(preprocess, remove_columns=split["train"].column_names)
    tokenized_eval = split["test"].map(preprocess, remove_columns=split["test"].column_names)
    return tokenized_train, tokenized_eval


def load_model(cfg: Config):
    quantization = cfg.quantization
    bnb_config = None
    torch_dtype = torch.bfloat16 if quantization.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    if quantization.get("load_in_4bit"):
        # Quantisation: load the base model with 4-bit weights via bitsandbytes to
        # reduce VRAM usage for local experimentation.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
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
    tokenizer = make_tokenizer(cfg)
    train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)
    model = load_model(cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training["epochs"],
        per_device_train_batch_size=cfg.training["micro_batch_size"],
        gradient_accumulation_steps=max(1, cfg.training["batch_size"] // cfg.training["micro_batch_size"]),
        learning_rate=cfg.training["learning_rate"],
        warmup_ratio=cfg.training.get("warmup_ratio", 0.03),
        logging_steps=cfg.training.get("logging_steps", 10),
        evaluation_strategy="steps",
        eval_steps=cfg.training.get("eval_steps", 50),
        save_steps=cfg.training.get("save_steps", 100),
        fp16=cfg.training.get("fp16", True),
        gradient_checkpointing=cfg.training.get("gradient_checkpointing", False),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
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
