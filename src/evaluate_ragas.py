from __future__ import annotations

"""RAGAS evaluation for the fine-tuned mortgage SLM.

Uses a local Ollama model as the LLM judge — no API key or cost required.
Ollama must be running locally with the judge model already pulled:
    ollama pull llama3.1

Data mapping (mortgage JSONL → RAGAS schema):
    instruction  → question
    input        → contexts   (product details act as the retrieved context)
    output       → ground_truth
    model output → answer

Metrics:
    Faithfulness       – does the answer stay within the product details? (hallucination check)
    AnswerRelevancy    – does the answer address the question asked?
    FactualCorrectness – accuracy vs ground truth
    SemanticSimilarity – semantic closeness to the reference answer

Usage:
    python src/evaluate_ragas.py
    python src/evaluate_ragas.py --max-samples 20
    python src/evaluate_ragas.py --judge-model mistral
    python src/evaluate_ragas.py --output artifacts/ragas_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a UK mortgage assistant. Only answer questions about UK mortgage "
    "products, interest rates, LTV, fees, eligibility, and related topics. "
    "For any other question, respond that the question is outside your knowledge scope."
)

DEFAULT_MODEL_DIR = "artifacts/exports/base-run-merged"
DEFAULT_DATA_PATH = "data/synthetic_mortgage_dataset.jsonl"
DEFAULT_JUDGE_MODEL = "llama3.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the mortgage SLM with RAGAS metrics via local Ollama judge")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to merged HF model folder")
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="JSONL eval dataset path")
    p.add_argument("--max-samples", type=int, default=50, help="Number of examples to evaluate (default: 50)")
    p.add_argument("--max-new-tokens", type=int, default=400, help="Max tokens to generate per answer")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Ollama model to use as LLM judge")
    p.add_argument("--output", default=None, help="Optional path to save JSON results")
    p.add_argument("--allow-downloads", action="store_true", help="Allow HF Hub downloads if files are missing")
    return p.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(path: str, max_samples: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def build_prompt(instruction: str, input_text: str, tokenizer: AutoTokenizer) -> str:
    user_content = instruction
    if input_text:
        user_content += "\n\n" + input_text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_new_tokens: int,
    device: str,
) -> list[str]:
    answers: list[str] = []
    for i, ex in enumerate(examples, 1):
        prompt = build_prompt(ex["instruction"], ex["input"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_token_ids = out[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        answers.append(completion)
        print(f"  [{i}/{len(examples)}] {len(completion.split())} words", file=sys.stderr)
    return answers


def build_ragas_dataset(examples: list[dict], answers: list[str]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=ex["instruction"],
            retrieved_contexts=[ex["input"]],  # product details act as the retrieved context
            reference=ex["output"],
            response=answer,
        )
        for ex, answer in zip(examples, answers)
    ]
    return EvaluationDataset(samples=samples)


def configure_metrics(judge_model: str) -> list:
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    llm = llm_factory(judge_model, client=ollama_client)
    embeddings = embedding_factory("openai", model=judge_model, client=ollama_client)

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        FactualCorrectness(llm=llm),
        SemanticSimilarity(embeddings=embeddings),
    ]


def main() -> None:
    args = parse_args()
    local_files_only = not args.allow_downloads

    print(f"Data              : {args.data} (max {args.max_samples})")
    examples = load_examples(args.data, args.max_samples)
    print(f"Examples loaded   : {len(examples)}")

    device = get_device()
    print(f"Device            : {device}")
    print(f"SLM               : {args.model_dir}")
    print(f"Judge             : Ollama / {args.judge_model}")

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

    print(f"\nGenerating answers ({len(examples)} examples)...")
    t0 = time.perf_counter()
    answers = generate_answers(model, tokenizer, examples, args.max_new_tokens, device)
    print(f"Done in {time.perf_counter() - t0:.1f}s\n")

    ragas_dataset = build_ragas_dataset(examples, answers)
    metrics = configure_metrics(args.judge_model)

    print("Running RAGAS evaluation...")
    results = evaluate(ragas_dataset, metrics=metrics)

    scores: dict[str, float] = results.to_pandas().mean(numeric_only=True).to_dict()

    print("\n=== RAGAS Results ===")
    for name, score in scores.items():
        print(f"  {name:<28} {score:.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(scores, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
