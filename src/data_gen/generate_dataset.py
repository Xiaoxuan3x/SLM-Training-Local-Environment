from __future__ import annotations

"""
Synthetic mortgage dataset generator.

Phase 1: Sample realistic UK mortgage product records (no AI, template-based).
Phase 2: Generate diverse Q&A pairs per product using a language model.
Phase 3: Filter pairs with an LLM-as-judge using domain-specific criteria.
Output: JSONL in the same {instruction, input, output} format as the existing dataset.

Supported providers (set in configs/data_generation.yaml):
  groq      — free tier, Llama 3.3 70B.  Export GROQ_API_KEY before running.
  anthropic — paid, Claude Sonnet/Haiku.  Export ANTHROPIC_API_KEY before running.

Usage (run from repo root):
    # Groq (free)
    GROQ_API_KEY=<key> python src/data_gen/generate_dataset.py

    # Anthropic (paid)
    ANTHROPIC_API_KEY=<key> python src/data_gen/generate_dataset.py

    # Custom config or flags
    GROQ_API_KEY=<key> python src/data_gen/generate_dataset.py --config configs/data_generation.yaml
    GROQ_API_KEY=<key> python src/data_gen/generate_dataset.py --skip-filter
    python src/data_gen/generate_dataset.py --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import build_client, api_key_env
from product_sampler import sample_products
from qa_generator import generate_qa_pairs
from quality_filter import score_example, passes_filter


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _progress(current: int, total: int, label: str = "") -> None:
    pct = current / total * 100
    bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
    print(f"\r[{bar}] {pct:5.1f}%  {label}", end="", flush=True)


def _resolve_api_key(provider: str) -> str:
    env_var = api_key_env(provider)
    key = os.environ.get(env_var)
    if not key:
        sys.exit(f"Error: {env_var} environment variable is not set.")
    return key


def run(config: dict, skip_filter: bool, dry_run: bool) -> None:
    pipeline_cfg = config["pipeline"]
    sampler_cfg  = config["product_sampler"]
    gen_cfg      = config["qa_generator"]
    filter_cfg   = config["quality_filter"]
    provider     = config["provider"]["name"]

    output_path = Path(pipeline_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Phase 1 — product sampling (no API)                                  #
    # ------------------------------------------------------------------ #
    print("Phase 1: Sampling mortgage product records...")
    products = sample_products(
        num_products=sampler_cfg["num_products"],
        base_rate=sampler_cfg["base_rate"],
        report_date=sampler_cfg["report_date"],
        seed=pipeline_cfg["seed"],
    )
    print(f"  Sampled {len(products)} products.\n")

    if dry_run:
        print("Dry run: printing first 2 products and exiting.")
        for p in products[:2]:
            print(p.to_input_text())
            print("---")
        return

    api_key = _resolve_api_key(provider)
    gen_client    = build_client(provider, api_key, gen_cfg["model"])
    filter_client = build_client(provider, api_key, filter_cfg["model"])

    print(f"Provider: {provider}  |  "
          f"Gen model: {gen_cfg['model']}  |  "
          f"Filter model: {filter_cfg['model']}\n")

    # ------------------------------------------------------------------ #
    # Phase 2 — Q&A generation                                            #
    # ------------------------------------------------------------------ #
    print("Phase 2: Generating Q&A pairs...")
    rng = random.Random(pipeline_cfg["seed"])
    all_pairs: list[dict[str, str]] = []

    for i, product in enumerate(products):
        _progress(i + 1, len(products), f"product {i+1}/{len(products)}")
        pairs = generate_qa_pairs(
            product=product,
            client=gen_client,
            questions_per_product=gen_cfg["questions_per_product"],
            question_types=gen_cfg["question_types"],
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"],
            rng=rng,
        )
        all_pairs.extend(pairs)
        time.sleep(gen_client.inter_request_sleep)

    print(f"\n  Generated {len(all_pairs)} raw pairs.\n")

    # ------------------------------------------------------------------ #
    # Phase 3 — quality filtering                                         #
    # ------------------------------------------------------------------ #
    if skip_filter:
        final_pairs = all_pairs
        print(f"Filter skipped. Keeping all {len(final_pairs)} pairs.\n")
    else:
        print("Phase 3: Filtering with LLM-as-judge...")
        final_pairs = []
        rejected = 0

        for i, pair in enumerate(all_pairs):
            _progress(i + 1, len(all_pairs), f"pair {i+1}/{len(all_pairs)}")
            scores = score_example(
                example=pair,
                client=filter_client,
                max_tokens=filter_cfg["max_tokens"],
                temperature=filter_cfg["temperature"],
            )
            if scores is None:
                rejected += 1
                continue
            if passes_filter(scores, min_score=filter_cfg["min_score"]):
                final_pairs.append(pair)
            else:
                rejected += 1
            time.sleep(filter_client.inter_request_sleep)

        print(f"\n  Kept {len(final_pairs)} / {len(all_pairs)} pairs "
              f"(rejected {rejected}).\n")

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    with open(output_path, "w") as f:
        for pair in final_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(final_pairs)} examples to {output_path}")
    _print_summary(final_pairs)


def _print_summary(pairs: list[dict[str, str]]) -> None:
    if not pairs:
        return
    print("\nSample instructions:")
    for pair in pairs[:5]:
        print(f"  - {pair['instruction'][:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic mortgage training data.")
    parser.add_argument(
        "--config",
        default="configs/data_generation.yaml",
        help="Path to data generation config (default: configs/data_generation.yaml)",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip Phase 3 quality filtering (faster, lower quality)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample products only, no API calls",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, skip_filter=args.skip_filter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
