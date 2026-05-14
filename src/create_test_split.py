from __future__ import annotations

"""Split a JSONL dataset into a training file and a held-out test file.

The test split is written once and must never be loaded during training.
Both output files use the same {instruction, input, output} schema as the source.

Usage:
    python src/create_test_split.py
    python src/create_test_split.py --test-ratio 0.1 --seed 42
    python src/create_test_split.py --source data/synthetic_mortgage_dataset.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split JSONL dataset into train and held-out test files")
    p.add_argument("--source", default="data/synthetic_mortgage_dataset.jsonl")
    p.add_argument("--train-out", default="data/train_dataset.jsonl")
    p.add_argument("--test-out", default="data/test_dataset.jsonl")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Fraction reserved for test (default: 0.1)")
    p.add_argument("--seed", type=int, default=99, help="Shuffle seed — use a value different from the training seed (42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)

    if test_out.exists():
        print(f"WARNING: {test_out} already exists.")
        print("Re-running this script changes the test set, which invalidates past scores.")
        answer = input("Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    rows = load_jsonl(source)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_test = max(1, round(len(rows) * args.test_ratio))
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]

    write_jsonl(train_out, train_rows)
    write_jsonl(test_out, test_rows)

    print(f"Source  : {source} ({len(rows)} examples)")
    print(f"Train   : {train_out} ({len(train_rows)} examples)")
    print(f"Test    : {test_out} ({len(test_rows)} examples)  ← never load this during training")


if __name__ == "__main__":
    main()
