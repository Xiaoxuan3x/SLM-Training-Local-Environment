from __future__ import annotations

"""Lightweight zero-shot scope classifier for the UK mortgage assistant.

Uses an NLI-based zero-shot-classification pipeline to decide whether a user
question is within the UK mortgage domain before it reaches the SLM.

Model: typeform/distilbert-base-uncased-mnli  (~256 MB, 6 layers)
This is roughly 6× smaller than facebook/bart-large-mnli while still
providing reliable zero-shot classification for domain gating.
"""

import argparse
import sys

from transformers import pipeline

CLASSIFIER_MODEL = "typeform/distilbert-base-uncased-mnli"
_IN_SCOPE_LABEL = "UK mortgage question"
_OUT_OF_SCOPE_LABEL = "unrelated question"
_CANDIDATE_LABELS = [_IN_SCOPE_LABEL, _OUT_OF_SCOPE_LABEL]
DEFAULT_THRESHOLD = 0.60

REFUSAL_MESSAGE = (
    "This question is outside my area of expertise. I am a UK mortgage assistant "
    "and can only help with questions about mortgage products, interest rates, LTV, "
    "fees, eligibility, and related topics."
)


class ScopeClassifier:
    """Zero-shot NLI gate that rejects questions outside the UK mortgage domain."""

    def __init__(
        self,
        model: str = CLASSIFIER_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        local_files_only: bool = False,
    ) -> None:
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model,
            local_files_only=local_files_only,
        )
        self.threshold = threshold

    def is_in_scope(self, text: str) -> bool:
        """Return True if text is a UK mortgage question."""
        result = self._pipe(text, candidate_labels=_CANDIDATE_LABELS)
        scores = dict(zip(result["labels"], result["scores"]))
        return scores.get(_IN_SCOPE_LABEL, 0.0) >= self.threshold

    def check(self, text: str) -> tuple[bool, str]:
        """Return (in_scope, message).

        If in_scope is True, message is empty — proceed to the SLM.
        If in_scope is False, message is the standard refusal reply.
        """
        if self.is_in_scope(text):
            return True, ""
        return False, REFUSAL_MESSAGE


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Test the scope classifier against a single question."
    )
    parser.add_argument("question", help="User question to classify.")
    parser.add_argument("--model", default=CLASSIFIER_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face model downloads.",
    )
    args = parser.parse_args()

    clf = ScopeClassifier(
        model=args.model,
        threshold=args.threshold,
        local_files_only=not args.allow_downloads,
    )
    in_scope, msg = clf.check(args.question)
    if in_scope:
        print("IN SCOPE — forwarding to SLM.")
    else:
        print(f"OUT OF SCOPE — {msg}")
    sys.exit(0 if in_scope else 1)


if __name__ == "__main__":
    _cli()
