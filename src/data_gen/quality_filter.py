from __future__ import annotations

import asyncio
import json

from llm_client import LLMClient

_JUDGE_SYSTEM = """You are a quality evaluator for UK mortgage training data.
Score each instruction-response pair on four criteria. Return only valid JSON — no explanation."""

_JUDGE_PROMPT = """Evaluate this training example for a UK mortgage language model.

INSTRUCTION: {instruction}

INPUT (mortgage product data):
{input}

OUTPUT (model response):
{output}

Score each criterion 1–5:
1. factual_accuracy  — Does the output correctly reference figures from the product data?
2. financial_reasoning — Is the financial advice or analysis sound and realistic for the UK market?
3. completeness — Does the output fully address what the instruction asks?
4. no_hallucination — Does the output avoid inventing rates, fees, or conditions not in the input?

Return JSON only:
{{"factual_accuracy": <1-5>, "financial_reasoning": <1-5>, "completeness": <1-5>, "no_hallucination": <1-5>}}"""


async def score_example(
    example: dict[str, str],
    client: LLMClient,
    max_tokens: int,
    temperature: float,
    max_retries: int = 2,
) -> dict[str, int] | None:
    prompt = _JUDGE_PROMPT.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    )

    for attempt in range(max_retries):
        try:
            text = await client.complete(
                system=_JUDGE_SYSTEM,
                user=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    return None


def passes_filter(scores: dict[str, int], min_score: int) -> bool:
    return all(v >= min_score for v in scores.values())
