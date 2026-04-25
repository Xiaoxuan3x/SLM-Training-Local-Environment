from __future__ import annotations

import asyncio
import json

from llm_client import LLMClient
from product_sampler import MortgageProduct

_QUESTION_TYPE_INSTRUCTIONS = {
    "summarisation": (
        "Write an instruction asking the reader to summarise the mortgage product "
        "and highlight the key lending terms."
    ),
    "eligibility": (
        "Write an instruction asking whether this mortgage is suitable for a specific "
        "borrower profile. Invent a realistic borrower scenario (e.g. first-time buyer "
        "with 10% deposit, home mover with 40% equity, self-employed contractor)."
    ),
    "rate_analysis": (
        "Write an instruction asking the reader to analyse whether the interest rate "
        "is competitive given the current base rate."
    ),
    "fee_tradeoff": (
        "Write an instruction asking the reader to reason about whether the booking fee "
        "is worth paying. Assume a specific loan size (e.g. £200,000, £350,000)."
    ),
    "risk_suitability": (
        "Write an instruction asking the reader to identify the main risks of this product "
        "for a particular type of borrower or in a specific economic scenario."
    ),
    "comparison": (
        "Write an instruction asking the reader to compare this product against an "
        "alternative mortgage type (e.g. tracker vs fixed, 2yr vs 5yr, fee vs fee-free)."
    ),
    "factual_correctness": (
        "Write a True/False instruction that makes a specific factual claim about the "
        "product (e.g. 'True or false: this product has no upfront costs'). "
        "Mix correct and incorrect claims."
    ),
    "terminology": (
        "Write an instruction asking the reader to explain a specific mortgage term "
        "that appears in this product record (e.g. APRC, LTV, fixed rate, booking fee)."
    ),
    "borrower_advice": (
        "Write an instruction asking what a borrower should check or consider before "
        "committing to this mortgage product."
    ),
}

_SYSTEM_PROMPT = """You are a UK mortgage expert with deep knowledge of the residential lending market.
Your task is to generate instruction-response training pairs about UK mortgage products.
All responses must be accurate, grounded in the product data provided, and reflect real UK market conventions.
Do not invent interest rates or fees not present in the product record.
Return only valid JSON — no markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """Here is a UK mortgage product record:

{product_text}

Generate {n} instruction-response training pairs about this product.
Each pair must cover a DIFFERENT question type from this list:

{type_instructions}

Rules:
- "instruction": a natural, varied question or task directive (do not always start with the same verb)
- "input": copy the product record EXACTLY as provided above, word for word
- "output": a thorough, accurate answer grounded only in the product data

Return a JSON array with exactly {n} objects, each with keys: "instruction", "input", "output".
Example format:
[
  {{"instruction": "...", "input": "...", "output": "..."}},
  ...
]"""


def _build_type_instructions(selected_types: list[str]) -> str:
    lines = []
    for i, qtype in enumerate(selected_types, 1):
        lines.append(f"{i}. [{qtype}] {_QUESTION_TYPE_INSTRUCTIONS[qtype]}")
    return "\n".join(lines)


def _parse_response(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def generate_qa_pairs(
    product: MortgageProduct,
    client: LLMClient,
    questions_per_product: int,
    question_types: list[str],
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> list[dict[str, str]]:
    selected_types = question_types
    product_text = product.to_input_text()

    prompt = _USER_PROMPT_TEMPLATE.format(
        product_text=product_text,
        n=len(selected_types),
        type_instructions=_build_type_instructions(selected_types),
    )

    for attempt in range(max_retries):
        try:
            text = await client.complete(
                system=_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            pairs = _parse_response(text)
            valid = []
            for pair in pairs:
                if not all(k in pair for k in ("instruction", "input", "output")):
                    continue
                pair["input"] = product_text
                valid.append(pair)
            return valid
        except (json.JSONDecodeError, IndexError, KeyError, ValueError):
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    return []
