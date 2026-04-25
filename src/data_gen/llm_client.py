from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}


class LLMClient(ABC):
    """Provider-agnostic async wrapper around a language model API."""

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        ...

    @property
    @abstractmethod
    def max_concurrency(self) -> int:
        """Max simultaneous in-flight requests for this provider."""
        ...


class AnthropicClient(LLMClient):
    max_concurrency = 10

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._RateLimitError = anthropic.RateLimitError

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        for attempt in range(4):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except self._RateLimitError:
                await asyncio.sleep(30 * (attempt + 1))
        raise RuntimeError("Anthropic: max rate-limit retries exceeded.")


class GroqClient(LLMClient):
    # Free tier is ~30 RPM; keep concurrency low and let retry logic handle 429s
    max_concurrency = 2

    def __init__(self, api_key: str, model: str) -> None:
        from groq import AsyncGroq, RateLimitError
        self._client = AsyncGroq(api_key=api_key, timeout=60.0)
        self._model = model
        self._RateLimitError = RateLimitError

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        import httpx
        for attempt in range(4):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content
            except self._RateLimitError:
                await asyncio.sleep(60 * (attempt + 1))
            except httpx.TimeoutException:
                await asyncio.sleep(10 * (attempt + 1))
        raise RuntimeError("Groq: max rate-limit retries exceeded.")


def build_client(provider: str, api_key: str, model: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    if provider == "groq":
        return GroqClient(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider {provider!r}. Choose 'anthropic' or 'groq'.")


def api_key_env(provider: str) -> str:
    try:
        return _API_KEY_ENV[provider]
    except KeyError:
        raise ValueError(f"Unknown provider {provider!r}.")
