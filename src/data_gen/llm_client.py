from __future__ import annotations

from abc import ABC, abstractmethod

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}


class RateLimited(Exception):
    """Raised by LLMClient.complete() when the provider returns a rate-limit error.

    Callers must catch this OUTSIDE any semaphore and sleep before retrying,
    so the slot is freed while backing off.
    """


class LLMClient(ABC):
    """Provider-agnostic async wrapper around a language model API."""

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        """Make a single attempt. Raises RateLimited on 429; callers handle retries."""
        ...

    @property
    @abstractmethod
    def max_concurrency(self) -> int:
        """Max simultaneous in-flight requests for this provider."""
        ...


class AnthropicClient(LLMClient):
    # 5 keeps us well inside Tier-1 RPM (~50 RPM) at typical 5-10s/call latency.
    # Tier-2+ users can raise this in the config.
    max_concurrency = 5

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._RateLimitError = anthropic.RateLimitError

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except self._RateLimitError as exc:
            raise RateLimited() from exc


class GroqClient(LLMClient):
    max_concurrency = 2

    def __init__(self, api_key: str, model: str) -> None:
        from groq import AsyncGroq, RateLimitError
        self._client = AsyncGroq(api_key=api_key, timeout=60.0)
        self._model = model
        self._RateLimitError = RateLimitError

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        import httpx
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
        except self._RateLimitError as exc:
            raise RateLimited() from exc
        except httpx.TimeoutException as exc:
            raise RateLimited() from exc


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
