from __future__ import annotations

import time
from abc import ABC, abstractmethod

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}


class LLMClient(ABC):
    """Provider-agnostic wrapper around a language model API.

    Each implementation handles its own rate-limit retries internally so
    callers never need to import provider-specific exceptions.
    """

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        ...

    @property
    @abstractmethod
    def inter_request_sleep(self) -> float:
        """Minimum seconds to sleep between consecutive requests."""
        ...


class AnthropicClient(LLMClient):
    inter_request_sleep = 0.3

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._RateLimitError = anthropic.RateLimitError

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        for attempt in range(4):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except self._RateLimitError:
                time.sleep(30 * (attempt + 1))
        raise RuntimeError("Anthropic: max rate-limit retries exceeded.")


class GroqClient(LLMClient):
    inter_request_sleep = 2.0  # free tier: ~30 RPM

    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq, RateLimitError
        self._client = Groq(api_key=api_key)
        self._model = model
        self._RateLimitError = RateLimitError

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        for attempt in range(4):
            try:
                resp = self._client.chat.completions.create(
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
                time.sleep(60 * (attempt + 1))
        raise RuntimeError("Groq: max rate-limit retries exceeded.")


def build_client(provider: str, api_key: str, model: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    if provider == "groq":
        return GroqClient(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider {provider!r}. Choose 'anthropic' or 'groq'.")


def api_key_env(provider: str) -> str:
    """Return the environment variable name that holds this provider's API key."""
    try:
        return _API_KEY_ENV[provider]
    except KeyError:
        raise ValueError(f"Unknown provider {provider!r}.")
