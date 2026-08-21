"""Backend language model.

The production backend is Llama 4-17B served through LiteLLM on TACC Tapis,
which exposes an OpenAI-compatible ``/chat/completions`` endpoint, so this is a
thin HTTP client rather than a model deployment.

``EchoBackend`` is the deterministic fake used by tests and by any bench run
that needs a miss path without spending real inference calls.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class LMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class LMBackend(Protocol):
    def complete(self, query: str) -> LMResponse:
        ...


class EchoBackend:
    """Deterministic stand-in. Test fake — never a reported number.

    ``latency_s`` simulates the cost the cache is meant to avoid, so hit-vs-miss
    latency separation is observable without a real model.
    """

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s

    def complete(self, query: str) -> LMResponse:
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        digest = hashlib.sha256(query.encode()).hexdigest()[:12]
        words = max(1, len(query.split()))
        return LMResponse(
            text=f"[echo:{digest}] answer to: {query}",
            prompt_tokens=words,
            completion_tokens=words * 3,
        )


class LiteLLMBackend:
    """OpenAI-compatible chat completion against the Tapis LiteLLM proxy."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ValueError("TAPIS_BASE_URL is required for BACKEND_IMPL=litellm")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.Client(
                base_url=self.base_url, headers=headers, timeout=self.timeout_s
            )
        return self._client

    def _messages(self, query: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        return messages

    def complete(self, query: str) -> LMResponse:
        payload = {
            "model": self.model,
            "messages": self._messages(query),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = self._http().post("/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage") or {}
        return LMResponse(
            text=body["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def complete_raw(self, messages: list[dict], *, model: str, temperature: float,
                     extra_body: Optional[dict] = None) -> str:
        """Escape hatch for the judge, which needs a different model and params."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        resp = self._http().post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def build_backend(settings) -> LMBackend:
    impl = settings.backend_impl.lower()
    if impl in {"echo", "fake"}:
        return EchoBackend()
    if impl in {"litellm", "tapis", "real"}:
        return LiteLLMBackend(
            base_url=settings.tapis_base_url,
            api_key=settings.tapis_api_key,
            model=settings.lm_model,
            temperature=settings.lm_temperature,
            max_tokens=settings.lm_max_tokens,
            timeout_s=settings.lm_timeout_s,
        )
    raise ValueError(f"Unknown BACKEND_IMPL: {settings.backend_impl!r}")
