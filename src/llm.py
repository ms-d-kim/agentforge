"""Thin LLM client wrapper over Anthropic / OpenAI (spec §4, §7).

A single ``complete(prompt) -> str`` surface keeps the workflow pseudocode in the
spec literal. The client tracks cumulative ``total_calls`` / ``total_tokens`` so
workflows can snapshot deltas for ``WorkflowResult``.

Responses are cached by ``(model, system, prompt, temperature, max_tokens)`` in a
local SQLite file (spec §17) so re-runs don't re-pay the API and logged token
counts stay stable across reruns. ``FakeLLMClient`` enables fully offline tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time

from .config import (
    ANTHROPIC_MODEL,
    LLM_CACHE_ENABLED,
    LLM_CACHE_PATH,
    LLM_PROVIDER,
    MAX_TOKENS,
    OPENAI_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    TEMPERATURE,
)


class ResponseCache:
    """SQLite-backed (key -> (text, tokens)) cache."""

    def __init__(self, path):
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, tokens INTEGER)"
        )
        self.con.commit()

    @staticmethod
    def make_key(model, system, prompt, temperature, max_tokens) -> str:
        blob = json.dumps(
            [model, system or "", prompt, temperature, max_tokens], sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key):
        row = self.con.execute(
            "SELECT text, tokens FROM cache WHERE key=?", (key,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key, text, tokens) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO cache (key, text, tokens) VALUES (?,?,?)",
            (key, text, tokens),
        )
        self.con.commit()


class BaseLLMClient:
    """Common counting + caching logic; subclasses implement ``_complete``."""

    def __init__(self, model: str, cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self.total_calls = 0
        self.total_tokens = 0
        self.cache_hits = 0

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        self.total_calls += 1
        key = None
        if self.cache is not None:
            key = self.cache.make_key(self.model, system, prompt, temperature, max_tokens)
            hit = self.cache.get(key)
            if hit is not None:
                text, tokens = hit
                self.total_tokens += tokens
                self.cache_hits += 1
                return text

        text, tokens = self._complete_with_retry(prompt, system, temperature, max_tokens)
        self.total_tokens += tokens
        if self.cache is not None and key is not None:
            self.cache.put(key, text, tokens)
        return text

    def _complete_with_retry(self, prompt, system, temperature, max_tokens, attempts: int = 4):
        """Retry transient API errors (429 / 5xx / network) with linear backoff."""
        last = None
        for i in range(attempts):
            try:
                return self._complete(prompt, system, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001 — survive the long sequential run
                last = exc
                if i == attempts - 1:
                    raise
                time.sleep(1.5 * (i + 1))
        raise last  # unreachable

    def _complete(self, prompt, system, temperature, max_tokens) -> tuple[str, int]:
        raise NotImplementedError


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = ANTHROPIC_MODEL, cache: ResponseCache | None = None):
        super().__init__(model, cache)
        from anthropic import Anthropic  # lazy: keep module importable without the SDK

        # Reads ANTHROPIC_API_KEY and (if set) ANTHROPIC_BASE_URL from the env.
        self._client = Anthropic()

    def _complete(self, prompt, system, temperature, max_tokens):
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return text, tokens


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions client. Also serves OpenAI-compatible gateways
    (e.g. OpenRouter) via ``base_url`` + ``api_key``."""

    def __init__(
        self,
        model: str = OPENAI_MODEL,
        cache: ResponseCache | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model, cache)
        from openai import OpenAI  # lazy

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)

    def _complete(self, prompt, system, temperature, max_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens


class FakeLLMClient(BaseLLMClient):
    """Offline client driven by a ``responder(prompt, system) -> str`` callable.

    Used by the dry-run harness and unit tests to exercise the full pipeline with
    zero API calls. Never caches.
    """

    def __init__(self, responder, model: str = "fake-llm"):
        super().__init__(model, cache=None)
        self._responder = responder

    def _complete(self, prompt, system, temperature, max_tokens):
        text = self._responder(prompt, system)
        return text, max(1, len(text) // 4)  # rough fake token estimate


def build_client(provider: str | None = None, cache_enabled: bool | None = None) -> BaseLLMClient:
    """Construct the configured provider's client with caching per config."""
    provider = provider or LLM_PROVIDER
    use_cache = LLM_CACHE_ENABLED if cache_enabled is None else cache_enabled
    cache = ResponseCache(LLM_CACHE_PATH) if use_cache else None
    if provider == "anthropic":
        return AnthropicClient(ANTHROPIC_MODEL, cache)
    if provider == "openai":
        return OpenAIClient(OPENAI_MODEL, cache)
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env: "
                "echo 'OPENROUTER_API_KEY=sk-or-...' >> .env"
            )
        return OpenAIClient(OPENROUTER_MODEL, cache, base_url=OPENROUTER_BASE_URL, api_key=key)
    raise ValueError(
        f"unknown provider: {provider!r} (expected 'anthropic', 'openai', or 'openrouter')"
    )
