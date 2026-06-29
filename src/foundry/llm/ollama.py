"""Ollama LLM pool — local inference via Ollama REST API (fully synchronous)."""

from __future__ import annotations

from typing import Any, Optional

import requests


class OllamaLLMPool:
    """
    LLM pool backed by a local Ollama instance.

    Uses the Ollama REST API directly (no async, no OpenAI SDK dependency).
    This avoids all event loop conflicts when called from CLI or sync code.

    Usage::

        pool = OllamaLLMPool(model="qwen2.5:3b")
        response = pool.generate("What is 2+2?")
        responses = pool.generate_ensemble("Name a color.", n=3)
    """

    BASE_URL = "http://localhost:11434"

    def __init__(self, model: str = "qwen2.5:3b", base_url: Optional[str] = None) -> None:
        self._model = model
        self._base_url = base_url or self.BASE_URL

    @property
    def available_models(self) -> list[str]:
        return [self._model]

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        """Generate a response synchronously."""
        resp = requests.post(f"{self._base_url}/api/chat", json={
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }, timeout=120)
        return resp.json().get("message", {}).get("content", "")

    # Alias for backward compatibility
    generate_sync = generate

    # Ollama supports schema-constrained decoding via the `format` field.
    supports_structured = True

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> Any:
        """Generate JSON constrained to ``schema`` (Ollama structured outputs).

        Returns the parsed object, or ``None`` if the response can't be parsed.
        Guarantees well-formed JSON for small models, eliminating the common
        double-wrapped-array failure.
        """
        resp = requests.post(f"{self._base_url}/api/chat", json={
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }, timeout=180)
        content = resp.json().get("message", {}).get("content", "")
        from foundry.llm.structured import extract_json
        return extract_json(content)

    def generate_ensemble(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
        """Generate N responses with varied temperatures for diversity."""
        temps = [max(0.0, temperature - 0.3), temperature, min(1.5, temperature + 0.3)]
        return [
            self.generate(prompt, temperature=temps[i % len(temps)], **kwargs)
            for i in range(n)
        ]
