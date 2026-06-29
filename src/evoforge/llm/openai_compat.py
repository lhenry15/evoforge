"""OpenAI-compatible LLM pool — synchronous, for any ``/v1/chat/completions`` endpoint.

A fully synchronous pool (no async, no OpenAI SDK) that talks to any
OpenAI-compatible chat endpoint — a hosted gateway, a local proxy, vLLM, LM Studio,
Ollama's OpenAI shim, etc. It implements the duck-typed pool interface the
synthesis subsystem expects (``generate`` / ``generate_json`` /
``supports_structured``), so it is the drop-in sync pool for
:class:`~evoforge.synthesis.pipeline.ScenarioSynthesizer` and
:class:`~evoforge.synthesis.DataSynthesizer`.

Structured output is obtained by embedding the JSON Schema in the system prompt and
robustly parsing the reply (works even when the endpoint lacks native
``response_format`` support).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from evoforge.llm.structured import extract_json


class OpenAICompatPool:
    """Synchronous LLM pool backed by an OpenAI-compatible chat endpoint.

    Usage::

        pool = OpenAICompatPool(base_url="http://localhost:8787", model="gpt-4o-mini")
        text = pool.generate("Say hi.")
        obj = pool.generate_json("List two colors.", {"type": "object", ...})

    ``base_url`` / ``model`` / ``api_key`` fall back to the ``OPENAI_BASE_URL``,
    ``OPENAI_MODEL`` and ``OPENAI_API_KEY`` environment variables, so callers (an
    example script or the evolution loop) can configure the endpoint without code
    changes.
    """

    DEFAULT_BASE_URL = "http://localhost:8787"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 180.0,
    ) -> None:
        self._model = model or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL
        self._base_url = (base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self._timeout = timeout
        self.calls = 0

    @property
    def available_models(self) -> list[str]:
        return [self._model]

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    supports_structured = True

    # ── internals ─────────────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _chat(self, system: str, user: str, model: Optional[str], temperature: float, max_tokens: int) -> str:
        self.calls += 1
        body: dict[str, Any] = {
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json=body, headers=self._headers(), timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"].get("content") or "") if data.get("choices") else ""

    # ── public sync interface ─────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        """Generate a response synchronously."""
        return self._chat(system, prompt, model, temperature, int(kwargs.get("max_tokens", 1024)))

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> Any:
        """Generate JSON conforming to ``schema``; returns the parsed object or ``None``.

        The schema is embedded in the system prompt and the reply is robustly parsed,
        so this works even on endpoints without native structured-output support.
        """
        import json as _json

        system2 = (
            f"{system}\n\nReturn ONLY valid minified JSON conforming to this JSON Schema "
            f"(no prose, no markdown fences):\n{_json.dumps(schema)}"
        )
        raw = self._chat(system2, prompt, model, temperature, int(kwargs.get("max_tokens", 1024)))
        parsed = extract_json(raw)
        return parsed

    def generate_ensemble(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
        """Generate N responses with varied temperatures for diversity."""
        temps = [max(0.0, temperature - 0.2), temperature, min(1.0, temperature + 0.2)]
        responses = [self.generate(prompt, temperature=temps[i % len(temps)], **kwargs) for i in range(n)]
        return responses
