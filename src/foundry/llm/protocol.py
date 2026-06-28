"""LLM pool protocol and GitHub Models implementation."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMPool(Protocol):
    """
    Multi-model LLM pool used for bootstrap generation, LLM judge ensemble, and UQ.

    Default implementation: GitHubModelsLLMPool (uses GitHub Models via OpenAI SDK).
    Swap out by passing a custom implementation to SDKConfig.
    """

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate from a specific model (or let the pool choose)."""
        ...

    async def generate_ensemble(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
        """Generate from N different models for UQ ensemble."""
        ...

    @property
    def available_models(self) -> list[str]:
        """List of model identifiers available in this pool."""
        ...


class GitHubModelsLLMPool:
    """
    LLM pool backed by GitHub Models (https://github.com/marketplace/models).

    Uses the OpenAI-compatible endpoint at models.inference.ai.azure.com.
    Requires GITHUB_TOKEN environment variable.

    Used for:
      - Zero-shot bootstrap data generation
      - LLM judge for eval scoring
      - Ensemble UQ (multiple temperatures / multiple calls to same model)

    Example::

        pool = GitHubModelsLLMPool()
        score = await pool.generate(judge_prompt, temperature=0)
    """

    BASE_URL = "https://models.inference.ai.azure.com"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        from openai import AsyncOpenAI

        token = api_key or os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN not set. Export it before initialising GitHubModelsLLMPool."
            )
        self._model = model
        self._client = AsyncOpenAI(base_url=self.BASE_URL, api_key=token)

    @property
    def available_models(self) -> list[str]:
        return [self._model]

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        r = await self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return r.choices[0].message.content or ""

    async def generate_ensemble(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> list[str]:
        """Run prompt N times in parallel (varied temperature for diversity)."""
        temps = [max(0.0, temperature - 0.2), temperature, min(1.0, temperature + 0.2)]
        tasks = [
            self.generate(prompt, temperature=temps[i % len(temps)], **kwargs)
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))


# Keep old name as alias for backward compat
CopilotLLMPool = GitHubModelsLLMPool
