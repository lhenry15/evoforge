"""DataFactory — generate targeted training data from capability gaps."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.core.types import DataFormat


class TrainingExample(BaseModel):
    """A single generated training example (SFT format)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    capability: str
    instruction: str              # user message(s)
    ideal_response: str           # what the agent should say/do
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    format: DataFormat = DataFormat.SFT
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataFactoryConfig(BaseModel):
    """Configuration for data generation."""
    examples_per_gap: int = 20              # how many examples per capability gap
    temperature: float = 0.8                # diversity in generation
    include_tool_traces: bool = True        # generate tool-calling examples
    difficulty_spread: bool = True          # mix easy/medium/hard
    max_retries: int = 2


class DataFactory:
    """
    Format training examples for fine-tuning backends (JSONL / chat format).

    NOTE: LLM-based generation of training data now lives in
    ``foundry.synthesis`` (mode-conditioned, quality-gated). This class is
    retained for converting :class:`TrainingExample` objects into the chat /
    JSONL format consumed by training backends.

    Usage::

        factory = DataFactory(pool=None)
        factory.save_training_data(examples, "train.jsonl", system_prompt="...")
    """

    def __init__(self, pool: Any = None, config: Optional[DataFactoryConfig] = None) -> None:
        self._pool = pool
        self._config = config or DataFactoryConfig()

    def format_for_training(
        self,
        examples: list[TrainingExample],
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """
        Convert TrainingExample objects to chat-format training data.

        Output format (compatible with mlx-lm and OpenAI fine-tune)::

            [
                {
                    "messages": [
                        {"role": "system", "content": "..."},
                        {"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}
                    ]
                },
                ...
            ]
        """
        formatted = []
        for ex in examples:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": ex.instruction})

            # If tool calls exist, format as assistant tool-calling turn
            if ex.tool_calls:
                # First: assistant decides to call tools
                tool_call_text = json.dumps(ex.tool_calls, indent=2)
                messages.append({
                    "role": "assistant",
                    "content": f"I'll help with that.\n\n[Tool calls]\n{tool_call_text}",
                })
                # Then: final response
                messages.append({"role": "assistant", "content": ex.ideal_response})
            else:
                messages.append({"role": "assistant", "content": ex.ideal_response})

            formatted.append({"messages": messages})
        return formatted

    def save_training_data(
        self,
        examples: list[TrainingExample],
        output_path: str,
        system_prompt: str = "",
    ) -> str:
        """
        Save formatted training data as JSONL (one JSON object per line).

        This is the format expected by mlx-lm and OpenAI fine-tuning.
        """
        from pathlib import Path

        formatted = self.format_for_training(examples, system_prompt)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")

        return str(path)
