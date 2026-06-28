"""DataFactory — generate targeted training data from capability gaps."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.core.types import (
    CapabilityGap,
    DataFormat,
)


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


_GENERATION_SYSTEM = """You are an expert training data generator for AI agents.
Your job is to create realistic, high-quality training examples that help an agent 
improve at a specific capability.

Output format: JSON array of objects, each with:
- "instruction": the user message(s) the agent would receive
- "ideal_response": what a perfect agent would respond
- "tool_calls": (optional) list of tool calls the agent should make, each with "name" and "arguments"

Requirements:
- Examples must be diverse (different scenarios, edge cases, phrasings)
- Responses must demonstrate expert-level handling of the capability
- Include both simple and complex cases
- Tool calls should be realistic and complete"""

_GENERATION_PROMPT = """Generate {n} training examples for an AI agent.

AGENT TASK: {task_spec}

TARGET CAPABILITY TO IMPROVE: {capability}
Current score: {score:.2f} (threshold: {threshold:.2f}) — the agent is failing here.

AVAILABLE TOOLS: {tools}

SYSTEM PROMPT THE AGENT USES: {system_prompt}

{difficulty_instruction}

Generate {n} diverse, high-quality examples that would teach the agent to handle this capability well.
Reply with ONLY a JSON array."""


class DataFactory:
    """
    Generate targeted training data from CapabilityGap signals.

    The factory uses an LLM to generate (instruction, ideal_response) pairs
    specifically targeted at capabilities where the agent is underperforming.

    Usage::

        factory = DataFactory(pool=OllamaLLMPool())
        examples = factory.generate_for_gaps(
            gaps=[CapabilityGap(capability="booking", score=0.0, ...)],
            task_spec="A flight booking assistant...",
            tools=["search_flights", "book_flight"],
        )
    """

    def __init__(self, pool: Any, config: Optional[DataFactoryConfig] = None) -> None:
        self._pool = pool
        self._config = config or DataFactoryConfig()

    def generate_for_gaps(
        self,
        gaps: list[CapabilityGap],
        task_spec: str,
        tools: list[Any] = None,
        system_prompt: str = "",
    ) -> list[TrainingExample]:
        """
        Generate training examples for all capability gaps.

        Returns a flat list of TrainingExample objects ready for training.
        """
        all_examples = []
        for gap in gaps:
            examples = self._generate_for_gap(gap, task_spec, tools or [], system_prompt)
            all_examples.extend(examples)
        return all_examples

    def _generate_for_gap(
        self,
        gap: CapabilityGap,
        task_spec: str,
        tools: list[Any],
        system_prompt: str,
    ) -> list[TrainingExample]:
        n = min(gap.suggested_n_examples, self._config.examples_per_gap)

        # Format tools for the prompt
        tool_desc = self._format_tools(tools) if tools else "No specific tools defined."

        difficulty_instruction = ""
        if self._config.difficulty_spread:
            difficulty_instruction = (
                "Mix difficulty levels:\n"
                f"- {n // 3} easy (straightforward requests)\n"
                f"- {n // 3} medium (multi-step or ambiguous)\n"
                f"- {n - 2 * (n // 3)} hard (edge cases, error recovery, complex scenarios)"
            )

        prompt = _GENERATION_PROMPT.format(
            n=n,
            task_spec=task_spec,
            capability=gap.capability,
            score=gap.score,
            threshold=gap.threshold,
            tools=tool_desc,
            system_prompt=system_prompt or "(not provided)",
            difficulty_instruction=difficulty_instruction,
        )

        # Generate with retries
        for attempt in range(self._config.max_retries + 1):
            try:
                raw = self._pool.generate(
                    prompt,
                    system=_GENERATION_SYSTEM,
                    temperature=self._config.temperature,
                    max_tokens=4096,
                )
                examples = self._parse_examples(raw, gap.capability)
                if examples:
                    return examples
            except Exception as e:
                if attempt == self._config.max_retries:
                    print(f"[DataFactory] Failed to generate for {gap.capability}: {e}")
                    return []

        return []

    def _parse_examples(self, raw: str, capability: str) -> list[TrainingExample]:
        """Parse LLM output into TrainingExample objects."""
        # Try to extract JSON array from response
        try:
            # Find JSON array in response (may be wrapped in markdown code block)
            cleaned = raw.strip()
            if "```" in cleaned:
                # Extract from code block
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            # Try to find array directly
            if not cleaned.startswith("["):
                match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group()

            items = json.loads(cleaned)
            if not isinstance(items, list):
                return []

            examples = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                instruction = item.get("instruction", "")
                response = item.get("ideal_response", item.get("response", ""))
                if not instruction or not response:
                    continue
                examples.append(TrainingExample(
                    capability=capability,
                    instruction=instruction,
                    ideal_response=response,
                    tool_calls=item.get("tool_calls", []),
                ))
            return examples

        except (json.JSONDecodeError, AttributeError):
            return []

    def _format_tools(self, tools: list[Any]) -> str:
        """Format tools list for the generation prompt."""
        descriptions = []
        for t in tools:
            if isinstance(t, str):
                descriptions.append(f"- {t}")
            elif hasattr(t, "name") and hasattr(t, "description"):
                descriptions.append(f"- {t.name}: {t.description}")
            else:
                descriptions.append(f"- {t}")
        return "\n".join(descriptions) if descriptions else "No tools."

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
