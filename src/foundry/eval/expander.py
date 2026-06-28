"""Eval expansion — generate harder cases for saturating capabilities."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from foundry.core.types import EvalCase, Message, ScoringMethod, SaturationSignal


_EXPANSION_SYSTEM = """You are an expert adversarial test designer for AI agents.
Given a capability where the agent is already performing well, generate HARDER test cases
that probe edge cases, failure modes, and subtle requirements the easy tests miss.

Reply with ONLY a JSON array:
[{
  "user_message": "a challenging user message",
  "expected": "what a correct response must include",
  "difficulty": "hard",
  "failure_mode": "what specific failure this tests (e.g. 'ambiguous input', 'conflicting constraints')",
  "scoring_method": "contains|llm_judge",
  "scoring_rubric": "strict criteria for judging"
}]

Requirements:
- These must be HARDER than existing cases — not just rephrased easy ones
- Target specific failure modes: ambiguity, multi-step reasoning, conflicting constraints,
  implicit requirements, adversarial inputs, uncommon scenarios
- Each case should test a distinct edge case (no duplicates)
- Expected should be precise and verifiable"""

_EXPANSION_PROMPT = """The agent is SATURATING on this capability (scoring {score:.0%}).
Generate {n} HARDER eval cases that will challenge it further.

AGENT TASK: {task_spec}
CAPABILITY: {capability} (current score: {score:.2f})
TOOLS: {tools}
SYSTEM PROMPT: {system_prompt}

EXISTING EASY CASES (agent passes these):
{existing_cases}

Generate {n} significantly harder cases that test:
- Edge cases the above tests don't cover
- Adversarial or ambiguous inputs
- Multi-step reasoning requirements
- Implicit constraints the agent might miss
- Uncommon but valid scenarios"""


class EvalExpander:
    """
    Generate harder eval cases for saturating capabilities.

    When a capability score exceeds the saturation threshold (default 0.85),
    the evolution engine triggers EXPAND_EVAL. This class generates
    adversarial/edge-case test scenarios to keep pushing the agent.

    Usage::

        expander = EvalExpander(pool=OllamaLLMPool())
        new_cases = expander.expand(
            saturation_signals=[SaturationSignal(...)],
            existing_cases=current_cases,
            task_spec="...",
        )
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def expand(
        self,
        saturation_signals: list[SaturationSignal],
        existing_cases: list[EvalCase],
        task_spec: str,
        tools: list[Any] = None,
        system_prompt: str = "",
        cases_per_signal: int = 5,
    ) -> list[EvalCase]:
        """Generate harder eval cases for all saturating capabilities."""
        all_new = []
        for signal in saturation_signals:
            # Get existing cases for this capability (to avoid duplicates)
            cap_cases = [c for c in existing_cases if c.capability == signal.capability]
            new_cases = self._expand_capability(
                signal, cap_cases, task_spec, tools or [], system_prompt, cases_per_signal,
            )
            all_new.extend(new_cases)
        return all_new

    def _expand_capability(
        self,
        signal: SaturationSignal,
        existing_cases: list[EvalCase],
        task_spec: str,
        tools: list[Any],
        system_prompt: str,
        n: int,
    ) -> list[EvalCase]:
        # Format existing cases for context
        existing_str = ""
        for c in existing_cases[:5]:
            existing_str += f'  - "{c.messages[0].content[:80]}"\n'
        if not existing_str:
            existing_str = "  (no existing cases)"

        tool_desc = "\n".join(f"  - {t}" for t in tools) if tools else "No tools."

        prompt = _EXPANSION_PROMPT.format(
            task_spec=task_spec,
            capability=signal.capability,
            score=signal.score,
            tools=tool_desc,
            system_prompt=system_prompt or "(generic)",
            existing_cases=existing_str,
            n=n,
        )

        raw = self._pool.generate(
            prompt, system=_EXPANSION_SYSTEM, temperature=0.8, max_tokens=4096,
        )

        return self._parse_cases(raw, signal.capability)

    def _parse_cases(self, raw: str, capability: str) -> list[EvalCase]:
        try:
            cleaned = raw.strip()
            if "```" in cleaned:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            if not cleaned.startswith("["):
                match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group()

            items = json.loads(cleaned)
            if not isinstance(items, list):
                return []

            cases = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                user_msg = item.get("user_message", "")
                if not user_msg:
                    continue

                method_str = item.get("scoring_method", "llm_judge")
                method = ScoringMethod.LLM_JUDGE if method_str == "llm_judge" else ScoringMethod.CONTAINS

                cases.append(EvalCase(
                    id=f"{capability[:8]}-hard-{str(uuid.uuid4())[:4]}",
                    capability=capability,
                    messages=[Message(role="user", content=user_msg)],
                    expected=item.get("expected", ""),
                    scoring_method=method,
                    scoring_rubric=item.get("scoring_rubric"),
                    metadata={
                        "difficulty": "hard",
                        "failure_mode": item.get("failure_mode", "edge_case"),
                        "expanded": True,
                    },
                ))
            return cases

        except (json.JSONDecodeError, TypeError):
            return []
