"""AdaptiveEvalExpander — generate eval cases that close coverage blind spots.

Targets failure modes seen in real traces but not yet probed by the benchmark.

Design for small models (e.g. qwen2.5:3b):
  1. Derive a strict success criterion once per (capability, mode).
  2. Ask the model only for diverse, realistic USER MESSAGES (what it's good at),
     via schema-constrained decoding (no malformed JSON).
  3. CONSTRUCT each case's ``expected`` + ``scoring_rubric`` deterministically from
     the criterion, so the discriminating power doesn't depend on the small model.
  4. Quality-gate messages (dedup, novelty, request-shape) and retry to hit target.

Generated cases are tagged with ``target_mode`` so re-mapping closes the blind spot.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import EvalCase, Message, ScoringMethod
from evoforge.coverage.case_quality import EvalCaseQualityGate
from evoforge.coverage.criterion import CriterionDeriver, SuccessCriterion
from evoforge.coverage.schema import Blindspot
from evoforge.llm.structured import coerce_strings, generate_structured
from evoforge.trace.schema import TraceLineage

_MESSAGES_SCHEMA = {
    "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
    "required": ["messages"],
}

_MESSAGES_SYSTEM = (
    "You generate realistic, diverse user messages to test an AI agent. "
    "Each message must be distinct, natural, and clearly exercise the target "
    "capability. Vary names, places, dates, and phrasing."
)


class ExpansionStats(BaseModel):
    """Diagnostics for an expansion run (yield/quality visibility)."""
    blindspots: int = 0
    requested: int = 0
    generated_messages: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = Field(default_factory=dict)
    attempts: int = 0

    @property
    def yield_rate(self) -> float:
        return round(self.accepted / self.generated_messages, 4) if self.generated_messages else 0.0


class AdaptiveEvalExpander:
    """Generate targeted, tagged eval cases for coverage blind spots."""

    def __init__(self, pool: Any, max_attempts: int = 3) -> None:
        self._pool = pool
        self._deriver = CriterionDeriver(pool)
        self._max_attempts = max_attempts

    # ── public API ────────────────────────────────────────────────────

    def expand(
        self,
        blindspots: list[Blindspot],
        task_spec: str,
        tools: list[Any] = None,
        system_prompt: str = "",
        cases_per_blindspot: int = 3,
        existing_messages: Optional[list[str]] = None,
    ) -> list[EvalCase]:
        cases, _ = self.expand_with_stats(
            blindspots, task_spec, tools, system_prompt,
            cases_per_blindspot, existing_messages,
        )
        return cases

    def expand_with_stats(
        self,
        blindspots: list[Blindspot],
        task_spec: str,
        tools: list[Any] = None,
        system_prompt: str = "",
        cases_per_blindspot: int = 3,
        existing_messages: Optional[list[str]] = None,
    ) -> tuple[list[EvalCase], ExpansionStats]:
        stats = ExpansionStats(blindspots=len(blindspots))
        gate = EvalCaseQualityGate(existing_messages=existing_messages or [])
        accepted_messages: list[str] = list(existing_messages or [])
        all_cases: list[EvalCase] = []

        for spot in blindspots:
            target = min(cases_per_blindspot, spot.suggested_cases) or cases_per_blindspot
            stats.requested += target
            criterion = self._deriver.derive(
                capability=spot.capability,
                mode=spot.mode,
                task_spec=task_spec,
                symptom="",
                failing_inputs=spot.example_inputs,
            )
            cases = self._expand_one(spot, criterion, task_spec, target, gate,
                                     accepted_messages, stats)
            all_cases.extend(cases)

        return all_cases, stats

    # ── per-blindspot generation with retry ───────────────────────────

    def _expand_one(
        self,
        spot: Blindspot,
        criterion: SuccessCriterion,
        task_spec: str,
        target: int,
        gate: EvalCaseQualityGate,
        accepted_messages: list[str],
        stats: ExpansionStats,
    ) -> list[EvalCase]:
        kept: list[EvalCase] = []

        for _attempt in range(self._max_attempts):
            if len(kept) >= target:
                break
            stats.attempts += 1
            # Over-generate to survive quality filtering.
            n_request = (target - len(kept)) * 2 + 1
            messages = self._generate_messages(spot, criterion, task_spec, n_request)
            stats.generated_messages += len(messages)

            for msg in messages:
                if len(kept) >= target:
                    break
                ok, reason = gate.accept(msg, accepted_messages)
                if not ok:
                    stats.rejected += 1
                    stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + 1
                    continue
                stats.accepted += 1
                accepted_messages.append(msg.strip())
                kept.append(self._construct_case(spot, criterion, msg.strip()))

        return kept

    def _generate_messages(
        self, spot: Blindspot, criterion: SuccessCriterion, task_spec: str, n: int
    ) -> list[str]:
        seeds = "\n".join(f'- "{s[:120]}"' for s in spot.example_inputs[:3]) or "(none)"
        prompt = (
            f"AGENT TASK: {task_spec}\n"
            f"CAPABILITY: {spot.capability}\n"
            f"WHAT SUCCESS REQUIRES: {criterion.success_signal}\n"
            f"KNOWN FAILURE: {criterion.fail_signal}\n"
            f"REAL FAILING INPUTS (for inspiration, do not copy):\n{seeds}\n\n"
            f"Generate {n} diverse, realistic user messages that require the agent to "
            f"fully satisfy the success requirement above. Each must be distinct."
        )
        parsed = generate_structured(
            self._pool, prompt, _MESSAGES_SCHEMA, system=_MESSAGES_SYSTEM,
            temperature=0.9, max_tokens=1500,
        )
        return coerce_strings(parsed, key="messages")

    def _construct_case(
        self, spot: Blindspot, criterion: SuccessCriterion, message: str
    ) -> EvalCase:
        return EvalCase(
            id=f"adapt-{spot.mode[:6]}-{uuid.uuid4().hex[:4]}",
            capability=spot.capability,
            messages=[Message(role="user", content=message)],
            expected=criterion.expected(),
            scoring_method=ScoringMethod.LLM_JUDGE,
            scoring_rubric=criterion.rubric(),
            metadata={
                "target_mode": spot.mode,
                "failure_mode": spot.mode,
                "difficulty": "hard",
                "adaptive": True,
                "blindspot": f"{spot.capability}/{spot.mode}",
                "success_keywords": criterion.success_keywords,
                "lineage": TraceLineage(
                    generation_method="adaptive_eval",
                    derived_from=f"{spot.capability}/{spot.mode}",
                    tags=[spot.mode, "coverage"],
                ).model_dump(),
            },
        )
