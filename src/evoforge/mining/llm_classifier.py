"""LLMModeClassifier — re-label unknown/low-confidence failures into real modes.

The heuristic :class:`FailureSignatureExtractor` only recognizes a few surface
patterns (errors, empty output, refusals, malformed structure). Many real
failures are subtler — e.g. the agent performed the *wrong action* (searched
instead of booking). This classifier uses an LLM, conditioned on the full trace
(input, expected behavior, agent response, judge reasoning), to assign a precise
:class:`FailureMode`. Results are cached and call-capped for efficiency.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from evoforge.llm.structured import extract_json
from evoforge.trace.schema import FailureMode, FailureSignature, TraceRecord

# Concise definitions so small local models classify reliably.
_MODE_DEFS = """- prompt_gap: instructions were missing/ambiguous; agent did something plausible but not what was asked
- tool_misuse: wrong tool, wrong arguments, or a required tool was never called (e.g. searched but never booked)
- missing_knowledge: a factual or domain gap led to a wrong answer
- policy_conflict: agent refused, deflected, or violated a constraint
- environment_fragility: a tool/environment error, crash, or timeout
- format_violation: output structure/format was wrong (e.g. invalid JSON)
- hallucination: agent fabricated information
- incomplete: agent stopped early and left the task unfinished"""

_SYSTEM = """You are a reliability engineer classifying why an AI agent failed a
task. Choose the single best failure mode from the list. Reply with ONLY JSON:
{"mode": "<one of the listed modes>", "symptom": "<= 10 words", "confidence": 0.0-1.0}"""


class LLMModeClassifier:
    """Classify a failing trace into a precise :class:`FailureMode` via an LLM."""

    def __init__(self, pool: Any, max_calls: int = 24) -> None:
        self._pool = pool
        self._max_calls = max_calls
        self._cache: dict[tuple, tuple[FailureMode, str, float]] = {}
        self.calls = 0

    def classify(self, trace: TraceRecord) -> Optional[tuple[FailureMode, str, float]]:
        """Return (mode, symptom, confidence) or None if unclassifiable/over budget."""
        user = self._user_input(trace)
        expected = str(trace.metadata.get("expected", ""))
        judge = str(trace.metadata.get("judge_reasoning", ""))
        response = trace.final_response or ""

        key = (trace.capability or "", response[:100], judge[:100])
        if key in self._cache:
            return self._cache[key]
        if self.calls >= self._max_calls:
            return None

        prompt = (
            f"FAILURE MODES:\n{_MODE_DEFS}\n\n"
            f"CAPABILITY: {trace.capability or 'n/a'}\n"
            f"USER REQUEST: {user[:200]}\n"
            f"EXPECTED BEHAVIOR: {expected[:200] or '(not specified)'}\n"
            f"AGENT RESPONSE: {response[:200] or '(empty)'}\n"
            f"JUDGE REASONING: {judge[:200] or '(none)'}\n\n"
            "Classify the failure."
        )
        self.calls += 1
        try:
            raw = self._pool.generate(prompt, system=_SYSTEM, temperature=0, max_tokens=80)
        except Exception:
            return None
        if inspect.isawaitable(raw):
            raise RuntimeError(
                "LLMModeClassifier received an async LLM pool in sync mode. "
                "Use a synchronous pool (for example, OllamaLLMPool)."
            )

        result = self._parse(str(raw))
        if result is not None:
            self._cache[key] = result
        return result

    @staticmethod
    def _user_input(trace: TraceRecord) -> str:
        return "\n".join(
            m.content for m in trace.input_messages if getattr(m, "role", "") == "user"
        )

    @staticmethod
    def _parse(raw: str) -> Optional[tuple[FailureMode, str, float]]:
        data = extract_json(raw)
        if not isinstance(data, dict):
            return None

        mode_str = str(data.get("mode", "")).strip().lower()
        try:
            mode = FailureMode(mode_str)
        except ValueError:
            return None
        if mode == FailureMode.UNKNOWN:
            return None
        symptom = str(data.get("symptom", "")).strip()[:120]
        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        return mode, symptom, confidence


def apply_classification(trace: TraceRecord, mode: FailureMode, symptom: str, confidence: float) -> None:
    """Rewrite a trace's failure signature with a re-classified mode.

    Regenerates the deterministic ``signature_id`` so re-classified failures
    cluster by their new mode rather than the stale ``unknown`` bucket.
    """
    sig = trace.failure_signature
    if sig is None:
        return
    new_symptom = symptom or sig.symptom
    sig.mode = mode
    sig.symptom = new_symptom
    sig.confidence = max(confidence, sig.confidence)
    sig.signature_id = FailureSignature.make_id(mode.value, trace.capability, new_symptom)
    sig.metadata["reclassified"] = True
    trace.lineage.failure_signature_id = sig.signature_id
