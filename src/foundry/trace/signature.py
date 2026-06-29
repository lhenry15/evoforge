"""Baseline failure-signature extraction.

This is a deterministic heuristic baseline so that every failing trace gets a
usable signature from day one. Phase 2 (failure-mode mining) augments/overrides
this with clustering and LLM-assisted root-cause labeling.
"""

from __future__ import annotations

from typing import Optional

from foundry.trace.schema import (
    FailureMode,
    FailureSignature,
    TraceOutcome,
    TraceRecord,
)

_ERROR_PREFIXES = ("[error", "error:", "exception", "traceback")
_REFUSAL_MARKERS = ("i cannot", "i can't", "i'm unable", "not able to", "as an ai")
_FORMAT_HINTS = ("json", "format", "schema", "must be valid")


class FailureSignatureExtractor:
    """Assign a coarse failure mode + symptom to a failing trace.

    The extractor is conservative: when it is unsure it emits ``UNKNOWN`` with
    low confidence rather than guessing, so that mining can later refine without
    fighting bad labels.
    """

    def extract(self, record: TraceRecord) -> Optional[FailureSignature]:
        if not record.is_failure:
            return None

        response = (record.final_response or "").strip()
        lower = response.lower()
        failed_tools = [t for t in record.tool_invocations if not t.succeeded]

        mode = FailureMode.UNKNOWN
        symptom = "response did not satisfy expectation"
        evidence: list[str] = []
        confidence = 0.3

        if record.outcome == TraceOutcome.ERROR or lower.startswith(_ERROR_PREFIXES):
            mode = FailureMode.ENVIRONMENT_FRAGILITY
            symptom = "agent raised an error during execution"
            evidence.append(response[:160])
            confidence = 0.6
        elif failed_tools:
            mode = FailureMode.TOOL_MISUSE
            names = ", ".join(t.tool_name for t in failed_tools)
            symptom = f"tool call failed: {names}"
            evidence.extend((t.error_type or t.tool_name) for t in failed_tools)
            confidence = 0.55
        elif not response:
            mode = FailureMode.INCOMPLETE
            symptom = "empty response"
            confidence = 0.6
        elif any(marker in lower for marker in _REFUSAL_MARKERS):
            mode = FailureMode.POLICY_CONFLICT
            symptom = "agent refused or deflected the request"
            evidence.append(response[:160])
            confidence = 0.45
        elif len(response) < 15:
            mode = FailureMode.INCOMPLETE
            symptom = "response too short to satisfy task"
            evidence.append(response)
            confidence = 0.45
        elif self._expected_format_missing(record):
            mode = FailureMode.FORMAT_VIOLATION
            symptom = "expected structured output appears malformed or missing"
            evidence.append(response[:160])
            confidence = 0.4
        else:
            evidence.append(response[:160])

        signature_id = FailureSignature.make_id(mode.value, record.capability, symptom)
        return FailureSignature(
            signature_id=signature_id,
            mode=mode,
            symptom=symptom,
            trigger=self._infer_trigger(record),
            evidence=[e for e in evidence if e][:5],
            capability=record.capability,
            confidence=confidence,
        )

    def _expected_format_missing(self, record: TraceRecord) -> bool:
        """Heuristic: task hints at structured output but response lacks structure."""
        hint_text = " ".join(
            m.content.lower() for m in record.input_messages if getattr(m, "role", "") == "user"
        )
        rubric = str(record.metadata.get("judge_reasoning", "")).lower()
        wants_structure = any(h in hint_text or h in rubric for h in _FORMAT_HINTS)
        if not wants_structure:
            return False
        resp = record.final_response or ""
        looks_structured = ("{" in resp and "}" in resp) or ("[" in resp and "]" in resp)
        return not looks_structured

    def _infer_trigger(self, record: TraceRecord) -> Optional[str]:
        user_turns = [
            m.content for m in record.input_messages if getattr(m, "role", "") == "user"
        ]
        if not user_turns:
            return None
        return user_turns[-1][:160]
