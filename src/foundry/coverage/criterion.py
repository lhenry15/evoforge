"""Success-criterion derivation for adaptive eval generation.

Small models are unreliable at writing *discriminating* pass/fail criteria. So we
derive the criterion once per (capability, failure-mode) and then *construct* each
case's ``expected`` and ``scoring_rubric`` deterministically from it — instead of
hoping the 3B model writes a strict rubric for every case (it doesn't; it tends to
write lenient rubrics that the failing agent would pass).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.llm.structured import generate_structured

_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "success_signal": {"type": "string"},
        "fail_signal": {"type": "string"},
        "success_keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["success_signal", "fail_signal", "success_keywords"],
}

_SYSTEM = (
    "You define STRICT, observable pass/fail criteria for evaluating one AI agent "
    "capability. Write success_signal in THIRD PERSON describing what the agent MUST "
    "do — e.g. 'The agent confirms the total price and asks the user to approve before "
    "booking.' Do NOT write it as the agent's reply and do NOT use first person ('I')."
)


class SuccessCriterion(BaseModel):
    """A discriminating definition of success/failure for a capability+mode."""
    capability: str
    mode: str
    success_signal: str          # what a CORRECT response must do (observable)
    fail_signal: str             # what the failing response does
    success_keywords: list[str] = Field(default_factory=list)

    def expected(self) -> str:
        return self.success_signal

    def rubric(self) -> str:
        kw = ", ".join(self.success_keywords[:6])
        kw_clause = f" Indicators of success include: {kw}." if kw else ""
        return (
            f"PASS only if the agent's response clearly does this: {self.success_signal}."
            f"{kw_clause} "
            f"FAIL if the response merely does this: {self.fail_signal}, "
            f"or omits the required success behavior."
        )


class CriterionDeriver:
    """Derive a :class:`SuccessCriterion` for a (capability, mode) blind spot."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._cache: dict[tuple[str, str], SuccessCriterion] = {}

    def derive(
        self,
        capability: str,
        mode: str,
        task_spec: str,
        symptom: str = "",
        failing_inputs: Optional[list[str]] = None,
    ) -> SuccessCriterion:
        key = (capability, mode)
        if key in self._cache:
            return self._cache[key]

        seeds = "\n".join(f'- "{s[:120]}"' for s in (failing_inputs or [])[:4]) or "(none)"
        prompt = (
            f"AGENT TASK: {task_spec}\n"
            f"CAPABILITY: {capability}\n"
            f"OBSERVED FAILURE MODE: {mode}\n"
            f"FAILURE SYMPTOM: {symptom or '(unspecified)'}\n"
            f"REAL FAILING INPUTS:\n{seeds}\n\n"
            "Define: success_signal (what a CORRECT response MUST do, observable), "
            "fail_signal (what the failing response does instead), and "
            "success_keywords (concrete words/phrases a passing response should contain)."
        )
        parsed = generate_structured(
            self._pool, prompt, _CRITERION_SCHEMA, system=_SYSTEM,
            temperature=0.3, max_tokens=400,
        )
        criterion = self._build(capability, mode, symptom, parsed)
        self._cache[key] = criterion
        return criterion

    def _build(self, capability: str, mode: str, symptom: str, parsed: Any) -> SuccessCriterion:
        success = fail = ""
        keywords: list[str] = []
        if isinstance(parsed, dict):
            success = str(parsed.get("success_signal", "")).strip()
            fail = str(parsed.get("fail_signal", "")).strip()
            kw = parsed.get("success_keywords", [])
            if isinstance(kw, list):
                keywords = [str(k).strip() for k in kw if str(k).strip()][:8]

        success = self._normalize_signal(success, capability)

        # Robust fallback if derivation was empty/garbled.
        if not success:
            success = (
                f"The agent fully completes the '{capability}' request and returns a "
                f"concrete, correct result rather than only partial or intermediate output."
            )
        if not fail:
            fail = symptom or f"exhibits the '{mode}' failure mode"

        return SuccessCriterion(
            capability=capability,
            mode=mode,
            success_signal=success,
            fail_signal=fail,
            success_keywords=keywords,
        )

    @staticmethod
    def _normalize_signal(success: str, capability: str) -> str:
        """Reframe role-played / first-person success signals into a third-person spec."""
        if not success:
            return success
        low = success.lstrip().lower()
        first_person = (
            low.startswith("i ") or low.startswith("i'") or low.startswith("i’")
            or low.startswith("here ") or low.startswith("sure")
        )
        if first_person:
            # The model wrote the agent's reply instead of a spec; wrap it as a spec.
            return f"The agent's response for '{capability}' must achieve: {success}"
        return success
