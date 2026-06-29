"""TraceNormalizer — convert raw trajectories and eval results into TraceRecords.

The rest of the predictive loop should never touch raw, framework-specific
shapes. Everything goes through the normalizer first so mining and forecasting
see one consistent representation.
"""

from __future__ import annotations

from typing import Any, Optional

from foundry.core.types import EvalCase, EvalCaseResult, EvalRunResult, Trajectory
from foundry.trace.schema import (
    ToolInvocation,
    TraceLineage,
    TraceOutcome,
    TraceRecord,
    TraceSource,
)
from foundry.trace.signature import FailureSignatureExtractor

_ERROR_MARKERS = (
    "[error",
    "error:",
    "exception",
    "traceback",
    "could not complete",
    "failed to",
)


class TraceNormalizer:
    """Turn raw agent artifacts into normalized, analyzable :class:`TraceRecord`s."""

    def __init__(self, signature_extractor: Optional[FailureSignatureExtractor] = None) -> None:
        self._sig = signature_extractor or FailureSignatureExtractor()

    # ── Trajectory (telemetry) ────────────────────────────────────────

    def from_trajectory(
        self,
        traj: Trajectory,
        capability: Optional[str] = None,
        source: TraceSource = TraceSource.TELEMETRY,
    ) -> TraceRecord:
        """Normalize a recorded live trajectory.

        Without an eval score the outcome is inferred heuristically from the
        response text and tool results.
        """
        tool_invocations = self._parse_tool_calls(traj.tool_calls)
        outcome = self._infer_outcome(traj.response, tool_invocations)

        record = TraceRecord(
            trace_id=traj.id,
            agent_name=traj.agent_name,
            source=source,
            capability=capability or traj.metadata.get("capability"),
            input_messages=list(traj.messages),
            final_response=traj.response,
            tool_invocations=tool_invocations,
            outcome=outcome,
            latency_ms=traj.latency_ms,
            context_hash=TraceRecord.make_context_hash(traj.messages),
            lineage=TraceLineage(generation_method="telemetry"),
            metadata=dict(traj.metadata),
        )
        return self._attach_signature(record)

    # ── Eval results ──────────────────────────────────────────────────

    def from_eval_result(
        self,
        result: EvalCaseResult,
        case: Optional[EvalCase] = None,
        agent_name: str = "",
        source: TraceSource = TraceSource.EVAL,
    ) -> TraceRecord:
        """Normalize a single eval case result (authoritative outcome via score)."""
        messages = list(case.messages) if case else []
        outcome = TraceOutcome.SUCCESS if result.passed else TraceOutcome.FAILURE

        metadata: dict[str, Any] = {}
        if result.judge_reasoning:
            metadata["judge_reasoning"] = result.judge_reasoning
        if case and case.expected:
            metadata["expected"] = case.expected

        record = TraceRecord(
            trace_id=f"eval-{result.case_id}",
            agent_name=agent_name,
            source=source,
            capability=result.capability,
            input_messages=messages,
            final_response=result.agent_response,
            outcome=outcome,
            score=result.score,
            latency_ms=result.latency_ms,
            context_hash=TraceRecord.make_context_hash(messages) if messages else "",
            lineage=TraceLineage(eval_case_id=result.case_id, generation_method="eval"),
            metadata=metadata,
        )
        return self._attach_signature(record)

    def from_eval_run(
        self,
        run_result: EvalRunResult,
        cases: Optional[list[EvalCase]] = None,
    ) -> list[TraceRecord]:
        """Normalize every case result in an eval run."""
        case_by_id = {c.id: c for c in (cases or [])}
        return [
            self.from_eval_result(
                r, case=case_by_id.get(r.case_id), agent_name=run_result.agent_name
            )
            for r in run_result.case_results
        ]

    def from_trajectories(
        self,
        trajectories: list[Trajectory],
        capability: Optional[str] = None,
    ) -> list[TraceRecord]:
        return [self.from_trajectory(t, capability=capability) for t in trajectories]

    # ── Internals ─────────────────────────────────────────────────────

    def _attach_signature(self, record: TraceRecord) -> TraceRecord:
        if record.is_failure:
            sig = self._sig.extract(record)
            if sig is not None:
                record.failure_signature = sig
                record.lineage.failure_signature_id = sig.signature_id
        return record

    def _parse_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolInvocation]:
        """Parse heterogeneous tool-call dict shapes into ToolInvocation objects."""
        invocations: list[ToolInvocation] = []
        for i, raw in enumerate(tool_calls or []):
            if not isinstance(raw, dict):
                continue
            name, arguments = self._extract_name_args(raw)
            if not name:
                continue
            error_type = raw.get("error_type") or raw.get("error")
            succeeded = bool(raw.get("success", error_type is None))
            invocations.append(
                ToolInvocation(
                    tool_name=str(name),
                    arguments=arguments,
                    succeeded=succeeded,
                    error_type=str(error_type) if error_type else None,
                    result_summary=self._summarize(raw.get("result")),
                    order=int(raw.get("order", i)),
                )
            )
        return invocations

    @staticmethod
    def _extract_name_args(raw: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
        # OpenAI-style: {"function": {"name": ..., "arguments": ...}}
        if isinstance(raw.get("function"), dict):
            fn = raw["function"]
            args = fn.get("arguments", {})
            return fn.get("name"), args if isinstance(args, dict) else {"raw": args}
        # Flat: {"name"/"tool_name": ..., "arguments"/"args": {...}}
        name = raw.get("name") or raw.get("tool_name") or raw.get("tool")
        args = raw.get("arguments", raw.get("args", {}))
        return name, args if isinstance(args, dict) else {"raw": args}

    @staticmethod
    def _summarize(result: Any) -> Optional[str]:
        if result is None:
            return None
        return str(result)[:200]

    @staticmethod
    def _infer_outcome(response: str, tool_invocations: list[ToolInvocation]) -> TraceOutcome:
        text = (response or "").strip().lower()
        if not text:
            return TraceOutcome.FAILURE
        if text.startswith("[error") or "[error:" in text:
            return TraceOutcome.ERROR
        if any(marker in text for marker in _ERROR_MARKERS):
            return TraceOutcome.FAILURE
        if any(not t.succeeded for t in tool_invocations):
            return TraceOutcome.PARTIAL
        # Without a score we cannot assert success; treat as success-ish but
        # leave UNKNOWN so downstream does not over-trust unscored telemetry.
        return TraceOutcome.UNKNOWN
