"""Trace intelligence schema.

Normalized representations that downstream mining, synthesis, and forecasting
all consume. Keeping these schemas stable is the contract that lets the rest of
the predictive loop evolve independently.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import Message


class TraceSource(str, Enum):
    """Where a trace originated."""
    TELEMETRY = "telemetry"       # recorded from a live agent call
    EVAL = "eval"                 # produced during a single-turn eval
    MULTI_TURN = "multi_turn"     # produced during a multi-turn eval
    SYNTHETIC = "synthetic"       # produced by data synthesis
    UNKNOWN = "unknown"


class TraceOutcome(str, Enum):
    """Outcome classification for a single trace."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"               # the agent raised / crashed
    UNKNOWN = "unknown"


class FailureMode(str, Enum):
    """Root-cause dimensions used by mining and forecasting.

    These are intentionally coarse and stable. Phase 2 mining assigns them with
    higher fidelity; downstream synthesis and forecasting condition on them.
    """
    PROMPT_GAP = "prompt_gap"                        # instructions missing/ambiguous
    TOOL_MISUSE = "tool_misuse"                      # wrong tool, wrong args, or not called
    MISSING_KNOWLEDGE = "missing_knowledge"          # factual / domain gap
    POLICY_CONFLICT = "policy_conflict"              # violated a constraint / forbidden action
    ENVIRONMENT_FRAGILITY = "environment_fragility"  # tool/env error, timeout
    FORMAT_VIOLATION = "format_violation"            # output format/structure wrong
    HALLUCINATION = "hallucination"                  # fabricated info
    INCOMPLETE = "incomplete"                        # task left unfinished
    UNKNOWN = "unknown"


class ToolInvocation(BaseModel):
    """A single tool call within a trace, normalized across frameworks."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool = True
    error_type: Optional[str] = None
    result_summary: Optional[str] = None
    order: int = 0


class FailureSignature(BaseModel):
    """A stable, comparable fingerprint of a failure.

    The ``signature_id`` is deterministic so the same failure shape collapses to
    the same id across runs — this is what makes recurrence measurable and what
    mining/forecasting key on.
    """
    signature_id: str
    mode: FailureMode = FailureMode.UNKNOWN
    symptom: str = ""                 # short description of the observed problem
    trigger: Optional[str] = None     # input pattern that triggered it
    evidence: list[str] = Field(default_factory=list)
    capability: Optional[str] = None
    confidence: float = 0.0           # 0-1 confidence in this signature
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(mode: str, capability: Optional[str], symptom: str) -> str:
        """Deterministic signature id from the failure's defining attributes."""
        key = f"{mode}|{capability or ''}|{symptom.strip().lower()[:120]}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


class TraceLineage(BaseModel):
    """Links a trace or generated artifact to its sources and derivations.

    This is the backbone of provenance: every eval case, synthetic example, and
    fix should be traceable back to the traces/failures that motivated it.
    """
    parent_trace_ids: list[str] = Field(default_factory=list)
    eval_case_id: Optional[str] = None
    failure_signature_id: Optional[str] = None
    generation_method: Optional[str] = None   # telemetry | eval | bootstrap | expansion | synthesis
    derived_from: Optional[str] = None         # artifact id this was derived from
    tags: list[str] = Field(default_factory=list)


class TraceRecord(BaseModel):
    """Normalized trace — the canonical input for analysis, mining, and forecasting."""
    trace_id: str
    agent_name: str
    source: TraceSource = TraceSource.UNKNOWN
    capability: Optional[str] = None
    input_messages: list[Message] = Field(default_factory=list)
    final_response: str = ""
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    outcome: TraceOutcome = TraceOutcome.UNKNOWN
    score: Optional[float] = None
    latency_ms: float = 0.0
    context_hash: str = ""             # stable hash of user input for grouping/dedup
    failure_signature: Optional[FailureSignature] = None
    lineage: TraceLineage = Field(default_factory=TraceLineage)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        """Whether this trace represents a non-success outcome worth analyzing."""
        return self.outcome in (
            TraceOutcome.FAILURE,
            TraceOutcome.ERROR,
            TraceOutcome.PARTIAL,
        )

    @property
    def tool_names(self) -> list[str]:
        return [t.tool_name for t in self.tool_invocations]

    @staticmethod
    def make_context_hash(messages: list[Message]) -> str:
        """Stable hash of the user-side input, used to group similar traces."""
        user_turns = [m.content for m in messages if getattr(m, "role", "") == "user"]
        joined = "\n".join(user_turns).strip().lower()
        return hashlib.sha1(joined.encode()).hexdigest()[:12]
