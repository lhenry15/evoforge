"""Schema + scoring constants for failure-mode mining."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.trace.schema import FailureMode

# Severity weights per mode — how damaging a failure of this kind is when it
# occurs. Impact ranking multiplies frequency by severity so a rare but severe
# mode can outrank a frequent but cosmetic one.
SEVERITY: dict[FailureMode, float] = {
    FailureMode.POLICY_CONFLICT: 1.0,
    FailureMode.HALLUCINATION: 0.9,
    FailureMode.TOOL_MISUSE: 0.8,
    FailureMode.MISSING_KNOWLEDGE: 0.7,
    FailureMode.ENVIRONMENT_FRAGILITY: 0.7,
    FailureMode.INCOMPLETE: 0.6,
    FailureMode.PROMPT_GAP: 0.6,
    FailureMode.FORMAT_VIOLATION: 0.5,
    FailureMode.UNKNOWN: 0.4,
}

# Maps a failure mode to the family of fix the downstream engine should prefer.
# This is the bridge to Phase 5 (proactive fixing).
FIX_TYPE: dict[FailureMode, str] = {
    FailureMode.PROMPT_GAP: "prompt",
    FailureMode.FORMAT_VIOLATION: "prompt",
    FailureMode.TOOL_MISUSE: "skill",
    FailureMode.MISSING_KNOWLEDGE: "training",
    FailureMode.POLICY_CONFLICT: "policy",
    FailureMode.ENVIRONMENT_FRAGILITY: "workflow",
    FailureMode.HALLUCINATION: "training",
    FailureMode.INCOMPLETE: "workflow",
    FailureMode.UNKNOWN: "investigate",
}


def severity_of(mode: FailureMode) -> float:
    return SEVERITY.get(mode, 0.4)


def fix_type_of(mode: FailureMode) -> str:
    return FIX_TYPE.get(mode, "investigate")


class FailureExample(BaseModel):
    """A representative failing instance within a cluster."""
    trace_id: str
    trigger: Optional[str] = None
    response: str = ""
    signature_id: str = ""


class FailureModeCluster(BaseModel):
    """A discovered cluster of related failures, treated as one actionable mode."""
    cluster_id: str
    mode: FailureMode
    capability: Optional[str] = None
    label: str = ""                 # human-readable root cause
    symptom_summary: str = ""
    suggested_fix_type: str = "investigate"

    signature_ids: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    size: int = 0

    # Scoring
    severity: float = 0.0
    impact_score: float = 0.0       # size * severity
    cohesion: float = 0.0           # dominance of the top signature in the cluster
    reproducibility: float = 0.0    # fraction of repeated inputs (same context_hash)
    stability: float = 0.0          # composite of cohesion + reproducibility
    confidence: float = 0.0

    examples: list[FailureExample] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MiningResult(BaseModel):
    """Output of a mining run over an agent's failing traces."""
    agent_name: str
    n_failures: int = 0
    n_clusters: int = 0
    coverage_top10: float = 0.0     # fraction of failures explained by top-10 clusters
    mode_distribution: dict[str, int] = Field(default_factory=dict)
    clusters: list[FailureModeCluster] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def top_clusters(self) -> list[FailureModeCluster]:
        return self.clusters[:10]

    def by_fix_type(self) -> dict[str, list[FailureModeCluster]]:
        out: dict[str, list[FailureModeCluster]] = {}
        for c in self.clusters:
            out.setdefault(c.suggested_fix_type, []).append(c)
        return out
