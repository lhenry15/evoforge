"""Schema for targeted synthetic data generation."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.core.types import DataFormat
from foundry.factory.data_factory import TrainingExample
from foundry.trace.schema import FailureMode, TraceLineage


class SynthFormat(str, Enum):
    """Output format for a synthetic example."""
    SFT = "sft"               # (instruction, ideal_response)
    DPO = "dpo"               # (instruction, chosen, rejected)
    TOOL_TRACE = "tool_trace"  # (instruction, ideal_response, tool_calls)


# Per-mode default format. Modes where the failure is a *behavioral preference*
# (refusing, hallucinating) benefit from DPO contrast against the real failing
# response; tool issues benefit from explicit tool traces; the rest use SFT.
DEFAULT_FORMAT: dict[FailureMode, SynthFormat] = {
    FailureMode.TOOL_MISUSE: SynthFormat.TOOL_TRACE,
    FailureMode.POLICY_CONFLICT: SynthFormat.DPO,
    FailureMode.HALLUCINATION: SynthFormat.DPO,
    FailureMode.PROMPT_GAP: SynthFormat.SFT,
    FailureMode.FORMAT_VIOLATION: SynthFormat.SFT,
    FailureMode.MISSING_KNOWLEDGE: SynthFormat.SFT,
    FailureMode.INCOMPLETE: SynthFormat.SFT,
    FailureMode.ENVIRONMENT_FRAGILITY: SynthFormat.SFT,
    FailureMode.UNKNOWN: SynthFormat.SFT,
}


def default_format_for(mode: FailureMode) -> SynthFormat:
    return DEFAULT_FORMAT.get(mode, SynthFormat.SFT)


class QualityReport(BaseModel):
    """Outcome of running a synthetic example through the quality gates."""
    accepted: bool = False
    confidence: float = 0.0       # combined novelty + consistency
    novelty: float = 0.0          # 0-1, higher = less similar to existing corpus
    consistency: float = 0.0      # 0-1, judge agreement the example is correct
    duplicate_of: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)


class SyntheticExample(BaseModel):
    """A single mode-conditioned synthetic example with provenance."""
    id: str = Field(default_factory=lambda: f"syn-{uuid.uuid4().hex[:8]}")
    target_mode: FailureMode
    target_cluster_id: str = ""
    capability: Optional[str] = None
    format: SynthFormat = SynthFormat.SFT

    instruction: str
    ideal_response: str = ""          # SFT / tool_trace
    chosen: Optional[str] = None      # DPO positive
    rejected: Optional[str] = None    # DPO negative (often a *real* failing response)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    lineage: TraceLineage = Field(default_factory=TraceLineage)
    quality: Optional[QualityReport] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_training_example(self) -> TrainingExample:
        """Convert to the existing TrainingExample for training backends."""
        fmt = {
            SynthFormat.SFT: DataFormat.SFT,
            SynthFormat.TOOL_TRACE: DataFormat.TOOL_TRACES,
            SynthFormat.DPO: DataFormat.DPO,
        }[self.format]
        response = self.ideal_response or self.chosen or ""
        return TrainingExample(
            id=self.id,
            capability=self.capability or "",
            instruction=self.instruction,
            ideal_response=response,
            tool_calls=self.tool_calls,
            format=fmt,
            metadata={
                "synthetic": True,
                "target_mode": self.target_mode.value,
                "target_cluster_id": self.target_cluster_id,
                "rejected": self.rejected,
                "confidence": self.quality.confidence if self.quality else None,
                "lineage": self.lineage.model_dump(),
            },
        )


class SynthesisResult(BaseModel):
    """Output of a synthesis run."""
    agent_name: str = ""
    accepted: list[SyntheticExample] = Field(default_factory=list)
    rejected: list[SyntheticExample] = Field(default_factory=list)
    n_generated: int = 0
    acceptance_rate: float = 0.0
    per_mode_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def training_examples(self) -> list[TrainingExample]:
        return [e.to_training_example() for e in self.accepted]
