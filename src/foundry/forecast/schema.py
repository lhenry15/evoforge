"""Schema for failure forecasting."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.core.types import Message


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ForecastRequest(BaseModel):
    """An incoming request to assess for failure risk (before/while running)."""
    messages: list[Message] = Field(default_factory=list)
    capability: Optional[str] = None
    tool_invocations: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, capability: Optional[str] = None) -> "ForecastRequest":
        return cls(messages=[Message(role="user", content=text)], capability=capability)

    @property
    def text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "user")


class Forecast(BaseModel):
    """The predicted risk for a single request."""
    p_failure: float
    risk_level: RiskLevel
    likely_mode: str = "unknown"
    mode_probabilities: dict[str, float] = Field(default_factory=dict)
    capability: Optional[str] = None
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    novelty: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastEvaluation(BaseModel):
    """Comparison of the model against naive baselines on a labeled holdout."""
    n: int
    base_rate: float
    model_brier: float
    majority_brier: float
    capability_brier: float
    model_accuracy: float
    model_auc: Optional[float] = None
    beats_majority: bool = False
    beats_capability: bool = False
    method: str = "resubstitution"   # resubstitution | cross_validation | holdout
    n_folds: Optional[int] = None
    honest: bool = False             # True when evaluated on held-out predictions
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationBin(BaseModel):
    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_rate: float


class CalibrationReport(BaseModel):
    """Reliability of predicted probabilities."""
    n: int
    brier: float
    ece: float                       # expected calibration error
    bins: list[CalibrationBin] = Field(default_factory=list)
    within_tolerance: bool = False   # ece <= tolerance


class DriftReport(BaseModel):
    """Distribution shift between a reference window and a recent window."""
    failure_rate_reference: float
    failure_rate_recent: float
    failure_rate_delta: float
    capability_js_divergence: float
    drift_score: float
    drifted: bool
    details: dict[str, Any] = Field(default_factory=dict)
