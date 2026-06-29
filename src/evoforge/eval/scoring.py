"""Eval scoring configuration."""

from __future__ import annotations

from pydantic import BaseModel


class ScoringConfig(BaseModel):
    """Weights and modes for the 3-layer eval scoring hierarchy."""
    layer1_weight: float = 0.6      # task completion (outcome)
    layer2_weight: float = 0.3      # trajectory quality (milestone-based)
    layer3_weight: float = 0.1      # step quality (diagnostic, PRM)
    individual_satisfaction_mode: str = "min"  # "min" | "avg" for multi-party
    parallelism: int = 4            # concurrent eval cases
