"""Eval runner types and configuration."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


class EvalType(str, Enum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    MULTI_PARTY = "multi_party"     # V1.0


class ScoringConfig(BaseModel):
    layer1_weight: float = 0.6      # task completion
    layer2_weight: float = 0.3      # trajectory quality (milestone-based)
    layer3_weight: float = 0.1      # step quality (diagnostic, PRM)
    individual_satisfaction_mode: str = "min"  # "min" | "avg" for multi-party
    parallelism: int = 4            # concurrent eval cases
