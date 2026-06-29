"""Telemetry collection configuration."""

from __future__ import annotations

from pydantic import BaseModel


class CollectConfig(BaseModel):
    trajectories: bool = True           # log all tool calls + reasoning steps
    user_feedback: bool = True          # capture thumbs up/down, corrections
    turn_level_scores: bool = False     # per-turn PRM scoring (V0.2)
    cost_and_latency: bool = True       # token costs + response latency
