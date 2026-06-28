"""Label strategy types."""

from __future__ import annotations

from enum import Enum


class LabelStrategy(str, Enum):
    AUTO = "auto"                   # select highest-trust strategy for task type
    EXECUTABLE = "executable"       # run code / verify math / env feedback
    ENSEMBLE = "ensemble"           # multi-model LLM judge consensus
    SINGLE_JUDGE = "single_judge"   # single LLM judge (Grade C max)
    REWARD_MODEL = "reward_model"   # trained reward model
    HUMAN = "human"                 # HITL queue
