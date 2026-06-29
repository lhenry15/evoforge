"""Training data curation configuration."""

from __future__ import annotations

from pydantic import BaseModel


class CurriculumConfig(BaseModel):
    progression: str = "easy_to_hard"   # "easy_to_hard" | "random" | "hard_first"
    replay_buffer_size: int = 1000


class TrainEvolutionConfig(BaseModel):
    fine_tune_backend: str = "lora"     # "lora" | "openai" | instance
    dpo_margin_threshold: float = 0.2   # minimum quality gap for valid DPO pair
    teacher_model: str = "gpt-4o"       # teacher for current-vs-teacher DPO
    curriculum: CurriculumConfig = CurriculumConfig()
    ab_test_before_promote: bool = True
