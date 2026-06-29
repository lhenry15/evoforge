"""Zero-shot bootstrap source — generates eval + train data from task spec only."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ZeroShotSource(BaseModel):
    """
    Bootstrap data from task spec + tool manifest with no human seed data.

    Uses multi-LLM generation (CopilotLLMPool by default) to create
    diverse, difficulty-calibrated eval and training data.

    Example::

        sdk.bootstrap(
            agent=my_agent,
            source=ZeroShotSource(
                num_eval_cases=200,
                difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
                persona_diversity=True,
            )
        )
    """

    num_eval_cases: int = 200
    num_train_cases: int = 500
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.2, "medium": 0.5, "hard": 0.3}
    )
    persona_diversity: bool = True           # inject varied user personas
    conflict_injection: bool = False         # for multi-party agents
    adversarial_ratio: float = 0.2          # fraction generated via self-play
    seed: Optional[int] = None                  # for reproducibility
