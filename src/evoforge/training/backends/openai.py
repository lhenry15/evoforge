"""OpenAI fine-tune API backend."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel
from evoforge.training.backends.protocol import TrainingJob


class OpenAIFineTuneConfig(BaseModel):
    model: str = "gpt-4o-mini"
    n_epochs: int | str = "auto"
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"


class OpenAIFineTuneBackend:
    """
    Fine-tune backend using the OpenAI fine-tune API. No GPU required.

    Requires: pip install evoforge[openai]

    Example::

        backend = OpenAIFineTuneBackend(model="gpt-4o-mini")
        sdk.evolve.configure(
            train_evolution=TrainEvolutionConfig(fine_tune_backend=backend)
        )
    """

    def __init__(self, **config_kwargs: Any) -> None:
        self.config = OpenAIFineTuneConfig(**config_kwargs)

    def launch(self, dataset_path: str, config: dict[str, Any]) -> TrainingJob:
        # TODO: implement OpenAI fine-tune API call
        raise NotImplementedError

    def status(self, job_id: str) -> TrainingJob:
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def get_model_id(self, job_id: str) -> str:
        raise NotImplementedError
