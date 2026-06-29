"""LoRA fine-tune backend via HuggingFace PEFT."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel
from foundry.training.backends.protocol import TrainingJob


class LoRAConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    output_dir: str = "./foundry-lora-output"


class LoRABackend:
    """
    Fine-tune backend using LoRA via HuggingFace PEFT + TRL.

    Requires: pip install evoforge[lora]

    Example::

        backend = LoRABackend(base_model="meta-llama/Llama-3-8B-Instruct", rank=16)
        sdk.evolve.configure(
            train_evolution=TrainEvolutionConfig(fine_tune_backend=backend)
        )
    """

    def __init__(self, **config_kwargs: Any) -> None:
        self.config = LoRAConfig(**config_kwargs)

    def launch(self, dataset_path: str, config: dict[str, Any]) -> TrainingJob:
        # TODO: implement HuggingFace PEFT + TRL training loop
        raise NotImplementedError

    def status(self, job_id: str) -> TrainingJob:
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def get_model_id(self, job_id: str) -> str:
        raise NotImplementedError
