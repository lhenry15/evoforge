"""Fine-tune backend protocol — implement to add new training backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Optional
from pydantic import BaseModel, Field


class TrainingJob(BaseModel):
    """Represents a fine-tuning job (in-progress or completed)."""
    job_id: str
    backend: str                      # "mlx_lora" | "openai" | "lora"
    status: str                       # pending | running | complete | failed
    model_id: Optional[str] = None    # adapter path or fine-tuned model ID
    base_model: Optional[str] = None  # the model that was fine-tuned
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Training metrics (populated on completion)
    train_loss: float = 0.0
    val_loss: float = 0.0
    iters_completed: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("complete", "failed")

    @property
    def succeeded(self) -> bool:
        return self.status == "complete"


@runtime_checkable
class FineTuneBackend(Protocol):
    """
    Protocol for fine-tune backends.

    Built-in implementations:
      - MLXLoRABackend   (Apple Silicon, local, mlx-lm)
      - LoRABackend      (HuggingFace PEFT, local GPU)
      - OpenAIFineTuneBackend (OpenAI fine-tune API, cloud)

    Lifecycle:
      1. launch()        — start a fine-tune job (sync or async depending on backend)
      2. status()        — poll job status (for async backends)
      3. get_model_id()  — retrieve fine-tuned model identifier
      4. validate()      — run a quick inference to verify the model works
    """

    def launch(self, dataset_path: str, config: dict[str, Any] = None) -> TrainingJob:
        """
        Launch a fine-tune job.

        Args:
            dataset_path: Path to JSONL training data (chat format).
            config:       Backend-specific overrides (iters, lr, etc.)

        Returns:
            TrainingJob — may be immediately complete (local backends)
            or pending/running (cloud backends).
        """
        ...

    def status(self, job_id: str) -> TrainingJob:
        """Poll job status. Returns current TrainingJob state."""
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel a running job. No-op if already terminal."""
        ...

    def get_model_id(self, job_id: str) -> str:
        """
        Return model identifier once job is complete.

        For local backends: adapter path or fused model directory.
        For cloud backends: fine-tuned model ID (e.g. "ft:gpt-4o:org:suffix").
        """
        ...

    def validate(self, job_id: str, prompt: str, system_prompt: str = "") -> str:
        """
        Run a single inference with the fine-tuned model to verify it works.

        Args:
            job_id:        Completed job to validate.
            prompt:        Test prompt to send.
            system_prompt: Optional system prompt.

        Returns:
            Model response string.

        Raises:
            RuntimeError if job is not complete or inference fails.
        """
        ...
