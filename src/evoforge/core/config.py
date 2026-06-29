"""SDK configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    backend: str = "local"
    path: Path = Path.home() / "agent-foundry" / ".foundry"
    project_path: Optional[Path] = None  # if set, uses ./.foundry/ in project dir

    model_config = {"arbitrary_types_allowed": True}


class LLMConfig(BaseModel):
    provider: str = "copilot"
    ensemble_size: int = 3          # number of models used for UQ ensemble
    temperature: float = 0.7
    max_tokens: int = 2048


class SDKConfig(BaseModel):
    task_spec: str
    llm_provider: str = "copilot"
    storage_backend: str = "local"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    project_name: Optional[str] = None
    verbose: bool = False
