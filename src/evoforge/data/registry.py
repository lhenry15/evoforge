"""Versioned data registry with lineage tracking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class DataVersion(BaseModel):
    """A single version of a data artifact."""
    version: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version: Optional[int] = None       # None for v1
    generation_method: str = "manual"          # "bootstrap", "expansion", "dpo", "manual"
    n_items: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: str = ""                         # relative path to the data file


class DataArtifact(BaseModel):
    """A versioned data artifact (eval set, train set, etc.)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                                   # e.g. "eval_cases", "train_sft", "train_dpo"
    artifact_type: str                          # "eval" | "train_sft" | "train_dpo" | "trajectories"
    agent_name: str
    versions: list[DataVersion] = Field(default_factory=list)

    @property
    def current_version(self) -> DataVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def version_count(self) -> int:
        return len(self.versions)


class DataRegistry:
    """
    Track versioned data artifacts with full lineage.

    Every eval set, train set, and DPO dataset gets versioned.
    Each version records: parent, generation method, size, timestamp.

    Usage::

        registry = DataRegistry(storage_path=".foundry/registry")
        
        # Register a new version
        registry.commit(
            name="eval_cases",
            agent_name="flight_agent",
            artifact_type="eval",
            n_items=20,
            generation_method="bootstrap",
            file_path="eval_cases/bootstrap.json",
        )
        
        # Query lineage
        history = registry.get_history("eval_cases", "flight_agent")
        print(registry.lineage_tree("eval_cases", "flight_agent"))
    """

    def __init__(self, storage_path: str) -> None:
        self._path = Path(storage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, DataArtifact] = {}
        self._load()

    def commit(
        self,
        name: str,
        agent_name: str,
        artifact_type: str,
        n_items: int,
        generation_method: str = "manual",
        file_path: str = "",
        metadata: dict[str, Any] = None,
    ) -> DataVersion:
        """
        Commit a new version of a data artifact.

        Returns the new DataVersion.
        """
        key = f"{agent_name}/{name}"
        artifact = self._artifacts.get(key)

        if artifact is None:
            artifact = DataArtifact(
                name=name, artifact_type=artifact_type, agent_name=agent_name,
            )
            self._artifacts[key] = artifact

        parent = artifact.version_count if artifact.version_count > 0 else None
        version = DataVersion(
            version=artifact.version_count + 1,
            parent_version=parent,
            generation_method=generation_method,
            n_items=n_items,
            file_path=file_path,
            metadata=metadata or {},
        )
        artifact.versions.append(version)
        self._save()
        return version

    def get_history(self, name: str, agent_name: str) -> list[DataVersion]:
        """Get full version history for an artifact."""
        key = f"{agent_name}/{name}"
        artifact = self._artifacts.get(key)
        return artifact.versions if artifact else []

    def get_current(self, name: str, agent_name: str) -> DataVersion | None:
        """Get the current (latest) version."""
        key = f"{agent_name}/{name}"
        artifact = self._artifacts.get(key)
        return artifact.current_version if artifact else None

    def list_artifacts(self, agent_name: Optional[str] = None) -> list[DataArtifact]:
        """List all artifacts, optionally filtered by agent."""
        if agent_name:
            return [a for a in self._artifacts.values() if a.agent_name == agent_name]
        return list(self._artifacts.values())

    def lineage_tree(self, name: str, agent_name: str) -> str:
        """Generate a human-readable lineage tree."""
        history = self.get_history(name, agent_name)
        if not history:
            return f"  (no history for {agent_name}/{name})"

        lines = [f"  {agent_name}/{name} — {len(history)} versions"]
        for v in history:
            parent = f"← v{v.parent_version}" if v.parent_version else "(root)"
            lines.append(
                f"    v{v.version} [{v.generation_method}] "
                f"{v.n_items} items, {v.timestamp[:16]} {parent}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """Full registry summary."""
        lines = [f"DataRegistry: {len(self._artifacts)} artifacts"]
        for key, artifact in sorted(self._artifacts.items()):
            cv = artifact.current_version
            lines.append(
                f"  {key} ({artifact.artifact_type}) — "
                f"v{artifact.version_count}, {cv.n_items if cv else 0} items"
            )
        return "\n".join(lines)

    def _save(self) -> None:
        data = {k: v.model_dump() for k, v in self._artifacts.items()}
        (self._path / "registry.json").write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        registry_file = self._path / "registry.json"
        if registry_file.exists():
            data = json.loads(registry_file.read_text())
            for k, v in data.items():
                self._artifacts[k] = DataArtifact.model_validate(v)
