"""AgentEvolutionHistory — persistent lifecycle tracking for a single agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvolutionEvent(BaseModel):
    """A single event in an agent's evolution history."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    event_type: str         # "eval" | "prompt_change" | "skill_added" | "skill_refined" |
                            # "skill_retired" | "training_started" | "training_complete" |
                            # "ab_test_passed" | "ab_test_failed" | "model_promoted" |
                            # "eval_expanded" | "bootstrap"
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentSnapshot(BaseModel):
    """A point-in-time snapshot of agent state."""
    version: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    system_prompt: str = ""
    skill_prompts: dict[str, str] = Field(default_factory=dict)
    model_id: str = ""
    adapter_path: Optional[str] = None
    eval_score: Optional[float] = None
    capability_scores: dict[str, float] = Field(default_factory=dict)
    trigger: str = ""           # what caused this snapshot ("initial", "prompt_evolution", "lora_training", etc.)


class AgentEvolutionHistory:
    """
    Persistent evolution history for a single agent.

    Tracks everything that happens to the agent over its lifetime:
      - Every eval run (scores, failures)
      - Every prompt change (before/after)
      - Every skill added/refined/retired
      - Every training job (data used, result)
      - Every promotion decision (A/B test outcome)
      - Snapshots at each version (full agent state)

    Storage: .foundry/agents/{agent_name}/history.json

    Usage::

        history = AgentEvolutionHistory(agent_name="flight_agent", storage_path=".foundry")
        history.record_eval(score=0.5, capability_scores={...}, failures=[...])
        history.record_prompt_change(old="...", new="...", reason="...")
        history.record_training(job_id="...", n_examples=12, result="complete")
        history.snapshot(system_prompt="...", model_id="...", eval_score=0.8)
    """

    def __init__(self, agent_name: str, storage_path: str) -> None:
        self.agent_name = agent_name
        self._path = Path(storage_path) / "agents" / agent_name
        self._path.mkdir(parents=True, exist_ok=True)
        self._history_file = self._path / "history.json"
        self._events: list[EvolutionEvent] = []
        self._snapshots: list[AgentSnapshot] = []
        self._load()

    # ── Recording events ──────────────────────────────────────────────

    def record_eval(
        self,
        score: float,
        capability_scores: dict[str, float],
        n_passed: int = 0,
        n_total: int = 0,
        failures: list[dict[str, Any]] = None,
    ) -> None:
        """Record an eval run."""
        self._events.append(EvolutionEvent(
            event_type="eval",
            detail={
                "score": score,
                "capability_scores": capability_scores,
                "n_passed": n_passed,
                "n_total": n_total,
                "failures": (failures or [])[:10],  # cap at 10
            },
        ))
        self._save()

    def record_prompt_change(
        self, old_prompt: str, new_prompt: str, reason: str
    ) -> None:
        """Record a system prompt change."""
        self._events.append(EvolutionEvent(
            event_type="prompt_change",
            detail={
                "old": old_prompt[:500],
                "new": new_prompt[:500],
                "reason": reason,
            },
        ))
        self._save()

    def record_skill_added(self, name: str, content: str, capability: str) -> None:
        self._events.append(EvolutionEvent(
            event_type="skill_added",
            detail={"name": name, "content": content[:300], "capability": capability},
        ))
        self._save()

    def record_skill_refined(self, name: str, old_content: str, new_content: str, reason: str) -> None:
        self._events.append(EvolutionEvent(
            event_type="skill_refined",
            detail={"name": name, "old": old_content[:200], "new": new_content[:200], "reason": reason},
        ))
        self._save()

    def record_skill_retired(self, name: str, reason: str) -> None:
        self._events.append(EvolutionEvent(
            event_type="skill_retired",
            detail={"name": name, "reason": reason},
        ))
        self._save()

    def record_training(
        self, job_id: str, n_examples: int, result: str,
        train_loss: float = 0, val_loss: float = 0, adapter_path: str = "",
    ) -> None:
        self._events.append(EvolutionEvent(
            event_type="training_complete" if result == "complete" else "training_started",
            detail={
                "job_id": job_id, "n_examples": n_examples, "result": result,
                "train_loss": train_loss, "val_loss": val_loss, "adapter_path": adapter_path,
            },
        ))
        self._save()

    def record_ab_test(self, passed: bool, old_score: float, new_score: float, regressions: list = None) -> None:
        self._events.append(EvolutionEvent(
            event_type="ab_test_passed" if passed else "ab_test_failed",
            detail={
                "old_score": old_score, "new_score": new_score,
                "improvement": new_score - old_score, "regressions": regressions or [],
            },
        ))
        self._save()

    def record_promotion(self, old_model: str, new_model: str) -> None:
        self._events.append(EvolutionEvent(
            event_type="model_promoted",
            detail={"old_model": old_model, "new_model": new_model},
        ))
        self._save()

    def record_eval_expanded(self, n_old: int, n_new: int, capabilities: list[str]) -> None:
        self._events.append(EvolutionEvent(
            event_type="eval_expanded",
            detail={"n_old": n_old, "n_new": n_new, "capabilities": capabilities},
        ))
        self._save()

    def record_bootstrap(self, n_cases: int, capabilities: list[str]) -> None:
        self._events.append(EvolutionEvent(
            event_type="bootstrap",
            detail={"n_cases": n_cases, "capabilities": capabilities},
        ))
        self._save()

    # ── Snapshots ─────────────────────────────────────────────────────

    def snapshot(
        self,
        system_prompt: str = "",
        skill_prompts: dict[str, str] = None,
        model_id: str = "",
        adapter_path: Optional[str] = None,
        eval_score: Optional[float] = None,
        capability_scores: dict[str, float] = None,
        trigger: str = "",
    ) -> AgentSnapshot:
        """Take a point-in-time snapshot of the agent's full state."""
        snap = AgentSnapshot(
            version=len(self._snapshots) + 1,
            system_prompt=system_prompt,
            skill_prompts=skill_prompts or {},
            model_id=model_id,
            adapter_path=adapter_path,
            eval_score=eval_score,
            capability_scores=capability_scores or {},
            trigger=trigger,
        )
        self._snapshots.append(snap)
        self._save()
        return snap

    # ── Queries ───────────────────────────────────────────────────────

    @property
    def events(self) -> list[EvolutionEvent]:
        return self._events

    @property
    def snapshots(self) -> list[AgentSnapshot]:
        return self._snapshots

    @property
    def current_snapshot(self) -> AgentSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def n_versions(self) -> int:
        return len(self._snapshots)

    def get_eval_history(self) -> list[dict[str, Any]]:
        """Get all eval events in chronological order."""
        return [e.detail for e in self._events if e.event_type == "eval"]

    def get_score_trend(self) -> list[float]:
        """Get score progression over time."""
        return [e.detail.get("score", 0) for e in self._events if e.event_type == "eval"]

    def get_events_by_type(self, event_type: str) -> list[EvolutionEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def summary(self) -> str:
        """Human-readable summary."""
        evals = self.get_events_by_type("eval")
        prompts = self.get_events_by_type("prompt_change")
        skills_added = self.get_events_by_type("skill_added")
        trainings = [e for e in self._events if "training" in e.event_type]
        promotions = self.get_events_by_type("model_promoted")

        lines = [
            f"Agent: {self.agent_name}",
            f"Versions: {self.n_versions}",
            f"Events: {len(self._events)} total",
            f"  Evals: {len(evals)}",
            f"  Prompt changes: {len(prompts)}",
            f"  Skills added: {len(skills_added)}",
            f"  Training jobs: {len(trainings)}",
            f"  Promotions: {len(promotions)}",
        ]
        if evals:
            scores = [e.detail.get("score", 0) for e in evals]
            lines.append(f"Score trend: {' → '.join(f'{s:.3f}' for s in scores)}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "agent_name": self.agent_name,
            "events": [e.model_dump() for e in self._events],
            "snapshots": [s.model_dump() for s in self._snapshots],
        }
        self._history_file.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if self._history_file.exists():
            data = json.loads(self._history_file.read_text())
            self._events = [EvolutionEvent.model_validate(e) for e in data.get("events", [])]
            self._snapshots = [AgentSnapshot.model_validate(s) for s in data.get("snapshots", [])]

    def to_dict(self) -> dict[str, Any]:
        """Export full history as dict (for dashboard)."""
        return {
            "agent_name": self.agent_name,
            "n_versions": self.n_versions,
            "events": [e.model_dump() for e in self._events],
            "snapshots": [s.model_dump() for s in self._snapshots],
            "score_trend": self.get_score_trend(),
        }
