"""DataNamespace — sdk.data interface for persisting eval cases and trajectories."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evoforge.core.sdk import FoundrySDK

from evoforge.core.types import EvalCase, EvalRunResult, Trajectory


class DataNamespace:
    """
    sdk.data — save and load eval cases, trajectories, and eval results.

    Usage::

        sdk.data.save_eval_cases(cases)
        cases = sdk.data.load_eval_cases()

        sdk.data.save_trajectory(traj)
        trajs = sdk.data.load_trajectories(agent_name="my_agent")
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk
        from evoforge.data.storage.local import LocalStorageBackend
        self._store = LocalStorageBackend(base_path=str(sdk.config.storage.path))

    # ── Eval cases ────────────────────────────────────────────────────

    def save_eval_cases(self, cases: list[EvalCase], tag: str = "default") -> str:
        """Persist eval cases. Returns the storage key."""
        key = f"eval_cases/{tag}.json"
        data = json.dumps([c.model_dump() for c in cases], indent=2).encode()
        self._store.write(key, data)
        return key

    def load_eval_cases(self, tag: str = "default") -> list[EvalCase]:
        """Load previously saved eval cases."""
        key = f"eval_cases/{tag}.json"
        raw = self._store.read(key)
        if raw is None:
            return []
        return [EvalCase(**d) for d in json.loads(raw)]

    def list_eval_tags(self) -> list[str]:
        keys = self._store.list("eval_cases/")
        return [k.replace("eval_cases/", "").replace(".json", "") for k in keys]

    # ── Trajectories ──────────────────────────────────────────────────

    def save_trajectory(self, traj: Trajectory) -> str:
        """Persist a single trajectory. Returns the storage key."""
        key = f"trajectories/{traj.agent_name}/{traj.id}.json"
        self._store.write(key, json.dumps(traj.model_dump(), indent=2).encode())
        return key

    def load_trajectories(self, agent_name: str) -> list[Trajectory]:
        """Load all trajectories recorded for an agent."""
        prefix = f"trajectories/{agent_name}/"
        keys = self._store.list(prefix)
        trajs = []
        for k in keys:
            raw = self._store.read(k)
            if raw:
                trajs.append(Trajectory(**json.loads(raw)))
        return trajs

    # ── Eval results ──────────────────────────────────────────────────

    def save_eval_result(self, result: EvalRunResult) -> str:
        """Persist an eval run result for historical comparison."""
        run_id = str(uuid.uuid4())[:8]
        key = f"eval_results/{result.agent_name}/{run_id}.json"
        self._store.write(key, json.dumps(result.model_dump(), indent=2).encode())
        return key

    def load_eval_results(self, agent_name: str) -> list[EvalRunResult]:
        """Load all historical eval run results for an agent."""
        prefix = f"eval_results/{agent_name}/"
        keys = self._store.list(prefix)
        results = []
        for k in keys:
            raw = self._store.read(k)
            if raw:
                results.append(EvalRunResult(**json.loads(raw)))
        return results
