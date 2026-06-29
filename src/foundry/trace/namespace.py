"""TraceNamespace — sdk.trace interface.

The integration glue that turns eval runs and recorded trajectories into
normalized, persisted traces ready for mining and forecasting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.core.types import EvalCase, EvalRunResult
from foundry.trace.normalizer import TraceNormalizer
from foundry.trace.schema import TraceRecord
from foundry.trace.store import TraceStore


class TraceNamespace:
    """sdk.trace — record, persist, and query normalized traces.

    Usage::

        # Convert an eval run into analyzable traces (with failure signatures)
        records = sdk.trace.record_eval_run(result, cases=cases)

        # Pull recorded telemetry into the trace store
        sdk.trace.record_trajectories("my_agent")

        # Inspect recurring failures
        sdk.trace.failures("my_agent")
        sdk.trace.recurrence_rate("my_agent")
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk
        self._normalizer = TraceNormalizer()
        self._store = TraceStore(storage_path=str(sdk.config.storage.path))

    @property
    def store(self) -> TraceStore:
        return self._store

    @property
    def normalizer(self) -> TraceNormalizer:
        return self._normalizer

    # ── Recording ─────────────────────────────────────────────────────

    def record_eval_run(
        self,
        run_result: EvalRunResult,
        cases: Optional[list[EvalCase]] = None,
        persist: bool = True,
    ) -> list[TraceRecord]:
        """Normalize an eval run into traces and (optionally) persist them."""
        records = self._normalizer.from_eval_run(run_result, cases=cases)
        if persist:
            self._store.save_many(records)
        return records

    def record_trajectories(
        self,
        agent_name: str,
        capability: Optional[str] = None,
        persist: bool = True,
    ) -> list[TraceRecord]:
        """Normalize all stored telemetry trajectories for an agent."""
        trajectories = self._sdk.data.load_trajectories(agent_name)
        records = self._normalizer.from_trajectories(trajectories, capability=capability)
        if persist:
            self._store.save_many(records)
        return records

    # ── Queries ───────────────────────────────────────────────────────

    def load(self, agent_name: str) -> list[TraceRecord]:
        return self._store.load(agent_name)

    def failures(self, agent_name: str) -> list[TraceRecord]:
        return self._store.failures(agent_name)

    def by_capability(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        return self._store.index_by_capability(agent_name)

    def by_tool(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        return self._store.index_by_tool(agent_name)

    def by_signature(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        return self._store.index_by_signature(agent_name)

    def signature_counts(self, agent_name: str) -> dict[str, int]:
        return self._store.signature_counts(agent_name)

    def recurrence_rate(self, agent_name: str) -> float:
        return self._store.recurrence_rate(agent_name)
