"""TraceStore — persist normalized traces and provide indexed retrieval.

Phase 1 deliverable: fast retrieval by capability, tool, and failure signature.
Storage is plain JSON files under ``<storage>/traces/<agent>/`` so it composes
with the existing local-first storage model and is trivially inspectable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from foundry.trace.schema import TraceRecord


class TraceStore:
    """File-backed store for :class:`TraceRecord`s with in-memory indexes."""

    def __init__(self, storage_path: str) -> None:
        self._root = Path(storage_path) / "traces"
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, record: TraceRecord) -> str:
        agent_dir = self._root / record.agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        path = agent_dir / f"{record.trace_id}.json"
        path.write_text(record.model_dump_json(indent=2))
        return str(path)

    def save_many(self, records: list[TraceRecord]) -> int:
        for record in records:
            self.save(record)
        return len(records)

    def load(self, agent_name: str) -> list[TraceRecord]:
        agent_dir = self._root / agent_name
        if not agent_dir.exists():
            return []
        records: list[TraceRecord] = []
        for f in sorted(agent_dir.glob("*.json")):
            try:
                records.append(TraceRecord.model_validate_json(f.read_text()))
            except (json.JSONDecodeError, ValueError):
                continue
        return records

    # ── Filtered views ────────────────────────────────────────────────

    def failures(self, agent_name: str) -> list[TraceRecord]:
        return [r for r in self.load(agent_name) if r.is_failure]

    # ── Indexes ───────────────────────────────────────────────────────

    def index_by_capability(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        index: dict[str, list[TraceRecord]] = defaultdict(list)
        for r in self.load(agent_name):
            index[r.capability or "unknown"].append(r)
        return dict(index)

    def index_by_tool(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        index: dict[str, list[TraceRecord]] = defaultdict(list)
        for r in self.load(agent_name):
            for name in r.tool_names:
                index[name].append(r)
        return dict(index)

    def index_by_signature(self, agent_name: str) -> dict[str, list[TraceRecord]]:
        index: dict[str, list[TraceRecord]] = defaultdict(list)
        for r in self.load(agent_name):
            if r.failure_signature is not None:
                index[r.failure_signature.signature_id].append(r)
        return dict(index)

    def signature_counts(self, agent_name: str) -> dict[str, int]:
        """Recurrence counts per failure signature (drives the recurrence KPI)."""
        return {
            sig_id: len(records)
            for sig_id, records in self.index_by_signature(agent_name).items()
        }

    def recurrence_rate(self, agent_name: str) -> float:
        """Fraction of failures that share a signature with another failure.

        This is the core Phase 2/3 KPI input: high recurrence means the loop is
        not actually fixing root causes.
        """
        failures = self.failures(agent_name)
        signed = [r for r in failures if r.failure_signature is not None]
        if not signed:
            return 0.0
        counts = defaultdict(int)
        for r in signed:
            counts[r.failure_signature.signature_id] += 1
        recurring = sum(n for n in counts.values() if n > 1)
        return round(recurring / len(signed), 4)
