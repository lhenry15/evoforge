"""CoverageMapper — build a demand-vs-supply coverage map.

Demand comes from mined failure modes (what actually breaks). Supply comes from
eval cases that are *tagged* with the failure mode they probe (``target_mode``
or ``failure_mode`` in case metadata). Untagged cases contribute capability-
level context but not mode-level coverage — which is exactly why blind spots are
detectable: a mode can have heavy demand and zero targeted supply.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from evoforge.core.types import EvalCase
from evoforge.coverage.schema import CoverageCell, CoverageMap, severity_of_value
from evoforge.mining.schema import MiningResult


class CoverageMapper:
    """Construct a :class:`CoverageMap` from eval cases + mined failures."""

    def build(
        self,
        eval_cases: list[EvalCase],
        mining_result: MiningResult,
        agent_name: str = "",
    ) -> CoverageMap:
        demand = self._demand(mining_result)
        supply, n_tagged = self._supply(eval_cases)

        keys = sorted(set(demand) | set(supply))
        cells: list[CoverageCell] = []
        for cap, mode in keys:
            dem = demand.get((cap, mode), {})
            sup = supply.get((cap, mode), {})
            cells.append(
                CoverageCell(
                    capability=cap,
                    mode=mode,
                    observed_failures=int(dem.get("count", 0)),
                    eval_cases=int(sup.get("count", 0)),
                    difficulty_counts=dict(sup.get("difficulty", {})),
                    metadata={
                        "severity": dem.get("severity", severity_of_value(mode)),
                        "example_inputs": dem.get("inputs", [])[:5],
                    },
                )
            )

        return CoverageMap(
            agent_name=agent_name or mining_result.agent_name,
            cells=cells,
            capabilities=sorted({c.capability for c in cells}),
            modes=sorted({c.mode for c in cells}),
            n_eval_cases=len(eval_cases),
            n_tagged_cases=n_tagged,
        )

    # ── demand / supply ───────────────────────────────────────────────

    @staticmethod
    def _demand(mining_result: MiningResult) -> dict[tuple[str, str], dict[str, Any]]:
        demand: dict[tuple[str, str], dict[str, Any]] = {}
        for cluster in mining_result.clusters:
            cap = cluster.capability or "unknown"
            mode = cluster.mode.value
            key = (cap, mode)
            entry = demand.setdefault(
                key, {"count": 0, "severity": cluster.severity, "inputs": []}
            )
            entry["count"] += cluster.size
            entry["inputs"].extend(e.trigger for e in cluster.examples if e.trigger)
        return demand

    @staticmethod
    def _supply(eval_cases: list[EvalCase]) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
        supply: dict[tuple[str, str], dict[str, Any]] = {}
        n_tagged = 0
        for case in eval_cases:
            mode = CoverageMapper._tagged_mode(case)
            if not mode:
                continue
            n_tagged += 1
            cap = case.capability or "unknown"
            difficulty = str(case.metadata.get("difficulty", "unknown"))
            entry = supply.setdefault((cap, mode), {"count": 0, "difficulty": Counter()})
            entry["count"] += 1
            entry["difficulty"][difficulty] += 1
        # Counters -> plain dicts
        for entry in supply.values():
            entry["difficulty"] = dict(entry["difficulty"])
        return supply, n_tagged

    @staticmethod
    def _tagged_mode(case: EvalCase) -> Optional[str]:
        return case.metadata.get("target_mode") or case.metadata.get("failure_mode")
