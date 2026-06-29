"""Schema for adaptive eval coverage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from foundry.mining.schema import severity_of
from foundry.trace.schema import FailureMode


def severity_of_value(mode_value: str) -> float:
    """Severity lookup tolerant of raw mode strings."""
    try:
        return severity_of(FailureMode(mode_value))
    except ValueError:
        return 0.4


class CoverageCell(BaseModel):
    """One (capability x failure-mode) cell of the coverage map."""
    capability: str
    mode: str
    observed_failures: int = 0           # demand — real failures of this kind
    eval_cases: int = 0                  # supply — eval cases probing this cell
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_blindspot(self) -> bool:
        """Real failures observed but no targeted eval coverage at all."""
        return self.observed_failures > 0 and self.eval_cases == 0

    @property
    def is_undercovered(self) -> bool:
        """More real failures than eval cases probing them."""
        return self.observed_failures > self.eval_cases


class Blindspot(BaseModel):
    """A (capability x mode) cell that needs targeted eval generation."""
    capability: str
    mode: str
    observed_failures: int
    severity: float
    impact: float                        # observed_failures * severity
    suggested_cases: int
    example_inputs: list[str] = Field(default_factory=list)


class CoverageMap(BaseModel):
    """Demand-vs-supply coverage across capabilities and failure modes."""
    agent_name: str = ""
    cells: list[CoverageCell] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)
    n_eval_cases: int = 0
    n_tagged_cases: int = 0

    def blindspots(self) -> list[Blindspot]:
        out: list[Blindspot] = []
        for c in self.cells:
            if not c.is_blindspot:
                continue
            sev = float(c.metadata.get("severity", severity_of_value(c.mode)))
            out.append(
                Blindspot(
                    capability=c.capability,
                    mode=c.mode,
                    observed_failures=c.observed_failures,
                    severity=sev,
                    impact=round(c.observed_failures * sev, 4),
                    suggested_cases=min(5, max(2, c.observed_failures)),
                    example_inputs=list(c.metadata.get("example_inputs", []))[:5],
                )
            )
        out.sort(key=lambda b: b.impact, reverse=True)
        return out

    def undercovered(self) -> list[CoverageCell]:
        return [c for c in self.cells if c.is_undercovered]

    def coverage_ratio(self) -> float:
        """Fraction of demand cells that have at least one targeted eval case."""
        demand = [c for c in self.cells if c.observed_failures > 0]
        if not demand:
            return 1.0
        covered = sum(1 for c in demand if c.eval_cases > 0)
        return round(covered / len(demand), 4)

    def matrix(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Heatmap-ready nested dict: capability -> mode -> cell summary."""
        m: dict[str, dict[str, dict[str, Any]]] = {cap: {} for cap in self.capabilities}
        for c in self.cells:
            m.setdefault(c.capability, {})[c.mode] = {
                "observed": c.observed_failures,
                "eval_cases": c.eval_cases,
                "blindspot": c.is_blindspot,
                "undercovered": c.is_undercovered,
            }
        return m
