"""DriftMonitor — detect distribution shift between reference and recent windows.

Drift on the input/failure distribution is an early warning that a forecaster
trained on past data may be going stale. We combine the change in failure rate
with the Jensen-Shannon divergence of the capability distribution.
"""

from __future__ import annotations

import math
from collections import Counter

from foundry.forecast.schema import DriftReport
from foundry.trace.schema import TraceRecord


class DriftMonitor:
    """Compare a recent window of traces against a fitted reference window."""

    def __init__(self, drift_threshold: float = 0.15) -> None:
        self._threshold = drift_threshold
        self._ref_failrate: float = 0.0
        self._ref_caps: dict[str, float] = {}
        self._fitted = False

    def fit(self, reference_traces: list[TraceRecord]) -> "DriftMonitor":
        labeled = [t for t in reference_traces if _labeled(t)]
        self._ref_failrate = _failure_rate(labeled)
        self._ref_caps = _distribution([t.capability or "unknown" for t in reference_traces])
        self._fitted = True
        return self

    def compare(self, recent_traces: list[TraceRecord]) -> DriftReport:
        labeled = [t for t in recent_traces if _labeled(t)]
        recent_failrate = _failure_rate(labeled)
        recent_caps = _distribution([t.capability or "unknown" for t in recent_traces])

        delta = round(recent_failrate - self._ref_failrate, 4)
        js = round(_js_divergence(self._ref_caps, recent_caps), 4)
        drift_score = round(0.5 * abs(delta) + 0.5 * js, 4)

        return DriftReport(
            failure_rate_reference=round(self._ref_failrate, 4),
            failure_rate_recent=round(recent_failrate, 4),
            failure_rate_delta=delta,
            capability_js_divergence=js,
            drift_score=drift_score,
            drifted=drift_score > self._threshold,
            details={
                "reference_capabilities": self._ref_caps,
                "recent_capabilities": recent_caps,
            },
        )


def _labeled(trace: TraceRecord) -> bool:
    from foundry.trace.schema import TraceOutcome

    return trace.outcome != TraceOutcome.UNKNOWN


def _failure_rate(traces: list[TraceRecord]) -> float:
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.is_failure) / len(traces)


def _distribution(values: list[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for k, pv in p.items():
        if pv <= 0:
            continue
        qv = q.get(k, 0.0)
        if qv <= 0:
            continue
        total += pv * math.log(pv / qv)
    return total
