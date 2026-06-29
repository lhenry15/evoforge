"""Failure-mode mining.

Phase 2 of the predictive loop: turn raw failing traces into a small set of
stable, ranked, actionable failure modes with root-cause labels, impact, and
stability scores. Consumes :class:`foundry.trace.TraceRecord`s.

Public surface::

    from foundry.mining import (
        FailureModeMiner, FailureClusterer,
        FailureModeCluster, MiningResult, FailureModeReport,
        SEVERITY, FIX_TYPE,
    )
"""

from foundry.mining.schema import (
    FIX_TYPE,
    SEVERITY,
    FailureModeCluster,
    MiningResult,
)
from foundry.mining.clusterer import FailureClusterer
from foundry.mining.miner import FailureModeMiner
from foundry.mining.llm_classifier import LLMModeClassifier
from foundry.mining.report import FailureModeReport

__all__ = [
    "FIX_TYPE",
    "SEVERITY",
    "FailureModeCluster",
    "MiningResult",
    "FailureClusterer",
    "FailureModeMiner",
    "LLMModeClassifier",
    "FailureModeReport",
]
