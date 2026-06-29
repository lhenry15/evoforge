"""Failure-mode mining.

Phase 2 of the predictive loop: turn raw failing traces into a small set of
stable, ranked, actionable failure modes with root-cause labels, impact, and
stability scores. Consumes :class:`evoforge.trace.TraceRecord`s.

Public surface::

    from evoforge.mining import (
        FailureModeMiner, FailureClusterer,
        FailureModeCluster, MiningResult, FailureModeReport,
        SEVERITY, FIX_TYPE,
    )
"""

from evoforge.mining.schema import (
    FIX_TYPE,
    SEVERITY,
    FailureModeCluster,
    MiningResult,
)
from evoforge.mining.clusterer import FailureClusterer
from evoforge.mining.miner import FailureModeMiner
from evoforge.mining.llm_classifier import LLMModeClassifier
from evoforge.mining.report import FailureModeReport

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
