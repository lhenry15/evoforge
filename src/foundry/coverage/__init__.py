"""Adaptive eval coverage.

Phase 3b of the predictive loop: keep the benchmark aligned with the failure
modes that actually occur. We compare *demand* (mined failures per
capability x mode) against *supply* (eval cases that probe that cell), surface
blind spots, and generate targeted eval cases that close them.

Public surface::

    from foundry.coverage import (
        CoverageMapper, CoverageMap, CoverageCell, Blindspot,
        AdaptiveEvalExpander, CoverageReport,
    )
"""

from foundry.coverage.schema import Blindspot, CoverageCell, CoverageMap
from foundry.coverage.mapper import CoverageMapper
from foundry.coverage.expander import AdaptiveEvalExpander
from foundry.coverage.report import CoverageReport

__all__ = [
    "Blindspot",
    "CoverageCell",
    "CoverageMap",
    "CoverageMapper",
    "AdaptiveEvalExpander",
    "CoverageReport",
]
