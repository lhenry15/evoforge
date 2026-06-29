"""Adaptive eval coverage.

Phase 3b of the predictive loop: keep the benchmark aligned with the failure
modes that actually occur. We compare *demand* (mined failures per
capability x mode) against *supply* (eval cases that probe that cell), surface
blind spots, and generate targeted eval cases that close them.

Public surface::

    from evoforge.coverage import (
        CoverageMapper, CoverageMap, CoverageCell, Blindspot,
        AdaptiveEvalExpander, CoverageReport,
    )
"""

from evoforge.coverage.schema import Blindspot, CoverageCell, CoverageMap
from evoforge.coverage.mapper import CoverageMapper
from evoforge.coverage.expander import AdaptiveEvalExpander
from evoforge.coverage.report import CoverageReport

__all__ = [
    "Blindspot",
    "CoverageCell",
    "CoverageMap",
    "CoverageMapper",
    "AdaptiveEvalExpander",
    "CoverageReport",
]
