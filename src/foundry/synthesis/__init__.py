"""Targeted synthetic data generation.

Phase 3a of the predictive loop: generate high-value, *mode-conditioned*
training data that specifically teaches the agent to handle discovered failure
modes — and pass it through quality gates so only novel, consistent, deduped
examples survive. Every example carries lineage back to the failure cluster it
targets.

Public surface::

    from foundry.synthesis import (
        DataSynthesizer, ModeConditionedGenerator, QualityGate,
        SyntheticExample, SynthesisResult, QualityReport, SynthFormat,
    )
"""

from foundry.synthesis.schema import (
    DEFAULT_FORMAT,
    QualityReport,
    SynthesisResult,
    SyntheticExample,
    SynthFormat,
    default_format_for,
)
from foundry.synthesis.generator import ModeConditionedGenerator
from foundry.synthesis.quality import QualityGate
from foundry.synthesis.synthesizer import DataSynthesizer

__all__ = [
    "DEFAULT_FORMAT",
    "QualityReport",
    "SynthesisResult",
    "SyntheticExample",
    "SynthFormat",
    "default_format_for",
    "ModeConditionedGenerator",
    "QualityGate",
    "DataSynthesizer",
]
