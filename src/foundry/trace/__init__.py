"""Trace intelligence — normalized traces, failure signatures, and lineage.

This package is the foundation for the predictive improvement loop:
analyze traces -> mine failure modes -> synthesize data -> forecast -> prevent.

Public surface::

    from foundry.trace import (
        TraceRecord, TraceOutcome, TraceSource,
        FailureSignature, FailureMode, TraceLineage,
        TraceNormalizer, FailureSignatureExtractor, TraceStore,
    )
"""

from foundry.trace.schema import (
    FailureMode,
    FailureSignature,
    ToolInvocation,
    TraceLineage,
    TraceOutcome,
    TraceRecord,
    TraceSource,
)
from foundry.trace.signature import FailureSignatureExtractor
from foundry.trace.normalizer import TraceNormalizer
from foundry.trace.store import TraceStore

__all__ = [
    "FailureMode",
    "FailureSignature",
    "ToolInvocation",
    "TraceLineage",
    "TraceOutcome",
    "TraceRecord",
    "TraceSource",
    "FailureSignatureExtractor",
    "TraceNormalizer",
    "TraceStore",
]
