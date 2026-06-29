"""Trace intelligence — normalized traces, failure signatures, and lineage.

This package is the foundation for the predictive improvement loop:
analyze traces -> mine failure modes -> synthesize data -> forecast -> prevent.

Public surface::

    from evoforge.trace import (
        TraceRecord, TraceOutcome, TraceSource,
        FailureSignature, FailureMode, TraceLineage,
        TraceNormalizer, FailureSignatureExtractor, TraceStore,
    )
"""

from evoforge.trace.schema import (
    FailureMode,
    FailureSignature,
    ToolInvocation,
    TraceLineage,
    TraceOutcome,
    TraceRecord,
    TraceSource,
)
from evoforge.trace.signature import FailureSignatureExtractor
from evoforge.trace.normalizer import TraceNormalizer
from evoforge.trace.store import TraceStore

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
