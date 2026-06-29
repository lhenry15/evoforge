"""Targeted synthetic data generation.

Phase 3a of the predictive loop: generate high-value, *mode-conditioned*
training data that specifically teaches the agent to handle discovered failure
modes — and pass it through quality gates so only novel, consistent, deduped
examples survive. Every example carries lineage back to the failure cluster it
targets.

Public surface::

    from evoforge.synthesis import (
        DataSynthesizer, ModeConditionedGenerator, QualityGate,
        SyntheticExample, SynthesisResult, QualityReport, SynthFormat,
    )

Scenario-driven synthesis (the seed backbone) adds three composable components:

    from evoforge.synthesis import (
        Seed, SeedGenerator, SimParticipant,           # (1) control surface
        ConversationGenerator, SimTranscript, SimTurn,  # (2) conversation
        Label, LabelSchema, LabelField, SchemaLabeler,  # (3) open labeler set
        REGISTRY, certify_labeler, label_transcript,
    )
"""

from evoforge.synthesis.schema import (
    DEFAULT_FORMAT,
    QualityReport,
    SynthesisResult,
    SyntheticExample,
    SynthFormat,
    default_format_for,
)
from evoforge.synthesis.seed import Seed, SeedGenerator, SimParticipant
from evoforge.synthesis.conversation import (
    ConversationGenerator,
    SimTranscript,
    SimTurn,
)
from evoforge.synthesis.labeler import (
    REGISTRY,
    AVOIDS_FAILURE_SCHEMA,
    CertProbe,
    CertReport,
    Label,
    Labeler,
    LabelerRegistry,
    LabelField,
    LabelSchema,
    PRESENCE_SCHEMA,
    SchemaLabeler,
    certify_labeler,
    label_transcript,
)
from evoforge.synthesis.pipeline import LabeledDataset, ScenarioSynthesizer
from evoforge.synthesis.generator import ModeConditionedGenerator
from evoforge.synthesis.quality import QualityGate
from evoforge.synthesis.synthesizer import DataSynthesizer

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
    # scenario-driven synthesis (the seed backbone)
    "Seed",
    "SeedGenerator",
    "SimParticipant",
    "ConversationGenerator",
    "SimTranscript",
    "SimTurn",
    "ScenarioSynthesizer",
    "LabeledDataset",
    "Label",
    "Labeler",
    "LabelField",
    "LabelSchema",
    "LabelerRegistry",
    "SchemaLabeler",
    "REGISTRY",
    "PRESENCE_SCHEMA",
    "AVOIDS_FAILURE_SCHEMA",
    "CertProbe",
    "CertReport",
    "certify_labeler",
    "label_transcript",
]
