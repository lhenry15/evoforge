"""
EvoForge: A data-centric SDK for self-evolving LLM agents.

Quick start:
    import evoforge

    sdk = evoforge.init(task_spec="A flight booking assistant")

    @sdk.agent(tools=[search_flights, book_flight])
    def my_agent(messages): ...

    sdk.bootstrap(agent=my_agent)
    sdk.evolve.run_cycle(agent=my_agent)
"""

from evoforge.core.config import SDKConfig
from evoforge.core.sdk import FoundrySDK
from evoforge.core.types import (
    DataFormat,
    Grade,
    TaskType,
    Message,
    PromotionEvent,
    CapabilityGap,
    SaturationSignal,
)
from evoforge.core.agent_config import AgentConfig, ModelConfig, ModelHost
from evoforge.bootstrap.zero_shot import ZeroShotSource
from evoforge.environment.protocol import EnvironmentProtocol
from evoforge.environment.connectors.http import HTTPSandboxConnector
from evoforge.environment.connectors.mcp import MCPConnector
from evoforge.eval.runner import EvalType
from evoforge.eval.simulator import (
    ConversationSimulator,
    ScriptedSimulator,
    SimParticipant,
    SimScenario,
    SimTranscript,
    SimTurn,
)
from evoforge.evolution.engine import EvolutionTriggers
from evoforge.training.backends.protocol import FineTuneBackend
from evoforge.training.backends.lora import LoRABackend
from evoforge.training.backends.openai import OpenAIFineTuneBackend
from evoforge.training.curator import CurriculumConfig, TrainEvolutionConfig
from evoforge.eval.scoring import ScoringConfig
from evoforge.factory.labeler import LabelStrategy
from evoforge.collection.telemetry import CollectConfig
from evoforge.core.privacy import PrivacyConfig, PrivacyRules
from evoforge.trace import (
    FailureMode,
    FailureSignature,
    TraceLineage,
    TraceNormalizer,
    TraceOutcome,
    TraceRecord,
    TraceSource,
    TraceStore,
)
from evoforge.mining import (
    FailureClusterer,
    FailureModeCluster,
    FailureModeMiner,
    FailureModeReport,
    MiningResult,
)
from evoforge.synthesis import (
    DataSynthesizer,
    ModeConditionedGenerator,
    QualityGate,
    SynthesisResult,
    SyntheticExample,
    SynthFormat,
)
from evoforge.coverage import (
    AdaptiveEvalExpander,
    Blindspot,
    CoverageCell,
    CoverageMap,
    CoverageMapper,
    CoverageReport,
)
from evoforge.forecast import (
    DriftMonitor,
    DriftReport,
    Forecast,
    ForecastEvaluation,
    ForecastRequest,
    RiskForecaster,
    RiskLevel,
)


def init(
    task_spec: str,
    llm_provider: str = "copilot",
    storage_backend: str = "local",
    **kwargs,
) -> FoundrySDK:
    """Initialize a Foundry SDK instance.

    Args:
        task_spec: Natural language description of what the agent does.
        llm_provider: LLM pool to use for multi-model generation.
                      Defaults to "copilot" (Copilot SDK).
        storage_backend: Where to persist data artifacts.
                         Defaults to "local" (filesystem).

    Returns:
        A configured FoundrySDK instance.

    Example:
        sdk = evoforge.init(task_spec="A customer support agent for a SaaS product")
    """
    config = SDKConfig(
        task_spec=task_spec,
        llm_provider=llm_provider,
        storage_backend=storage_backend,
        **kwargs,
    )
    return FoundrySDK(config)


__all__ = [
    "init",
    "SDKConfig",
    "FoundrySDK",
    "DataFormat",
    "Grade",
    "TaskType",
    "Message",
    "ZeroShotSource",
    "EnvironmentProtocol",
    "HTTPSandboxConnector",
    "MCPConnector",
    "EvalType",
    "ConversationSimulator",
    "ScriptedSimulator",
    "SimParticipant",
    "SimScenario",
    "SimTranscript",
    "SimTurn",
    "EvolutionTriggers",
    "FineTuneBackend",
    "LoRABackend",
    "OpenAIFineTuneBackend",
    "CurriculumConfig",
    "TrainEvolutionConfig",
    "ScoringConfig",
    "LabelStrategy",
    "CollectConfig",
    "PrivacyConfig",
    "PrivacyRules",
    "AgentConfig",
    "ModelConfig",
    "ModelHost",
    "PromotionEvent",
    "CapabilityGap",
    "SaturationSignal",
    "FailureMode",
    "FailureSignature",
    "TraceLineage",
    "TraceNormalizer",
    "TraceOutcome",
    "TraceRecord",
    "TraceSource",
    "TraceStore",
    "FailureClusterer",
    "FailureModeCluster",
    "FailureModeMiner",
    "FailureModeReport",
    "MiningResult",
    "DataSynthesizer",
    "ModeConditionedGenerator",
    "QualityGate",
    "SynthesisResult",
    "SyntheticExample",
    "SynthFormat",
    "AdaptiveEvalExpander",
    "Blindspot",
    "CoverageCell",
    "CoverageMap",
    "CoverageMapper",
    "CoverageReport",
    "DriftMonitor",
    "DriftReport",
    "Forecast",
    "ForecastEvaluation",
    "ForecastRequest",
    "RiskForecaster",
    "RiskLevel",
]
