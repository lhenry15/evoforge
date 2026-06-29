"""
Foundry: A data-centric SDK for self-evolving LLM agents.

Quick start:
    import foundry

    sdk = foundry.init(task_spec="A flight booking assistant")

    @sdk.agent(tools=[search_flights, book_flight])
    def my_agent(messages): ...

    sdk.bootstrap(agent=my_agent)
    sdk.evolve.run_cycle(agent=my_agent)
"""

from foundry.core.config import SDKConfig
from foundry.core.sdk import FoundrySDK
from foundry.core.types import (
    DataFormat,
    Grade,
    TaskType,
    Message,
    PromotionEvent,
    CapabilityGap,
    SaturationSignal,
)
from foundry.core.agent_config import AgentConfig, ModelConfig, ModelHost
from foundry.bootstrap.zero_shot import ZeroShotSource
from foundry.environment.protocol import EnvironmentProtocol
from foundry.environment.connectors.http import HTTPSandboxConnector
from foundry.environment.connectors.mcp import MCPConnector
from foundry.eval.runner import EvalType
from foundry.evolution.engine import EvolutionTriggers
from foundry.training.backends.protocol import FineTuneBackend
from foundry.training.backends.lora import LoRABackend
from foundry.training.backends.openai import OpenAIFineTuneBackend
from foundry.training.curator import CurriculumConfig, TrainEvolutionConfig
from foundry.eval.scoring import ScoringConfig
from foundry.factory.labeler import LabelStrategy
from foundry.collection.telemetry import CollectConfig
from foundry.core.privacy import PrivacyConfig, PrivacyRules
from foundry.trace import (
    FailureMode,
    FailureSignature,
    TraceLineage,
    TraceNormalizer,
    TraceOutcome,
    TraceRecord,
    TraceSource,
    TraceStore,
)
from foundry.mining import (
    FailureClusterer,
    FailureModeCluster,
    FailureModeMiner,
    FailureModeReport,
    MiningResult,
)
from foundry.synthesis import (
    DataSynthesizer,
    ModeConditionedGenerator,
    QualityGate,
    SynthesisResult,
    SyntheticExample,
    SynthFormat,
)
from foundry.coverage import (
    AdaptiveEvalExpander,
    Blindspot,
    CoverageCell,
    CoverageMap,
    CoverageMapper,
    CoverageReport,
)
from foundry.forecast import (
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
        sdk = foundry.init(task_spec="A customer support agent for a SaaS product")
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
