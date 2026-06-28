"""Evolution engine triggers and configuration."""

from __future__ import annotations

from pydantic import BaseModel
from foundry.core.types import PromotionEvent, CapabilityGap, SaturationSignal


class EvolutionTriggers(BaseModel):
    """Thresholds that govern when each evolution type fires."""

    # Eval evolution: capability score above this → too easy → expand
    eval_saturation_threshold: float = 0.85

    # Train evolution: capability score below this → failing → target
    train_failure_threshold: float = 0.60

    # Fine-tune trigger: accumulate this many new trajectories before considering FT
    train_data_threshold: int = 500

    # Drift trigger: input distribution shift above this → trigger cycle
    drift_sensitivity: float = 0.15

    # Eval expansion: grow eval set by this factor when saturated
    eval_expansion_factor: float = 1.5

    # Promotion: new model must beat current by this margin to be promoted
    promotion_threshold: float = 0.05


def _can_fine_tune(agent: object) -> tuple[bool, str]:
    """
    Check whether an agent has sufficient AgentConfig for auto fine-tuning.

    Returns (can_fine_tune, reason_if_not).
    """
    config = getattr(agent, "_foundry_agent_config", None)
    if config is None:
        return False, "no AgentConfig provided — add config=AgentConfig(model=...) to @sdk.agent"
    if config.model is None:
        return False, "AgentConfig.model is None — set model=ModelConfig(id=..., host=...) to enable"
    return True, ""


def _can_auto_promote(agent: object) -> tuple[bool, str]:
    """
    Check whether an agent has a swap_model fn for auto-promotion.

    Returns (can_promote, reason_if_not).
    """
    config = getattr(agent, "_foundry_agent_config", None)
    if config is None or config.swap_model is None:
        return False, (
            "no swap_model fn in AgentConfig — "
            "Foundry will emit PromotionEvent instead of auto-swapping"
        )
    return True, ""


def _get_system_prompt(agent: object) -> str | None:
    """Extract system_prompt from AgentConfig if available."""
    config = getattr(agent, "_foundry_agent_config", None)
    if config is None:
        return None
    return config.system_prompt


def _get_model_id(agent: object) -> str | None:
    """Extract model.id from AgentConfig if available."""
    config = getattr(agent, "_foundry_agent_config", None)
    if config is None or config.model is None:
        return None
    return config.model.id

