"""Agent configuration — optionally expose agent internals to unlock advanced features."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional
from pydantic import BaseModel, Field


class ModelHost(str, Enum):
    """Where the agent's model is hosted."""
    OPENAI      = "openai"
    AZURE       = "azure"
    ANTHROPIC   = "anthropic"
    TOGETHER    = "together"
    HUGGINGFACE = "huggingface"
    LOCAL       = "local"     # Ollama, vLLM, LM Studio
    CUSTOM      = "custom"    # any arbitrary endpoint


class ModelConfig(BaseModel):
    """Identity and hosting of the agent's underlying model."""
    id: str                           # e.g. "gpt-4o", "llama-3-8b-instruct"
    host: ModelHost = ModelHost.OPENAI
    endpoint: Optional[str] = None       # custom base URL (required for CUSTOM/LOCAL)
    context_window: Optional[int] = None # used to truncate data to fit context


class AgentConfig(BaseModel):
    """
    Optional configuration exposing agent internals to Foundry.

    Without this, Foundry works as a black box (data collection + eval).
    With this, Foundry can also do prompt evolution and auto fine-tuning.

    Fields
    ------
    system_prompt:
        The agent's system prompt. Enables the SkillRegistry to version,
        evolve, and A/B test prompt changes independently of the model.

    model:
        The model identity and hosting. Enables FineTuneTrigger to select
        the right backend and know which base model to fine-tune.

    skill_prompts:
        Named sub-prompts used by the agent (e.g. tool-specific instructions).
        Each entry is independently versioned in the SkillRegistry.

    swap_model:
        A callable invoked by Foundry when a fine-tuned model passes A/B
        testing and is ready to promote.

        Signature: (new_model_id: str) -> None

        If None, Foundry emits a PromotionEvent and the developer
        handles the swap. The fine-tuned model ID is always available
        in PromotionEvent.new_model_id regardless.

    Example::

        AgentConfig(
            system_prompt="You are a helpful flight booking assistant...",
            model=ModelConfig(id="gpt-4o", host=ModelHost.OPENAI),
            skill_prompts={
                "handle_timeout": "If a tool call times out, retry once...",
                "multi_leg":      "For multi-leg flights, search each leg...",
            },
            swap_model=lambda new_id: my_agent.set_model(new_id),
        )

    Degradation without AgentConfig
    --------------------------------
    No system_prompt  →  skill evolution suggests prompts; developer applies
    No model config   →  training data curated; developer triggers fine-tune
    No swap_model fn  →  PromotionEvent emitted; developer swaps model
    """

    model_config = {"arbitrary_types_allowed": True}

    system_prompt: Optional[str] = None
    model: ModelConfig | None = None
    skill_prompts: dict[str, str] = Field(default_factory=dict)
    swap_model: Callable[[str], None] | None = None
