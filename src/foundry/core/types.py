"""Shared types used across all Foundry modules."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CODE = "code"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    RAG = "rag"
    PLANNING = "planning"
    CONVERSATION = "conversation"
    COMPOSITE = "composite"


class DataFormat(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    PRM = "prm"
    TOOL_TRACES = "tool_traces"


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class EvalLayer(str, Enum):
    TASK_COMPLETION = "task_completion"
    TRAJECTORY_QUALITY = "trajectory_quality"
    STEP_QUALITY = "step_quality"


class GenerationMode(str, Enum):
    SCRATCH = "scratch"
    MUTATION = "mutation"
    TRAJECTORY_MINING = "trajectory_mining"
    SELF_PLAY = "self_play"
    DISTILLATION = "distillation"


class ConflictType(str, Enum):
    RESOURCE = "resource"
    PRIORITY = "priority"
    PREFERENCE = "preference"
    PERMISSION = "permission"


class ScoringMethod(str, Enum):
    """How to score a single eval case."""
    EXACT_MATCH = "exact_match"          # response == expected (case-insensitive)
    CONTAINS = "contains"                # expected substring in response
    LLM_JUDGE = "llm_judge"             # LLM rates response vs rubric
    REGEX = "regex"                      # response matches regex pattern
    TOOL_CALL_CHECK = "tool_call_check"  # correct tools were called


class EvolutionAction(str, Enum):
    """Action recommended by the evolution engine after a cycle."""
    GENERATE_TRAIN_DATA = "generate_train_data"   # capability failing → data gen
    EXPAND_EVAL = "expand_eval"                   # capability saturating → add cases
    TRIGGER_FINE_TUNE = "trigger_fine_tune"        # enough data accumulated → FT
    EMIT_PROMOTION_EVENT = "emit_promotion_event" # FT done, no swap_model fn
    NO_ACTION = "no_action"                        # all scores in healthy range


class Message(BaseModel):
    role: str                   # "user" | "assistant" | "tool" | "system"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolResult(BaseModel):
    call_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabelResult(BaseModel):
    label: Any
    score: float
    grade: Grade
    confidence: float
    strategy_used: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Eval types ────────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    """A single evaluation case."""
    id: str
    messages: list[Message]           # conversation to send to agent
    expected: str                      # expected answer / reference
    capability: str                    # e.g. "flight_search", "error_recovery"
    scoring_method: ScoringMethod = ScoringMethod.LLM_JUDGE
    scoring_rubric: Optional[str] = None   # extra instructions for LLM judge
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    """Score for a single eval case."""
    case_id: str
    capability: str
    agent_response: str
    score: float                       # 0.0 - 1.0
    passed: bool                       # score >= 0.6
    judge_reasoning: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunResult(BaseModel):
    """Aggregate results from a full eval run."""
    agent_name: str
    overall_score: float
    capability_scores: dict[str, float]  # capability → avg score
    case_results: list[EvalCaseResult]
    n_passed: int
    n_total: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """A recorded agent interaction — stored for training data curation."""
    id: str
    agent_name: str
    messages: list[Message]
    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvolutionDecision(BaseModel):
    """Output of a single evolution cycle."""
    agent_name: str
    actions: list[EvolutionAction]
    capability_gaps: list["CapabilityGap"]
    saturation_signals: list["SaturationSignal"]
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Evolution signal types ────────────────────────────────────────────────────

class PromotionEvent(BaseModel):
    agent_name: str
    old_model_id: str
    new_model_id: str
    eval_score_before: float
    eval_score_after: float
    improvement: float
    fine_tune_job_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityGap(BaseModel):
    capability: str
    score: float
    threshold: float
    suggested_n_examples: int


class SaturationSignal(BaseModel):
    capability: str
    score: float
    threshold: float
    suggested_expansion: int
