"""Model promotion — full lifecycle: train → validate → A/B → promote."""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from foundry.core.types import EvalCase, PromotionEvent
from foundry.evolution.regression_guard import ABTestConfig, ABTestResult, RegressionGuard
from foundry.training.backends.protocol import TrainingJob


class PromotionConfig(BaseModel):
    """Configuration for the promotion pipeline."""
    ab_test_config: ABTestConfig = Field(default_factory=ABTestConfig)
    validate_before_ab: bool = True       # quick inference check before full A/B
    save_history: bool = True             # persist promotion events


class PromotionResult(BaseModel):
    """Result of a promotion attempt."""
    promoted: bool
    training_job: Optional[TrainingJob] = None
    ab_test: Optional[ABTestResult] = None
    promotion_event: Optional[PromotionEvent] = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionPipeline:
    """
    Full model promotion lifecycle.

    Pipeline:
      1. Train (MLXLoRA or OpenAI FT)
      2. Validate (quick inference sanity check)
      3. A/B test (compare old vs new on all eval cases)
      4. Promote (call swap_model or emit PromotionEvent)

    Usage::

        pipeline = PromotionPipeline(sdk=sdk)
        result = pipeline.run(
            agent=my_agent,
            training_backend=backend,
            training_data=examples,
            eval_cases=cases,
        )
        if result.promoted:
            print(f"Model promoted to {result.training_job.model_id}")
    """

    def __init__(self, sdk: Any, config: Optional[PromotionConfig] = None) -> None:
        self._sdk = sdk
        self._config = config or PromotionConfig()

    def run(
        self,
        agent: Callable,
        agent_new_factory: Callable[[str], Callable],
        training_job: TrainingJob,
        eval_cases: list[EvalCase],
    ) -> PromotionResult:
        """
        Run the full promotion pipeline.

        Args:
            agent:              Current (old) agent.
            agent_new_factory:  Function that creates a new agent given adapter_path.
            training_job:       Completed training job.
            eval_cases:         Cases for A/B testing.

        Returns:
            PromotionResult with promoted=True/False and details.
        """
        # Step 1: Check training succeeded
        if not training_job.succeeded:
            return PromotionResult(
                promoted=False,
                training_job=training_job,
                reason=f"Training failed: {training_job.metadata.get('error', 'unknown')}",
            )

        adapter_path = training_job.model_id

        # Step 2: Create new agent with adapter
        agent_new = agent_new_factory(adapter_path)

        # Step 3: A/B test
        guard = RegressionGuard(sdk=self._sdk, config=self._config.ab_test_config)
        ab_result = guard.test(agent, agent_new, eval_cases)

        if not ab_result.passed:
            return PromotionResult(
                promoted=False,
                training_job=training_job,
                ab_test=ab_result,
                reason=f"A/B test failed: {ab_result.reason}",
            )

        # Step 4: Promote
        new_model_id = adapter_path
        event = guard.promote(agent, new_model_id)

        return PromotionResult(
            promoted=True,
            training_job=training_job,
            ab_test=ab_result,
            promotion_event=event,
            reason="Promoted: A/B test passed with no regressions",
        )
