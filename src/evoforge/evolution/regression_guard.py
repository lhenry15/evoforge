"""Regression guard — A/B test before model promotion."""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import EvalCase, PromotionEvent


class ABTestConfig(BaseModel):
    """Configuration for A/B testing."""
    promotion_margin: float = 0.05   # new must beat old by this much
    regression_threshold: float = 0.1  # max allowed drop per capability
    min_cases: int = 3                 # minimum eval cases to trust result


class ABTestResult(BaseModel):
    """Result of an A/B comparison between old and new model."""
    passed: bool
    old_score: float
    new_score: float
    improvement: float
    regressions: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


class RegressionGuard:
    """
    Prevents model promotion when new adapter causes regressions.

    Logic:
      1. Eval current agent on all cases → baseline scores
      2. Eval new agent (with adapter) on same cases → new scores
      3. Compare per-capability:
         - Overall must improve by >= promotion_margin
         - No single capability can drop by > regression_threshold
      4. If passes → promote (call swap_model or emit PromotionEvent)
         If fails → reject adapter, keep current model

    Usage::

        guard = RegressionGuard(sdk=sdk, config=ABTestConfig())
        result = guard.test(
            agent_old=current_agent,
            agent_new=finetuned_agent,
            cases=eval_cases,
        )
        if result.passed:
            guard.promote(agent_old, new_model_id)
    """

    def __init__(self, sdk: Any, config: Optional[ABTestConfig] = None) -> None:
        self._sdk = sdk
        self._config = config or ABTestConfig()

    def test(
        self,
        agent_old: Callable,
        agent_new: Callable,
        cases: list[EvalCase],
    ) -> ABTestResult:
        """
        Run A/B test comparing old agent vs new agent.

        Returns ABTestResult indicating whether promotion is safe.
        """
        if len(cases) < self._config.min_cases:
            return ABTestResult(
                passed=False, old_score=0, new_score=0, improvement=0,
                reason=f"Not enough eval cases ({len(cases)} < {self._config.min_cases})",
            )

        # Eval both agents on same cases
        old_result = self._sdk.eval.run(agent=agent_old, cases=cases, parallelism=1)
        new_result = self._sdk.eval.run(agent=agent_new, cases=cases, parallelism=1)

        improvement = new_result.overall_score - old_result.overall_score

        # Check per-capability regressions
        regressions = []
        for cap, old_score in old_result.capability_scores.items():
            new_score = new_result.capability_scores.get(cap, 0.0)
            drop = old_score - new_score
            if drop > self._config.regression_threshold:
                regressions.append({
                    "capability": cap,
                    "old_score": old_score,
                    "new_score": new_score,
                    "drop": drop,
                })

        # Decision
        if regressions:
            return ABTestResult(
                passed=False,
                old_score=old_result.overall_score,
                new_score=new_result.overall_score,
                improvement=improvement,
                regressions=regressions,
                reason=f"Regression in {len(regressions)} capabilities: {[r['capability'] for r in regressions]}",
            )

        if improvement < self._config.promotion_margin:
            return ABTestResult(
                passed=False,
                old_score=old_result.overall_score,
                new_score=new_result.overall_score,
                improvement=improvement,
                reason=f"Improvement {improvement:.3f} < margin {self._config.promotion_margin}",
            )

        return ABTestResult(
            passed=True,
            old_score=old_result.overall_score,
            new_score=new_result.overall_score,
            improvement=improvement,
            reason="A/B test passed: improvement meets margin with no regressions",
        )

    def promote(self, agent: Callable, new_model_id: str) -> PromotionEvent | None:
        """
        Promote the agent to the new model.

        If swap_model fn exists → call it directly.
        If not → emit PromotionEvent for developer to handle.
        """
        config = getattr(agent, "_foundry_agent_config", None)
        old_model_id = ""
        if config and config.model:
            old_model_id = config.model.id

        if config and config.swap_model:
            config.swap_model(new_model_id)
            return None  # swap done directly

        # Emit event for developer
        event = PromotionEvent(
            agent_name=agent.__name__,
            old_model_id=old_model_id,
            new_model_id=new_model_id,
            eval_score_before=0.0,
            eval_score_after=0.0,
            improvement=0.0,
            fine_tune_job_id="",
        )
        return event
