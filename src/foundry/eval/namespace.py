"""EvalNamespace — sdk.eval interface (fully synchronous)."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.core.types import (
    EvalCase,
    EvalCaseResult,
    EvalRunResult,
    ScoringMethod,
)


_LLM_JUDGE_SYSTEM = """You are an impartial AI evaluator.
Score how well the agent response satisfies the expected answer / rubric.
Reply with ONLY a JSON object: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}
Be strict: 1.0 = perfect, 0.0 = completely wrong/irrelevant."""

_LLM_JUDGE_PROMPT = """Task context: {task_spec}

Expected answer / rubric:
{expected}

{rubric_section}

Agent response:
{response}

Score the agent response."""


class EvalNamespace:
    """
    sdk.eval — run eval cases against an agent and score results.

    All methods are synchronous. Uses thread pool for parallelism.
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk

    def run(
        self,
        agent: Callable,
        cases: list[EvalCase],
        parallelism: int = 4,
    ) -> EvalRunResult:
        """Run eval cases against agent."""
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            results = list(pool.map(lambda c: self._eval_case(agent, c), cases))

        return self._aggregate(agent.__name__, results)

    def run_multi_turn(self, agent: Callable, scenarios: list[Any]) -> EvalRunResult:
        """Run multi-turn eval scenarios."""
        from foundry.eval.multi_turn import UserSimulator
        sim = UserSimulator()
        return sim.run_scenarios(agent=agent, scenarios=scenarios)

    def run_full(
        self,
        agent: Callable,
        cases: list[EvalCase] = None,
        scenarios: list[Any] = None,
        parallelism: int = 4,
    ) -> EvalRunResult:
        """Run both single-turn and multi-turn eval, merge results."""
        results: list[EvalCaseResult] = []

        if cases:
            st = self.run(agent, cases, parallelism)
            results.extend(st.case_results)

        if scenarios:
            mt = self.run_multi_turn(agent, scenarios)
            results.extend(mt.case_results)

        return self._aggregate(agent.__name__, results)

    def _eval_case(self, agent: Callable, case: EvalCase) -> EvalCaseResult:
        """Evaluate a single case (synchronous)."""
        t0 = time.perf_counter()
        try:
            response = agent(case.messages)
        except Exception as e:
            response = f"[ERROR: {e}]"
        latency_ms = (time.perf_counter() - t0) * 1000

        score, reasoning = self._score(case, str(response))
        return EvalCaseResult(
            case_id=case.id,
            capability=case.capability,
            agent_response=str(response),
            score=score,
            passed=score >= 0.6,
            judge_reasoning=reasoning,
            latency_ms=round(latency_ms, 1),
        )

    def _score(self, case: EvalCase, response: str) -> tuple[float, str]:
        """Score a response using the configured method."""
        method = case.scoring_method

        if method == ScoringMethod.EXACT_MATCH:
            match = response.strip().lower() == case.expected.strip().lower()
            return (1.0 if match else 0.0), ("exact match" if match else "no match")

        if method == ScoringMethod.CONTAINS:
            found = case.expected.lower() in response.lower()
            return (1.0 if found else 0.0), ("found" if found else "not found")

        if method == ScoringMethod.REGEX:
            found = bool(re.search(case.expected, response, re.IGNORECASE))
            return (1.0 if found else 0.0), ("pattern matched" if found else "no match")

        if method == ScoringMethod.TOOL_CALL_CHECK:
            required = [t.strip() for t in case.expected.split(",")]
            hit = sum(1 for t in required if t.lower() in response.lower())
            score = hit / len(required) if required else 0.0
            return score, f"{hit}/{len(required)} required tools called"

        # Default: LLM_JUDGE
        return self._llm_judge(case, response)

    def _llm_judge(self, case: EvalCase, response: str) -> tuple[float, str]:
        """Score using LLM judge (synchronous Ollama call)."""
        from foundry.llm.ollama import OllamaLLMPool

        pool = OllamaLLMPool()
        rubric_section = (
            f"Additional rubric:\n{case.scoring_rubric}" if case.scoring_rubric else ""
        )
        prompt = _LLM_JUDGE_PROMPT.format(
            task_spec=self._sdk.config.task_spec,
            expected=case.expected,
            rubric_section=rubric_section,
            response=response,
        )
        raw = pool.generate(prompt, system=_LLM_JUDGE_SYSTEM, temperature=0)
        try:
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            obj = json.loads(m.group() if m else raw)
            return float(obj.get("score", 0.0)), obj.get("reasoning", "")
        except Exception:
            return 0.5, f"judge parse error: {raw[:100]}"

    def _aggregate(self, agent_name: str, results: list[EvalCaseResult]) -> EvalRunResult:
        """Aggregate case results into EvalRunResult."""
        cap_buckets: dict[str, list[float]] = {}
        for r in results:
            cap_buckets.setdefault(r.capability, []).append(r.score)
        capability_scores = {k: sum(v) / len(v) for k, v in cap_buckets.items()}
        overall = sum(capability_scores.values()) / len(capability_scores) if capability_scores else 0.0

        return EvalRunResult(
            agent_name=agent_name,
            overall_score=round(overall, 4),
            capability_scores={k: round(v, 4) for k, v in capability_scores.items()},
            case_results=results,
            n_passed=sum(1 for r in results if r.passed),
            n_total=len(results),
        )
