"""DataSynthesizer — orchestrate mode-conditioned synthesis with quality gates."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from evoforge.mining.schema import MiningResult
from evoforge.synthesis.generator import ModeConditionedGenerator
from evoforge.synthesis.quality import QualityGate
from evoforge.synthesis.schema import (
    SynthesisResult,
    SyntheticExample,
    SynthFormat,
    default_format_for,
)
from evoforge.trace.schema import FailureMode


class DataSynthesizer:
    """Generate, gate, and collect synthetic training data from mined failures.

    Usage::

        synth = DataSynthesizer(pool=gen_pool, judge_pool=judge_pool)
        result = synth.synthesize(
            mining_result, task_spec="A flight booking assistant",
            tools=["search_flights", "book_flight"],
            corpus_instructions=[c.messages[-1].content for c in eval_cases],
        )
        print(result.acceptance_rate, len(result.accepted))
    """

    def __init__(
        self,
        pool: Any,
        judge_pool: Any = None,
        generator: Optional[ModeConditionedGenerator] = None,
        per_cluster: int = 5,
        max_clusters: int = 5,
        near_dup_threshold: float = 0.85,
        min_novelty: float = 0.2,
        min_consistency: float = 0.5,
        judge_votes: int = 3,
    ) -> None:
        self._generator = generator or ModeConditionedGenerator(pool=pool)
        self._judge_pool = judge_pool
        self._per_cluster = per_cluster
        self._max_clusters = max_clusters
        self._near_dup_threshold = near_dup_threshold
        self._min_novelty = min_novelty
        self._min_consistency = min_consistency
        self._judge_votes = judge_votes

    def synthesize(
        self,
        mining_result: MiningResult,
        task_spec: str,
        tools: Optional[list[Any]] = None,
        system_prompt: str = "",
        corpus_instructions: Optional[list[str]] = None,
        formats: Optional[dict[FailureMode, SynthFormat]] = None,
    ) -> SynthesisResult:
        gate = QualityGate(
            corpus_instructions=corpus_instructions,
            judge_pool=self._judge_pool,
            near_dup_threshold=self._near_dup_threshold,
            min_novelty=self._min_novelty,
            min_consistency=self._min_consistency,
            judge_votes=self._judge_votes,
        )

        accepted: list[SyntheticExample] = []
        rejected: list[SyntheticExample] = []
        per_mode: Counter[str] = Counter()

        for cluster in mining_result.clusters[: self._max_clusters]:
            fmt = (formats or {}).get(cluster.mode) or default_format_for(cluster.mode)
            candidates = self._generator.generate(
                cluster=cluster,
                task_spec=task_spec,
                tools=tools or [],
                system_prompt=system_prompt,
                n=self._per_cluster,
                fmt=fmt,
            )
            for candidate in candidates:
                report = gate.evaluate(candidate, accepted)
                candidate.quality = report
                if report.accepted:
                    accepted.append(candidate)
                    per_mode[cluster.mode.value] += 1
                else:
                    rejected.append(candidate)

        total = len(accepted) + len(rejected)
        rate = round(len(accepted) / total, 4) if total else 0.0

        return SynthesisResult(
            agent_name=mining_result.agent_name,
            accepted=accepted,
            rejected=rejected,
            n_generated=total,
            acceptance_rate=rate,
            per_mode_counts=dict(per_mode),
        )
