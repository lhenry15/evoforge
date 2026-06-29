"""SynthesisNamespace — sdk.synth interface.

Turns mined failure modes into gated, lineage-stamped synthetic training data.
Pulls the existing eval corpus as the novelty/dedup reference automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.mining.schema import MiningResult
from foundry.synthesis.schema import SynthesisResult
from foundry.synthesis.synthesizer import DataSynthesizer


class SynthesisNamespace:
    """sdk.synth — generate targeted synthetic data from mined failures.

    Usage::

        result = sdk.synth.run("my_agent", pool=gen_pool, judge_pool=judge_pool)
        examples = result.training_examples()  # ready for training backends
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk

    def run(
        self,
        agent_name: str,
        pool: Any,
        judge_pool: Any = None,
        mining_result: Optional[MiningResult] = None,
        task_spec: Optional[str] = None,
        tools: Optional[list[Any]] = None,
        system_prompt: str = "",
        per_cluster: int = 5,
        max_clusters: int = 5,
        eval_tag: str = "bootstrap",
    ) -> SynthesisResult:
        if mining_result is None:
            mining_result = self._sdk.mine.run(agent_name)

        task_spec = task_spec or self._sdk.config.task_spec
        corpus = self._corpus_instructions(eval_tag)

        synthesizer = DataSynthesizer(
            pool=pool,
            judge_pool=judge_pool,
            per_cluster=per_cluster,
            max_clusters=max_clusters,
        )
        return synthesizer.synthesize(
            mining_result=mining_result,
            task_spec=task_spec,
            tools=tools,
            system_prompt=system_prompt,
            corpus_instructions=corpus,
        )

    def _corpus_instructions(self, eval_tag: str) -> list[str]:
        """Existing eval-case inputs used as the novelty/dedup reference."""
        try:
            cases = self._sdk.data.load_eval_cases(tag=eval_tag)
        except Exception:
            return []
        return [c.messages[-1].content for c in cases if c.messages]
