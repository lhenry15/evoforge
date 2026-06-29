"""MiningNamespace — sdk.mine interface.

Runs failure-mode mining over an agent's persisted traces and returns ranked,
labeled failure modes ready for synthesis, eval expansion, and proactive fixes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.mining.miner import FailureModeMiner
from foundry.mining.report import FailureModeReport
from foundry.mining.schema import MiningResult


class MiningNamespace:
    """sdk.mine — discover stable failure modes from traces.

    Usage::

        result = sdk.mine.run("my_agent")              # deterministic, no LLM
        result = sdk.mine.run("my_agent", pool=pool)   # + LLM root-cause labels
        print(sdk.mine.report("my_agent"))
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk

    def run(
        self,
        agent_name: str,
        pool: Any = None,
        max_labeled_clusters: int = 8,
        persist: bool = False,
    ) -> MiningResult:
        traces = self._sdk.trace.load(agent_name)
        miner = FailureModeMiner(pool=pool, max_labeled_clusters=max_labeled_clusters)
        result = miner.mine(traces, agent_name=agent_name)
        # Persist re-classified signatures so coverage/forecast/dashboard benefit.
        if persist and pool is not None and miner.n_reclassified:
            self._sdk.trace.store.save_many(traces)
        return result

    def report(
        self,
        agent_name: str,
        pool: Any = None,
        as_dict: bool = False,
    ) -> Any:
        result = self.run(agent_name, pool=pool)
        renderer = FailureModeReport(result)
        return renderer.to_dict() if as_dict else renderer.to_text()
