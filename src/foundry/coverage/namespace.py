"""CoverageNamespace — sdk.coverage interface.

Map benchmark coverage against real failure modes, expose blind spots, and
generate targeted eval cases that close them (optionally persisting so the loop
closes on the next cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.core.types import EvalCase
from foundry.coverage.expander import AdaptiveEvalExpander
from foundry.coverage.mapper import CoverageMapper
from foundry.coverage.report import CoverageReport
from foundry.coverage.schema import Blindspot, CoverageMap
from foundry.mining.schema import MiningResult


class CoverageNamespace:
    """sdk.coverage — adaptive eval coverage against mined failure modes.

    Usage::

        cmap = sdk.coverage.map("my_agent")
        print(sdk.coverage.report("my_agent"))
        new_cases = sdk.coverage.expand("my_agent", pool=pool, persist=True)
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk

    def map(
        self,
        agent_name: str,
        mining_result: Optional[MiningResult] = None,
        eval_tag: str = "bootstrap",
        pool: Any = None,
    ) -> CoverageMap:
        # Mine with the pool (and persist) when available so demand uses the same
        # re-classified mode taxonomy the supply (tagged eval cases) was built on.
        if mining_result is None:
            mining_result = self._sdk.mine.run(agent_name, pool=pool, persist=pool is not None)
        cases = self._load_cases(eval_tag)
        return CoverageMapper().build(cases, mining_result, agent_name=agent_name)

    def blindspots(
        self,
        agent_name: str,
        mining_result: Optional[MiningResult] = None,
        eval_tag: str = "bootstrap",
        pool: Any = None,
    ) -> list[Blindspot]:
        return self.map(agent_name, mining_result, eval_tag, pool=pool).blindspots()

    def report(
        self,
        agent_name: str,
        mining_result: Optional[MiningResult] = None,
        eval_tag: str = "bootstrap",
        as_dict: bool = False,
        pool: Any = None,
    ) -> Any:
        coverage_map = self.map(agent_name, mining_result, eval_tag, pool=pool)
        renderer = CoverageReport(coverage_map)
        return renderer.to_dict() if as_dict else renderer.to_text()

    def expand(
        self,
        agent_name: str,
        pool: Any,
        mining_result: Optional[MiningResult] = None,
        eval_tag: str = "bootstrap",
        cases_per_blindspot: int = 3,
        tools: Optional[list[Any]] = None,
        system_prompt: str = "",
        persist: bool = False,
    ) -> list[EvalCase]:
        # Use the pool for the demand-side mining so blind spots are based on
        # re-classified (and persisted) modes that the supply will be tagged with.
        coverage_map = self.map(agent_name, mining_result, eval_tag, pool=pool)
        blindspots = coverage_map.blindspots()
        existing = self._load_cases(eval_tag)
        existing_messages = [c.messages[-1].content for c in existing if c.messages]
        expander = AdaptiveEvalExpander(pool=pool)
        new_cases = expander.expand(
            blindspots=blindspots,
            task_spec=self._sdk.config.task_spec,
            tools=tools,
            system_prompt=system_prompt,
            cases_per_blindspot=cases_per_blindspot,
            existing_messages=existing_messages,
        )
        if persist and new_cases:
            self._sdk.data.save_eval_cases(existing + new_cases, tag=eval_tag)
        return new_cases

    def _load_cases(self, eval_tag: str) -> list[EvalCase]:
        try:
            return self._sdk.data.load_eval_cases(tag=eval_tag)
        except Exception:
            return []
