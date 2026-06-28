"""Architecture evolution — search over agent structures to find what works best."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from foundry.core.types import EvalCase, Message


class AgentArchitecture(BaseModel):
    """Defines an agent architecture pattern."""
    name: str                       # e.g. "single", "debate", "cot", "decompose"
    description: str
    wrapper: Optional[str] = None   # internal key for the wrapper function


class ArchitectureSearchResult(BaseModel):
    """Result of searching over architectures for a capability."""
    capability: str
    results: dict[str, float] = Field(default_factory=dict)  # arch_name → score
    winner: str = ""
    improvement_over_baseline: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Built-in architecture patterns ───────────────────────────────────────────

ARCHITECTURES = [
    AgentArchitecture(
        name="single",
        description="Direct single-pass: prompt → response (baseline)",
    ),
    AgentArchitecture(
        name="cot",
        description="Chain-of-Thought: think step-by-step before answering",
    ),
    AgentArchitecture(
        name="debate",
        description="Self-debate: generate 2 responses, critique each, pick best",
    ),
    AgentArchitecture(
        name="self_consistency",
        description="Self-consistency: generate N responses, majority vote",
    ),
    AgentArchitecture(
        name="decompose",
        description="Decompose-then-solve: break into subtasks, solve each, combine",
    ),
    AgentArchitecture(
        name="verify",
        description="Generate-then-verify: produce answer, then self-check for errors",
    ),
]


class ArchitectureEvolver:
    """
    Search over agent architectures to find what works best per capability.

    Given eval cases for a specific capability, wraps the base agent in different
    architectural patterns and measures which scores highest.

    Algorithm:
      For each architecture:
        1. Wrap the base agent in that pattern (e.g. add CoT prefix, or debate)
        2. Run eval on the wrapped agent
        3. Record score
      Pick the architecture with highest score.

    Usage::

        evolver = ArchitectureEvolver(pool=OllamaLLMPool())
        result = evolver.search(
            base_agent=my_agent,
            cases=[...],  # cases for one capability
            capability="error_handling",
        )
        print(f"Best: {result.winner} ({result.results})")
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def search(
        self,
        base_agent: Callable,
        cases: list[EvalCase],
        capability: str,
        architectures: list[AgentArchitecture] = None,
        sdk: Any = None,
    ) -> ArchitectureSearchResult:
        """
        Search all architectures and find the best one for this capability.

        Args:
            base_agent:     The current agent (used as baseline and wrapped)
            cases:          Eval cases for the target capability
            capability:     Which capability we're optimizing
            architectures:  List to try (defaults to all built-in)
            sdk:            FoundrySDK instance for running eval

        Returns:
            ArchitectureSearchResult with per-architecture scores and winner
        """
        if not sdk:
            raise ValueError("sdk is required to run eval")

        archs = architectures or ARCHITECTURES
        results: dict[str, float] = {}

        for arch in archs:
            wrapped = self._wrap_agent(base_agent, arch)
            eval_result = sdk.eval.run(agent=wrapped, cases=cases, parallelism=1)
            score = eval_result.capability_scores.get(capability, eval_result.overall_score)
            results[arch.name] = score

        # Find winner
        winner = max(results, key=results.get) if results else "single"
        baseline_score = results.get("single", 0)
        improvement = results[winner] - baseline_score

        return ArchitectureSearchResult(
            capability=capability,
            results=results,
            winner=winner,
            improvement_over_baseline=improvement,
        )

    def search_all_capabilities(
        self,
        base_agent: Callable,
        cases: list[EvalCase],
        sdk: Any,
    ) -> dict[str, ArchitectureSearchResult]:
        """Search architectures for each capability independently."""
        # Group cases by capability
        by_cap: dict[str, list[EvalCase]] = {}
        for c in cases:
            by_cap.setdefault(c.capability, []).append(c)

        results = {}
        for cap, cap_cases in by_cap.items():
            results[cap] = self.search(base_agent, cap_cases, cap, sdk=sdk)

        return results

    def _wrap_agent(self, base_agent: Callable, arch: AgentArchitecture) -> Callable:
        """Wrap the base agent in an architectural pattern."""
        if arch.name == "single":
            return base_agent

        if arch.name == "cot":
            return self._make_cot_wrapper(base_agent)

        if arch.name == "debate":
            return self._make_debate_wrapper(base_agent)

        if arch.name == "self_consistency":
            return self._make_consistency_wrapper(base_agent)

        if arch.name == "decompose":
            return self._make_decompose_wrapper(base_agent)

        if arch.name == "verify":
            return self._make_verify_wrapper(base_agent)

        return base_agent

    def _make_cot_wrapper(self, base_agent: Callable) -> Callable:
        """Add 'think step by step' to the user message."""
        def cot_agent(messages: list[Message]) -> str:
            # Prepend CoT instruction to last user message
            modified = list(messages)
            if modified:
                last = modified[-1]
                modified[-1] = Message(
                    role=last.role,
                    content=last.content + "\n\nThink step by step before answering.",
                )
            return base_agent(modified)
        cot_agent.__name__ = f"{base_agent.__name__}_cot"
        # Copy foundry metadata
        for attr in ['_foundry_agent_config', '_foundry_tools', '_foundry_task_spec', '_foundry_sdk', '_foundry_multi_party']:
            if hasattr(base_agent, attr):
                setattr(cot_agent, attr, getattr(base_agent, attr))
        return cot_agent

    def _make_debate_wrapper(self, base_agent: Callable) -> Callable:
        """Generate 2 responses, use LLM to pick the better one."""
        pool = self._pool

        def debate_agent(messages: list[Message]) -> str:
            # Generate two responses
            r1 = base_agent(messages)
            r2 = base_agent(messages)

            # Pick the better one
            judge_prompt = (
                f'Which response is better?\n'
                f'Question: "{messages[-1].content[:100]}"\n'
                f'A: "{r1[:150]}"\n'
                f'B: "{r2[:150]}"\n'
                f'Reply with just the better response (copy it exactly):'
            )
            chosen = asyncio.run(pool.generate(judge_prompt, temperature=0, max_tokens=300))
            # Return whichever is more similar to chosen
            return r1 if len(set(r1[:50]) & set(chosen[:50])) > len(set(r2[:50]) & set(chosen[:50])) else r2

        debate_agent.__name__ = f"{base_agent.__name__}_debate"
        for attr in ['_foundry_agent_config', '_foundry_tools', '_foundry_task_spec', '_foundry_sdk', '_foundry_multi_party']:
            if hasattr(base_agent, attr):
                setattr(debate_agent, attr, getattr(base_agent, attr))
        return debate_agent

    def _make_consistency_wrapper(self, base_agent: Callable) -> Callable:
        """Generate 3 responses, pick the most common / longest."""
        def consistency_agent(messages: list[Message]) -> str:
            responses = [base_agent(messages) for _ in range(3)]
            # Pick the longest (usually most complete)
            return max(responses, key=len)

        consistency_agent.__name__ = f"{base_agent.__name__}_consistency"
        for attr in ['_foundry_agent_config', '_foundry_tools', '_foundry_task_spec', '_foundry_sdk', '_foundry_multi_party']:
            if hasattr(base_agent, attr):
                setattr(consistency_agent, attr, getattr(base_agent, attr))
        return consistency_agent

    def _make_decompose_wrapper(self, base_agent: Callable) -> Callable:
        """Add decomposition instruction to the prompt."""
        def decompose_agent(messages: list[Message]) -> str:
            modified = list(messages)
            if modified:
                last = modified[-1]
                modified[-1] = Message(
                    role=last.role,
                    content=(
                        last.content +
                        "\n\nBreak this into steps: "
                        "1) What information do I need? "
                        "2) What tools should I call? "
                        "3) What's the final answer?"
                    ),
                )
            return base_agent(modified)

        decompose_agent.__name__ = f"{base_agent.__name__}_decompose"
        for attr in ['_foundry_agent_config', '_foundry_tools', '_foundry_task_spec', '_foundry_sdk', '_foundry_multi_party']:
            if hasattr(base_agent, attr):
                setattr(decompose_agent, attr, getattr(base_agent, attr))
        return decompose_agent

    def _make_verify_wrapper(self, base_agent: Callable) -> Callable:
        """Generate response, then self-verify for errors."""
        pool = self._pool

        def verify_agent(messages: list[Message]) -> str:
            # First pass
            response = base_agent(messages)

            # Verify
            verify_prompt = (
                f'Check this response for errors:\n'
                f'Question: "{messages[-1].content[:100]}"\n'
                f'Response: "{response[:200]}"\n'
                f'If correct, reply with the same response. If errors found, fix them:'
            )
            verified = asyncio.run(pool.generate(verify_prompt, temperature=0, max_tokens=300))
            return verified if len(verified) > 10 else response

        verify_agent.__name__ = f"{base_agent.__name__}_verify"
        for attr in ['_foundry_agent_config', '_foundry_tools', '_foundry_task_spec', '_foundry_sdk', '_foundry_multi_party']:
            if hasattr(base_agent, attr):
                setattr(verify_agent, attr, getattr(base_agent, attr))
        return verify_agent
