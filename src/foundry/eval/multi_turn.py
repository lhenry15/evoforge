"""Multi-turn eval with UserSimulator."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from foundry.core.types import EvalCaseResult, EvalRunResult, Message


class Milestone(BaseModel):
    """A checkpoint in a multi-turn conversation that must be reached."""
    description: str              # e.g. "Agent searched for flights"
    check: str                    # substring or keyword to verify in agent response
    turn: Optional[int] = None   # expected turn (None = any turn)
    required: bool = True         # if False, milestone is bonus points


class MultiTurnScenario(BaseModel):
    """A multi-turn eval scenario with user simulator behavior."""
    id: str
    capability: str
    initial_message: str                    # first user message
    user_responses: list[str]               # subsequent user messages (scripted)
    milestones: list[Milestone]             # checkpoints to score
    max_turns: int = 6
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiTurnResult(BaseModel):
    """Result of a single multi-turn evaluation."""
    scenario_id: str
    capability: str
    turns: list[dict[str, str]]             # full conversation
    milestones_hit: list[str]
    milestones_missed: list[str]
    score: float                            # % milestones hit
    passed: bool


class UserSimulator:
    """
    Simulates user behavior in multi-turn conversations.

    Two modes:
      1. Scripted: follows pre-defined user_responses sequence
      2. LLM-driven: uses an LLM to generate contextual user replies

    Usage::

        sim = UserSimulator(pool=OllamaLLMPool())
        result = sim.run_scenario(agent=my_agent, scenario=scenario)
    """

    def __init__(self, pool: Any = None) -> None:
        self._pool = pool

    def run_scenario(
        self,
        agent: Callable,
        scenario: MultiTurnScenario,
    ) -> MultiTurnResult:
        """Run a multi-turn conversation and score against milestones."""
        turns: list[dict[str, str]] = []
        messages: list[Message] = []

        # User's first message
        user_msg = scenario.initial_message
        user_responses = list(scenario.user_responses)

        for turn_idx in range(scenario.max_turns):
            # User turn
            messages.append(Message(role="user", content=user_msg))
            turns.append({"role": "user", "content": user_msg})

            # Agent turn
            try:
                agent_response = agent(messages)
            except Exception as e:
                agent_response = f"[ERROR: {e}]"

            messages.append(Message(role="assistant", content=agent_response))
            turns.append({"role": "assistant", "content": agent_response})

            # Next user message (scripted or LLM)
            if user_responses:
                user_msg = user_responses.pop(0)
            elif self._pool:
                user_msg = self._generate_user_reply(turns, scenario)
            else:
                break  # no more user messages

        # Score milestones
        milestones_hit = []
        milestones_missed = []

        for ms in scenario.milestones:
            hit = False
            for t in turns:
                if t["role"] == "assistant" and ms.check.lower() in t["content"].lower():
                    if ms.turn is None or turns.index(t) // 2 == ms.turn:
                        hit = True
                        break
            if hit:
                milestones_hit.append(ms.description)
            else:
                milestones_missed.append(ms.description)

        required = [m for m in scenario.milestones if m.required]
        required_hit = sum(1 for m in required if m.description in milestones_hit)
        score = required_hit / len(required) if required else 1.0

        return MultiTurnResult(
            scenario_id=scenario.id,
            capability=scenario.capability,
            turns=turns,
            milestones_hit=milestones_hit,
            milestones_missed=milestones_missed,
            score=score,
            passed=score >= 0.6,
        )

    def run_scenarios(
        self,
        agent: Callable,
        scenarios: list[MultiTurnScenario],
    ) -> EvalRunResult:
        """Run multiple scenarios and aggregate into EvalRunResult."""
        results = [self.run_scenario(agent, s) for s in scenarios]

        case_results = [
            EvalCaseResult(
                case_id=r.scenario_id,
                capability=r.capability,
                agent_response=str(r.turns[-1]["content"])[:200] if r.turns else "",
                score=r.score,
                passed=r.passed,
                judge_reasoning=f"Hit: {r.milestones_hit}, Missed: {r.milestones_missed}",
            )
            for r in results
        ]

        # Aggregate by capability
        cap_buckets: dict[str, list[float]] = {}
        for r in results:
            cap_buckets.setdefault(r.capability, []).append(r.score)
        capability_scores = {k: sum(v) / len(v) for k, v in cap_buckets.items()}
        overall = sum(capability_scores.values()) / len(capability_scores) if capability_scores else 0.0

        return EvalRunResult(
            agent_name=agent.__name__,
            overall_score=round(overall, 4),
            capability_scores={k: round(v, 4) for k, v in capability_scores.items()},
            case_results=case_results,
            n_passed=sum(1 for r in results if r.passed),
            n_total=len(results),
        )

    def _generate_user_reply(
        self,
        turns: list[dict[str, str]],
        scenario: MultiTurnScenario,
    ) -> str:
        """Generate a contextual user reply using LLM."""
        conversation = "\n".join(f"{t['role']}: {t['content'][:100]}" for t in turns[-4:])
        prompt = f"""You are simulating a user talking to a flight booking assistant.
Continue this conversation naturally. Reply as the user would.

Conversation so far:
{conversation}

Reply with ONLY the user's next message (one sentence, natural language):"""

        raw = self._pool.generate(prompt, temperature=0.7, max_tokens=100)
        if inspect.isawaitable(raw):
            raise RuntimeError(
                "UserSimulator received an async LLM pool in sync mode. "
                "Use a synchronous pool (for example, OllamaLLMPool)."
            )
        return str(raw)
