"""Generic conversation-simulator seam for EvoForge eval.

This module is the *seam* between the eval subsystem and any conversation
generator (a built-in scripted reference, the native multi-party generator, or
an external plug-in). The conversation *types* themselves now live
in the synthesis subsystem:

    * component (1) -- the control surface -- ``evoforge.synthesis.seed``
      (``Seed`` / ``SimParticipant``)
    * component (2) -- the conversation -- ``evoforge.synthesis.conversation``
      (``SimTurn`` / ``SimTranscript``)

They are re-exported here so the eval seam, external connectors, and examples
keep importing them from ``evoforge.eval.simulator`` unchanged. ``SimScenario``
is kept as a backward-compatible alias of ``Seed``.

A simulator is anything implementing :class:`ConversationSimulator` -- given a
seed it returns a labeled :class:`SimTranscript`. The seam stays
feature-agnostic: callers inject scenario/labeling behavior; the eval subsystem
only consumes the resulting transcripts.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Conversation types are owned by the synthesis subsystem and re-exported here.
from evoforge.synthesis.conversation import SimTranscript, SimTurn
from evoforge.synthesis.seed import Seed, SimParticipant

# Backward-compatible alias: the simulator seam historically called the seed a
# "SimScenario". Seed is the richer control surface; the alias keeps existing
# callers (connectors, examples, tests) working unchanged.
SimScenario = Seed

AgentFn = Callable[..., Any]


@runtime_checkable
class ConversationSimulator(Protocol):
    """A pluggable producer of labeled conversations.

    Implementations may be single- or multi-party, scripted or LLM-driven, and
    may live in EvoForge (native) or behind an optional dependency.
    The eval subsystem only depends on this protocol, never on a concrete
    implementation.
    """

    def simulate(self, scenario: SimScenario, agent: Optional[AgentFn] = None) -> SimTranscript:
        """Produce one labeled transcript for ``scenario``."""
        ...


def simulate_many(
    simulator: ConversationSimulator,
    scenarios: list[SimScenario],
    agent: Optional[AgentFn] = None,
) -> list[SimTranscript]:
    """Run ``simulator`` over many scenarios, returning the transcripts.

    Failures on a single scenario are isolated: a transcript with an ``error``
    in its metadata is emitted instead of aborting the batch.
    """
    transcripts: list[SimTranscript] = []
    for scenario in scenarios:
        try:
            transcript = simulator.simulate(scenario, agent=agent)
        except Exception as exc:  # noqa: BLE001 - isolate per-scenario failures
            transcript = SimTranscript(
                scenario_id=scenario.id,
                capability=scenario.capability,
                turns=[],
                expected=scenario.expected,
                scoring_method=scenario.scoring_method,
                simulator=type(simulator).__name__,
                metadata={"error": str(exc)},
            )
        transcripts.append(transcript)
    return transcripts


def transcripts_to_eval_cases(transcripts: list[SimTranscript]) -> list[Any]:
    """Lower a batch of transcripts to eval cases for scoring."""
    cases = [transcript.to_eval_case() for transcript in transcripts]
    return cases


class ScriptedSimulator:
    """Offline reference simulator driven entirely by the seed.

    The seed (or its scenario metadata) carries a ``script`` -- a list of
    ``(speaker, role, content)`` or ``(speaker, role, content, labels)`` tuples.
    This is the deterministic, no-LLM reference used by tests and the offline
    example; it implements :class:`ConversationSimulator` structurally.
    """

    name: str = "scripted"

    def simulate(self, scenario: SimScenario, agent: Optional[AgentFn] = None) -> SimTranscript:
        script = list(scenario.metadata.get("script", []))
        turns: list[SimTurn] = []
        for index, step in enumerate(script):
            speaker, role, content, labels = self._unpack(step)
            is_agent = role == "assistant"
            turns.append(
                SimTurn(
                    party_id=speaker,
                    role=role,
                    content=content,
                    is_agent=is_agent,
                    labels=labels,
                    metadata={"turn_index": index},
                )
            )
        carried = {k: v for k, v in scenario.metadata.items() if k not in ("script", "labels")}
        transcript = SimTranscript(
            scenario_id=scenario.id,
            capability=scenario.capability,
            turns=turns,
            labels=dict(scenario.metadata.get("labels", {})),
            expected=scenario.expected,
            scoring_method=scenario.scoring_method,
            simulator=self.name,
            metadata={"source": "scripted", **carried},
        )
        return transcript

    @staticmethod
    def _unpack(step: Any) -> tuple[str, str, str, dict[str, Any]]:
        """Normalize a script step (tuple or mapping) to (speaker, role, content, labels)."""
        if isinstance(step, dict):
            speaker = str(step.get("speaker", step.get("party_id", "user")))
            role = str(step.get("role", "assistant" if step.get("is_agent") else "user"))
            content = str(step.get("content", ""))
            labels = dict(step.get("labels", {}))
            return speaker, role, content, labels
        speaker = str(step[0]) if len(step) > 0 else "user"
        role = str(step[1]) if len(step) > 1 else "user"
        content = str(step[2]) if len(step) > 2 else ""
        labels = dict(step[3]) if len(step) > 3 and step[3] else {}
        return speaker, role, content, labels


__all__ = [
    "AgentFn",
    "ConversationSimulator",
    "ScriptedSimulator",
    "Seed",
    "SimParticipant",
    "SimScenario",
    "SimTranscript",
    "SimTurn",
    "simulate_many",
    "transcripts_to_eval_cases",
]
