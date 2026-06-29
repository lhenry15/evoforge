"""Tests for the generic conversation-simulator plug-in seam (offline, no LLM)."""

from __future__ import annotations

from typing import Any

from evoforge.core.types import ScoringMethod
from evoforge.eval.simulator import (
    ConversationSimulator,
    ScriptedSimulator,
    SimParticipant,
    SimScenario,
    SimTranscript,
    simulate_many,
    transcripts_to_eval_cases,
)


def _scenario(**kwargs: Any) -> SimScenario:
    base: dict[str, Any] = dict(
        id="s1",
        capability="preference_recall",
        goal="apply the stated preference",
        expected="bullets",
        scoring_method=ScoringMethod.CONTAINS,
        participants=[
            SimParticipant(id="alice", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ],
        metadata={
            "script": [
                ("alice", "user", "Always summarise in bullet points.", {"preference_present": True}),
                ("agent", "assistant", "Got it — bullets from now on."),
            ],
            "labels": {"verdict": "reflected"},
        },
    )
    base.update(kwargs)
    return SimScenario(**base)


def test_scripted_simulator_satisfies_protocol() -> None:
    sim = ScriptedSimulator()
    assert isinstance(sim, ConversationSimulator)
    assert sim.name == "scripted"


def test_scripted_simulator_builds_labeled_transcript() -> None:
    sim = ScriptedSimulator()
    transcript = sim.simulate(_scenario())

    assert isinstance(transcript, SimTranscript)
    assert transcript.scenario_id == "s1"
    assert transcript.simulator == "scripted"
    assert len(transcript.turns) == 2
    # multi-party roles preserved
    assert transcript.turns[0].party_id == "alice" and transcript.turns[0].role == "user"
    assert transcript.turns[1].is_agent is True
    # per-turn label round-trips
    assert transcript.turns[0].labels == {"preference_present": True}
    # conversation-level label round-trips
    assert transcript.labels == {"verdict": "reflected"}
    # script/labels stripped from carried-over metadata
    assert "script" not in transcript.metadata


def test_agent_response_picks_last_agent_turn() -> None:
    transcript = ScriptedSimulator().simulate(_scenario())
    assert transcript.agent_response() == "Got it — bullets from now on."


def test_transcript_to_eval_case_carries_labels() -> None:
    transcript = ScriptedSimulator().simulate(_scenario())
    case = transcript.to_eval_case()

    assert case.id == "s1"
    assert case.capability == "preference_recall"
    assert case.scoring_method == ScoringMethod.CONTAINS
    assert len(case.messages) == 2
    assert case.metadata["simulated"] is True
    assert case.metadata["labels"] == {"verdict": "reflected"}
    assert case.messages[1].metadata["party_id"] == "agent"


def test_simulate_many_and_batch_conversion() -> None:
    sim = ScriptedSimulator()
    scenarios = [_scenario(id="a"), _scenario(id="b")]
    transcripts = simulate_many(sim, scenarios)
    cases = transcripts_to_eval_cases(transcripts)
    assert [c.id for c in cases] == ["a", "b"]
