"""
EvoForge Example: Multi-party conversation simulator seam (offline)

Demonstrates the generic ``ConversationSimulator`` plug-in seam without any
external dependencies, using the built-in :class:`ScriptedSimulator`. The same
seam is implemented by richer connectors (e.g. the native multi-party simulator)
which generate *labeled* multi-party conversations with an LLM simulator.

Run:

    python examples/multiparty_simulator.py

What it shows:
  * Building generic :class:`SimScenario` objects (domain payloads stay opaque
    in ``metadata``).
  * Producing :class:`SimTranscript` objects with per-turn ``labels``.
  * Converting transcripts to EvoForge ``EvalCase`` objects for scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoforge.core.types import ScoringMethod  # noqa: E402
from evoforge.eval.simulator import (  # noqa: E402
    ScriptedSimulator,
    SimParticipant,
    SimScenario,
    simulate_many,
    transcripts_to_eval_cases,
)

# A multi-party scenario: two humans + the agent under test. The "script" travels
# opaquely in metadata so the seam never needs to know the domain. Each turn may
# carry labels — here a tiny "preference_present" flag illustrating the kind of
# per-turn annotation a real labeler attaches.
SCENARIO = SimScenario(
    id="pref-demo-0",
    capability="preference_recall",
    goal="apply the standing formatting preference the team stated",
    participants=[
        SimParticipant(id="alice", role="user", persona="team lead"),
        SimParticipant(id="bob", role="user", persona="engineer"),
        SimParticipant(id="agent", role="assistant", is_agent=True),
    ],
    expected="bullets",
    scoring_method=ScoringMethod.CONTAINS,
    metadata={
        "script": [
            ("alice", "user", "Going forward, always summarise standups in bullet points.",
             {"preference_present": {"present": True, "axis": "format", "value": "bullets"}}),
            ("bob", "user", "Sounds good, that's easier to skim.",
             {"preference_present": {"present": False}}),
            ("agent", "assistant", "Got it — I'll summarise standups in bullets from now on.",
             {"preference_present": {"present": False}}),
        ],
    },
)


def main() -> int:
    simulator = ScriptedSimulator()
    transcripts = simulate_many(simulator, [SCENARIO])

    for transcript in transcripts:
        speakers = sorted({turn.party_id for turn in transcript.turns})
        labeled = sum(1 for turn in transcript.turns if turn.labels)
        print(f"transcript={transcript.scenario_id} simulator={transcript.simulator}")
        print(f"  turns={len(transcript.turns)} speakers={speakers} labeled_turns={labeled}")
        for turn in transcript.turns:
            flag = turn.labels.get("preference_present", {})
            mark = "PRESENT" if isinstance(flag, dict) and flag.get("present") else "-"
            print(f"  [{turn.party_id:>6}/{turn.role:<9}] {mark:<8} {turn.content}")

    cases = transcripts_to_eval_cases(transcripts)
    print(f"\nconverted {len(cases)} transcript(s) -> EvalCase(s); "
          f"first case scoring_method={cases[0].scoring_method.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
