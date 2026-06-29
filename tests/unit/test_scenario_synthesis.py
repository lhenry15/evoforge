"""Offline tests for scenario-driven synthesis: seed -> conversation -> labeler.

No real LLM: a deterministic FakePool drives component (2) generation and
component (3) judging so the seed backbone, the open labeler set, the verbatim
guard, the self-certification separation battery, and the native simulator are
all exercised without network access.
"""

from __future__ import annotations

from evoforge.eval.connectors.native import NativeMultiPartySimulator
from evoforge.eval.simulator import ConversationSimulator
from evoforge.mining.schema import FailureExample, FailureModeCluster
from evoforge.synthesis.conversation import ConversationGenerator, SimTurn
from evoforge.synthesis.labeler import (
    PRESENCE_SCHEMA,
    REGISTRY,
    CertProbe,
    LabelField,
    LabelSchema,
    SchemaLabeler,
    certify_labeler,
    label_transcript,
)
from evoforge.synthesis.pipeline import LabeledDataset, ScenarioSynthesizer
from evoforge.synthesis.seed import Seed, SeedGenerator, SimParticipant
from evoforge.trace.schema import FailureMode


class FakePool:
    """Deterministic pool: emits a 2-turn conversation and keyword-based judgments."""

    supports_structured = True

    def __init__(self, keyword: str = "bullets") -> None:
        self._keyword = keyword

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kwargs):
        props = schema.get("properties", {})
        if "turns" in props:
            return {
                "turns": [
                    {"speaker": "alex", "content": f"Going forward always use {self._keyword}."},
                    {"speaker": "agent", "content": f"Understood, I'll use {self._keyword} from now on."},
                ]
            }
        # judge schema: present iff the message under review mentions the keyword
        review = prompt.split("Message under review")[-1].lower()
        present = self._keyword in review
        return {"present": present, "verbatim_only": False, "reason": "stub"}

    def generate(self, prompt, system="", temperature=0.0, max_tokens=128, **kwargs):
        return "stub"


def _cluster() -> FailureModeCluster:
    return FailureModeCluster(
        cluster_id="fm-1",
        mode=FailureMode.ENVIRONMENT_FRAGILITY,
        capability="booking",
        label="tool errors drop the booking",
        symptom_summary="booking lost when the tool errors",
        suggested_fix_type="prompt",
        signature_ids=["s1"],
        trace_ids=["t1"],
        size=1,
        examples=[
            FailureExample(trace_id="t1", trigger="Book UA123", response="[ERROR] tool exploded", signature_id="s1"),
        ],
    )


# ── component (1): the seed ──────────────────────────────────────────────────


def test_seed_from_failure_cluster_is_failure_seed() -> None:
    seed = Seed.from_failure_cluster(_cluster(), complexity=0.7)
    assert seed.is_failure_seed() is True
    assert seed.failure_mode == FailureMode.ENVIRONMENT_FRAGILITY
    assert seed.capability == "booking"
    assert seed.failure_examples and seed.failure_examples[0].trigger == "Book UA123"
    assert "prompt" in " ".join(seed.conditions)


def test_seed_generator_from_failures_spreads_complexity() -> None:
    seeds = SeedGenerator(FakePool()).from_failures(_cluster(), n=3, complexity=0.5)
    assert [s.id for s in seeds] == ["seed-fm-1-0", "seed-fm-1-1", "seed-fm-1-2"]
    # complexity rises across the requested copies for varied hardness
    assert seeds[0].complexity < seeds[2].complexity


def test_seed_multiparty_detection() -> None:
    single = Seed(id="s", participants=[SimParticipant(id="u"), SimParticipant(id="a", is_agent=True)])
    multi = Seed(
        id="m",
        participants=[SimParticipant(id="u1"), SimParticipant(id="u2"), SimParticipant(id="a", is_agent=True)],
    )
    assert single.is_multiparty() is False
    assert multi.is_multiparty() is True


# ── component (2): the conversation ──────────────────────────────────────────


def test_conversation_generator_builds_transcript_from_seed() -> None:
    seed = Seed.from_failure_cluster(_cluster())
    transcript = ConversationGenerator(FakePool()).generate(seed)
    assert transcript.scenario_id == seed.id
    assert transcript.capability == "booking"
    assert len(transcript.turns) == 2
    # agent speaker mapped to an assistant/agent turn
    assert transcript.turns[-1].is_agent is True
    assert transcript.turns[0].is_agent is False


# ── component (3): the open, self-certified labeler set ──────────────────────


def test_schema_labeler_votes_and_judges_human_party() -> None:
    labeler = SchemaLabeler(PRESENCE_SCHEMA, FakePool(keyword="bullets"), votes=3)
    # human turn mentioning the keyword -> present
    human = [SimTurn(party_id="alex", role="user", content="Please use bullets going forward.")]
    label = labeler.label(human, Seed(id="s", goal="use bullets"))
    assert label is not None and label.present is True
    assert label.votes == "3/3"
    # an agent turn is skipped by a human-party labeler
    agent = [SimTurn(party_id="agent", role="assistant", content="Sure, bullets.", is_agent=True)]
    assert labeler.label(agent, Seed(id="s", goal="use bullets")) is None


def test_certify_labeler_separates_positive_and_negative() -> None:
    labeler = SchemaLabeler(PRESENCE_SCHEMA, FakePool(keyword="bullets"), votes=1)
    seed = Seed(id="s", goal="format")  # goal avoids the keyword so it can't leak
    probes = [
        CertProbe(name="pos", prefix=[SimTurn(party_id="u", role="user", content="Use bullets please.")], expect_present=True),
        CertProbe(name="neg", prefix=[SimTurn(party_id="u", role="user", content="Hello there.")], expect_present=False),
    ]
    report = certify_labeler(labeler, seed, probes)
    assert report.passed is True
    assert report.n_correct == 2


def test_label_transcript_sweeps_prefixes() -> None:
    seed = Seed(id="s", goal="format")
    transcript = ConversationGenerator(FakePool(keyword="bullets")).generate(seed)
    labeler = SchemaLabeler(PRESENCE_SCHEMA, FakePool(keyword="bullets"), votes=1)
    label_transcript(transcript, [labeler], seed)
    # the human turn mentioning the keyword carries a presence label
    assert transcript.turns[0].labels.get("target_presence", {}).get("present") is True


# ── the native (no-deps) multi-party simulator ───────────────────────────────


def test_native_simulator_satisfies_protocol_and_labels() -> None:
    sim = NativeMultiPartySimulator(FakePool(keyword="bullets"), votes=1)
    assert isinstance(sim, ConversationSimulator)
    assert sim.name == "native"
    seed = Seed.from_failure_cluster(_cluster())
    transcript = sim.simulate(seed)
    assert len(transcript.turns) == 2
    # failure seed -> agent turn judged for avoids_failure
    assert "avoids_failure" in transcript.turns[-1].labels


def test_registry_is_open_and_expandable() -> None:
    assert "target_presence" in REGISTRY.available()
    assert "avoids_failure" in REGISTRY.available()
    REGISTRY.register_schema(
        LabelSchema(
            name="mentions_deadline",
            question="Does the message mention a deadline?",
            fields=[LabelField(name="mentions_deadline", kind="bool")],
            present_field="mentions_deadline",
        )
    )
    assert "mentions_deadline" in REGISTRY.available()
    created = REGISTRY.create("mentions_deadline", FakePool(), votes=1)
    assert created.name == "mentions_deadline"


# ── the seed -> conversation -> label pipeline ───────────────────────────────


def test_scenario_synthesizer_simulate_labels_turns() -> None:
    synth = ScenarioSynthesizer(FakePool(keyword="bullets"), votes=1)
    seed = Seed(
        id="s", goal="format",
        participants=[
            SimParticipant(id="alex", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ],
    )
    transcript = synth.simulate(seed, labelers=["target_presence"])
    assert len(transcript.turns) == 2
    assert "target_presence" in transcript.turns[0].labels


def test_scenario_synthesizer_defaults_avoids_failure_for_failure_seed() -> None:
    synth = ScenarioSynthesizer(FakePool(keyword="bullets"), votes=1)
    failure_seed = Seed.from_failure_cluster(_cluster())
    transcript = synth.simulate(failure_seed)  # no labelers -> default by seed kind
    assert "avoids_failure" in transcript.turns[-1].labels


def test_build_dataset_and_jsonl_roundtrip(tmp_path) -> None:
    synth = ScenarioSynthesizer(FakePool(keyword="bullets"), votes=1)
    seeds = [Seed(id="a", goal="format"), Seed(id="b", goal="format")]
    dataset = synth.build_dataset(seeds, labelers=["target_presence"])
    assert len(dataset) == 2
    assert dataset.to_eval_cases()[0].id == "a"
    assert "target_presence" in dataset.label_names()

    path = tmp_path / "ds.jsonl"
    assert dataset.write_jsonl(path) == 2
    reloaded = LabeledDataset.read_jsonl(path)
    assert len(reloaded) == 2
    assert reloaded.conversations[0].scenario_id == "a"


def test_labeled_dataset_add_merges() -> None:
    left = LabeledDataset(metadata={"x": 1})
    right = LabeledDataset(metadata={"y": 2})
    merged = left + right
    assert merged.metadata == {"x": 1, "y": 2}
    assert len(merged) == 0


def test_conversation_generator_transcript_fallback_parses_markdown() -> None:
    """When the model returns Markdown (not JSON), the speaker-labeled fallback parses it."""

    class _MarkdownPool:
        supports_structured = True

        def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kwargs):
            return None  # force the transcript fallback

        def generate(self, prompt, system="", temperature=0.0, max_tokens=512, **kwargs):
            return (
                "alex: Going forward, always use bullets.\n"
                "agent: ```markdown\n## Preferences\n- bullets\n```\n"
                "Got it — recorded."
            )

    seed = Seed(
        id="s",
        participants=[
            SimParticipant(id="alex", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ],
    )
    transcript = ConversationGenerator(_MarkdownPool()).generate(seed)
    assert len(transcript.turns) == 2
    assert transcript.turns[0].party_id == "alex" and transcript.turns[0].is_agent is False
    # multi-line fenced content is preserved under the agent turn
    assert transcript.turns[1].is_agent is True
    assert "## Preferences" in transcript.turns[1].content
