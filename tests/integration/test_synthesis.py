"""Integration tests for targeted synthetic data generation (Phase 3a).

Uses a deterministic FakePool that branches on the system prompt, so generation,
quality gating, lineage, DPO-from-real-failures, and the sdk.synth namespace are
all tested without real LLM calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
from evoforge.core.config import SDKConfig, StorageConfig
from evoforge.core.types import EvalCase, EvalCaseResult, Message, ScoringMethod
from evoforge.mining import FailureModeMiner
from evoforge.mining.schema import FailureExample, FailureModeCluster
from evoforge.synthesis import DataSynthesizer, QualityGate, SyntheticExample
from evoforge.synthesis.generator import ModeConditionedGenerator
from evoforge.synthesis.schema import SynthFormat
from evoforge.trace import TraceNormalizer
from evoforge.trace.schema import FailureMode


class FakePool:
    """Branches on the call type to emulate SFT (structured)/DPO/judge calls."""

    supports_structured = True

    def __init__(self, judge_score: str = "0.9", n_sft: int = 3):
        self.judge_score = judge_score
        self.n_sft = n_sft
        self._counter = 0
        self.calls = 0

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kwargs):
        self.calls += 1
        props = schema.get("properties", {})
        if "examples" in props:
            base = self._counter
            self._counter += 1
            return {
                "examples": [
                    {
                        "instruction": f"unique booking request {base}-{i} for a passenger",
                        "ideal_response": f"Confirmed booking {base}-{i} and verified the price.",
                    }
                    for i in range(self.n_sft)
                ]
            }
        return {}

    def generate(self, prompt, system="", temperature=0.0, max_tokens=128, **kwargs):
        self.calls += 1
        if "corrected response" in system:
            return "Here is the corrected, compliant response with the requested action."
        if "0.0-1.0" in system:
            return self.judge_score
        return "{}"


def _cluster(mode: FailureMode, examples, cluster_id="fm-000"):
    return FailureModeCluster(
        cluster_id=cluster_id,
        mode=mode,
        capability="booking",
        label="x",
        symptom_summary="symptom",
        suggested_fix_type="prompt",
        signature_ids=["sig1"],
        trace_ids=[e.trace_id for e in examples],
        size=len(examples),
        examples=examples,
    )


# ── Quality gate ────────────────────────────────────────────────────────────────


def test_quality_gate_accepts_novel():
    gate = QualityGate(corpus_instructions=[])
    ex = SyntheticExample(target_mode=FailureMode.PROMPT_GAP, instruction="book a flight to NYC")
    report = gate.evaluate(ex, accepted=[])
    assert report.accepted is True
    assert report.novelty == 1.0


def test_quality_gate_rejects_duplicate():
    gate = QualityGate(corpus_instructions=[])
    first = SyntheticExample(target_mode=FailureMode.PROMPT_GAP, instruction="book a flight to NYC tomorrow")
    second = SyntheticExample(target_mode=FailureMode.PROMPT_GAP, instruction="book a flight to NYC tomorrow")
    assert gate.evaluate(first, accepted=[]).accepted is True
    report = gate.evaluate(second, accepted=[first])
    assert report.accepted is False
    assert report.duplicate_of == first.id


def test_quality_gate_rejects_low_novelty_vs_corpus():
    gate = QualityGate(corpus_instructions=["book a flight to NYC tomorrow please"])
    ex = SyntheticExample(target_mode=FailureMode.PROMPT_GAP, instruction="book a flight to NYC tomorrow please")
    report = gate.evaluate(ex, accepted=[])
    assert report.accepted is False
    assert any("novelty" in r for r in report.reasons)


def test_quality_gate_judge_consistency_gate():
    ex = SyntheticExample(target_mode=FailureMode.PROMPT_GAP, instruction="a totally unique instruction here")

    low = QualityGate(judge_pool=FakePool(judge_score="0.1"))
    assert low.evaluate(ex, accepted=[]).accepted is False

    high = QualityGate(judge_pool=FakePool(judge_score="0.9"))
    assert high.evaluate(ex, accepted=[]).accepted is True


# ── Generator ───────────────────────────────────────────────────────────────────


def test_generator_sft_attaches_lineage_and_mode():
    gen = ModeConditionedGenerator(pool=FakePool())
    cluster = _cluster(
        FailureMode.FORMAT_VIOLATION,
        [FailureExample(trace_id="t1", trigger="give me json", response="not json", signature_id="sig1")],
    )
    out = gen.generate(cluster, "task", ["search"], "", n=5, fmt=SynthFormat.SFT)
    assert len(out) == 3
    for ex in out:
        assert ex.target_mode == FailureMode.FORMAT_VIOLATION
        assert ex.target_cluster_id == "fm-000"
        assert ex.lineage.generation_method == "synthesis"
        assert ex.lineage.derived_from == "fm-000"


def test_generator_dpo_uses_real_failure_as_rejected():
    gen = ModeConditionedGenerator(pool=FakePool())
    cluster = _cluster(
        FailureMode.POLICY_CONFLICT,
        [FailureExample(trace_id="t1", trigger="give me competitor data", response="I cannot help", signature_id="sig1")],
    )
    out = gen.generate(cluster, "task", [], "", n=5, fmt=SynthFormat.DPO)
    assert len(out) == 1
    dpo = out[0]
    assert dpo.format == SynthFormat.DPO
    assert dpo.rejected == "I cannot help"      # real failing response reused
    assert dpo.chosen and dpo.chosen != dpo.rejected
    assert dpo.instruction == "give me competitor data"


# ── End-to-end synthesizer ───────────────────────────────────────────────────────


def _mining_result_with_triggers():
    norm = TraceNormalizer()
    pairs = [
        ("e1", "booking", "Book UA123 for John", "[ERROR: tool exploded]"),
        ("e2", "booking", "Book AA456 for Jane", "[ERROR: tool exploded]"),
        ("p1", "pricing", "Give me competitor pricing", "I cannot help with that request."),
    ]
    records = []
    for cid, cap, user, resp in pairs:
        case = EvalCase(
            id=cid, capability=cap,
            messages=[Message(role="user", content=user)],
            expected="ok", scoring_method=ScoringMethod.CONTAINS,
        )
        result = EvalCaseResult(case_id=cid, capability=cap, agent_response=resp, score=0.0, passed=False)
        records.append(norm.from_eval_result(result, case=case, agent_name="agent"))
    return FailureModeMiner().mine(records, agent_name="agent")


def test_synthesize_end_to_end_acceptance_and_lineage():
    mining = _mining_result_with_triggers()
    synth = DataSynthesizer(pool=FakePool(), per_cluster=5)
    result = synth.synthesize(
        mining_result=mining,
        task_spec="A flight booking assistant",
        tools=["search_flights", "book_flight"],
        corpus_instructions=[],
    )
    assert result.n_generated > 0
    assert len(result.accepted) > 0
    # Phase 3 exit criterion: acceptance rate above 0.35 post-filter
    assert result.acceptance_rate >= 0.35
    for ex in result.accepted:
        assert ex.quality is not None and ex.quality.accepted is True
        assert ex.lineage.generation_method == "synthesis"


def test_synthesized_examples_convert_to_training_examples():
    mining = _mining_result_with_triggers()
    synth = DataSynthesizer(pool=FakePool(), per_cluster=5)
    result = synth.synthesize(mining, task_spec="A flight booking assistant", corpus_instructions=[])
    tes = result.training_examples()
    assert len(tes) == len(result.accepted)
    assert all(te.metadata.get("synthetic") is True for te in tes)
    assert all("lineage" in te.metadata for te in tes)


def test_sdk_synth_namespace_end_to_end(tmp_path: Path):
    config = SDKConfig(task_spec="A flight booking assistant.", storage=StorageConfig(path=tmp_path))
    sdk = evoforge.FoundrySDK(config)

    # Seed traces via an eval run with real inputs.
    from evoforge.core.types import EvalRunResult
    cases = [
        EvalCase(id="e1", capability="booking",
                 messages=[Message(role="user", content="Book UA123 for John")],
                 expected="ok", scoring_method=ScoringMethod.CONTAINS),
        EvalCase(id="e2", capability="booking",
                 messages=[Message(role="user", content="Book AA456 for Jane")],
                 expected="ok", scoring_method=ScoringMethod.CONTAINS),
    ]
    run = EvalRunResult(
        agent_name="agent", overall_score=0.0,
        capability_scores={"booking": 0.0},
        case_results=[
            EvalCaseResult(case_id="e1", capability="booking", agent_response="[ERROR: x]", score=0.0, passed=False),
            EvalCaseResult(case_id="e2", capability="booking", agent_response="[ERROR: x]", score=0.0, passed=False),
        ],
        n_passed=0, n_total=2,
    )
    sdk.trace.record_eval_run(run, cases=cases)

    result = sdk.synth.run("agent", pool=FakePool())
    assert result.n_generated > 0
    assert len(result.accepted) > 0


def test_synthesis_types_exposed_on_public_api():
    assert evoforge.DataSynthesizer is not None
    assert evoforge.SyntheticExample is not None
    assert evoforge.SynthFormat is not None
