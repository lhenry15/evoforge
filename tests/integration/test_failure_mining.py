"""Integration tests for failure-mode mining (Phase 2).

Deterministic tests (no real LLM) covering clustering, scoring (impact /
stability), ranking, coverage, the report renderer, optional LLM labeling via a
fake pool, and the sdk.mine namespace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
import evoforge.mining
from evoforge.core.config import SDKConfig, StorageConfig
from evoforge.core.types import EvalCaseResult, EvalRunResult, Message, ScoringMethod
from evoforge.mining import FailureModeMiner, FailureModeReport
from evoforge.trace import TraceNormalizer
from evoforge.trace.schema import FailureMode


def _failure(case_id: str, capability: str, response: str) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case_id,
        capability=capability,
        agent_response=response,
        score=0.0,
        passed=False,
    )


def _records():
    """Build a mixed set of failing traces with a few distinct modes."""
    norm = TraceNormalizer()
    raw = [
        _failure("a1", "booking", "[ERROR: tool exploded]"),
        _failure("a2", "booking", "[ERROR: tool exploded]"),
        _failure("a3", "booking", "[ERROR: tool exploded]"),
        _failure("b1", "search", ""),              # incomplete (empty)
        _failure("b2", "search", ""),              # incomplete (empty)
        _failure("c1", "pricing", "I cannot help with that request."),  # policy/refusal
    ]
    return [norm.from_eval_result(r, agent_name="agent") for r in raw]


class FakePool:
    """Deterministic stand-in for an LLM pool used for labeling."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system="", temperature=0.0, max_tokens=128, **kwargs):
        self.calls += 1
        return '{"root_cause": "tool backend crashes on booking", "suggested_fix": "add retry and validation"}'


def test_mining_clusters_by_mode():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")

    assert result.n_failures == 6
    # 3 distinct modes -> 3 clusters (env_fragility, incomplete, policy_conflict)
    assert result.n_clusters == 3
    modes = {c.mode for c in result.clusters}
    assert FailureMode.ENVIRONMENT_FRAGILITY in modes
    assert FailureMode.INCOMPLETE in modes
    assert FailureMode.POLICY_CONFLICT in modes


def test_clusters_ranked_by_impact():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")
    impacts = [c.impact_score for c in result.clusters]
    assert impacts == sorted(impacts, reverse=True)


def test_largest_cluster_has_expected_size_and_stability():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")
    # The 3 identical booking errors share a signature -> cohesion 1.0,
    # identical inputs (empty) -> high reproducibility -> high stability.
    booking = next(c for c in result.clusters if c.mode == FailureMode.ENVIRONMENT_FRAGILITY)
    assert booking.size == 3
    assert booking.cohesion == 1.0
    assert booking.stability >= 0.6
    assert booking.suggested_fix_type == "workflow"


def test_coverage_metric():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")
    # All clusters fit within top-10 -> coverage is 100%
    assert result.coverage_top10 == 1.0


def test_mode_distribution_counts():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")
    assert result.mode_distribution.get("environment_fragility") == 3
    assert result.mode_distribution.get("incomplete") == 2
    assert result.mode_distribution.get("policy_conflict") == 1


def test_no_failures_returns_empty_result():
    miner = FailureModeMiner()
    result = miner.mine([], agent_name="agent")
    assert result.n_failures == 0
    assert result.clusters == []


def test_llm_labeling_with_fake_pool():
    pool = FakePool()
    miner = FailureModeMiner(pool=pool)
    result = miner.mine(_records(), agent_name="agent")
    assert pool.calls >= 1
    top = result.clusters[0]
    assert top.label == "tool backend crashes on booking"
    assert top.metadata.get("suggested_fix") == "add retry and validation"


def test_report_text_and_dict():
    miner = FailureModeMiner()
    result = miner.mine(_records(), agent_name="agent")
    report = FailureModeReport(result)

    text = report.to_text()
    assert "Failure-mode report" in text
    assert "environment_fragility" in text

    d = report.to_dict()
    assert d["n_failures"] == 6
    assert len(d["clusters"]) == 3
    assert d["clusters"][0]["suggested_fix_type"] in {"workflow", "policy", "prompt", "skill", "training", "investigate"}


def test_sdk_mine_namespace_end_to_end(tmp_path: Path):
    config = SDKConfig(
        task_spec="A flight booking assistant.",
        storage=StorageConfig(path=tmp_path),
    )
    sdk = evoforge.FoundrySDK(config)

    run = EvalRunResult(
        agent_name="agent",
        overall_score=0.0,
        capability_scores={"booking": 0.0, "search": 0.0},
        case_results=[
            _failure("a1", "booking", "[ERROR: tool exploded]"),
            _failure("a2", "booking", "[ERROR: tool exploded]"),
            _failure("b1", "search", ""),
        ],
        n_passed=0,
        n_total=3,
    )
    sdk.trace.record_eval_run(run)

    result = sdk.mine.run("agent")
    assert result.n_failures == 3
    assert result.n_clusters == 2

    text = sdk.mine.report("agent")
    assert "Failure-mode report" in text


def test_mining_types_exposed_on_public_api():
    assert evoforge.FailureModeMiner is not None
    assert evoforge.FailureModeReport is not None
    assert evoforge.MiningResult is not None


# ── LLM re-classification of unknown failures ────────────────────────────────────


def _unknown_failure(case_id, capability, user, response, judge):
    """A failure the heuristic extractor cannot categorize (mode=unknown)."""
    from evoforge.core.types import EvalCase
    norm = TraceNormalizer()
    case = EvalCase(
        id=case_id, capability=capability,
        messages=[Message(role="user", content=user)],
        expected="Agent completes the booking and returns a reference number.",
        scoring_method=ScoringMethod.LLM_JUDGE,
    )
    result = EvalCaseResult(
        case_id=case_id, capability=capability, agent_response=response,
        score=0.0, passed=False, judge_reasoning=judge,
    )
    return norm.from_eval_result(result, case=case, agent_name="agent")


class ModePool:
    """Fake pool that classifies failures as tool_misuse (searched, didn't book)."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system="", temperature=0.0, max_tokens=80, **kwargs):
        self.calls += 1
        if "Classify the failure" in prompt:
            return '{"mode": "tool_misuse", "symptom": "searched but never booked", "confidence": 0.8}'
        # cluster labeling call
        return '{"root_cause": "agent never calls book_flight", "suggested_fix": "force booking tool"}'


def test_unknown_failures_are_reclassified_by_llm():
    # Two "searched instead of booked" failures the heuristic marks unknown.
    failures = [
        _unknown_failure("b1", "booking", "Book a flight from NY to LA",
                         "Here are available flights: UA123 $320.",
                         "Response does not confirm the booking or provide a reference number."),
        _unknown_failure("b2", "booking", "Book a non-stop from Boston to Seattle",
                         "There are several options available.",
                         "Response does not confirm the booking or provide a reference number."),
    ]
    # Heuristic baseline: unknown.
    assert all(f.failure_signature.mode == FailureMode.UNKNOWN for f in failures)

    pool = ModePool()
    result = FailureModeMiner(pool=pool).mine(failures, agent_name="agent")

    # Re-classified to a real, actionable mode -> clustered together.
    assert result.metadata.get("n_reclassified") == 2
    assert "tool_misuse" in result.mode_distribution
    assert "unknown" not in result.mode_distribution
    top = result.clusters[0]
    assert top.mode == FailureMode.TOOL_MISUSE
    assert top.size == 2
    assert top.suggested_fix_type == "skill"   # tool_misuse -> skill fix


def test_reclassification_skipped_without_pool():
    failures = [
        _unknown_failure("b1", "booking", "Book a flight", "Here are flights...", "no booking"),
    ]
    result = FailureModeMiner(pool=None).mine(failures, agent_name="agent")
    assert result.metadata.get("n_reclassified", 0) == 0
    assert "unknown" in result.mode_distribution


def test_confident_modes_not_reclassified():
    # A clear ERROR -> environment_fragility at confidence 0.6 (above threshold).
    norm = TraceNormalizer()
    from evoforge.core.types import EvalCase
    case = EvalCase(id="e1", capability="booking", messages=[Message(role="user", content="book")],
                    expected="ok", scoring_method=ScoringMethod.CONTAINS)
    rec = norm.from_eval_result(
        EvalCaseResult(case_id="e1", capability="booking", agent_response="[ERROR: crashed]",
                       score=0.0, passed=False),
        case=case, agent_name="agent",
    )
    assert rec.failure_signature.mode == FailureMode.ENVIRONMENT_FRAGILITY
    pool = ModePool()
    FailureModeMiner(pool=pool, reclassify=True).mine([rec], agent_name="agent")
    # No classify call needed for an already-confident mode.
    assert rec.failure_signature.mode == FailureMode.ENVIRONMENT_FRAGILITY


def test_sdk_mine_persist_writes_reclassified_modes(tmp_path: Path):
    config = SDKConfig(task_spec="A flight agent.", storage=StorageConfig(path=tmp_path))
    sdk = evoforge.FoundrySDK(config)
    sdk.trace.store.save_many([
        _unknown_failure("b1", "booking", "Book a flight NY to LA",
                         "Here are flights: UA123 $320.", "did not book"),
    ])
    # Persisted modes start as unknown...
    assert sdk.trace.load("agent")[0].failure_signature.mode == FailureMode.UNKNOWN

    sdk.mine.run("agent", pool=ModePool(), persist=True)

    # ...and are upgraded on disk after mining with persist.
    reloaded = sdk.trace.load("agent")[0]
    assert reloaded.failure_signature.mode == FailureMode.TOOL_MISUSE
    assert reloaded.failure_signature.metadata.get("reclassified") is True


def test_llm_classifier_exposed_on_public_api():
    assert evoforge.mining.LLMModeClassifier is not None
