"""Integration tests for the trace intelligence foundation (Phase 1).

These tests use no LLM calls — they verify normalization, signature extraction,
lineage, storage/indexing, and the sdk.trace namespace deterministically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import foundry
from foundry.core.config import SDKConfig, StorageConfig
from foundry.core.types import (
    EvalCase,
    EvalCaseResult,
    EvalRunResult,
    Message,
    ScoringMethod,
    Trajectory,
)
from foundry.trace import (
    FailureMode,
    TraceNormalizer,
    TraceOutcome,
    TraceRecord,
    TraceSource,
    TraceStore,
)


def _sdk(tmp_path: Path):
    config = SDKConfig(
        task_spec="A flight booking assistant.",
        storage=StorageConfig(path=tmp_path),
    )
    return foundry.FoundrySDK(config)


def _failing_case_result(case_id="c1", capability="booking") -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case_id,
        capability=capability,
        agent_response="[ERROR: tool exploded]",
        score=0.0,
        passed=False,
        judge_reasoning="agent crashed",
        latency_ms=12.0,
    )


def _passing_case_result(case_id="c2", capability="search") -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case_id,
        capability=capability,
        agent_response="Here are flights: UA123 $320",
        score=1.0,
        passed=True,
        latency_ms=9.0,
    )


# ── Normalization ──────────────────────────────────────────────────────────────


def test_normalizes_eval_result_to_trace_record():
    norm = TraceNormalizer()
    case = EvalCase(
        id="c1",
        capability="booking",
        messages=[Message(role="user", content="Book UA123 for John")],
        expected="Confirmed",
        scoring_method=ScoringMethod.CONTAINS,
    )
    record = norm.from_eval_result(_failing_case_result(), case=case, agent_name="flight_agent")

    assert isinstance(record, TraceRecord)
    assert record.source == TraceSource.EVAL
    assert record.outcome == TraceOutcome.FAILURE
    assert record.capability == "booking"
    assert record.lineage.eval_case_id == "c1"
    assert record.context_hash  # non-empty


def test_failure_gets_signature_with_stable_id():
    norm = TraceNormalizer()
    r1 = norm.from_eval_result(_failing_case_result(), agent_name="a")
    r2 = norm.from_eval_result(_failing_case_result(), agent_name="a")

    assert r1.failure_signature is not None
    assert r1.failure_signature.mode == FailureMode.ENVIRONMENT_FRAGILITY
    # Deterministic: same failure shape -> same signature id
    assert r1.failure_signature.signature_id == r2.failure_signature.signature_id
    assert r1.lineage.failure_signature_id == r1.failure_signature.signature_id


def test_passing_result_has_no_signature():
    norm = TraceNormalizer()
    record = norm.from_eval_result(_passing_case_result(), agent_name="a")
    assert record.outcome == TraceOutcome.SUCCESS
    assert record.failure_signature is None


def test_empty_response_classified_incomplete():
    norm = TraceNormalizer()
    result = EvalCaseResult(
        case_id="c3", capability="booking", agent_response="",
        score=0.0, passed=False,
    )
    record = norm.from_eval_result(result, agent_name="a")
    assert record.failure_signature.mode == FailureMode.INCOMPLETE


def test_trajectory_tool_calls_parsed_openai_and_flat_shapes():
    norm = TraceNormalizer()
    traj = Trajectory(
        id="t1",
        agent_name="flight_agent",
        messages=[Message(role="user", content="Find flights SFO to NYC")],
        response="Here are options",
        tool_calls=[
            {"function": {"name": "search_flights", "arguments": {"origin": "SFO"}}},
            {"name": "book_flight", "arguments": {"flight_id": "UA123"}, "error": "invalid"},
        ],
    )
    record = norm.from_trajectory(traj)
    assert record.tool_names == ["search_flights", "book_flight"]
    assert record.tool_invocations[0].arguments == {"origin": "SFO"}
    assert record.tool_invocations[1].succeeded is False
    # A failed tool -> partial outcome -> failure signature present
    assert record.outcome == TraceOutcome.PARTIAL
    assert record.failure_signature.mode == FailureMode.TOOL_MISUSE


def test_eval_run_normalization_maps_all_cases():
    norm = TraceNormalizer()
    run = EvalRunResult(
        agent_name="flight_agent",
        overall_score=0.5,
        capability_scores={"booking": 0.0, "search": 1.0},
        case_results=[_failing_case_result(), _passing_case_result()],
        n_passed=1,
        n_total=2,
    )
    records = norm.from_eval_run(run)
    assert len(records) == 2
    assert {r.outcome for r in records} == {TraceOutcome.FAILURE, TraceOutcome.SUCCESS}


# ── Store + indexing ────────────────────────────────────────────────────────────


def test_store_persists_and_indexes(tmp_path: Path):
    store = TraceStore(storage_path=str(tmp_path))
    norm = TraceNormalizer()
    records = [
        norm.from_eval_result(_failing_case_result("c1", "booking"), agent_name="a"),
        norm.from_eval_result(_failing_case_result("c2", "booking"), agent_name="a"),
        norm.from_eval_result(_passing_case_result("c3", "search"), agent_name="a"),
    ]
    store.save_many(records)

    loaded = store.load("a")
    assert len(loaded) == 3
    assert len(store.failures("a")) == 2
    assert set(store.index_by_capability("a").keys()) == {"booking", "search"}

    # Both booking failures share a signature -> recurrence detected
    counts = store.signature_counts("a")
    assert max(counts.values()) == 2
    assert store.recurrence_rate("a") == 1.0


# ── Namespace integration ───────────────────────────────────────────────────────


def test_sdk_trace_namespace_records_eval_run(tmp_path: Path):
    sdk = _sdk(tmp_path)
    run = EvalRunResult(
        agent_name="flight_agent",
        overall_score=0.5,
        capability_scores={"booking": 0.0},
        case_results=[_failing_case_result()],
        n_passed=0,
        n_total=1,
    )
    records = sdk.trace.record_eval_run(run)
    assert len(records) == 1
    assert len(sdk.trace.failures("flight_agent")) == 1
    assert sdk.trace.recurrence_rate("flight_agent") == 0.0  # single failure, no recurrence


def test_trace_types_exposed_on_public_api():
    assert foundry.TraceRecord is not None
    assert foundry.FailureMode is not None
    assert foundry.TraceNormalizer is not None
