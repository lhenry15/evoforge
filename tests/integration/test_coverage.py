"""Integration tests for adaptive eval coverage (Phase 3b).

Deterministic tests (FakePool for generation) covering the coverage map,
blind-spot detection/ranking, the heatmap, the adaptive expander's tagging, and
the headline property: generating targeted cases *closes* the blind spot.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import foundry
from foundry.core.config import SDKConfig, StorageConfig
from foundry.core.types import EvalCase, EvalCaseResult, Message, ScoringMethod
from foundry.coverage import AdaptiveEvalExpander, CoverageMapper, CoverageReport
from foundry.mining import FailureModeMiner
from foundry.trace import TraceNormalizer


class FakePool:
    """Two-step fake pool: derives a criterion, then generates user messages.

    Mirrors the real pipeline (criterion derivation -> message generation) using
    schema-constrained ``generate_json``.
    """

    supports_structured = True

    def __init__(self):
        self._msg_counter = 0

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kwargs):
        props = schema.get("properties", {})
        if "success_signal" in props:
            return {
                "success_signal": "the agent completes the booking and returns a reference number",
                "fail_signal": "the agent only searches and lists flights",
                "success_keywords": ["confirmation", "reference", "booked"],
            }
        if "messages" in props:
            base = self._msg_counter
            self._msg_counter += 1
            return {
                "messages": [
                    f"Please book a flight from city {base}-{i} to destination {base}-{i} for John Smith"
                    for i in range(4)
                ]
            }
        return {}

    def generate(self, prompt, system="", temperature=0.0, max_tokens=128, **kwargs):
        # Fallback path: emit a schema-less messages object.
        return json.dumps({"messages": ["please book a flight from NYC to LA for John"]})


def _mining(rows):
    """rows: list of (case_id, capability, response) -> MiningResult."""
    norm = TraceNormalizer()
    records = []
    for cid, cap, resp in rows:
        case = EvalCase(
            id=cid, capability=cap,
            messages=[Message(role="user", content=f"input for {cid}")],
            expected="ok", scoring_method=ScoringMethod.CONTAINS,
        )
        result = EvalCaseResult(case_id=cid, capability=cap, agent_response=resp, score=0.0, passed=False)
        records.append(norm.from_eval_result(result, case=case, agent_name="agent"))
    return FailureModeMiner().mine(records, agent_name="agent")


def _tagged_case(cap, mode, difficulty="hard"):
    return EvalCase(
        id=f"{cap}-{mode}",
        capability=cap,
        messages=[Message(role="user", content="probe")],
        expected="ok",
        scoring_method=ScoringMethod.LLM_JUDGE,
        metadata={"target_mode": mode, "difficulty": difficulty},
    )


# ── Mapper + blind spots ────────────────────────────────────────────────────────


def test_blindspot_detected_when_no_tagged_supply():
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("e2", "booking", "[ERROR: x]")])
    cmap = CoverageMapper().build([], mining, agent_name="agent")

    cell = next(c for c in cmap.cells if c.capability == "booking")
    assert cell.observed_failures == 2
    assert cell.eval_cases == 0
    assert cell.is_blindspot is True
    assert cmap.coverage_ratio() == 0.0
    assert len(cmap.blindspots()) == 1


def test_tagged_case_counts_as_supply_and_closes_cell():
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("e2", "booking", "[ERROR: x]")])
    cmap = CoverageMapper().build([_tagged_case("booking", "environment_fragility")], mining, "agent")

    cell = next(c for c in cmap.cells if c.capability == "booking")
    assert cell.eval_cases == 1
    assert cell.is_blindspot is False
    assert cmap.coverage_ratio() == 1.0
    assert cmap.blindspots() == []


def test_blindspots_ranked_by_impact():
    mining = _mining([
        ("e1", "booking", "[ERROR: x]"),     # env_fragility, sev 0.7
        ("p1", "pricing", "I cannot help"),  # policy_conflict, sev 1.0
    ])
    cmap = CoverageMapper().build([], mining, "agent")
    spots = cmap.blindspots()
    assert len(spots) == 2
    # policy_conflict (sev 1.0) outranks env_fragility (sev 0.7) at equal size
    assert spots[0].mode == "policy_conflict"


def test_matrix_structure():
    mining = _mining([("e1", "booking", "[ERROR: x]")])
    cmap = CoverageMapper().build([], mining, "agent")
    matrix = cmap.matrix()
    assert "booking" in matrix
    assert "environment_fragility" in matrix["booking"]
    assert matrix["booking"]["environment_fragility"]["blindspot"] is True


# ── Adaptive expander ───────────────────────────────────────────────────────────


def test_adaptive_expander_tags_cases_with_mode_and_lineage():
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("e2", "booking", "[ERROR: x]")])
    spots = CoverageMapper().build([], mining, "agent").blindspots()
    cases = AdaptiveEvalExpander(FakePool()).expand(spots, task_spec="A flight agent")

    assert len(cases) >= 1
    for c in cases:
        assert c.capability == "booking"
        assert c.metadata["target_mode"] == "environment_fragility"
        assert c.metadata["adaptive"] is True
        assert c.metadata["lineage"]["generation_method"] == "adaptive_eval"
        # Discriminating parts are CONSTRUCTED from the derived criterion.
        assert "completes the booking" in c.expected
        assert "PASS only if" in c.scoring_rubric
        assert "FAIL" in c.scoring_rubric
        assert c.metadata["success_keywords"]  # carried for downstream checks


def test_adaptive_expander_dedups_and_reports_stats():
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("e2", "booking", "[ERROR: x]")])
    spots = CoverageMapper().build([], mining, "agent").blindspots()
    cases, stats = AdaptiveEvalExpander(FakePool()).expand_with_stats(
        spots, task_spec="A flight agent", cases_per_blindspot=3
    )
    # Unique messages only (FakePool varies by counter, so dedup keeps distinct ones).
    texts = [c.messages[0].content for c in cases]
    assert len(texts) == len(set(texts))
    assert stats.accepted == len(cases)
    assert stats.generated_messages >= stats.accepted


def test_expansion_closes_blindspot_loop():
    """Headline property: generating targeted cases closes the blind spot."""
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("e2", "booking", "[ERROR: x]")])

    cmap_before = CoverageMapper().build([], mining, "agent")
    assert cmap_before.coverage_ratio() == 0.0
    spots = cmap_before.blindspots()
    assert len(spots) == 1

    new_cases = AdaptiveEvalExpander(FakePool()).expand(spots, task_spec="A flight agent")

    cmap_after = CoverageMapper().build(new_cases, mining, "agent")
    assert cmap_after.coverage_ratio() == 1.0
    assert not any(c.is_blindspot for c in cmap_after.cells)


# ── Report ──────────────────────────────────────────────────────────────────────


def test_coverage_report_text_and_dict():
    mining = _mining([("e1", "booking", "[ERROR: x]"), ("p1", "pricing", "I cannot help")])
    cmap = CoverageMapper().build([], mining, "agent")
    report = CoverageReport(cmap)

    text = report.to_text()
    assert "Coverage map" in text
    assert "Blind spots" in text

    d = report.to_dict()
    assert d["coverage_ratio"] == 0.0
    assert len(d["blindspots"]) == 2
    assert "matrix" in d


# ── Namespace ───────────────────────────────────────────────────────────────────


def test_sdk_coverage_namespace_end_to_end(tmp_path: Path):
    config = SDKConfig(task_spec="A flight booking assistant.", storage=StorageConfig(path=tmp_path))
    sdk = foundry.FoundrySDK(config)

    from foundry.core.types import EvalRunResult
    cases = [
        EvalCase(id="e1", capability="booking",
                 messages=[Message(role="user", content="Book UA123")],
                 expected="ok", scoring_method=ScoringMethod.CONTAINS),
        EvalCase(id="e2", capability="booking",
                 messages=[Message(role="user", content="Book AA456")],
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

    # No tagged eval cases yet -> blind spot exists.
    spots = sdk.coverage.blindspots("agent")
    assert len(spots) >= 1

    # Expand + persist, then re-map: coverage improves.
    new_cases = sdk.coverage.expand("agent", pool=FakePool(), persist=True)
    assert len(new_cases) >= 1

    text = sdk.coverage.report("agent")
    assert "Coverage map" in text


def test_coverage_types_exposed_on_public_api():
    assert foundry.CoverageMapper is not None
    assert foundry.AdaptiveEvalExpander is not None
    assert foundry.CoverageMap is not None
