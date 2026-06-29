"""Unit tests for success-criterion derivation + eval-case quality gate."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from foundry.coverage.case_quality import EvalCaseQualityGate
from foundry.coverage.criterion import CriterionDeriver, SuccessCriterion


class _CriterionPool:
    supports_structured = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kw):
        self.calls += 1
        return self.payload


def test_criterion_builds_expected_and_strict_rubric():
    pool = _CriterionPool({
        "success_signal": "The agent completes the booking and returns a reference number",
        "fail_signal": "The agent only lists flights",
        "success_keywords": ["confirmation", "reference"],
    })
    crit = CriterionDeriver(pool).derive("book_flight", "prompt_gap", "A flight agent")
    assert "reference number" in crit.expected()
    rubric = crit.rubric()
    assert rubric.startswith("PASS only if")
    assert "FAIL" in rubric
    assert "only lists flights" in rubric
    assert "confirmation" in rubric  # keywords surfaced


def test_criterion_is_cached_per_capability_mode():
    pool = _CriterionPool({
        "success_signal": "x", "fail_signal": "y", "success_keywords": [],
    })
    deriver = CriterionDeriver(pool)
    deriver.derive("book_flight", "prompt_gap", "task")
    deriver.derive("book_flight", "prompt_gap", "task")
    assert pool.calls == 1  # second call served from cache


def test_first_person_success_signal_is_reframed():
    pool = _CriterionPool({
        "success_signal": "I have checked the prices and confirmed your booking.",
        "fail_signal": "only searches",
        "success_keywords": [],
    })
    crit = CriterionDeriver(pool).derive("price_confirmation", "prompt_gap", "task")
    # Role-played reply wrapped into a third-person spec.
    assert crit.success_signal.startswith("The agent's response")


def test_criterion_fallback_when_empty():
    pool = _CriterionPool({})
    crit = CriterionDeriver(pool).derive("book_flight", "prompt_gap", "task")
    assert crit.success_signal  # non-empty fallback
    assert "book_flight" in crit.success_signal
    assert crit.fail_signal


def test_quality_gate_rejects_short_and_nonrequests():
    gate = EvalCaseQualityGate(existing_messages=[])
    ok, reason = gate.accept("hi", [])
    assert not ok and reason == "too short"
    ok, reason = gate.accept("The weather in Tokyo is nice this month.", [])
    assert not ok and reason == "not a user request"


def test_quality_gate_accepts_clear_requests():
    gate = EvalCaseQualityGate(existing_messages=[])
    for msg in [
        "Please book a flight from NYC to LA for John on May 3rd.",
        "I require a business class seat from SF to Tokyo next month.",
        "Can you confirm the total price before booking?",
    ]:
        ok, reason = gate.accept(msg, [])
        assert ok, f"should accept: {msg} ({reason})"


def test_quality_gate_near_dup_and_novelty():
    gate = EvalCaseQualityGate(existing_messages=["book a flight from nyc to la for john"])
    # Near-duplicate of existing corpus -> low novelty.
    ok, reason = gate.accept("Book a flight from NYC to LA for John", [])
    assert not ok and "novelty" in reason

    # Distinct request accepted, then its near-dup rejected within the batch.
    acc = []
    ok, _ = gate.accept("Please reserve a window seat to Denver next Friday", acc)
    assert ok
    acc.append("Please reserve a window seat to Denver next Friday")
    ok, reason = gate.accept("Please reserve a window seat to Denver next Friday!", acc)
    assert not ok and "near-duplicate" in reason


def test_success_criterion_rubric_without_keywords():
    crit = SuccessCriterion(
        capability="c", mode="m", success_signal="does X", fail_signal="does Y",
    )
    rubric = crit.rubric()
    assert "does X" in rubric and "does Y" in rubric
