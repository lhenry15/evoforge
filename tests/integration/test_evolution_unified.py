"""Integration test for the rewired EvolveNamespace.execute_cycle.

Verifies that execute_cycle now delegates training-data generation to synthesis
(mode-conditioned) and eval growth to coverage expansion — the unified path —
using a deterministic structured fake pool (no real LLM for those steps).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
from evoforge.core.config import SDKConfig, StorageConfig
from evoforge.core.types import (
    EvalCase,
    EvalCaseResult,
    EvalRunResult,
    Message,
    ScoringMethod,
)


class StructuredFakePool:
    """Deterministic structured pool covering mining, synthesis, and coverage."""

    supports_structured = True

    def __init__(self):
        self._n = 0

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kw):
        props = schema.get("properties", {})
        if "success_signal" in props:
            return {
                "success_signal": "the agent completes the booking and returns a reference number",
                "fail_signal": "the agent only lists flights",
                "success_keywords": ["confirmation", "reference"],
            }
        if "messages" in props:
            self._n += 1
            return {"messages": [
                f"Please book flight {self._n}-{i} from NYC to LA for passenger {self._n}-{i}"
                for i in range(4)
            ]}
        if "examples" in props:
            self._n += 1
            return {"examples": [
                {"instruction": f"book flight {self._n}-{i} for John",
                 "ideal_response": f"Confirmed booking {self._n}-{i}, reference R{self._n}{i}."}
                for i in range(3)
            ]}
        return {}

    def generate(self, prompt, system="", temperature=0.0, max_tokens=128, **kw):
        # Mode classification during mining.
        if "Classify the failure" in prompt:
            return '{"mode": "tool_misuse", "symptom": "searched but did not book", "confidence": 0.8}'
        if "corrected response" in system:
            return "Confirmed your booking with reference R123."
        return "{}"


def _seed_traces(sdk):
    cases = [
        EvalCase(id=f"c{i}", capability="book_flight",
                 messages=[Message(role="user", content=f"Book a flight {i} from NYC to LA")],
                 expected="Agent books and returns a reference number.",
                 scoring_method=ScoringMethod.LLM_JUDGE)
        for i in range(3)
    ]
    sdk.data.save_eval_cases(cases, tag="bootstrap")
    results = [
        EvalCaseResult(case_id=f"c{i}", capability="book_flight",
                       agent_response="Here are flights: UA123 $320.",
                       score=0.0, passed=False, judge_reasoning="did not book")
        for i in range(3)
    ]
    run = EvalRunResult(agent_name="agent", overall_score=0.0,
                        capability_scores={"book_flight": 0.0},
                        case_results=results, n_passed=0, n_total=3)
    sdk.trace.record_eval_run(run, cases=cases)
    return run


def test_execute_cycle_uses_synthesis_and_coverage(tmp_path: Path):
    sdk = evoforge.FoundrySDK(SDKConfig(
        task_spec="A flight booking assistant that books flights.",
        storage=StorageConfig(path=tmp_path),
    ))
    run = _seed_traces(sdk)

    # Build a decorated-ish agent stub the evolution path can introspect.
    def agent(messages):
        return "ok"
    agent.__name__ = "agent"
    agent._foundry_tools = ["search_flights", "book_flight"]
    agent._foundry_agent_config = None

    pool = StructuredFakePool()
    cycle = sdk.evolve.execute_cycle(
        agent=agent,
        eval_result=run,
        llm_pool=pool,
        training_backend=None,   # skip actual training
        system_prompt="You are a flight booking assistant.",
        tools=["search_flights", "book_flight"],
    )

    # Synthesis produced targeted training examples (mode-conditioned path).
    assert cycle.training_examples_generated > 0
    # Coverage expansion produced tagged eval cases (blind-spot path).
    assert len(cycle.expanded_eval_cases) > 0
    for c in cycle.expanded_eval_cases:
        assert c.metadata.get("adaptive") is True
        assert c.metadata.get("target_mode")

    # No errors from the synthesis/coverage delegation.
    synth_cov_errors = [
        e for e in cycle.errors
        if "Data generation failed" in e or "Eval expansion failed" in e
    ]
    assert synth_cov_errors == []


def test_old_generator_modules_are_removed():
    import importlib
    for mod in ("evoforge.factory.strategies", "evoforge.eval.expander"):
        try:
            importlib.import_module(mod)
            assert False, f"{mod} should have been removed"
        except ModuleNotFoundError:
            pass
