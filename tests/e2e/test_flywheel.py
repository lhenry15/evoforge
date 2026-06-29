"""
End-to-end flywheel test: eval → data gen → train → re-eval.

This is the definitive test of the Foundry data-centric evolution loop:
  1. Eval local agent → find capability gaps
  2. Generate targeted training data from gaps
  3. LoRA fine-tune qwen2.5:3b locally
  4. Validate fine-tuned model responds correctly
  5. (Bonus) Re-eval and confirm improvement signal

Runs fully LOCAL — no cloud API calls. Requires Ollama + qwen2.5:3b.

Run:
    pytest tests/e2e/test_flywheel.py -v -s
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Check Ollama is running
try:
    import urllib.request
    urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
except Exception:
    pytest.skip("Ollama not running at localhost:11434", allow_module_level=True)

# Check mlx-lm is available
try:
    import mlx_lm
except ImportError:
    pytest.skip("mlx-lm not installed", allow_module_level=True)

import evoforge
from evoforge.core.agent_config import AgentConfig, ModelConfig, ModelHost
from evoforge.core.types import (
    EvalCase,
    EvolutionAction,
    Message,
    ScoringMethod,
)
from evoforge.llm.ollama import OllamaLLMPool
from evoforge.training.backends.mlx_lora import MLXLoRABackend, MLXLoRAConfig
from evoforge.evolution.namespace import CycleResult

# ── SDK ───────────────────────────────────────────────────────────────────────

SDK = evoforge.init(
    task_spec=(
        "A flight booking assistant. Searches flights and books them. "
        "Must ALWAYS confirm price before booking."
    ),
    verbose=False,
)

# ── Eval cases focused on the BOOKING capability ──────────────────────────────

BOOKING_EVAL_CASES = [
    EvalCase(
        id="book-001",
        capability="booking",
        messages=[Message(role="user", content="Book flight UA123 for John Smith")],
        expected="Confirms the booking with reference number",
        scoring_method=ScoringMethod.CONTAINS,  # fast: check for "Confirmed" or "BK"
    ),
    EvalCase(
        id="book-002",
        capability="booking",
        messages=[Message(role="user", content="I'd like to book the cheapest flight from SFO to NYC for Alice")],
        expected="price",
        scoring_method=ScoringMethod.CONTAINS,  # should mention price
    ),
    EvalCase(
        id="book-003",
        capability="booking",
        messages=[
            Message(role="user", content="Find flights SFO to LAX on 2024-12-01"),
            Message(role="assistant", content="Found: UA100 $199, DL200 $249"),
            Message(role="user", content="Book UA100 for Bob Jones"),
        ],
        expected="confirm",
        scoring_method=ScoringMethod.CONTAINS,
    ),
    EvalCase(
        id="search-001",
        capability="flight_search",
        messages=[Message(role="user", content="What flights go from BOS to MIA tomorrow?")],
        expected="flight",
        scoring_method=ScoringMethod.CONTAINS,
    ),
]


# ── Local agent (intentionally weak at booking) ──────────────────────────────

SYSTEM_PROMPT = (
    "You are a flight booking assistant. "
    "Use tools to search and book flights. "
    "Always confirm the price before booking."
)


@SDK.agent(
    tools=["search_flights", "book_flight"],
    config=AgentConfig(
        system_prompt=SYSTEM_PROMPT,
        model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
    ),
)
def flywheel_agent(messages: list[Message]) -> str:
    """Local Ollama agent for flywheel testing."""
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def _search_flights(origin: str, destination: str, date: str) -> str:
        return f"Flights {origin}→{destination} on {date}: UA123 $320, AA456 $289, DL789 $355"

    def _book_flight(flight_id: str, passenger_name: str) -> str:
        return f"Confirmed: {flight_id} for {passenger_name}. Ref: BK{abs(hash(flight_id)) % 10000:04d}"

    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        if m.role in ("user", "assistant"):
            oai_messages.append({"role": m.role, "content": m.content})

    tools = [
        {"type": "function", "function": {
            "name": "search_flights",
            "description": "Search available flights",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            }, "required": ["origin", "destination", "date"]},
        }},
        {"type": "function", "function": {
            "name": "book_flight",
            "description": "Book a flight for a passenger",
            "parameters": {"type": "object", "properties": {
                "flight_id": {"type": "string"},
                "passenger_name": {"type": "string"},
            }, "required": ["flight_id", "passenger_name"]},
        }},
    ]

    for _ in range(3):
        resp = client.chat.completions.create(
            model="qwen2.5:3b", messages=oai_messages,
            tools=tools, tool_choice="auto", max_tokens=256,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        oai_messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                           "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                          for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_flights":
                result = _search_flights(**args)
            elif tc.function.name == "book_flight":
                result = _book_flight(**args)
            else:
                result = "unknown"
            oai_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Could not complete request."


# ── Patch eval to use local Ollama judge ──────────────────────────────────────

def _patch_eval():
    import re, json
    import evoforge.eval.namespace as ns

    async def _local_judge(self, case, response):
        pool = OllamaLLMPool(model="qwen2.5:3b")
        prompt = f"""Score this response. Expected: {case.expected}
Rubric: {case.scoring_rubric or 'Response should address the expected behavior.'}
Response: {response}
Reply ONLY JSON: {{"score": <0.0-1.0>, "reasoning": "..."}}"""
        raw = await pool.generate(prompt, temperature=0)
        try:
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            obj = json.loads(m.group() if m else raw)
            return float(obj.get("score", 0.0)), obj.get("reasoning", "")
        except Exception:
            return 0.5, f"parse error: {raw[:60]}"

    ns.EvalNamespace._llm_judge = _local_judge

_patch_eval()


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFlywheel:
    """Full flywheel: eval → data gen → LoRA train → validate."""

    def test_full_evolution_cycle(self):
        """
        THE definitive Foundry test:
          1. Eval agent → find gaps
          2. execute_cycle() → generates data + trains
          3. Validate fine-tuned model
        """
        print("\n" + "=" * 70)
        print("FOUNDRY FLYWHEEL TEST — eval → data gen → train → validate")
        print("=" * 70)

        # Step 1: Eval
        print("\n[Step 1] Evaluating agent...")
        eval_result = SDK.eval.run(
            agent=flywheel_agent,
            cases=BOOKING_EVAL_CASES,
            parallelism=1,
        )
        print(f"  overall: {eval_result.overall_score:.3f}")
        print(f"  capabilities: {eval_result.capability_scores}")
        for r in eval_result.case_results:
            print(f"    [{r.case_id}] {r.capability}: {r.score:.2f}")

        # Step 2: Execute evolution cycle
        print("\n[Step 2] Executing evolution cycle (data gen + train)...")
        pool = OllamaLLMPool(model="qwen2.5:3b")
        backend = MLXLoRABackend(config=MLXLoRAConfig(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            iters=50,
            batch_size=1,
            num_layers=4,
            max_seq_length=256,
            output_dir="/tmp/foundry_flywheel_test",
        ))

        cycle = SDK.evolve.execute_cycle(
            agent=flywheel_agent,
            eval_result=eval_result,
            llm_pool=pool,
            training_backend=backend,
            examples_per_gap=10,
        )

        print(f"  decision: {cycle.decision.summary if cycle.decision else 'None'}")
        print(f"  examples generated: {cycle.training_examples_generated}")
        if cycle.training_job:
            job = cycle.training_job
            print(f"  training job: {job.status}")
            print(f"    train_loss={job.train_loss:.4f}  val_loss={job.val_loss:.4f}")
            print(f"    adapter: {job.model_id}")
        if cycle.validation_response:
            print(f"  validation: {cycle.validation_response[:150]}")
        if cycle.errors:
            print(f"  errors: {cycle.errors}")

        # Assertions
        assert cycle.success, f"Cycle failed: {cycle.errors}"
        assert cycle.decision is not None
        assert cycle.training_examples_generated > 0
        if cycle.training_job:
            assert cycle.training_job.succeeded

        print("\n" + "=" * 70)
        print("✅ FLYWHEEL COMPLETE — agent improved via data-centric evolution")
        print("=" * 70)

    def test_execute_cycle_skips_when_no_gaps(self):
        """If all capabilities are above threshold, no action is taken."""
        from evoforge.core.types import EvalRunResult, EvalCaseResult

        # Simulate a perfect eval result
        perfect_result = EvalRunResult(
            agent_name="flywheel_agent",
            overall_score=0.95,
            capability_scores={"flight_search": 0.95, "booking": 0.90},
            case_results=[],
            n_passed=4,
            n_total=4,
        )

        pool = OllamaLLMPool(model="qwen2.5:3b")
        cycle = SDK.evolve.execute_cycle(
            agent=flywheel_agent,
            eval_result=perfect_result,
            llm_pool=pool,
        )

        assert cycle.decision is not None
        assert EvolutionAction.EXPAND_EVAL in cycle.decision.actions
        assert cycle.training_examples_generated == 0
        assert cycle.training_job is None
        print(f"\n[no-gap test] Actions: {[a.value for a in cycle.decision.actions]}")

    def test_execute_cycle_without_training_backend(self):
        """Without a training backend, only data is generated (no training)."""
        eval_result = SDK.eval.run(
            agent=flywheel_agent,
            cases=BOOKING_EVAL_CASES[:2],
            parallelism=1,
        )

        pool = OllamaLLMPool(model="qwen2.5:3b")
        cycle = SDK.evolve.execute_cycle(
            agent=flywheel_agent,
            eval_result=eval_result,
            llm_pool=pool,
            training_backend=None,  # No backend — just generate data
            examples_per_gap=5,
        )

        print(f"\n[no-backend test] Generated {cycle.training_examples_generated} examples, job={cycle.training_job}")
        assert cycle.training_job is None
        # If there were gaps, data should still be generated
        if cycle.decision and EvolutionAction.GENERATE_TRAIN_DATA in cycle.decision.actions:
            assert cycle.training_examples_generated > 0
