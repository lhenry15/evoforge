"""
End-to-end test using fully local Ollama model (qwen2.5:3b).

This runs the complete Foundry pipeline with ZERO cloud API calls:
  Agent inference: Ollama qwen2.5:3b
  LLM Judge:       Ollama qwen2.5:3b
  Storage:         Local filesystem

Demonstrates the offline-first workflow for cost-sensitive developers.

Run:
    pytest tests/e2e/test_local_ollama.py -v -s
"""

from __future__ import annotations

import os
import sys
import pytest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Check Ollama is running
try:
    import urllib.request
    urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
except Exception:
    pytest.skip("Ollama not running at localhost:11434", allow_module_level=True)

import foundry
from foundry.core.agent_config import AgentConfig, ModelConfig, ModelHost
from foundry.core.types import EvalCase, EvolutionAction, Message, ScoringMethod
from foundry.llm.ollama import OllamaLLMPool

# ── SDK setup (local storage) ─────────────────────────────────────────────────

SDK = foundry.init(
    task_spec="A flight booking assistant that searches and books flights.",
    verbose=False,
)

# ── Eval cases ────────────────────────────────────────────────────────────────

EVAL_CASES = [
    EvalCase(
        id="local-001",
        capability="flight_search",
        messages=[Message(role="user", content="Find flights from SFO to JFK on 2024-12-01")],
        expected="flight options with prices",
        scoring_method=ScoringMethod.CONTAINS,  # fast: no LLM judge needed
    ),
    EvalCase(
        id="local-002",
        capability="flight_search",
        messages=[Message(role="user", content="What flights go from LAX to ORD tomorrow?")],
        expected="flights",
        scoring_method=ScoringMethod.CONTAINS,
    ),
    EvalCase(
        id="local-003",
        capability="booking",
        messages=[Message(role="user", content="Book flight UA123 for John Smith")],
        expected="Confirmed",
        scoring_method=ScoringMethod.CONTAINS,
    ),
    EvalCase(
        id="local-004",
        capability="error_handling",
        messages=[Message(role="user", content="Find flights from INVALID to NOWHERE on 2024-13-45")],
        expected="Handles invalid input gracefully",
        scoring_method=ScoringMethod.LLM_JUDGE,
        scoring_rubric="Agent should acknowledge the invalid airport codes or date format",
    ),
]


# ── Local Ollama agent ────────────────────────────────────────────────────────

@SDK.agent(
    tools=["search_flights", "book_flight"],
    config=AgentConfig(
        system_prompt=(
            "You are a flight booking assistant. "
            "Use the available tools to search and book flights. "
            "If input is invalid, explain what's wrong."
        ),
        model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
    ),
)
def local_flight_agent(messages: list[Message]) -> str:
    """
    Fully local agent using Ollama qwen2.5:3b with tool calling.
    Uses OpenAI-compatible endpoint.
    """
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def _search_flights(origin: str, destination: str, date: str) -> str:
        return f"Flights {origin}→{destination} on {date}: UA123 $320, AA456 $289, DL789 $355"

    def _book_flight(flight_id: str, passenger_name: str) -> str:
        return f"Confirmed: {flight_id} for {passenger_name}. Ref: BK{abs(hash(flight_id)) % 10000:04d}"

    sys_prompt = local_flight_agent._foundry_agent_config.system_prompt
    oai_messages = [{"role": "system", "content": sys_prompt}]
    for m in messages:
        if m.role in ("user", "assistant"):
            oai_messages.append({"role": m.role, "content": m.content})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search available flights between airports",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "departure IATA code"},
                        "destination": {"type": "string", "description": "arrival IATA code"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a flight for a passenger",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {"type": "string"},
                        "passenger_name": {"type": "string"},
                    },
                    "required": ["flight_id", "passenger_name"],
                },
            },
        },
    ]

    for _ in range(3):
        resp = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=oai_messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        # Append assistant message with tool calls
        oai_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_flights":
                result = _search_flights(**args)
            elif tc.function.name == "book_flight":
                result = _book_flight(**args)
            else:
                result = f"Unknown tool: {tc.function.name}"
            oai_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Could not complete request."


# ── Override eval judge to use local Ollama ───────────────────────────────────

# Monkey-patch the eval namespace to use OllamaLLMPool instead of GitHubModelsLLMPool
_original_llm_judge = None


def _patch_eval_to_use_ollama():
    """Replace the LLM judge in EvalNamespace with local Ollama."""
    import foundry.eval.namespace as ns
    import re
    import json

    async def _local_llm_judge(self, case, response):
        pool = OllamaLLMPool(model="qwen2.5:3b")
        rubric = f"\nAdditional rubric:\n{case.scoring_rubric}" if case.scoring_rubric else ""
        prompt = f"""Score this agent response.
Expected: {case.expected}{rubric}

Agent response: {response}

Reply ONLY with JSON: {{"score": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

        raw = await pool.generate(prompt, temperature=0)
        try:
            m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            obj = json.loads(m.group() if m else raw)
            return float(obj.get("score", 0.0)), obj.get("reasoning", "")
        except Exception:
            return 0.5, f"parse error: {raw[:80]}"

    ns.EvalNamespace._llm_judge = _local_llm_judge


_patch_eval_to_use_ollama()

# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLocalOllamaPipeline:
    """Full pipeline test — zero cloud calls."""

    def test_agent_runs_locally(self):
        msg = [Message(role="user", content="Find flights from SFO to JFK on 2024-12-01")]
        response = local_flight_agent(msg)
        print(f"\n[local agent] {response[:200]}")
        assert isinstance(response, str)
        assert len(response) > 5

    def test_telemetry_records(self):
        from pathlib import Path
        from foundry.data.storage.local import LocalStorageBackend

        msg = [Message(role="user", content="Book UA123 for Bob")]
        local_flight_agent(msg)

        store = LocalStorageBackend(Path.home() / ".agent-foundry")
        keys = store.list("trajectories/local_flight_agent/")
        print(f"\n[local trajectories] {len(keys)}")
        assert len(keys) >= 1

    def test_eval_fully_local(self):
        """Run eval with both rule-based and LLM judge scoring — all local."""
        result = SDK.eval.run(agent=local_flight_agent, cases=EVAL_CASES, parallelism=1)
        print(f"\n[local eval]")
        print(f"  overall:      {result.overall_score:.3f}")
        print(f"  capabilities: {result.capability_scores}")
        print(f"  passed:       {result.n_passed}/{result.n_total}")
        for r in result.case_results:
            print(f"  [{r.case_id}] {r.capability:16s} score={r.score:.2f} latency={r.latency_ms:.0f}ms")

        assert 0.0 <= result.overall_score <= 1.0
        assert result.n_total == 4

    def test_evolution_decisions_local(self):
        """Evolution engine works with local eval results."""
        result = SDK.eval.run(agent=local_flight_agent, cases=EVAL_CASES, parallelism=1)
        decision = SDK.evolve.run_cycle(agent=local_flight_agent, eval_result=result)
        print(f"\n[local evolution] {decision.summary}")
        assert len(decision.actions) >= 1

    def test_data_persistence(self):
        """Save and reload eval cases."""
        SDK.data.save_eval_cases(EVAL_CASES, tag="local_test")
        loaded = SDK.data.load_eval_cases(tag="local_test")
        assert len(loaded) == len(EVAL_CASES)
