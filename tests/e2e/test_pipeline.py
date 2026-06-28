"""
End-to-end pipeline test using real LLM calls via GitHub Models.

Tests the full Foundry cycle:
  1. Agent wrapping (smolagents, pydantic-ai, plain function)
  2. Eval run with LLM judge + rule-based scoring
  3. Telemetry collection (trajectories saved to disk)
  4. Evolution decisions (gaps, saturations, actions)

Requires: GITHUB_TOKEN env var set (GitHub Models endpoint).

Run:
    GITHUB_TOKEN=... pytest tests/e2e/test_pipeline.py -v -s
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Skip entire module if no token
if not os.environ.get("GITHUB_TOKEN"):
    pytest.skip("GITHUB_TOKEN not set", allow_module_level=True)

import foundry
from foundry.core.agent_config import AgentConfig, ModelConfig, ModelHost
from foundry.core.types import (
    EvalCase,
    EvolutionAction,
    Message,
    ScoringMethod,
)

# ── Shared SDK ────────────────────────────────────────────────────────────────

SDK = foundry.init(
    task_spec=(
        "A flight booking assistant. Given a user request, search for "
        "available flights and book the best option. Always confirm price."
    ),
    verbose=False,
)

# ── Eval cases (used by all agents) ──────────────────────────────────────────

EVAL_CASES = [
    EvalCase(
        id="fc-001",
        capability="flight_search",
        messages=[Message(role="user", content="Find flights from SFO to JFK on 2024-12-01")],
        expected="Provides at least one flight option with price information",
        scoring_method=ScoringMethod.LLM_JUDGE,
        scoring_rubric="Response should include flight details like airline, price, or flight number",
    ),
    EvalCase(
        id="fc-002",
        capability="flight_search",
        messages=[Message(role="user", content="What flights are available from LAX to ORD tomorrow?")],
        expected="Lists flight options with relevant details",
        scoring_method=ScoringMethod.LLM_JUDGE,
    ),
    EvalCase(
        id="fc-003",
        capability="booking",
        messages=[Message(role="user", content="Book flight UA123 for John Smith")],
        expected="Confirms the booking with a reference number or confirmation",
        scoring_method=ScoringMethod.LLM_JUDGE,
        scoring_rubric="Must include booking confirmation or reference, not just say it will book",
    ),
    EvalCase(
        id="fc-004",
        capability="error_handling",
        messages=[Message(role="user", content="Find flights from INVALID to NOWHERE on 2024-13-45")],
        expected="Handles the invalid request gracefully with a helpful message",
        scoring_method=ScoringMethod.LLM_JUDGE,
        scoring_rubric="Should acknowledge the invalid input and ask for clarification or explain the issue",
    ),
    EvalCase(
        id="fc-005",
        capability="price_confirmation",
        messages=[
            Message(role="user", content="Find flights SFO to NYC on 2024-12-15"),
            Message(role="assistant", content="I found UA456 at $299 and AA789 at $245."),
            Message(role="user", content="Book the cheapest one for Alice Johnson"),
        ],
        expected="Books AA789 (cheapest) for Alice Johnson and confirms the $245 price",
        scoring_method=ScoringMethod.LLM_JUDGE,
        scoring_rubric="Must book the cheaper flight and confirm the price before or during booking",
    ),
]


# ── Agent 1: Plain Python function agent (uses GitHub Models directly) ────────

@SDK.agent(
    tools=["search_flights", "book_flight"],
    config=AgentConfig(
        system_prompt=(
            "You are a helpful flight booking assistant. "
            "Search for flights and book them when asked. "
            "Always confirm price before booking. "
            "Be concise and helpful."
        ),
        model=ModelConfig(id="gpt-4o-mini", host=ModelHost.OPENAI),
    ),
)
def plain_flight_agent(messages: list[Message]) -> str:
    """
    Direct GitHub Models agent — no framework dependency.
    Acts as our baseline / control agent.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["GITHUB_TOKEN"],
    )

    # Synthetic tool implementation (no real API)
    def _search_flights(origin: str, destination: str, date: str) -> str:
        return (
            f"Flights {origin}→{destination} on {date}: "
            f"UA123 $320 (09:00), AA456 $289 (11:30), DL789 $355 (14:00)"
        )

    def _book_flight(flight_id: str, passenger: str) -> str:
        return f"Confirmed: {flight_id} for {passenger}. Ref: BK{abs(hash(flight_id)) % 10000:04d}"

    lc_messages = [
        {"role": "system", "content": plain_flight_agent._foundry_agent_config.system_prompt}
    ]
    for m in messages:
        if m.role in ("user", "assistant"):
            lc_messages.append({"role": m.role, "content": m.content})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search available flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a specific flight",
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

    import json

    # Agentic loop (max 3 tool calls to avoid runaway)
    for _ in range(3):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=lc_messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )
        msg = resp.choices[0].message
        lc_messages.append(msg.to_dict() if hasattr(msg, "to_dict") else {"role": "assistant", "content": msg.content or "", "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else {} for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            return msg.content or ""

        # Execute tool calls
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_flights":
                result = _search_flights(**args)
            elif tc.function.name == "book_flight":
                result = _book_flight(**args)
            else:
                result = f"Unknown tool: {tc.function.name}"
            lc_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I was unable to complete the request within the allowed steps."


# ── Agent 2: smolagents wrapper ───────────────────────────────────────────────

def _make_smol_agent():
    from smolagents import ToolCallingAgent, InferenceClientModel, tool

    @tool
    def search_flights(origin: str, destination: str, date: str) -> str:
        """Search available flights between two cities on a given date.

        Args:
            origin: departure airport IATA code (e.g. SFO)
            destination: arrival airport IATA code (e.g. JFK)
            date: travel date in YYYY-MM-DD format
        """
        return f"Flights {origin}→{destination} on {date}: UA123 $320, AA456 $289, DL789 $355"

    @tool
    def book_flight(flight_id: str, passenger_name: str) -> str:
        """Book a specific flight for a passenger.

        Args:
            flight_id: flight identifier such as UA123
            passenger_name: full legal name of the passenger
        """
        return f"Confirmed: {flight_id} for {passenger_name}. Ref: BK{abs(hash(flight_id)) % 10000:04d}"

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        token=os.environ.get("HF_TOKEN", ""),
    )
    return ToolCallingAgent(tools=[search_flights, book_flight], model=model)


_smol_agent_instance = None


@SDK.agent(
    tools=["search_flights", "book_flight"],
    config=AgentConfig(
        system_prompt="You are a helpful flight booking assistant.",
        model=ModelConfig(id="Qwen/Qwen2.5-72B-Instruct", host=ModelHost.HUGGINGFACE),
    ),
)
def smol_flight_agent(messages: list[Message]) -> str:
    """smolagents ToolCallingAgent wrapper."""
    global _smol_agent_instance
    if _smol_agent_instance is None:
        _smol_agent_instance = _make_smol_agent()
    task = messages[-1].content if messages else ""
    try:
        return str(_smol_agent_instance.run(task))
    except Exception as e:
        return f"[smolagents error: {e}]"


# ── Agent 3: pydantic-ai wrapper ──────────────────────────────────────────────

def _make_pai_agent():
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["GITHUB_TOKEN"],
    )
    model = OpenAIChatModel("gpt-4o-mini", provider=provider)
    agent = Agent(
        model,
        system_prompt=(
            "You are a helpful flight booking assistant. "
            "Use tools to search and book flights. Always confirm price."
        ),
    )

    @agent.tool_plain
    def search_flights(origin: str, destination: str, date: str) -> str:
        """Search available flights."""
        return f"Flights {origin}→{destination} on {date}: UA123 $320, AA456 $289, DL789 $355"

    @agent.tool_plain
    def book_flight(flight_id: str, passenger_name: str) -> str:
        """Book a flight."""
        return f"Confirmed: {flight_id} for {passenger_name}. Ref: BK{abs(hash(flight_id)) % 10000:04d}"

    return agent


_pai_agent_instance = None


@SDK.agent(
    tools=["search_flights", "book_flight"],
    config=AgentConfig(
        system_prompt="You are a helpful flight booking assistant. Always confirm price.",
        model=ModelConfig(id="gpt-4o-mini", host=ModelHost.OPENAI),
        swap_model=lambda new_id: print(f"[pydantic-ai] swap_model called: {new_id}"),
    ),
)
def pai_flight_agent(messages: list[Message]) -> str:
    """pydantic-ai Agent wrapper."""
    global _pai_agent_instance
    if _pai_agent_instance is None:
        _pai_agent_instance = _make_pai_agent()
    task = messages[-1].content if messages else ""
    try:
        result = _pai_agent_instance.run_sync(task)
        return result.output
    except Exception as e:
        return f"[pydantic-ai error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPlainAgentPipeline:
    """Full pipeline test with plain Python agent (most reliable, no extra deps)."""

    def test_agent_runs_and_returns_string(self):
        msg = [Message(role="user", content="Find flights from SFO to JFK on 2024-12-01")]
        response = plain_flight_agent(msg)
        print(f"\n[plain agent response] {response[:200]}")
        assert isinstance(response, str)
        assert len(response) > 10

    def test_telemetry_records_trajectory(self):
        """Verify that calling the agent persists a trajectory to disk."""
        from pathlib import Path
        from foundry.data.storage.local import LocalStorageBackend

        msg = [Message(role="user", content="Any flights SFO to NYC on 2024-12-10?")]
        plain_flight_agent(msg)

        store = LocalStorageBackend(Path.home() / ".agent-foundry")
        keys = store.list("trajectories/plain_flight_agent/")
        print(f"\n[trajectories stored] {len(keys)}")
        assert len(keys) >= 1

    def test_eval_run_returns_scores(self):
        """Run eval cases and verify scores come back."""
        result = SDK.eval.run(agent=plain_flight_agent, cases=EVAL_CASES)
        print(f"\n[eval result]")
        print(f"  overall_score:      {result.overall_score:.3f}")
        print(f"  capability_scores:  {result.capability_scores}")
        print(f"  passed:             {result.n_passed}/{result.n_total}")
        for r in result.case_results:
            print(f"  [{r.case_id}] {r.capability:20s} score={r.score:.2f}  latency={r.latency_ms:.0f}ms")
            if r.judge_reasoning:
                print(f"           judge: {r.judge_reasoning}")

        assert 0.0 <= result.overall_score <= 1.0
        assert set(result.capability_scores.keys()) == {
            "flight_search", "booking", "error_handling", "price_confirmation"
        }
        assert result.n_total == len(EVAL_CASES)

    def test_eval_result_saved_to_storage(self):
        """Verify eval results persist."""
        result = SDK.eval.run(agent=plain_flight_agent, cases=EVAL_CASES[:2])
        key = SDK.data.save_eval_result(result)
        loaded = SDK.data.load_eval_results(agent_name="plain_flight_agent")
        assert len(loaded) >= 1
        assert loaded[-1].agent_name == "plain_flight_agent"

    def test_evolution_cycle_identifies_gaps(self):
        """Run a full evolution cycle and verify decisions are reasonable."""
        result = SDK.eval.run(agent=plain_flight_agent, cases=EVAL_CASES)
        decision = SDK.evolve.run_cycle(agent=plain_flight_agent, eval_result=result)

        print(f"\n[evolution decision]")
        print(f"  summary: {decision.summary}")
        print(f"  actions: {[a.value for a in decision.actions]}")
        print(f"  gaps:    {[(g.capability, g.score) for g in decision.capability_gaps]}")
        print(f"  sats:    {[(s.capability, s.score) for s in decision.saturation_signals]}")

        assert len(decision.actions) >= 1
        assert decision.agent_name == "plain_flight_agent"
        # With a capable LLM and good prompts, expect no catastrophic failures
        assert EvolutionAction.NO_ACTION in decision.actions or \
               EvolutionAction.GENERATE_TRAIN_DATA in decision.actions or \
               EvolutionAction.EXPAND_EVAL in decision.actions

    def test_eval_cases_persist_and_reload(self):
        """Verify save/load round-trip for eval cases."""
        SDK.data.save_eval_cases(EVAL_CASES, tag="test")
        loaded = SDK.data.load_eval_cases(tag="test")
        assert len(loaded) == len(EVAL_CASES)
        assert loaded[0].id == EVAL_CASES[0].id
        assert loaded[0].capability == EVAL_CASES[0].capability


class TestPydanticAIPipeline:
    """Pipeline test with pydantic-ai agent (GitHub Models gpt-4o-mini)."""

    def test_agent_runs(self):
        msg = [Message(role="user", content="Find flights from BOS to MIA on 2024-12-05")]
        response = pai_flight_agent(msg)
        print(f"\n[pydantic-ai response] {response[:200]}")
        assert isinstance(response, str)
        assert len(response) > 10

    def test_eval_run(self):
        result = SDK.eval.run(agent=pai_flight_agent, cases=EVAL_CASES[:3], parallelism=1)
        print(f"\n[pydantic-ai eval] overall={result.overall_score:.3f}  passed={result.n_passed}/{result.n_total}")
        assert 0.0 <= result.overall_score <= 1.0

    def test_evolution_cycle(self):
        result = SDK.eval.run(agent=pai_flight_agent, cases=EVAL_CASES[:3], parallelism=1)
        decision = SDK.evolve.run_cycle(agent=pai_flight_agent, eval_result=result)
        print(f"\n[pydantic-ai evolution] {decision.summary}")
        assert len(decision.actions) >= 1


class TestSmolagentsPipeline:
    """
    Pipeline test with smolagents (HuggingFace Inference).

    smolagents uses HF Inference API which requires a separate HF token.
    If unavailable, the agent gracefully returns an error string and
    we still verify the Foundry pipeline handles it correctly.
    """

    def test_agent_runs_or_fails_gracefully(self):
        msg = [Message(role="user", content="Find flights SFO to SEA on 2024-12-20")]
        response = smol_flight_agent(msg)
        print(f"\n[smolagents response] {response[:200]}")
        # Should always return a string — either a real response or graceful error
        assert isinstance(response, str)

    def test_eval_handles_errors_gracefully(self):
        """Even if smolagents fails, eval pipeline should complete with scores."""
        result = SDK.eval.run(agent=smol_flight_agent, cases=EVAL_CASES[:2])
        print(f"\n[smolagents eval] overall={result.overall_score:.3f}")
        assert isinstance(result.overall_score, float)
        assert result.n_total == 2


class TestCrossAgentComparison:
    """Compare all agents on the same eval set."""

    def test_compare_agents(self):
        agents = [plain_flight_agent, pai_flight_agent, smol_flight_agent]
        results = []
        for agent in agents:
            r = SDK.eval.run(agent=agent, cases=EVAL_CASES)
            results.append((agent.__name__, r))
            SDK.data.save_eval_result(r)

        print("\n[cross-agent comparison]")
        print(f"{'Agent':<30} {'Overall':>8} {'flight_search':>14} {'booking':>8} {'error':>8}")
        print("-" * 75)
        for name, r in results:
            fs = r.capability_scores.get("flight_search", 0)
            bk = r.capability_scores.get("booking", 0)
            eh = r.capability_scores.get("error_handling", 0)
            print(f"{name:<30} {r.overall_score:>8.3f} {fs:>14.3f} {bk:>8.3f} {eh:>8.3f}")

        # Both agents should produce valid scores
        for _, r in results:
            assert 0.0 <= r.overall_score <= 1.0
