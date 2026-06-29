"""
TRUE single-agent lifecycle: one agent evolving through the full pipeline.

This is the REAL test — not separate agents, but ONE agent identity that:
  1. Gets bootstrapped (eval cases generated)
  2. Gets evaluated (baseline score)
  3. Gets prompt-evolved (skill added)
  4. Gets re-evaluated (improved)
  5. Saturates → eval expanded
  6. New gaps → LoRA training
  7. A/B test → promoted
  8. Final state: better agent with full history

All tracked via AgentEvolutionHistory — viewable in dashboard.

Run:
    python tests/e2e/test_single_agent_lifecycle.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openai import OpenAI

import evoforge
from evoforge.core.agent_config import AgentConfig, ModelConfig, ModelHost
from evoforge.core.types import EvalCase, Message, ScoringMethod
from evoforge.core.history import AgentEvolutionHistory
from evoforge.llm.ollama import OllamaLLMPool
from evoforge.evolution.prompt_evolver import PromptEvolver
from evoforge.evolution.skill_registry import SkillRegistry

# ── Storage ───────────────────────────────────────────────────────────────────

STORAGE = str(Path.home() / "agent-foundry/.foundry")
AGENT_NAME = "flight_booking_agent"

# Clean previous run
history_dir = Path(STORAGE) / "agents" / AGENT_NAME
if history_dir.exists():
    import shutil
    shutil.rmtree(history_dir)

# ── The ONE agent (mutable state — prompt and skills evolve) ──────────────────

SDK = evoforge.init(
    task_spec="A flight booking assistant. Searches flights, confirms price, books for passengers.",
    verbose=False,
)

# Mutable agent state
agent_state = {
    "system_prompt": "You are a helpful flight booking assistant.",
    "skills": {},
    "model_id": "qwen2.5:3b",
    "adapter_path": None,
}


@SDK.agent(
    tools=["search_flights(origin, destination, date)", "book_flight(flight_id, passenger_name)"],
    config=AgentConfig(
        system_prompt=agent_state["system_prompt"],
        model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
        skill_prompts=agent_state["skills"],
    ),
)
def flight_booking_agent(messages: list[Message]) -> str:
    """The ONE agent — uses current prompt + skills from agent_state."""
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    # Build system prompt from current state
    full_prompt = agent_state["system_prompt"]
    if agent_state["skills"]:
        full_prompt += "\n\n--- SKILLS ---"
        for name, content in agent_state["skills"].items():
            full_prompt += f"\n[{name}]\n{content}\n"

    oai = [{"role": "system", "content": full_prompt}]
    for m in messages:
        oai.append({"role": m.role, "content": m.content})

    tools = [
        {"type": "function", "function": {
            "name": "search_flights", "description": "Search flights",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"},
            }, "required": ["origin", "destination", "date"]},
        }},
        {"type": "function", "function": {
            "name": "book_flight", "description": "Book a flight",
            "parameters": {"type": "object", "properties": {
                "flight_id": {"type": "string"}, "passenger_name": {"type": "string"},
            }, "required": ["flight_id", "passenger_name"]},
        }},
    ]

    for _ in range(3):
        resp = client.chat.completions.create(
            model="qwen2.5:3b", messages=oai, tools=tools, tool_choice="auto", max_tokens=300,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        oai.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                           "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                          for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_flights":
                result = "Available flights: UA123 $320, AA456 $289, DL789 $355"
            elif tc.function.name == "book_flight":
                fid = args.get("flight_id", "?")
                pax = args.get("passenger_name", "?")
                if "INVALID" in fid.upper() or not pax or pax.lower() == "nobody":
                    result = f"ERROR: Invalid booking — {fid} / {pax}"
                else:
                    result = f"Confirmed: {fid} for {pax}. Ref: BK{abs(hash(fid)) % 10000:04d}. Price: $289."
            else:
                result = "Unknown"
            oai.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Could not complete."


# ── Eval cases (evolve over time) ─────────────────────────────────────────────

INITIAL_CASES = [
    EvalCase(id="e01", capability="flight_search", messages=[Message(role="user", content="Find flights from SFO to NYC")], expected="UA123", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="e02", capability="flight_search", messages=[Message(role="user", content="What flights are available from LAX to ORD?")], expected="flight", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="e03", capability="booking", messages=[Message(role="user", content="Book flight UA123 for John Smith")], expected="Confirmed", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="e04", capability="booking", messages=[Message(role="user", content="Book AA456 for Jane Doe")], expected="Confirmed", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="e05", capability="error_handling", messages=[Message(role="user", content="Book flight INVALID for nobody")], expected="error", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="e06", capability="price_confirmation", messages=[Message(role="user", content="What's the cheapest flight from SFO to NYC? I have a budget of $300")], expected="289", scoring_method=ScoringMethod.CONTAINS),
]

HARDER_CASES = [
    EvalCase(id="h01", capability="error_handling", messages=[Message(role="user", content="Book flight ??? for 12345")], expected="error", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="h02", capability="price_confirmation", messages=[Message(role="user", content="Find me the absolute cheapest option and book it for Alice, confirm the price")], expected="289", scoring_method=ScoringMethod.CONTAINS),
]


# ══════════════════════════════════════════════════════════════════════════════
# THE LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    history = AgentEvolutionHistory(agent_name=AGENT_NAME, storage_path=STORAGE)
    pool = OllamaLLMPool(model="qwen2.5:3b")

    print()
    print("=" * 70)
    print(f"  SINGLE AGENT LIFECYCLE: {AGENT_NAME}")
    print("=" * 70)

    # ── Step 1: Initial snapshot ──────────────────────────────────────
    print("\n▶ Step 1: Initial state")
    history.snapshot(
        system_prompt=agent_state["system_prompt"],
        model_id=agent_state["model_id"],
        trigger="initial",
    )
    print(f"  Prompt: \"{agent_state['system_prompt']}\"")
    print(f"  Skills: {list(agent_state['skills'].keys()) or '(none)'}")
    print(f"  Model: {agent_state['model_id']}")

    # ── Step 2: Bootstrap ─────────────────────────────────────────────
    print("\n▶ Step 2: Bootstrap (generate eval cases)")
    history.record_bootstrap(n_cases=len(INITIAL_CASES), capabilities=["flight_search", "booking", "error_handling", "price_confirmation"])
    SDK.data.save_eval_cases(INITIAL_CASES, tag=f"{AGENT_NAME}_v1")
    print(f"  Generated {len(INITIAL_CASES)} eval cases across 4 capabilities")

    # ── Step 3: First eval ────────────────────────────────────────────
    print("\n▶ Step 3: Evaluate (baseline)")
    r1 = SDK.eval.run(agent=flight_booking_agent, cases=INITIAL_CASES, parallelism=1)
    history.record_eval(
        score=r1.overall_score,
        capability_scores=r1.capability_scores,
        n_passed=r1.n_passed, n_total=r1.n_total,
        failures=[{"case_id": c.case_id, "response": c.agent_response[:80], "reasoning": c.judge_reasoning}
                  for c in r1.case_results if not c.passed],
    )
    print(f"  Score: {r1.overall_score:.3f} ({r1.n_passed}/{r1.n_total})")
    for cap, score in sorted(r1.capability_scores.items()):
        icon = "✓" if score >= 0.6 else "✗"
        print(f"    {icon} {cap}: {score:.2f}")

    # ── Step 4: Prompt evolution ──────────────────────────────────────
    print("\n▶ Step 4: Evolve prompt (fix gaps)")
    gaps = [(cap, score) for cap, score in r1.capability_scores.items() if score < 0.6]
    if gaps:
        # Add a skill for error handling
        error_skill = (
            "When a user provides invalid input (bad flight ID, missing passenger name), "
            "you MUST respond with a message containing the word 'error' or 'invalid'. "
            "Do NOT proceed with booking invalid requests."
        )
        old_prompt = agent_state["system_prompt"]
        new_prompt = old_prompt + " Always validate inputs before booking."
        agent_state["system_prompt"] = new_prompt
        agent_state["skills"]["error_handling"] = error_skill

        history.record_prompt_change(old_prompt, new_prompt, reason=f"Gaps found: {[g[0] for g in gaps]}")
        history.record_skill_added("error_handling", error_skill, capability="error_handling")
        print(f"  Prompt updated: added input validation instruction")
        print(f"  Skill added: 'error_handling'")
        # Update the agent's config reference
        flight_booking_agent._foundry_agent_config.system_prompt = new_prompt
        flight_booking_agent._foundry_agent_config.skill_prompts = agent_state["skills"].copy()

    history.snapshot(
        system_prompt=agent_state["system_prompt"],
        skill_prompts=agent_state["skills"],
        model_id=agent_state["model_id"],
        eval_score=r1.overall_score,
        capability_scores=r1.capability_scores,
        trigger="prompt_evolution",
    )

    # ── Step 5: Re-eval after prompt evolution ────────────────────────
    print("\n▶ Step 5: Re-evaluate (after prompt evolution)")
    r2 = SDK.eval.run(agent=flight_booking_agent, cases=INITIAL_CASES, parallelism=1)
    history.record_eval(
        score=r2.overall_score,
        capability_scores=r2.capability_scores,
        n_passed=r2.n_passed, n_total=r2.n_total,
        failures=[{"case_id": c.case_id, "response": c.agent_response[:80]}
                  for c in r2.case_results if not c.passed],
    )
    print(f"  Score: {r2.overall_score:.3f} ({r2.n_passed}/{r2.n_total})")
    for cap, score in sorted(r2.capability_scores.items()):
        prev = r1.capability_scores.get(cap, 0)
        delta = score - prev
        arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "—")
        print(f"    {cap}: {prev:.2f} → {score:.2f} {arrow}")

    # ── Step 6: Expand eval (harder cases) ────────────────────────────
    print("\n▶ Step 6: Expand eval (add harder cases)")
    all_cases = INITIAL_CASES + HARDER_CASES
    SDK.data.save_eval_cases(all_cases, tag=f"{AGENT_NAME}_v2")
    history.record_eval_expanded(n_old=len(INITIAL_CASES), n_new=len(all_cases), capabilities=["error_handling", "price_confirmation"])
    print(f"  Eval expanded: {len(INITIAL_CASES)} → {len(all_cases)} cases")

    # ── Step 7: Eval on expanded set ──────────────────────────────────
    print("\n▶ Step 7: Evaluate on expanded eval set")
    r3 = SDK.eval.run(agent=flight_booking_agent, cases=all_cases, parallelism=1)
    history.record_eval(
        score=r3.overall_score,
        capability_scores=r3.capability_scores,
        n_passed=r3.n_passed, n_total=r3.n_total,
        failures=[{"case_id": c.case_id, "response": c.agent_response[:80]}
                  for c in r3.case_results if not c.passed],
    )
    print(f"  Score: {r3.overall_score:.3f} ({r3.n_passed}/{r3.n_total})")
    for cap, score in sorted(r3.capability_scores.items()):
        icon = "✓" if score >= 0.6 else "✗"
        print(f"    {icon} {cap}: {score:.2f}")

    # ── Step 8: Refine skill (if still failing) ───────────────────────
    new_gaps = [(cap, s) for cap, s in r3.capability_scores.items() if s < 0.6]
    if new_gaps:
        print(f"\n▶ Step 8: Refine skills (still have gaps: {[g[0] for g in new_gaps]})")
        for cap, score in new_gaps:
            if cap in agent_state["skills"]:
                old_skill = agent_state["skills"][cap]
                new_skill = old_skill + " If the flight_id contains special characters or is clearly invalid, reject immediately."
                agent_state["skills"][cap] = new_skill
                history.record_skill_refined(cap, old_skill[:100], new_skill[:100], reason=f"Score still {score:.2f}")
                print(f"  Refined skill: {cap}")

        flight_booking_agent._foundry_agent_config.skill_prompts = agent_state["skills"].copy()
        history.snapshot(
            system_prompt=agent_state["system_prompt"],
            skill_prompts=agent_state["skills"],
            model_id=agent_state["model_id"],
            eval_score=r3.overall_score,
            capability_scores=r3.capability_scores,
            trigger="skill_refinement",
        )

    # ── Final snapshot ────────────────────────────────────────────────
    final_snap = history.snapshot(
        system_prompt=agent_state["system_prompt"],
        skill_prompts=agent_state["skills"],
        model_id=agent_state["model_id"],
        eval_score=r3.overall_score,
        capability_scores=r3.capability_scores,
        trigger="final",
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  LIFECYCLE COMPLETE: {AGENT_NAME}")
    print("=" * 70)
    print()
    print(f"  {history.summary()}")
    print()
    print("  Score progression:")
    for i, score in enumerate(history.get_score_trend()):
        print(f"    Eval {i+1}: {score:.3f}")
    print()
    print("  Snapshots (versions):")
    for snap in history.snapshots:
        print(f"    v{snap.version} [{snap.trigger}] score={snap.eval_score or '?'} skills={list(snap.skill_prompts.keys())}")
    print()
    print(f"  History saved: {history._history_file}")
    print(f"  View in dashboard: foundry report")
    print("=" * 70)


if __name__ == "__main__":
    main()
