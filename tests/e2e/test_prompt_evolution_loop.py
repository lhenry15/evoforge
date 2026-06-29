"""
End-to-end: Prompt + Skill evolution loop.

Proves the full flywheel using PROMPT evolution (not LoRA):
  1. Eval agent → identify gaps
  2. Evolve prompts/skills → fix gaps (instant, free)
  3. Re-eval agent WITH the new prompt → verify improvement
  4. Saturating → expand eval → harder cases
  5. Re-eval with harder cases → find new gaps
  6. Evolve again → fix new gaps

This is the LIGHTWEIGHT flywheel — no model training, just prompt engineering.

Run:
    python tests/e2e/test_prompt_evolution_loop.py
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
from evoforge.llm.ollama import OllamaLLMPool
from evoforge.evolution.prompt_evolver import PromptEvolver

# ── SDK ───────────────────────────────────────────────────────────────────────

SDK = evoforge.init(
    task_spec="A flight booking assistant. Searches and books flights. Must validate inputs and confirm price.",
    verbose=False,
)

# ── Agent that uses system_prompt + skill_prompts at runtime ──────────────────

INITIAL_SYSTEM_PROMPT = "You are a helpful flight booking assistant."


def make_prompted_agent(system_prompt: str, skill_prompts: dict[str, str] = None):
    """Create an agent that incorporates system_prompt + skill_prompts into its behavior."""

    # Build the full prompt from system + skills
    full_prompt = system_prompt
    if skill_prompts:
        full_prompt += "\n\n--- SKILLS ---\n"
        for name, content in skill_prompts.items():
            full_prompt += f"\n[{name}]\n{content}\n"

    @SDK.agent(
        tools=["search_flights", "book_flight"],
        config=AgentConfig(
            system_prompt=system_prompt,
            model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
            skill_prompts=skill_prompts or {},
        ),
    )
    def prompted_agent(messages: list[Message]) -> str:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
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
                model="qwen2.5:3b", messages=oai,
                tools=tools, tool_choice="auto", max_tokens=300,
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
                    result = f"Flights: UA123 $320, AA456 $289, DL789 $355"
                elif tc.function.name == "book_flight":
                    fid = args.get("flight_id", "?")
                    pax = args.get("passenger_name", "?")
                    # Simulate validation — INVALID flights fail
                    if "INVALID" in fid.upper() or not pax or pax == "nobody":
                        result = f"ERROR: Invalid booking request — flight_id={fid}, passenger={pax}"
                    else:
                        result = f"Confirmed: {fid} for {pax}. Ref: BK{abs(hash(fid)) % 10000:04d}"
                else:
                    result = "Unknown"
                oai.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "Could not complete."

    return prompted_agent


# ── Eval cases ────────────────────────────────────────────────────────────────

ROUND1_CASES = [
    EvalCase(id="r1-01", capability="booking",
        messages=[Message(role="user", content="Book flight UA123 for John Smith")],
        expected="Confirmed", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="r1-02", capability="booking",
        messages=[Message(role="user", content="Book AA456 for Jane Doe")],
        expected="Confirmed", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="r1-03", capability="error_handling",
        messages=[Message(role="user", content="Book flight INVALID for nobody")],
        expected="error", scoring_method=ScoringMethod.CONTAINS),  # should mention error/invalid
    EvalCase(id="r1-04", capability="error_handling",
        messages=[Message(role="user", content="Book flight XYZ123 for")],
        expected="invalid", scoring_method=ScoringMethod.CONTAINS),  # missing passenger
    EvalCase(id="r1-05", capability="price_confirmation",
        messages=[Message(role="user", content="Book the cheapest flight for Alice. Options: UA123 $320, AA456 $289")],
        expected="289", scoring_method=ScoringMethod.CONTAINS),  # should pick cheapest
    EvalCase(id="r1-06", capability="flight_search",
        messages=[Message(role="user", content="What flights are available from SFO to NYC?")],
        expected="flight", scoring_method=ScoringMethod.CONTAINS),
]

ROUND2_HARDER_CASES = [
    EvalCase(id="r2-01", capability="error_handling",
        messages=[Message(role="user", content="Book flight UA123 for !!!@#$%")],
        expected="invalid", scoring_method=ScoringMethod.CONTAINS),
    EvalCase(id="r2-02", capability="price_confirmation",
        messages=[Message(role="user", content="I want to fly from SFO to LAX but my budget is only $200. What can I get?")],
        expected="289", scoring_method=ScoringMethod.CONTAINS),  # cheapest is $289 > $200, should inform
    EvalCase(id="r2-03", capability="multi_step",
        messages=[Message(role="user", content="Search flights SFO to NYC, then book the cheapest one for Bob")],
        expected="Confirmed", scoring_method=ScoringMethod.CONTAINS),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 70)
    print("  FOUNDRY — PROMPT + SKILL EVOLUTION LOOP")
    print("  Lightweight flywheel: no model training, just prompt engineering")
    print("=" * 70)

    pool = OllamaLLMPool(model="qwen2.5:3b")

    # ══════════════════════════════════════════════════════════════════════
    # CYCLE 1: Eval with basic prompt → identify gaps → evolve prompt
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  CYCLE 1: Evaluate → Identify Gaps → Evolve Prompt")
    print("▓" * 70)

    # 1a: Eval with initial prompt
    print("\n  [1a] Evaluating with INITIAL prompt...")
    agent_v0 = make_prompted_agent(INITIAL_SYSTEM_PROMPT)
    r1 = SDK.eval.run(agent=agent_v0, cases=ROUND1_CASES, parallelism=1)

    print(f"  Overall: {r1.overall_score:.3f} ({r1.n_passed}/{r1.n_total} passed)")
    for cap, score in sorted(r1.capability_scores.items()):
        icon = "✓" if score >= 0.6 else "✗"
        print(f"    {icon} {cap}: {score:.2f}")
    print()
    for cr in r1.case_results:
        icon = "✓" if cr.passed else "✗"
        print(f"    {icon} [{cr.case_id}] {cr.agent_response[:50]}")

    # 1b: Evolve prompt based on failures
    print("\n  [1b] Evolving prompt based on failures...")
    decision = SDK.evolve.run_cycle(agent=agent_v0, eval_result=r1)
    print(f"  Decision: {[a.value for a in decision.actions]}")
    print(f"  Gaps: {[(g.capability, g.score) for g in decision.capability_gaps]}")

    if decision.capability_gaps:
        evolver = PromptEvolver(pool=pool)
        evo_result = evolver.evolve(
            agent=agent_v0,
            eval_result=r1,
            gaps=decision.capability_gaps,
            task_spec=SDK.config.task_spec,
        )
        print(f"\n  Prompt patches: {len(evo_result.patches)}")
        if evo_result.patches:
            new_prompt = evo_result.patches[0].revised
            print(f"  New system prompt: \"{new_prompt[:80]}...\"")
        else:
            new_prompt = INITIAL_SYSTEM_PROMPT
        print(f"  New skills: {[s.name for s in evo_result.new_skills]}")
        new_skills = {s.name: s.content for s in evo_result.new_skills}
    else:
        new_prompt = INITIAL_SYSTEM_PROMPT
        new_skills = {}

    # 1c: Re-eval with evolved prompt
    print("\n  [1c] Re-evaluating with EVOLVED prompt...")
    agent_v1 = make_prompted_agent(new_prompt, new_skills)
    r2 = SDK.eval.run(agent=agent_v1, cases=ROUND1_CASES, parallelism=1)

    print(f"  Overall: {r2.overall_score:.3f} ({r2.n_passed}/{r2.n_total} passed)")
    for cap, score in sorted(r2.capability_scores.items()):
        prev = r1.capability_scores.get(cap, 0)
        delta = score - prev
        arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "—")
        print(f"    {cap}: {prev:.2f} → {score:.2f} ({delta:+.2f}) {arrow}")
    print()
    for cr in r2.case_results:
        icon = "✓" if cr.passed else "✗"
        print(f"    {icon} [{cr.case_id}] {cr.agent_response[:50]}")

    # ══════════════════════════════════════════════════════════════════════
    # CYCLE 2: Saturating → expand eval → harder cases → new gaps
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  CYCLE 2: Expand Eval → Harder Cases → Find New Gaps")
    print("▓" * 70)

    saturating = {c: s for c, s in r2.capability_scores.items() if s > 0.85}
    print(f"\n  Saturating: {saturating}")
    print(f"  Adding {len(ROUND2_HARDER_CASES)} harder cases...")

    all_cases = ROUND1_CASES + ROUND2_HARDER_CASES
    print(f"\n  [2a] Evaluating v1 on expanded eval ({len(all_cases)} cases)...")
    r3 = SDK.eval.run(agent=agent_v1, cases=all_cases, parallelism=1)

    print(f"  Overall: {r3.overall_score:.3f} ({r3.n_passed}/{r3.n_total} passed)")
    for cap, score in sorted(r3.capability_scores.items()):
        icon = "✓" if score >= 0.6 else "✗"
        print(f"    {icon} {cap}: {score:.2f}")
    print()
    for cr in r3.case_results:
        icon = "✓" if cr.passed else "✗"
        print(f"    {icon} [{cr.case_id}] {cr.agent_response[:50]}")

    # ══════════════════════════════════════════════════════════════════════
    # CYCLE 3: Evolve prompt again for new gaps
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  CYCLE 3: Evolve Prompt Again for New Gaps")
    print("▓" * 70)

    decision3 = SDK.evolve.run_cycle(agent=agent_v1, eval_result=r3)
    print(f"\n  Decision: {[a.value for a in decision3.actions]}")
    print(f"  Gaps: {[(g.capability, g.score) for g in decision3.capability_gaps]}")

    if decision3.capability_gaps:
        evo_result3 = evolver.evolve(
            agent=agent_v1,
            eval_result=r3,
            gaps=decision3.capability_gaps,
            task_spec=SDK.config.task_spec,
        )
        if evo_result3.patches:
            final_prompt = evo_result3.patches[0].revised
            print(f"  New prompt: \"{final_prompt[:80]}...\"")
        else:
            final_prompt = new_prompt
        final_skills = new_skills.copy()
        for s in evo_result3.new_skills:
            final_skills[s.name] = s.content
        print(f"  Skills: {list(final_skills.keys())}")
    else:
        final_prompt = new_prompt
        final_skills = new_skills

    # 3b: Re-eval with v2 prompt
    print(f"\n  [3b] Re-evaluating with v2 prompt on full eval...")
    agent_v2 = make_prompted_agent(final_prompt, final_skills)
    r4 = SDK.eval.run(agent=agent_v2, cases=all_cases, parallelism=1)

    print(f"  Overall: {r4.overall_score:.3f} ({r4.n_passed}/{r4.n_total} passed)")
    for cap, score in sorted(r4.capability_scores.items()):
        prev = r3.capability_scores.get(cap, 0)
        delta = score - prev
        arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "—")
        print(f"    {cap}: {prev:.2f} → {score:.2f} ({delta:+.2f}) {arrow}")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY — Prompt Evolution Loop (3 cycles)")
    print("=" * 70)
    print()
    print(f"  {'Stage':<40} {'Score':>6} {'Cases':>6} {'Passed':>7}")
    print(f"  {'─'*40} {'─'*6} {'─'*6} {'─'*7}")
    print(f"  {'v0: Initial prompt':<40} {r1.overall_score:>6.3f} {r1.n_total:>6} {r1.n_passed:>7}")
    print(f"  {'v1: After prompt evolution (cycle 1)':<40} {r2.overall_score:>6.3f} {r2.n_total:>6} {r2.n_passed:>7}")
    print(f"  {'v1: On harder eval (cycle 2)':<40} {r3.overall_score:>6.3f} {r3.n_total:>6} {r3.n_passed:>7}")
    print(f"  {'v2: After 2nd evolution (cycle 3)':<40} {r4.overall_score:>6.3f} {r4.n_total:>6} {r4.n_passed:>7}")
    print()
    print(f"  Prompt versions: 3 (initial → v1 → v2)")
    print(f"  Skills generated: {len(final_skills)} ({list(final_skills.keys())})")
    print(f"  Model training: NONE (all improvement from prompts)")
    print()

    total_delta = r4.overall_score - r1.overall_score
    if total_delta > 0:
        print(f"  ✅ IMPROVEMENT via prompt evolution alone: {r1.overall_score:.3f} → {r4.overall_score:.3f} (Δ {total_delta:+.3f})")
    else:
        print(f"  Result: {r1.overall_score:.3f} → {r4.overall_score:.3f} (Δ {total_delta:+.3f})")
        print("  Note: Prompt evolution has limits — LoRA training may be needed for deeper behavioral changes.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
