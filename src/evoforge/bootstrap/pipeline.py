"""Bootstrap pipeline — auto-generate eval cases from task_spec + tools (sync)."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import EvalCase, Message, ScoringMethod
from evoforge.llm.structured import coerce_records, extract_json
from evoforge.text import format_tools


_CAPABILITY_SYSTEM = """You are an expert AI evaluation designer.
Given an agent's task specification and available tools, identify the distinct 
capabilities the agent needs to perform well.

Reply with ONLY a JSON array of objects:
[{"name": "capability_name", "description": "what this capability tests"}]

Requirements:
- EXACTLY 3 or 4 capabilities (no more!)
- Broad categories only (e.g. "search", "booking", "error_handling")
- Each name must be a single word or two words with underscore
- Do NOT create sub-capabilities or granular ones"""

_EVAL_GEN_SYSTEM = """You are an expert at creating evaluation test cases for AI agents.
Generate diverse, realistic test cases that thoroughly test a specific capability.

Reply with ONLY a JSON array of objects:
[{
  "user_message": "the message a user would send",
  "expected": "BEHAVIORAL description of success (e.g. 'agent lists flights with prices', NOT specific response text)",
  "difficulty": "easy|medium|hard",
  "scoring_rubric": "specific pass/fail criteria (e.g. 'response mentions at least one flight number and price')"
}]

CRITICAL RULES:
- "expected" must describe WHAT the agent should DO, not exact words it should say
- WRONG: "I can confirm the price of your flight at $320"  
- RIGHT: "Agent confirms the booking price to the user"
- Keep user_messages realistic and diverse"""


class BootstrapConfig(BaseModel):
    num_eval_cases: int = 20
    min_per_capability: int = 3
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.3, "medium": 0.4, "hard": 0.3}
    )
    scoring_preference: str = "llm_judge"


class BootstrapResult(BaseModel):
    capabilities: list[dict[str, str]]
    eval_cases: list[EvalCase]
    multi_turn_scenarios: list[Any] = Field(default_factory=list)
    n_generated: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class BootstrapPipeline:
    """Auto-generate eval cases from task_spec + tools (fully synchronous)."""

    def __init__(self, pool: Any, config: Optional[BootstrapConfig] = None) -> None:
        self._pool = pool
        self._config = config or BootstrapConfig()

    def run(
        self,
        task_spec: str,
        tools: list[Any] = None,
        system_prompt: str = "",
        agent: Any = None,
    ) -> BootstrapResult:
        tools = tools or []

        # Step 1: Infer capabilities
        capabilities = self._infer_capabilities(task_spec, tools, system_prompt)

        # Step 2: Calibrate (run agent on sample prompts)
        calibration = {}
        if agent is not None:
            calibration = self._calibrate_agent(agent, capabilities, task_spec)

        # Step 3: Generate eval cases
        n_per_cap = max(
            self._config.min_per_capability,
            self._config.num_eval_cases // len(capabilities) if capabilities else self._config.num_eval_cases,
        )

        all_cases: list[EvalCase] = []
        for cap in capabilities:
            cases = self._generate_cases(task_spec, cap, tools, n_per_cap,
                                         calibration.get(cap.get("name", ""), ""))
            all_cases.extend(cases)

        # Step 4: Multi-turn scenarios
        scenarios = self._generate_multi_turn(task_spec, capabilities, tools)

        return BootstrapResult(
            capabilities=capabilities,
            eval_cases=all_cases,
            multi_turn_scenarios=scenarios,
            n_generated=len(all_cases) + len(scenarios),
        )

    def _infer_capabilities(self, task_spec, tools, system_prompt) -> list[dict[str, str]]:
        tool_desc = self._format_tools(tools)
        prompt = f"""Identify the distinct capabilities for this agent:
TASK: {task_spec}
TOOLS: {tool_desc}
SYSTEM PROMPT: {system_prompt or '(not provided)'}
What capabilities should we evaluate?"""

        raw = self._pool.generate(prompt, system=_CAPABILITY_SYSTEM, temperature=0.3)
        caps = coerce_records(extract_json(raw))
        if caps:
            return caps[:4]  # Cap at 4
        return [{"name": str(t)[:20], "description": f"Tests {t}"} for t in tools[:3]]

    def _calibrate_agent(self, agent, capabilities, task_spec) -> dict[str, str]:
        from evoforge.core.types import Message
        calibration = {}
        samples = {
            "search": "Find flights from SFO to NYC",
            "book": "Book flight UA123 for John Smith",
            "price": "What is the cheapest flight?",
            "error": "Book an invalid flight",
        }
        for cap in capabilities:
            cap_name = cap.get("name", "")
            prompt = next((p for k, p in samples.items() if k in cap_name.lower()), f"Help with {cap_name}")
            try:
                response = agent([Message(role="user", content=prompt)])
                calibration[cap_name] = str(response)[:200]
            except Exception:
                pass
        return calibration

    def _generate_cases(self, task_spec, capability, tools, n, calibration="") -> list[EvalCase]:
        cap_name = capability.get("name", "unknown")
        cap_desc = capability.get("description", "")

        cal_section = ""
        if calibration:
            cal_section = f'\nACTUAL AGENT OUTPUT EXAMPLE:\n"{calibration[:150]}"\nCalibrate expectations to match this style.'

        prompt = f"""Generate {n} eval cases for this capability:
AGENT TASK: {task_spec}
CAPABILITY: {cap_name} — {cap_desc}
TOOLS: {self._format_tools(tools)}
{cal_section}
Generate {n} diverse test cases for "{cap_name}"."""

        raw = self._pool.generate(prompt, system=_EVAL_GEN_SYSTEM, temperature=0.7, max_tokens=2048)
        return self._parse_cases(raw, cap_name)

    def _parse_cases(self, raw: str, capability: str) -> list[EvalCase]:
        items = coerce_records(extract_json(raw))
        cases = []
        for item in items:
            user_msg = item.get("user_message", "")
            if not user_msg:
                continue
            method = ScoringMethod.LLM_JUDGE if self._config.scoring_preference == "llm_judge" else ScoringMethod.CONTAINS
            cases.append(EvalCase(
                id=f"{capability[:8]}-{uuid.uuid4().hex[:4]}",
                capability=capability,
                messages=[Message(role="user", content=user_msg)],
                expected=item.get("expected", ""),
                scoring_method=method,
                scoring_rubric=item.get("scoring_rubric"),
                metadata={"difficulty": item.get("difficulty", "medium"), "bootstrapped": True},
            ))
        return cases

    def _generate_multi_turn(self, task_spec, capabilities, tools) -> list[Any]:
        from evoforge.eval.multi_turn import MultiTurnScenario, Milestone

        prompt = f"""Design 2 multi-turn conversation scenarios for this agent.
TASK: {task_spec}
TOOLS: {self._format_tools(tools)}

Reply with JSON array:
[{{"id": "scenario_id", "capability": "single_capability_name", "initial_message": "first user msg",
  "user_responses": ["2nd msg", "3rd msg"],
  "milestones": [{{"description": "what should happen", "check": "keyword"}}]}}]"""

        raw = self._pool.generate(prompt, temperature=0.7, max_tokens=2048)
        items = coerce_records(extract_json(raw))
        scenarios = []
        for item in items:
            if not item.get("initial_message"):
                continue
            milestones = []
            for m in item.get("milestones", []):
                if isinstance(m, dict) and m.get("check"):
                    check = m["check"]
                    if isinstance(check, list):
                        check = check[0] if check else ""
                    if check:
                        milestones.append(Milestone(description=m.get("description", ""), check=str(check)))
            if not milestones:
                continue
            cap = item.get("capability", "multi_turn")
            if isinstance(cap, list):
                cap = cap[0] if cap else "multi_turn"
            scenarios.append(MultiTurnScenario(
                id=item.get("id", f"mt-{uuid.uuid4().hex[:4]}"),
                capability=str(cap).split(",")[0].strip(),
                initial_message=item["initial_message"],
                user_responses=item.get("user_responses", []),
                milestones=milestones,
            ))
        return scenarios

    def _format_tools(self, tools: list[Any]) -> str:
        return format_tools(tools)

    @staticmethod
    def mine_scenarios_from_trajectories(trajectories: list[Any]) -> list[Any]:
        from evoforge.eval.multi_turn import MultiTurnScenario, Milestone
        scenarios = []
        for traj in trajectories:
            messages = traj.messages if hasattr(traj, 'messages') else []
            user_msgs = [m.content for m in messages if m.role == "user"]
            if len(user_msgs) < 2:
                continue
            response = traj.response if hasattr(traj, 'response') else ""
            keywords = [w for w in response.split() if len(w) > 4 and w.isalpha()][:2]
            milestones = [Milestone(description="Agent responds", check=kw) for kw in keywords] or [Milestone(description="Agent responds", check="the")]
            scenarios.append(MultiTurnScenario(
                id=f"mined-{getattr(traj, 'id', uuid.uuid4().hex[:4])}",
                capability="mined_workflow",
                initial_message=user_msgs[0],
                user_responses=user_msgs[1:],
                milestones=milestones,
            ))
        return scenarios
