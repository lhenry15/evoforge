"""Workflow/pipeline evolution — suggest structural changes to agent behavior."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from foundry.core.types import CapabilityGap, EvalRunResult


class WorkflowStep(BaseModel):
    """A step in the agent's workflow."""
    name: str                       # e.g. "validate_input", "search_flights"
    action: str                     # "call_tool" | "check_condition" | "retry" | "fallback"
    tool: Optional[str] = None      # tool name if action=call_tool
    condition: Optional[str] = None # condition if action=check_condition
    on_failure: Optional[str] = None  # what to do if step fails


class WorkflowPatch(BaseModel):
    """A suggested change to the agent's workflow."""
    patch_type: str                 # "insert_step" | "add_retry" | "add_validation" | "add_fallback"
    description: str
    before_step: Optional[str] = None   # insert before this step
    after_step: Optional[str] = None    # insert after this step
    new_steps: list[WorkflowStep] = Field(default_factory=list)
    reasoning: str = ""
    capability_targeted: str = ""


class WorkflowEvolutionResult(BaseModel):
    """Result of workflow analysis."""
    patches: list[WorkflowPatch] = Field(default_factory=list)
    current_workflow: list[WorkflowStep] = Field(default_factory=list)
    suggested_workflow: list[WorkflowStep] = Field(default_factory=list)
    summary: str = ""


_WORKFLOW_SYSTEM = """You are an expert at designing agent workflows and pipelines.
Given an agent's current behavior and its failures, suggest structural improvements
to the agent's workflow (tool ordering, validation steps, retry logic, fallbacks).

Reply with ONLY JSON:
{
  "analysis": "why the current workflow fails",
  "patches": [
    {
      "patch_type": "insert_step|add_retry|add_validation|add_fallback",
      "description": "what this change does",
      "reasoning": "why this fixes the failure",
      "new_step": {"name": "step_name", "action": "call_tool|check_condition|retry|fallback", "tool": "tool_name_or_null"}
    }
  ],
  "suggested_workflow": [
    {"name": "step_name", "action": "action_type", "tool": "tool_or_null"}
  ]
}"""

_WORKFLOW_PROMPT = """Analyze this agent's failures and suggest workflow improvements.

AGENT TASK: {task_spec}
TOOLS: {tools}
SYSTEM PROMPT: {system_prompt}

CURRENT INFERRED WORKFLOW:
{current_workflow}

CAPABILITY GAPS:
{gaps}

SAMPLE FAILURES:
{failures}

Suggest structural improvements (new steps, retries, validations, fallbacks).
Focus on WORKFLOW changes, not prompt changes."""


class WorkflowEvolver:
    """
    Suggest workflow/pipeline structural changes based on failures.

    Analyzes:
      - Missing validation steps (e.g. validate input before tool call)
      - Missing retry logic (e.g. retry failed tool call with different params)
      - Missing fallback paths (e.g. if search returns empty, try broader query)
      - Suboptimal tool ordering (e.g. confirm price BEFORE booking, not after)

    Usage::

        evolver = WorkflowEvolver(pool=OllamaLLMPool())
        result = evolver.analyze(
            agent=my_agent,
            eval_result=result,
            gaps=decision.capability_gaps,
        )
        for patch in result.patches:
            print(f"{patch.patch_type}: {patch.description}")
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def analyze(
        self,
        agent: Callable,
        eval_result: EvalRunResult,
        gaps: list[CapabilityGap],
        task_spec: str = "",
    ) -> WorkflowEvolutionResult:
        """Analyze failures and suggest workflow changes."""
        return asyncio.run(self._analyze_async(agent, eval_result, gaps, task_spec))

    async def _analyze_async(
        self,
        agent: Callable,
        eval_result: EvalRunResult,
        gaps: list[CapabilityGap],
        task_spec: str,
    ) -> WorkflowEvolutionResult:
        config = getattr(agent, "_foundry_agent_config", None)
        tools = getattr(agent, "_foundry_tools", [])
        system_prompt = config.system_prompt if config else ""

        # Infer current workflow from tools
        current_workflow = self._infer_workflow(tools)

        gaps_str = "\n".join(f"  - {g.capability}: {g.score:.2f}" for g in gaps)
        failures_str = ""
        for r in eval_result.case_results[:5]:
            if not r.passed:
                failures_str += f"  - [{r.capability}] response: \"{r.agent_response[:80]}\"\n"

        workflow_str = "\n".join(f"  {i+1}. {s.name} ({s.action})" for i, s in enumerate(current_workflow))

        prompt = _WORKFLOW_PROMPT.format(
            task_spec=task_spec,
            tools=", ".join(str(t) for t in tools),
            system_prompt=system_prompt or "(generic)",
            current_workflow=workflow_str or "  (not defined)",
            gaps=gaps_str,
            failures=failures_str or "  (no details)",
        )

        raw = await self._pool.generate(prompt, system=_WORKFLOW_SYSTEM, temperature=0.3, max_tokens=2048)
        return self._parse(raw, current_workflow)

    def _infer_workflow(self, tools: list[Any]) -> list[WorkflowStep]:
        """Infer basic workflow from tool list."""
        steps = []
        for t in tools:
            name = str(t) if isinstance(t, str) else getattr(t, "name", str(t))
            steps.append(WorkflowStep(name=name, action="call_tool", tool=name))
        return steps

    def _parse(self, raw: str, current_workflow: list[WorkflowStep]) -> WorkflowEvolutionResult:
        result = WorkflowEvolutionResult(current_workflow=current_workflow)

        try:
            cleaned = raw.strip()
            if "```" in cleaned:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            if not cleaned.startswith("{"):
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group()

            data = json.loads(cleaned)
            result.summary = data.get("analysis", "")

            for p in data.get("patches", []):
                new_step = p.get("new_step", {})
                steps = []
                if new_step:
                    steps.append(WorkflowStep(
                        name=new_step.get("name", "unknown"),
                        action=new_step.get("action", "check_condition"),
                        tool=new_step.get("tool"),
                    ))
                result.patches.append(WorkflowPatch(
                    patch_type=p.get("patch_type", "insert_step"),
                    description=p.get("description", ""),
                    reasoning=p.get("reasoning", ""),
                    new_steps=steps,
                ))

            for s in data.get("suggested_workflow", []):
                result.suggested_workflow.append(WorkflowStep(
                    name=s.get("name", "?"),
                    action=s.get("action", "call_tool"),
                    tool=s.get("tool"),
                ))

        except (json.JSONDecodeError, TypeError, AttributeError):
            result.summary = f"Parse error: {raw[:100]}"

        return result
