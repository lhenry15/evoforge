"""Prompt evolution — rewrite prompts to fix capability gaps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import CapabilityGap, EvalRunResult


class PromptPatch(BaseModel):
    """A suggested change to a prompt."""
    target: str                     # "system_prompt" or skill name
    original: str                   # current prompt text
    revised: str                    # suggested new text
    reasoning: str                  # why the change was made
    capability_targeted: str        # which gap this addresses
    confidence: float = 0.0         # 0-1, how confident we are this helps


class SkillFile(BaseModel):
    """A generated/updated skill instruction file."""
    name: str                       # e.g. "error_handling", "price_confirmation"
    path: str                       # relative path (e.g. "skills/error_handling.md")
    content: str                    # markdown content
    version: int = 1
    capability_targeted: str


class PromptEvolutionResult(BaseModel):
    """Result of prompt/skill evolution."""
    patches: list[PromptPatch] = Field(default_factory=list)
    new_skills: list[SkillFile] = Field(default_factory=list)
    auto_applied: list[str] = Field(default_factory=list)  # which patches were auto-applied
    metadata: dict[str, Any] = Field(default_factory=dict)


_ANALYSIS_SYSTEM = """You are an expert prompt engineer analyzing agent failures.
Given eval results showing capability gaps, analyze the failures and suggest 
precise prompt modifications that would fix the issues.

You must output JSON with this structure:
{
  "failure_analysis": "brief analysis of why the agent is failing",
  "system_prompt_patch": {
    "add_instructions": ["instruction to add"],
    "remove_instructions": ["instruction to remove if counterproductive"],
    "revised_full": "the complete revised system prompt"
  },
  "new_skills": [
    {
      "name": "skill_name",
      "content": "# Skill: Name\\n\\nMarkdown instructions for this specific capability..."
    }
  ]
}"""

_ANALYSIS_PROMPT = """Analyze these agent failures and suggest prompt improvements.

CURRENT SYSTEM PROMPT:
{system_prompt}

CURRENT SKILL PROMPTS:
{skill_prompts}

TASK SPEC: {task_spec}

CAPABILITY GAPS (agent is failing on these):
{gaps}

SAMPLE FAILURES:
{failures}

Analyze WHY the agent fails on these capabilities and suggest:
1. System prompt modifications (be precise — add/remove specific instructions)
2. New skill files (markdown docs the agent can reference for specific capabilities)

Focus on the ROOT CAUSE — don't just add "try harder" instructions."""


class PromptEvolver:
    """
    Evolve prompts and skill files to fix capability gaps.

    This is faster than LoRA — changes are instant and free.
    Used when:
      - A capability gap is detected but the fix is likely behavioral (not knowledge)
      - The agent has the right tools but uses them incorrectly
      - Error handling, formatting, or process issues

    Usage::

        evolver = PromptEvolver(pool=OllamaLLMPool())
        result = evolver.evolve(
            agent=my_agent,
            eval_result=eval_result,
            gaps=decision.capability_gaps,
        )
        # Auto-apply skill_prompts, suggest system_prompt changes
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def evolve(
        self,
        agent: Callable,
        eval_result: EvalRunResult,
        gaps: list[CapabilityGap],
        task_spec: str = "",
        skills_dir: Optional[str] = None,
    ) -> PromptEvolutionResult:
        """Analyze failures and generate prompt/skill patches (sync)."""
        import json as _json
        import requests

        config = getattr(agent, "_foundry_agent_config", None)
        system_prompt = config.system_prompt if config else ""
        skill_prompts = config.skill_prompts if config else {}

        gaps_str = "\n".join(f"  - {g.capability}: {g.score:.2f}" for g in gaps)
        failures_str = ""
        for r in eval_result.case_results[:5]:
            if not r.passed:
                failures_str += f"  [{r.capability}] \"{r.agent_response[:80]}\"\n"

        prompt = _ANALYSIS_PROMPT.format(
            system_prompt=system_prompt or "(not provided)",
            skill_prompts=_json.dumps(skill_prompts) if skill_prompts else "(none)",
            task_spec=task_spec,
            gaps=gaps_str,
            failures=failures_str[:1000] or "(no details)",
        )

        # Use Ollama REST API directly (no async issues)
        try:
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model": "qwen2.5:3b",
                "messages": [
                    {"role": "system", "content": _ANALYSIS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            }, timeout=60)
            raw = resp.json().get("message", {}).get("content", "")
        except Exception as e:
            return PromptEvolutionResult(metadata={"error": str(e)})

        return self._parse_result(raw, system_prompt, skill_prompts, gaps, skills_dir)

    def _parse_result(
        self,
        raw: str,
        current_system_prompt: str,
        current_skills: dict[str, str],
        gaps: list[CapabilityGap],
        skills_dir: Optional[str],
    ) -> PromptEvolutionResult:
        result = PromptEvolutionResult()

        try:
            # Extract JSON from response
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

            # System prompt patch
            sp_patch = data.get("system_prompt_patch", {})
            revised = sp_patch.get("revised_full", "")
            if revised and revised != current_system_prompt:
                result.patches.append(PromptPatch(
                    target="system_prompt",
                    original=current_system_prompt,
                    revised=revised,
                    reasoning=data.get("failure_analysis", ""),
                    capability_targeted=gaps[0].capability if gaps else "unknown",
                    confidence=0.7,
                ))

            # New skills
            for skill_data in data.get("new_skills", []):
                name = skill_data.get("name", "")
                content = skill_data.get("content", "")
                if name and content:
                    path = f"skills/{name}.md"
                    result.new_skills.append(SkillFile(
                        name=name,
                        path=path,
                        content=content,
                        capability_targeted=gaps[0].capability if gaps else "unknown",
                    ))

        except (json.JSONDecodeError, TypeError, AttributeError):
            result.metadata["parse_error"] = raw[:200]

        return result

    def apply_skills(
        self,
        result: PromptEvolutionResult,
        skills_dir: str,
    ) -> list[str]:
        """Write skill files to disk. Returns list of created paths."""
        created = []
        base = Path(skills_dir)
        base.mkdir(parents=True, exist_ok=True)

        for skill in result.new_skills:
            path = base / f"{skill.name}.md"
            path.write_text(skill.content)
            created.append(str(path))

        return created

    def apply_skill_prompts(
        self,
        agent: Callable,
        result: PromptEvolutionResult,
    ) -> list[str]:
        """Auto-apply skill_prompt patches to the agent's config."""
        applied = []
        config = getattr(agent, "_foundry_agent_config", None)
        if not config:
            return applied

        for skill in result.new_skills:
            config.skill_prompts[skill.name] = skill.content
            applied.append(skill.name)

        result.auto_applied = applied
        return applied
