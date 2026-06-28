"""SkillRegistry — versioned skill lifecycle management."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class SkillVersion(BaseModel):
    """A single version of a skill."""
    version: int
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    eval_score: Optional[float] = None      # score when this version was active
    change_reason: str = ""                  # why this version was created


class SkillStatus(str):
    ACTIVE = "active"           # currently in use
    REFINED = "refined"         # has a newer version but kept for rollback
    MERGED = "merged"           # absorbed into another skill
    RETIRED = "retired"         # model has internalized it via LoRA


class Skill(BaseModel):
    """A versioned skill with full lifecycle tracking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                               # e.g. "error_handling"
    capability: str                         # which capability this addresses
    status: str = "active"                  # active | refined | merged | retired
    versions: list[SkillVersion] = Field(default_factory=list)
    merged_into: Optional[str] = None       # skill ID if merged
    retired_reason: Optional[str] = None    # e.g. "internalized by LoRA adapter v2"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def current_version(self) -> SkillVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def current_content(self) -> str:
        return self.current_version.content if self.current_version else ""

    @property
    def version_count(self) -> int:
        return len(self.versions)


class SkillRegistry:
    """
    Manages the lifecycle of skill files:
      CREATE  → new skill from capability gap
      REFINE  → update skill based on continued failures (v1 → v2 → v3)
      MERGE   → combine overlapping skills into one
      RETIRE  → mark as internalized after LoRA training proves it's learned

    Usage::

        registry = SkillRegistry(storage_path=".foundry/skills")
        registry.create("error_handling", content="...", capability="error_handling")
        registry.refine("error_handling", new_content="...", reason="Still failing")
        registry.retire("error_handling", reason="LoRA v2 scored 0.95 without this skill")
    """

    def __init__(self, storage_path: str, pool: Any = None) -> None:
        self._path = Path(storage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._pool = pool
        self._skills: dict[str, Skill] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(self, name: str, content: str, capability: str, reason: str = "initial") -> Skill:
        """Create a new skill (v1)."""
        if name in self._skills and self._skills[name].status == "active":
            # Already exists — refine instead
            return self.refine(name, content, reason)

        skill = Skill(
            name=name,
            capability=capability,
            status="active",
            versions=[SkillVersion(version=1, content=content, change_reason=reason)],
        )
        self._skills[name] = skill
        self._save_skill(skill)
        return skill

    def refine(self, name: str, new_content: str, reason: str = "") -> Skill:
        """Create a new version of an existing skill."""
        skill = self._skills.get(name)
        if not skill:
            return self.create(name, new_content, capability="unknown", reason=reason)

        new_version = SkillVersion(
            version=skill.version_count + 1,
            content=new_content,
            change_reason=reason,
        )
        skill.versions.append(new_version)
        self._save_skill(skill)
        return skill

    def merge(self, source_name: str, target_name: str, merged_content: str) -> Skill:
        """Merge source skill into target skill."""
        source = self._skills.get(source_name)
        target = self._skills.get(target_name)

        if not source or not target:
            raise ValueError(f"Skills not found: {source_name}, {target_name}")

        # Refine target with merged content
        self.refine(target_name, merged_content, reason=f"merged with {source_name}")

        # Mark source as merged
        source.status = "merged"
        source.merged_into = target.id
        self._save_skill(source)

        return target

    def retire(self, name: str, reason: str) -> Skill:
        """Retire a skill (model has internalized it)."""
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Skill not found: {name}")

        skill.status = "retired"
        skill.retired_reason = reason
        self._save_skill(skill)
        return skill

    def rollback(self, name: str, to_version: int) -> Skill:
        """Rollback a skill to a previous version."""
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Skill not found: {name}")
        if to_version > skill.version_count or to_version < 1:
            raise ValueError(f"Version {to_version} not found")

        # Add a new version with the old content
        old_content = skill.versions[to_version - 1].content
        self.refine(name, old_content, reason=f"rollback to v{to_version}")
        return self._skills[name]

    def record_score(self, name: str, score: float) -> None:
        """Record eval score for the current version of a skill."""
        skill = self._skills.get(name)
        if skill and skill.current_version:
            skill.current_version.eval_score = score
            self._save_skill(skill)

    # ── Queries ───────────────────────────────────────────────────────

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_active(self) -> list[Skill]:
        """Get all active skills."""
        return [s for s in self._skills.values() if s.status == "active"]

    def get_active_content(self) -> dict[str, str]:
        """Get name→content mapping for all active skills (for injection into agent)."""
        return {s.name: s.current_content for s in self.get_active()}

    def get_by_capability(self, capability: str) -> list[Skill]:
        """Get all skills for a capability (any status)."""
        return [s for s in self._skills.values() if s.capability == capability]

    def get_stale(self, max_score: float = 0.5, min_versions: int = 2) -> list[Skill]:
        """Find skills that have been refined multiple times but still score poorly."""
        stale = []
        for s in self.get_active():
            if s.version_count >= min_versions:
                if s.current_version and s.current_version.eval_score is not None:
                    if s.current_version.eval_score <= max_score:
                        stale.append(s)
        return stale

    def should_retire(self, name: str, score_without_skill: float, threshold: float = 0.85) -> bool:
        """Check if a skill can be retired (model performs well without it)."""
        return score_without_skill >= threshold

    # ── Auto-refine ───────────────────────────────────────────────────

    def auto_refine(
        self,
        name: str,
        eval_result: Any,
        failures: list[str],
    ) -> Skill | None:
        """Use LLM to refine a skill based on continued failures."""
        if not self._pool:
            return None

        skill = self._skills.get(name)
        if not skill:
            return None

        prompt = f"""This skill instruction is not working well enough. Refine it.

CURRENT SKILL ({skill.name} v{skill.version_count}):
{skill.current_content}

CONTINUED FAILURES (agent still fails on these):
{chr(10).join(f'  - {f}' for f in failures[:5])}

Requirements:
- Keep the same general purpose but make instructions MORE SPECIFIC
- Add concrete examples of correct behavior
- Add explicit "DO NOT" rules for the failure patterns
- Keep it concise (under 200 words)

Reply with ONLY the refined skill content (no JSON wrapping, just the markdown)."""

        raw = asyncio.run(self._pool.generate(prompt, temperature=0.3, max_tokens=1024))
        refined_content = raw.strip()

        if refined_content and refined_content != skill.current_content:
            return self.refine(name, refined_content, reason=f"auto-refined after failures: {failures[:2]}")
        return skill

    # ── Auto-merge ────────────────────────────────────────────────────

    def find_merge_candidates(self) -> list[tuple[str, str]]:
        """Find skills that overlap and could be merged."""
        active = self.get_active()
        candidates = []
        for i, s1 in enumerate(active):
            for s2 in active[i + 1:]:
                if s1.capability == s2.capability:
                    candidates.append((s1.name, s2.name))
        return candidates

    # ── Persistence ───────────────────────────────────────────────────

    def _save_skill(self, skill: Skill) -> None:
        """Save a skill to disk."""
        # Save metadata
        meta_dir = self._path / ".registry"
        meta_dir.mkdir(exist_ok=True)
        (meta_dir / f"{skill.name}.json").write_text(skill.model_dump_json(indent=2))

        # Save current content as .md (for easy reading)
        if skill.status == "active" and skill.current_content:
            (self._path / f"{skill.name}.md").write_text(skill.current_content)
        elif skill.status in ("retired", "merged"):
            md_path = self._path / f"{skill.name}.md"
            if md_path.exists():
                md_path.rename(self._path / f"{skill.name}.{skill.status}.md")

    def _load(self) -> None:
        """Load all skills from disk."""
        meta_dir = self._path / ".registry"
        if not meta_dir.exists():
            return
        for f in meta_dir.glob("*.json"):
            try:
                skill = Skill.model_validate_json(f.read_text())
                self._skills[skill.name] = skill
            except Exception:
                pass

    def summary(self) -> str:
        """Human-readable summary of all skills."""
        lines = [f"SkillRegistry: {len(self._skills)} skills ({len(self.get_active())} active)"]
        for s in self._skills.values():
            score = f"score={s.current_version.eval_score:.2f}" if s.current_version and s.current_version.eval_score is not None else "unscored"
            lines.append(f"  [{s.status:8}] {s.name} v{s.version_count} ({score}) — {s.capability}")
        return "\n".join(lines)
