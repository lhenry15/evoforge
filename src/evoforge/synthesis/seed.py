"""Component (1) of scenario-driven synthesis: the **Seed** and its generator.

A :class:`Seed` is the single control surface that decides *what* conversation
gets generated. It is what a human author **or** the evolution loop manipulates to
steer synthesis. It bundles the dials the other two components consume:

* **Scenario** — the situation/gist to dramatize.
* **FailureMode** (optional) — set it and the seed becomes a *failure seed*: the
  situation is shaped to provoke a known failure so a corrective example can be
  mined from how the agent handles it. Real failing ``(trigger, response)`` pairs
  from traces ride along as :attr:`Seed.failure_examples`.
* **Complexity** — a 0..1 difficulty dial (distractors, ambiguity, pressure).
* **Conditions** — explicit constraints/triggers that harden the conversation.
* **User / Agent context** — who the user(s) are and the agent under test.
* **Participants** — the speakers. One human ⇒ single-party; many ⇒ multi-party.

The seed is deliberately data — no LLM, no polymorphism — so it serializes as plain
JSON and can be authored by hand, by :class:`SeedGenerator`, or by the evolution
loop. Component (2) (``ConversationGenerator``) turns a seed into a transcript;
component (3) (the labeler set) annotates that transcript.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import ScoringMethod
from evoforge.llm.structured import coerce_records, generate_structured
from evoforge.mining.schema import FailureExample, FailureModeCluster
from evoforge.trace.schema import FailureMode


class SimParticipant(BaseModel):
    """One party in a (possibly multi-party) conversation the seed will generate."""

    id: str
    role: str = "user"                  # "user" | "assistant" | "agent" | custom party label
    persona: Optional[str] = None       # free-text persona / behaviour brief
    is_agent: bool = False              # True for the party under test
    metadata: dict[str, Any] = Field(default_factory=dict)


def _complexity_band(complexity: float) -> str:
    """Map a 0..1 complexity dial to a short instruction band for prompts."""
    if complexity >= 0.8:
        return "very hard: heavy distractions, ambiguity, conflicting goals, and pressure"
    if complexity >= 0.6:
        return "hard: notable distractions, some ambiguity, and competing considerations"
    if complexity >= 0.4:
        return "moderate: a few distractions and mild ambiguity"
    if complexity >= 0.2:
        return "easy: mostly straightforward with light noise"
    return "trivial: clean and direct"


class Seed(BaseModel):
    """The controllable seed for one synthesized conversation (component 1).

    A *plain* seed (``failure_mode is None``) yields an ordinary conversation. A
    *failure* seed (``failure_mode`` set) yields a conversation shaped to provoke
    that failure, from which a corrective example is labeled — the richer the
    ``complexity``/``conditions``/contexts, the more informative and challenging
    the resulting corrective example.
    """

    id: str
    capability: str = "general"                              # capability bucket for aggregation
    goal: str = ""                                           # what a correct agent must achieve

    # ── the scenario / situation ──────────────────────────────────────────────
    scenario: str = ""                                       # the gist/situation to dramatize
    conditions: list[str] = Field(default_factory=list)      # constraints/triggers that harden it
    complexity: float = 0.5                                  # 0..1 difficulty dial

    # ── the failure axis (optional) — makes this a corrective/failure seed ─────
    failure_mode: Optional[FailureMode] = None
    failure_examples: list[FailureExample] = Field(default_factory=list)

    # ── who is in the room ────────────────────────────────────────────────────
    user_context: str = ""                                   # the user side / situation
    agent_context: str = ""                                  # the agent under test (role/policies)
    participants: list[SimParticipant] = Field(default_factory=list)

    # ── length + scoring + open knobs ─────────────────────────────────────────
    min_turns: int = 4
    max_turns: int = 8
    expected: str = ""                                       # reference answer / oracle target
    scoring_method: ScoringMethod = ScoringMethod.LLM_JUDGE
    controls: dict[str, Any] = Field(default_factory=dict)   # off_topic_ratio, conflicting_parties, …
    metadata: dict[str, Any] = Field(default_factory=dict)   # opaque domain payload

    def human_participants(self) -> list[SimParticipant]:
        """Participants that are not the agent under test."""
        humans = [p for p in self.participants if not p.is_agent]
        return humans

    def is_multiparty(self) -> bool:
        """True when more than one human party speaks (⇒ multi-party generation)."""
        multiparty = len(self.human_participants()) > 1
        return multiparty

    def is_failure_seed(self) -> bool:
        """True when this seed targets a specific failure mode (corrective example)."""
        is_failure = self.failure_mode is not None
        return is_failure

    def complexity_band(self) -> str:
        """Human-readable difficulty band for prompting."""
        band = _complexity_band(self.complexity)
        return band

    @classmethod
    def from_failure_cluster(
        cls,
        cluster: FailureModeCluster,
        complexity: float = 0.6,
        conditions: Optional[list[str]] = None,
        participants: Optional[list[SimParticipant]] = None,
        max_examples: int = 3,
    ) -> "Seed":
        """Build a *failure seed* from a mined failure cluster (pure, no LLM).

        The cluster's symptom becomes the scenario, its real failing examples ride
        along for grounded (non-invented) contrast, and ``suggested_fix_type``
        seeds an initial hardening condition. Callers may raise ``complexity`` or
        add ``conditions`` to make the corrective example more challenging.
        """
        examples = list(cluster.examples or [])[:max_examples]
        derived_conditions = list(conditions or [])
        if cluster.suggested_fix_type and cluster.suggested_fix_type != "investigate":
            derived_conditions.append(f"exercise the '{cluster.suggested_fix_type}' fix path")
        default_parties = participants or [
            SimParticipant(id="user", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ]
        seed = cls(
            id=f"seed-{cluster.cluster_id}",
            capability=cluster.capability or "general",
            goal=f"handle the situation without the failure: {cluster.label or cluster.symptom_summary}",
            scenario=cluster.symptom_summary or cluster.label or "a situation that previously failed",
            conditions=derived_conditions,
            complexity=complexity,
            failure_mode=getattr(cluster, "mode", None),
            failure_examples=examples,
            agent_context=f"An agent for: {cluster.capability or 'general tasks'}.",
            participants=default_parties,
            scoring_method=ScoringMethod.LLM_JUDGE,
            metadata={"cluster_id": cluster.cluster_id, "from_failure_cluster": True},
        )
        return seed


_SEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seeds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "user_context": {"type": "string"},
                    "agent_context": {"type": "string"},
                    "goal": {"type": "string"},
                },
                "required": ["scenario"],
            },
        }
    },
    "required": ["seeds"],
}

_SEED_SYSTEM = """You design SEEDS for synthetic conversations used to train and
evaluate an AI agent. A seed is a realistic situation plus the conditions that make
it informative. Make seeds diverse and grounded; do NOT write the conversation
itself — only the situation, conditions, and who is involved."""


class SeedGenerator:
    """Component (1): author :class:`Seed` objects from a short brief (uses an LLM).

    This is the controllable front door of synthesis. A human or the evolution loop
    calls :meth:`generate` (free-form situations) or :meth:`from_failures`
    (corrective/failure seeds grounded in mined clusters), tuning ``complexity``,
    ``conditions``, participants, and contexts to steer everything downstream.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def generate(
        self,
        brief: str,
        capability: str = "general",
        n: int = 3,
        complexity: float = 0.5,
        multiparty: bool = False,
        participants: Optional[list[SimParticipant]] = None,
        conditions: Optional[list[str]] = None,
        controls: Optional[dict[str, Any]] = None,
        min_turns: int = 4,
        max_turns: int = 8,
        temperature: float = 0.8,
    ) -> list[Seed]:
        """Author ``n`` seeds from a natural-language ``brief``.

        ``participants`` (or ``multiparty``) decides single- vs multi-party
        generation downstream; everything else (complexity, conditions, contexts)
        becomes the dials component (2) reads.
        """
        parties = participants or self._default_participants(multiparty)
        prompt = (
            f"Brief: {brief}\n"
            f"Capability: {capability}\n"
            f"Number of seeds: {n}\n"
            f"Target difficulty — {_complexity_band(complexity)}.\n"
            f"Participants: {', '.join(p.id + ('/agent' if p.is_agent else '') for p in parties)}.\n"
            "For each seed give: a concrete scenario, a few hardening conditions, the "
            "user_context, the agent_context, and the goal a correct agent must achieve."
        )
        parsed = generate_structured(
            self._pool, prompt, _SEED_SCHEMA, system=_SEED_SYSTEM,
            temperature=temperature, max_tokens=1024,
        )
        records = coerce_records(parsed, key="seeds")
        seeds: list[Seed] = []
        for index, record in enumerate(records[:n]):
            merged_conditions = list(conditions or []) + [
                str(c) for c in (record.get("conditions") or []) if str(c).strip()
            ]
            seed = Seed(
                id=f"seed-{capability}-{index}",
                capability=capability,
                goal=str(record.get("goal", "")) or brief,
                scenario=str(record.get("scenario", "")).strip(),
                conditions=merged_conditions,
                complexity=complexity,
                user_context=str(record.get("user_context", "")).strip(),
                agent_context=str(record.get("agent_context", "")).strip(),
                participants=list(parties),
                min_turns=min_turns,
                max_turns=max_turns,
                controls=dict(controls or {}),
            )
            seeds.append(seed)
        return seeds

    def from_failures(
        self,
        cluster: FailureModeCluster,
        n: int = 1,
        complexity: float = 0.6,
        conditions: Optional[list[str]] = None,
        participants: Optional[list[SimParticipant]] = None,
    ) -> list[Seed]:
        """Build ``n`` failure seeds from one cluster (pure; deterministic).

        ``n`` lets a caller request several seeds at increasing complexity for the
        same failure — the base seed is grounded in the cluster's real examples.
        """
        seeds: list[Seed] = []
        for index in range(max(1, n)):
            # Spread complexity upward across requested copies for varied hardness.
            spread = min(1.0, complexity + 0.1 * index)
            seed = Seed.from_failure_cluster(
                cluster, complexity=spread, conditions=conditions, participants=participants
            )
            seed.id = f"{seed.id}-{index}"
            seeds.append(seed)
        return seeds

    @staticmethod
    def _default_participants(multiparty: bool) -> list[SimParticipant]:
        if multiparty:
            return [
                SimParticipant(id="alex", role="user"),
                SimParticipant(id="riley", role="user"),
                SimParticipant(id="agent", role="assistant", is_agent=True),
            ]
        return [
            SimParticipant(id="user", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ]
