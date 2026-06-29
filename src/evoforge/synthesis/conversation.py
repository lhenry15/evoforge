"""Component (2) of scenario-driven synthesis: the **ConversationGenerator**.

Consumes a :class:`~evoforge.synthesis.seed.Seed` (component 1) and produces a
:class:`SimTranscript`. One generator handles **both** shapes, decided by the seed:

* **single-party** — one human + the agent under test (the classic instruction →
  response shape, e.g. a corrective example for a failure seed); and
* **multi-party** — several human speakers plus (optionally) the agent.

The generator **does not label** anything — labeling is component (3). This keeps
the honest-label separation: the generator proposes a conversation, an independent
certified labeler judges it.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from evoforge.core.types import EvalCase, Message, ScoringMethod
from evoforge.llm.structured import coerce_records, generate_structured
from evoforge.synthesis.seed import Seed, SimParticipant


class SimTurn(BaseModel):
    """A single turn in a generated conversation, optionally labeled (by component 3)."""

    party_id: str                                           # speaker id
    role: str                                               # "user" | "assistant" | "tool" | "system"
    content: str
    is_agent: bool = False
    labels: dict[str, Any] = Field(default_factory=dict)    # per-turn labels (presence, verdict, …)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimTranscript(BaseModel):
    """The conversation a generator produces — the unit exchanged with EvoForge.

    Convertible to an :class:`~evoforge.core.types.EvalCase` for scoring or mined
    for training data. Per-turn and conversation-level ``labels`` are attached by
    the labeler set (component 3), never by the generator.
    """

    scenario_id: str                                       # id of the seed/scenario this came from
    capability: str
    turns: list[SimTurn] = Field(default_factory=list)
    labels: dict[str, Any] = Field(default_factory=dict)    # conversation-level labels / verdict
    expected: str = ""                                      # reference answer / oracle target
    scoring_method: ScoringMethod = ScoringMethod.LLM_JUDGE
    simulator: str = ""                                     # name of the producing generator
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_messages(self) -> list[Message]:
        """Render turns as EvoForge :class:`~evoforge.core.types.Message` objects."""
        messages = [
            Message(
                role=turn.role,
                content=turn.content,
                metadata={"party_id": turn.party_id, "is_agent": turn.is_agent, **turn.metadata},
            )
            for turn in self.turns
        ]
        return messages

    def agent_response(self) -> str:
        """The 'response under test': the last agent/assistant turn (or last turn)."""
        for turn in reversed(self.turns):
            if turn.is_agent or turn.role == "assistant":
                return turn.content
        response = self.turns[-1].content if self.turns else ""
        return response

    def to_eval_case(self) -> EvalCase:
        """Convert this transcript into an EvoForge eval case (labels kept in metadata)."""
        case = EvalCase(
            id=self.scenario_id,
            messages=self.to_messages(),
            expected=self.expected,
            capability=self.capability,
            scoring_method=self.scoring_method,
            metadata={
                "simulated": True,
                "simulator": self.simulator,
                "labels": self.labels,
                **self.metadata,
            },
        )
        return case


_CONV_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["speaker", "content"],
            },
        }
    },
    "required": ["turns"],
}

_CONV_SYSTEM = """You write realistic conversations to train and evaluate an AI
agent. Stay in character for every speaker, keep it natural, and honor the scenario
and conditions exactly. Output only the dialogue turns."""

_TRANSCRIPT_SYSTEM = """You write realistic conversations to train and evaluate an AI
agent. Stay in character for every speaker, keep it natural, and honor the scenario
and conditions exactly.

Write the conversation as a plain transcript: each turn on its own block, starting
with the speaker id then a colon, e.g.

    alex: <message>
    agent: <message, which MAY include multi-line markdown or fenced code>

Use ONLY the listed speaker ids. Do not number the turns or add commentary."""


def _parse_transcript(raw: str, seed: Seed) -> list[dict[str, Any]]:
    """Parse a speaker-labeled transcript into ``{speaker, content}`` records.

    Robust to models that emit natural Markdown instead of JSON: detects lines that
    begin with a known speaker id (optionally bold/bracketed) followed by ``:`` or
    ``-``, and treats everything until the next such marker as that turn's content
    (so fenced code / multi-line markdown is preserved verbatim).
    """
    speaker_ids = [participant.id for participant in seed.participants] or ["user", "agent", "assistant"]
    marker = re.compile(
        r"^\s*[*_>#\-\[]*\s*(" + "|".join(re.escape(s) for s in speaker_ids) + r")\b[*_\]]*\s*[:\-]\s*(.*)$",
        re.IGNORECASE,
    )
    records: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for line in raw.splitlines():
        match = marker.match(line)
        if match:
            if current is not None:
                records.append(current)
            current = {"speaker": match.group(1), "content": match.group(2)}
        elif current is not None:
            current["content"] = f"{current['content']}\n{line}"
    if current is not None:
        records.append(current)
    for record in records:
        record["content"] = str(record["content"]).strip()
    return [record for record in records if record["content"]]


class ConversationGenerator:
    """Component (2): turn a :class:`Seed` into a :class:`SimTranscript`."""

    def __init__(self, pool: Any, name: str = "native") -> None:
        self._pool = pool
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, seed: Seed, agent: Optional[Any] = None) -> SimTranscript:
        """Generate one conversation for ``seed`` (single- or multi-party by the seed).

        ``agent`` is accepted for interface symmetry with the simulator seam; the
        native generator simulates every party itself. Generation prefers a
        schema-constrained JSON pass, but falls back to parsing a speaker-labeled
        transcript when the model returns natural Markdown instead of JSON (common
        for rich content like fenced files/code), so rich scenarios still yield turns.
        """
        prompt = self._build_prompt(seed)
        temperature = 0.7 + 0.2 * min(1.0, seed.complexity)
        # Budget scales with the requested length so multi-turn, content-heavy
        # conversations (files/code per turn) are not truncated mid-output.
        budget = max(2048, min(6000, seed.max_turns * 400))

        parsed = generate_structured(
            self._pool, prompt, _CONV_SCHEMA, system=_CONV_SYSTEM,
            temperature=temperature, max_tokens=budget,
        )
        records = coerce_records(parsed, key="turns")
        if not records:
            # Fallback: ask for a plain transcript and parse speaker-labeled blocks.
            raw = self._pool.generate(
                prompt, system=_TRANSCRIPT_SYSTEM, temperature=temperature, max_tokens=budget,
            )
            records = _parse_transcript(str(raw), seed)

        turns = self._records_to_turns(seed, records)
        transcript = SimTranscript(
            scenario_id=seed.id,
            capability=seed.capability,
            turns=turns,
            expected=seed.expected,
            scoring_method=seed.scoring_method,
            simulator=self._name,
            metadata={
                "multiparty": seed.is_multiparty(),
                "failure_mode": seed.failure_mode.value if seed.failure_mode else None,
                "complexity": seed.complexity,
                **seed.metadata,
            },
        )
        return transcript

    # ── prompt construction ───────────────────────────────────────────────────
    def _build_prompt(self, seed: Seed) -> str:
        roster = self._roster(seed)
        lines = [
            f"Scenario: {seed.scenario or seed.goal or seed.capability}",
            f"Capability under test: {seed.capability}",
            f"Difficulty — {seed.complexity_band()}.",
            f"Speakers: {roster}.",
            f"Produce between {seed.min_turns} and {seed.max_turns} turns.",
        ]
        if seed.user_context:
            lines.append(f"User context: {seed.user_context}")
        if seed.agent_context:
            lines.append(f"Agent under test: {seed.agent_context}")
        if seed.conditions:
            lines.append("Conditions to honor: " + "; ".join(seed.conditions))
        if seed.controls:
            knobs = ", ".join(f"{k}={v}" for k, v in seed.controls.items())
            lines.append(f"Control knobs: {knobs}")
        if seed.is_failure_seed():
            lines.append(
                f"This situation should naturally tempt the failure mode "
                f"'{seed.failure_mode.value if seed.failure_mode else 'unknown'}'. Build that "
                "temptation into the human turns, then have the agent speaker respond CORRECTLY "
                "(the ideal response that avoids the failure) as the final turn."
            )
            if seed.failure_examples:
                shown = "; ".join(
                    (ex.trigger or "").strip()[:160] for ex in seed.failure_examples[:3] if ex.trigger
                )
                if shown:
                    lines.append(f"Real situations that previously triggered the failure: {shown}")
        elif seed.is_multiparty():
            lines.append("Make it a genuine multi-party discussion; speakers should react to each other.")
        return "\n".join(lines)

    @staticmethod
    def _roster(seed: Seed) -> str:
        parts = seed.participants or [
            SimParticipant(id="user", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ]
        described = []
        for participant in parts:
            tag = "the agent under test" if participant.is_agent else "a human"
            persona = f" — {participant.persona}" if participant.persona else ""
            described.append(f"{participant.id} ({tag}{persona})")
        return ", ".join(described)

    # ── turn assembly ─────────────────────────────────────────────────────────
    @staticmethod
    def _records_to_turns(seed: Seed, records: list[dict[str, Any]]) -> list[SimTurn]:
        by_id = {p.id.lower(): p for p in seed.participants}
        agent_ids = {p.id.lower() for p in seed.participants if p.is_agent}
        turns: list[SimTurn] = []
        for index, record in enumerate(records):
            speaker = str(record.get("speaker", "")).strip() or "participant"
            content = str(record.get("content", "")).strip()
            if not content:
                continue
            matched = by_id.get(speaker.lower())
            is_agent = bool(matched and matched.is_agent) or speaker.lower() in agent_ids
            turns.append(
                SimTurn(
                    party_id=matched.id if matched else speaker,
                    role="assistant" if is_agent else "user",
                    content=content,
                    is_agent=is_agent,
                    metadata={"turn_index": index},
                )
            )
        return turns
