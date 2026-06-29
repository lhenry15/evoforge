"""Component (3) of scenario-driven synthesis: the **open labeler set**.

Honestly labels conversations produced by component (2) — single- or multi-party.
Three properties make a label *honest* (none of which need a polymorphic type
registry):

1. **Generation ≠ labeling** — labels come from this independent pass, never the
   generator.
2. **Per-turn / per-prefix ground truth** — a labeler judges each prefix's latest
   relevant turn.
3. **Self-certified** — :func:`certify_labeler` runs a *separation battery*
   (constructed positives vs negatives); a labeler that cannot separate them is
   rejected, not trusted.

The set is **open and expandable** two ways, both gated by certification:

* **schema-as-data** — author a :class:`LabelSchema` (name, question, fields,
  guards) and :class:`SchemaLabeler` compiles it into a labeler. New label = new
  JSON, no code, no new class.
* **code** — implement the :class:`Labeler` ``Protocol`` and ``register`` it.

A label is a plain :class:`Label` (an explicit ``label_name`` string, ordinary
JSON) — no discriminator field, no import-time registry coupling to rehydrate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from evoforge.llm.structured import generate_structured
from evoforge.synthesis.conversation import SimTranscript, SimTurn
from evoforge.synthesis.seed import Seed


class Label(BaseModel):
    """An honest, portable per-turn label (plain JSON; no polymorphic ``_type``)."""

    label_name: str
    present: bool = False                                   # mechanical verdict (gated by guards)
    value: Any = None                                       # primary extracted value, if any
    verbatim_only: bool = False                             # only a copy-pasted marker ⇒ absent
    votes: str = ""                                         # majority tally, e.g. "3/3"
    basis: str = ""                                         # one-line judge justification
    fields: dict[str, Any] = Field(default_factory=dict)    # schema-authored fields: key -> {value, evidence}
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabelField(BaseModel):
    """One field a :class:`LabelSchema` asks the judge to extract."""

    name: str
    kind: str = "bool"                                      # "bool" | "enum" | "text"
    description: str = ""
    options: list[str] = Field(default_factory=list)        # for kind == "enum"


class LabelSchema(BaseModel):
    """A labeler authored as DATA — compiled by :class:`SchemaLabeler`."""

    name: str
    question: str                                           # the judgment question (the rubric)
    fields: list[LabelField] = Field(default_factory=list)
    present_field: str = ""                                 # bool field that gates presence ("" => "present")
    verbatim_guard: bool = True                             # a verbatim-only trace ⇒ absent
    judge_party: str = "human"                              # "human" | "agent" | "any": whose turn to judge
    target_key: str = "goal"                                # which Seed attribute is the target to look for


@runtime_checkable
class Labeler(Protocol):
    """Open interface for any labeler (schema-compiled or hand-written)."""

    @property
    def name(self) -> str:
        ...

    def label(self, prefix: list[SimTurn], seed: Seed) -> Optional[Label]:
        """Judge the latest relevant turn of ``prefix``; ``None`` if not applicable."""
        ...


def _seed_target(seed: Seed, target_key: str) -> str:
    """Resolve the 'thing to look for' from the seed (goal/expected/metadata)."""
    direct = getattr(seed, target_key, None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    meta_value = seed.metadata.get(target_key)
    if isinstance(meta_value, str) and meta_value.strip():
        return meta_value.strip()
    return (seed.goal or seed.scenario or seed.expected or "").strip()


class SchemaLabeler:
    """Compile a :class:`LabelSchema` into a voting, guarded :class:`Labeler`."""

    def __init__(self, schema: LabelSchema, pool: Any, votes: int = 3) -> None:
        self._schema = schema
        self._pool = pool
        self._votes = max(1, int(votes))

    @property
    def name(self) -> str:
        return self._schema.name

    # ── per-prefix delta: judge only the newest relevant turn ──────────────────
    def label(self, prefix: list[SimTurn], seed: Seed) -> Optional[Label]:
        if not prefix:
            return None
        newest = prefix[-1]
        if self._schema.judge_party == "human" and newest.is_agent:
            return None
        if self._schema.judge_party == "agent" and not newest.is_agent:
            return None
        content = (newest.content or "").strip()
        if not content:
            return None
        target = _seed_target(seed, self._schema.target_key)
        context = "\n".join(
            f"{t.party_id}: {t.content}" for t in prefix[:-1] if (t.content or "").strip()
        )
        results = [self._judge_once(target, context, newest) for _ in range(self._votes)]
        present, verbatim_only, fields, basis = self._aggregate(results)
        gated_present = bool(present and not (self._schema.verbatim_guard and verbatim_only))
        n_present = sum(1 for r in results if r.get("present") == present)
        label = Label(
            label_name=self._schema.name,
            present=gated_present,
            value=fields.get(self._present_field(), {}).get("value") if fields else present,
            verbatim_only=verbatim_only,
            votes=f"{n_present}/{self._votes}",
            basis=basis,
            fields=fields,
        )
        return label

    # ── internals ─────────────────────────────────────────────────────────────
    def _present_field(self) -> str:
        if self._schema.present_field:
            return self._schema.present_field
        for field in self._schema.fields:
            if field.kind == "bool":
                return field.name
        return ""

    def _judge_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for field in self._schema.fields:
            if field.kind == "bool":
                props[field.name] = {"type": "boolean"}
            elif field.kind == "enum":
                props[field.name] = {"type": "string", "enum": field.options}
            else:
                props[field.name] = {"type": "string"}
        props["present"] = {"type": "boolean"}
        props["verbatim_only"] = {"type": "boolean"}
        props["reason"] = {"type": "string"}
        return {"type": "object", "properties": props, "required": ["present"]}

    def _judge_once(self, target: str, context: str, turn: SimTurn) -> dict[str, Any]:
        system = (
            f"You are a strict, literal labeler. {self._schema.question}\n"
            "Judge ONLY the message under review, using prior context for reference. "
            "A copy-pasted marker with no genuine surrounding statement is NOT genuine "
            "expression (set verbatim_only=true). Answer as JSON for the requested keys."
        )
        user = (
            f"Target to look for:\n  {target}\n\n"
            f"Prior context:\n{context or '(none)'}\n\n"
            f"Message under review ({turn.party_id}):\n  {turn.content}"
        )
        try:
            parsed = generate_structured(
                self._pool, user, self._judge_schema(), system=system,
                temperature=0.0, max_tokens=500,
            )
            if isinstance(parsed, dict):
                return parsed
            return {"present": False, "reason": "unparseable"}
        except Exception as exc:  # conservative ABSENT on any failure
            return {"present": False, "verbatim_only": False, "reason": f"judge_error: {exc}"}

    def _aggregate(
        self, results: list[dict[str, Any]]
    ) -> tuple[bool, bool, dict[str, Any], str]:
        present = self._majority([bool(r.get("present", False)) for r in results], default=False)
        verbatim_only = self._majority([bool(r.get("verbatim_only", False)) for r in results], default=False)
        fields: dict[str, Any] = {}
        for field in self._schema.fields:
            values = [r.get(field.name) for r in results if field.name in r]
            if not values:
                continue
            try:
                chosen = Counter(values).most_common(1)[0][0]
            except TypeError:
                chosen = values[0]
            evidence = next(
                (str(r.get("reason", "")) for r in results if r.get(field.name) == chosen and r.get("reason")),
                "",
            )
            fields[field.name] = {"value": chosen, "evidence": evidence}
        basis = next((str(r.get("reason", "")) for r in results if r.get("reason")), "")
        return present, verbatim_only, fields, basis

    @staticmethod
    def _majority(values: list[bool], default: bool) -> bool:
        if not values:
            return default
        decision = Counter(values).most_common(1)[0][0]
        return bool(decision)


# ── open registry ─────────────────────────────────────────────────────────────
LabelerFactory = Callable[..., Labeler]


class LabelerRegistry:
    """A plain-dict registry of labeler factories — open, no import-time coupling."""

    def __init__(self) -> None:
        self._factories: dict[str, LabelerFactory] = {}

    def register(self, name: str, factory: Optional[LabelerFactory] = None) -> Any:
        """Register a factory. Usable directly or as a decorator."""
        if factory is not None:
            self._factories[name] = factory
            return factory

        def _decorator(func: LabelerFactory) -> LabelerFactory:
            self._factories[name] = func
            return func

        return _decorator

    def register_schema(self, schema: LabelSchema) -> None:
        """Register a schema-as-data labeler (compiled lazily with the caller's pool)."""
        self._factories[schema.name] = lambda pool, votes=3: SchemaLabeler(schema, pool, votes=votes)

    def create(self, name: str, pool: Any, **kwargs: Any) -> Labeler:
        if name not in self._factories:
            raise KeyError(f"Unknown labeler {name!r}. Known: {self.available()}")
        return self._factories[name](pool, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)


REGISTRY = LabelerRegistry()


# A feature-neutral default: "did a human naturally express the target?" Authored as
# DATA so other features just register their own schema instead of subclassing.
PRESENCE_SCHEMA = LabelSchema(
    name="target_presence",
    question="Decide whether the message genuinely expresses the target (not a verbatim marker dump).",
    fields=[
        LabelField(name="expresses_target", kind="bool", description="message genuinely expresses the target"),
    ],
    present_field="expresses_target",
    judge_party="human",
    target_key="goal",
)

# The corrective-example counterpart: "did the AGENT avoid the seed's failure mode?"
AVOIDS_FAILURE_SCHEMA = LabelSchema(
    name="avoids_failure",
    question="Decide whether the agent's response correctly handles the situation and AVOIDS the failure mode.",
    fields=[
        LabelField(name="avoids_failure", kind="bool", description="agent response avoids the failure mode"),
    ],
    present_field="avoids_failure",
    verbatim_guard=False,
    judge_party="agent",
    target_key="goal",
)

REGISTRY.register_schema(PRESENCE_SCHEMA)
REGISTRY.register_schema(AVOIDS_FAILURE_SCHEMA)


# ── self-certification: the separation battery ─────────────────────────────────
class CertProbe(BaseModel):
    """One constructed probe for certification: a prefix + the expected verdict."""

    name: str
    prefix: list[SimTurn]
    expect_present: bool


class CertReport(BaseModel):
    """Outcome of running a labeler through its separation battery."""

    labeler: str
    passed: bool
    n_correct: int
    n_total: int
    failures: list[str] = Field(default_factory=list)


def certify_labeler(labeler: Labeler, seed: Seed, probes: list[CertProbe]) -> CertReport:
    """Run a labeler over constructed positive/negative probes and require separation.

    A labeler that cannot label every positive PRESENT and every negative ABSENT is
    rejected (``passed=False``) — the gate that keeps an open labeler set honest.
    """
    correct = 0
    failures: list[str] = []
    for probe in probes:
        label = labeler.label(list(probe.prefix), seed)
        got = bool(label and label.present)
        if got == probe.expect_present:
            correct += 1
        else:
            failures.append(f"{probe.name}: expected present={probe.expect_present}, got {got}")
    report = CertReport(
        labeler=labeler.name,
        passed=not failures,
        n_correct=correct,
        n_total=len(probes),
        failures=failures,
    )
    return report


# ── applying labelers to a whole transcript (prefix sweep) ─────────────────────
def label_transcript(transcript: SimTranscript, labelers: list[Labeler], seed: Seed) -> SimTranscript:
    """Attach labels to every turn by sweeping prefixes (honest per-turn ground truth)."""
    for index in range(len(transcript.turns)):
        prefix = transcript.turns[: index + 1]
        for labeler in labelers:
            label = labeler.label(prefix, seed)
            if label is not None:
                transcript.turns[index].labels[label.label_name] = label.model_dump(mode="json")
    return transcript
