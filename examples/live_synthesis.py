"""Live demo of EvoForge scenario-driven synthesis — two richer, realistic cases.

A THIN wiring script over the synthesis pipeline. It shows that producing your own
labeled dataset is just: (1) describe the **seed** (the situation + who's in it),
and (2) author the **labelers** (what to measure). Everything else — generating the
conversation, voting/guarding the labels, writing JSONL — is the reusable
:class:`~evoforge.synthesis.ScenarioSynthesizer` / :class:`~evoforge.synthesis.LabeledDataset`
machinery, which the evolution loop drives the same way.

Two cases:

  A. **Group channel.md** (multi-party) — the assistant maintains a shared
     ``channel.md`` (## Preferences / ## Todo); members state standing channel
     preferences and tasks; the assistant must RECORD them into channel.md and
     later APPLY the recorded preferences.

  B. **Skill fail -> fix -> works** (single user) — the assistant's first attempt
     FAILS for lack of a working skill, it then AUTHORS/FIXES the skill, and the
     retry SUCCEEDS.

Run (with a venv that has evoforge + requests installed)::

    $env:PROXY_BASE_URL="http://127.0.0.1:8787"; $env:PROXY_MODEL="gpt-5.2-chat"
    python examples/live_synthesis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

from evoforge.llm.openai_compat import OpenAICompatPool
from evoforge.mining.schema import FailureExample
from evoforge.synthesis import (
    REGISTRY,
    CertProbe,
    LabeledDataset,
    LabelField,
    LabelSchema,
    ScenarioSynthesizer,
    SchemaLabeler,
    Seed,
    SimParticipant,
    SimTurn,
    certify_labeler,
)
from evoforge.trace.schema import FailureMode

_DATASETS_DIR = Path(__file__).parent / "datasets"


# ── labelers — authored as DATA and registered (the open labeler set) ──────────
# Case A: a human STATES a standing channel item; the agent RECORDS it into
# channel.md and later APPLIES recorded preferences.
STATES_CHANNEL_ITEM = LabelSchema(
    name="states_channel_item",
    question=(
        "Decide whether THIS human message states a STANDING channel preference (a durable "
        "'always / from now on' formatting or workflow rule) OR adds a concrete TODO/task to "
        "track. A one-off question, an acknowledgement, or small talk is neither."
    ),
    fields=[
        LabelField(name="states_channel_item", kind="bool",
                   description="states a standing preference or a todo to record"),
        LabelField(name="item_kind", kind="enum", options=["preference", "todo", "none"],
                   description="which kind of channel item, if any"),
    ],
    present_field="states_channel_item",
    judge_party="human",
    target_key="scenario",
)
UPDATES_CHANNEL_MD = LabelSchema(
    name="updates_channel_md",
    question=(
        "Decide whether the assistant's response UPDATES the shared channel.md to RECORD a "
        "preference or todo — i.e. it shows the updated channel.md (a '## Preferences' or "
        "'## Todo' entry) reflecting the newly stated item."
    ),
    fields=[LabelField(name="updates_channel_md", kind="bool",
                       description="response records the item into channel.md")],
    present_field="updates_channel_md",
    verbatim_guard=False,
    judge_party="agent",
    target_key="scenario",
)
APPLIES_PREFERENCE = LabelSchema(
    name="applies_preference",
    question=(
        "Decide whether the assistant's response APPLIES a previously-recorded channel "
        "preference to its actual output (e.g. it formats the summary/list the way a recorded "
        "preference requires). If no preference was recorded earlier, answer false."
    ),
    fields=[LabelField(name="applies_preference", kind="bool",
                       description="response applies a recorded preference")],
    present_field="applies_preference",
    verbatim_guard=False,
    judge_party="agent",
    target_key="scenario",
)
# Case B: the agent CREATES/FIXES a skill, and later the skill WORKS.
CREATES_OR_UPDATES_SKILL = LabelSchema(
    name="creates_or_updates_skill",
    question=(
        "Decide whether the assistant's response CREATES a new skill or FIXES/IMPROVES an "
        "existing one to close a capability gap — i.e. it authors or edits a skill module "
        "(code / SKILL.md / a reusable function), rather than just doing the one-off task by hand."
    ),
    fields=[LabelField(name="creates_or_updates_skill", kind="bool",
                       description="response authors or fixes a reusable skill")],
    present_field="creates_or_updates_skill",
    verbatim_guard=False,
    judge_party="agent",
    target_key="scenario",
)
SKILL_WORKS = LabelSchema(
    name="skill_works",
    question=(
        "Decide whether the assistant's response DEMONSTRATES the skill now WORKING — it "
        "produces the correct, well-formed result for the requested task (after the fix). A "
        "failing attempt, an error, or malformed output is NOT working."
    ),
    fields=[LabelField(name="skill_works", kind="bool",
                       description="response shows the skill producing a correct result")],
    present_field="skill_works",
    verbatim_guard=False,
    judge_party="agent",
    target_key="scenario",
)
for _schema in (STATES_CHANNEL_ITEM, UPDATES_CHANNEL_MD, APPLIES_PREFERENCE,
                CREATES_OR_UPDATES_SKILL, SKILL_WORKS):
    REGISTRY.register_schema(_schema)


# ── seeds — the control surface (hand-authored for precise, interesting cases) ─
def _channel_md_seed() -> Seed:
    return Seed(
        id="channel-md-group-0",
        capability="channel_preferences",
        scenario=(
            "A product team works in a group chat where the assistant maintains a shared "
            "channel.md with two sections: '## Preferences' (standing formatting/workflow "
            "rules) and '## Todo' (a running checkbox task list). Members drop preferences "
            "and tasks across the conversation."
        ),
        goal=("Record each stated preference/todo into channel.md, and later apply the "
              "recorded preferences when producing output."),
        conditions=[
            "When a member states a standing preference, update channel.md's Preferences "
            "section and post the updated file fenced as markdown.",
            "When a member adds a task, append it to channel.md's Todo section as an "
            "unchecked '- [ ]' item.",
            "When later asked to produce a summary or list, apply the recorded formatting "
            "preferences.",
        ],
        agent_context=(
            "You maintain channel.md (sections '## Preferences' and '## Todo'). On each "
            "relevant turn you post the UPDATED channel.md fenced as markdown plus a one-line "
            "confirmation."
        ),
        user_context="Alex (team lead) and Riley (engineer) coordinating in a group channel.",
        participants=[
            SimParticipant(id="alex", role="user", persona="team lead"),
            SimParticipant(id="riley", role="user", persona="engineer"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ],
        complexity=0.6,
        min_turns=6,
        max_turns=10,
    )


def _skill_fix_seed() -> Seed:
    return Seed(
        id="skill-fix-csv-0",
        capability="self_editing_skills",
        scenario=(
            "A single user asks the assistant to export a small report to CSV. The assistant "
            "initially has no working csv-export skill, so its first attempt FAILS (malformed "
            "output). It then AUTHORS/FIXES a reusable csv_export skill and RETRIES."
        ),
        goal="fix the missing/broken skill, then produce correct CSV output for the user.",
        failure_mode=FailureMode.MISSING_KNOWLEDGE,
        failure_examples=[FailureExample(
            trace_id="t1", trigger="export the report to CSV",
            response="(no csv_export skill; produced a malformed, comma-mangled blob)",
            signature_id="s1")],
        conditions=[
            "Turn 1 (user): asks to export the small report to CSV.",
            "Turn 2 (agent): ATTEMPTS with the current capability and it FAILS — show the "
            "malformed output and name the gap.",
            "Turn 3 (agent): AUTHORS or FIXES a reusable csv_export skill — show the skill "
            "module (a SKILL.md + a small function).",
            "Turn 4 (agent): RETRIES using the skill and produces CORRECT, well-formed CSV.",
        ],
        agent_context=(
            "You can author and edit your own skills (skill modules with code). When a skill "
            "is missing or broken you create or fix it, then use it."
        ),
        user_context="A single user who needs a CSV export of a short report.",
        participants=[
            SimParticipant(id="user", role="user"),
            SimParticipant(id="agent", role="assistant", is_agent=True),
        ],
        complexity=0.6,
        min_turns=4,
        max_turns=7,
    )


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _print_dataset(dataset: LabeledDataset) -> None:
    for transcript in dataset.conversations:
        print(f"\n[{transcript.scenario_id}] turns={len(transcript.turns)} "
              f"labels={dataset_labels(transcript)}")
        for turn in transcript.turns:
            labels = ", ".join(
                f"{name}={info.get('present')}" for name, info in turn.labels.items()
            ) or "(unlabeled)"
            who = "AGENT" if turn.is_agent else "user "
            print(f"  [{turn.party_id:>6}/{who}] {labels:<46} {turn.content[:54]}")


def dataset_labels(transcript) -> list[str]:
    names: set[str] = set()
    for turn in transcript.turns:
        names.update(turn.labels.keys())
    return sorted(names)


def main() -> int:
    base_url = os.getenv("PROXY_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("PROXY_MODEL") or os.getenv("OPENAI_MODEL")
    pool = OpenAICompatPool(base_url=base_url, model=model)
    print(f"endpoint = {pool.base_url}   model = {pool.model}")
    try:
        requests.get(pool.base_url + "/v1/models", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: endpoint not reachable ({exc}). Start the proxy or set PROXY_BASE_URL.")
        return 1

    synth = ScenarioSynthesizer(pool, votes=2)

    # ── self-certify one labeler per case before trusting the run ─────────────
    _hr("labeler self-certification (constructed positive vs negative)")
    reports = [
        certify_labeler(
            SchemaLabeler(STATES_CHANNEL_ITEM, pool, votes=2),
            _channel_md_seed(),
            probes=[
                CertProbe(name="pos", expect_present=True, prefix=[SimTurn(
                    party_id="alex", role="user",
                    content="Going forward, always post summaries as bullet points.")]),
                CertProbe(name="neg", expect_present=False, prefix=[SimTurn(
                    party_id="riley", role="user", content="Thanks, talk later!")]),
            ],
        ),
        certify_labeler(
            SchemaLabeler(SKILL_WORKS, pool, votes=2),
            _skill_fix_seed(),
            probes=[
                CertProbe(name="pos", expect_present=True, prefix=[SimTurn(
                    party_id="agent", role="assistant", is_agent=True,
                    content="name,age\\nJordan,34\\n  (valid CSV produced by the new skill)")]),
                CertProbe(name="neg", expect_present=False, prefix=[SimTurn(
                    party_id="agent", role="assistant", is_agent=True,
                    content="Sorry, I couldn't export it — the output came out mangled.")]),
            ],
        ),
    ]
    for report in reports:
        print(f"  labeler={report.labeler:<22} passed={report.passed} {report.n_correct}/{report.n_total}")

    # ── Case A: group channel.md (multi-party) ────────────────────────────────
    _hr("Case A — group channel.md: record + apply preferences/todos (multi-party)")
    channel = synth.build_dataset(
        [_channel_md_seed()],
        labelers=["states_channel_item", "updates_channel_md", "applies_preference"],
    )
    _print_dataset(channel)
    channel_path = _DATASETS_DIR / "channel_md_sample.jsonl"
    print(f"\n-> {channel_path.relative_to(Path(__file__).parent)} ({channel.write_jsonl(channel_path)} records)")

    # ── Case B: skill fail -> fix -> works (single user) ──────────────────────
    _hr("Case B — skill fail -> fix -> works (single user, corrective)")
    skill = synth.build_dataset(
        [_skill_fix_seed()],
        labelers=["creates_or_updates_skill", "skill_works"],
    )
    _print_dataset(skill)
    skill_path = _DATASETS_DIR / "skill_fix_sample.jsonl"
    print(f"\n-> {skill_path.relative_to(Path(__file__).parent)} ({skill.write_jsonl(skill_path)} records)")

    _hr("SUMMARY")
    ok = all(r.passed for r in reports) and len(channel) >= 1 and len(skill) >= 1
    print(f"endpoint calls={pool.calls} cert_passed={all(r.passed for r in reports)} "
          f"channel_convos={len(channel)} skill_convos={len(skill)}")
    print("RESULT:", "PASS" if ok else "CHECK OUTPUT ABOVE")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
