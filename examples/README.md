# Multi-party simulator & synthesis examples

EvoForge can ingest **labeled, multi-party, multi-turn conversations** through a
generic `ConversationSimulator` seam (`evoforge.eval.simulator`). A simulator
consumes a generic `SimScenario` and returns a `SimTranscript` whose turns and/or
whole conversation may carry **labels** (presence flags, verdicts, rubric scores,
…). EvoForge then turns transcripts into `EvalCase` objects for scoring or mines
them for training data — staying decoupled from *how* any particular simulator
works.

It also ships a **native, scenario-driven synthesis** backbone (no extra deps):
a `Seed` (the control surface) → `ConversationGenerator` (single- or multi-party)
→ an open, self-certified labeler set. `examples/live_synthesis.py` exercises that
whole backbone against a real LLM.

## Files

| File | Deps | What it shows |
| --- | --- | --- |
| [`multiparty_simulator.py`](multiparty_simulator.py) | none | Offline, runnable demo of the seam using the built-in `ScriptedSimulator`: build `SimScenario`s, produce labeled `SimTranscript`s, convert to `EvalCase`s. Instant (no LLM in the loop). |
| [`live_synthesis.py`](live_synthesis.py) | a chat endpoint + `requests` | **Live** end-to-end demo of the `ScenarioSynthesizer` pipeline over a real LLM. Two cases: **(A) group `channel.md`** — members state standing preferences/todos, the agent records them into `channel.md` and applies them (multi-party); **(B) skill fail→fix→works** — the agent's first attempt fails, it authors/fixes a skill, the retry succeeds (single user). Writes a labeled JSONL per case. |

## Run the offline demo

```bash
python examples/multiparty_simulator.py
```

## Run the live synthesis demo (real LLM)

```bash
# Prereqs: an OpenAI-compatible /v1/chat/completions endpoint; requests installed.
$env:PROXY_BASE_URL="http://127.0.0.1:8787"; $env:PROXY_MODEL="gpt-5.2-chat"
python examples/live_synthesis.py
```

It is a thin wiring script over the reusable pipeline: it (1) authors the
**seeds** (the situation + participants), (2) authors + self-certifies the
**labelers** (what to measure), then calls `ScenarioSynthesizer.build_dataset(...)`
to generate and honestly label each conversation, writing
`datasets/channel_md_sample.jsonl` and `datasets/skill_fix_sample.jsonl`. **To
produce your own dataset, change only the seeds and the labelers** — the same
surface the evolution loop drives. Labels use an n-of-m **vote ensemble** with a
verbatim guard.

## Plugging in your own simulator

The seam is feature-agnostic: implement `ConversationSimulator` (a
`simulate(scenario)` method returning a `SimTranscript`) and inject it into
`evoforge.eval`'s `run_simulated`, or drive it directly:

```python
from evoforge.eval.simulator import SimScenario, simulate_many
from evoforge.eval.connectors import NativeMultiPartySimulator

sim = NativeMultiPartySimulator(pool=my_pool)        # generate + label, no extra deps
transcripts = simulate_many(sim, scenarios)
for t in transcripts:
    print(t.model_dump_json())
```

## Sample transcript shape

A turn's `labels` map carries the per-turn annotation, for example a certified
presence label:

```jsonc
{
  "party_id": "alice", "role": "user", "is_agent": false,
  "content": "Going forward, keep the summary as a bulleted list.",
  "labels": {
    "target_presence": {
      "present": true, "verbatim_only": false, "votes": "3/3",
      "fields": { "expresses_target": {"value": true, "evidence": "..."} }
    }
  }
}
```
