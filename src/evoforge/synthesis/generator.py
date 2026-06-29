"""ModeConditionedGenerator — generate corrective examples per failure mode.

The generator is *scenario-driven*: a failure cluster is first lifted into a
:class:`~evoforge.synthesis.seed.Seed` (a *failure seed*, component 1), and the
seed — its scenario, hardening conditions, complexity, and the real failing
``(trigger, response)`` pairs it carries — is what conditions generation. This
keeps the seed as the single backbone of *all* synthesis: a corrective example is
just generation over a failure seed. For DPO it uses the observed failing response
as the ``rejected`` negative and synthesizes only the corrected ``chosen`` — so the
contrast is grounded in reality, not invented.
"""

from __future__ import annotations

import inspect
from typing import Any

from evoforge.llm.structured import coerce_records, generate_structured
from evoforge.mining.schema import FailureModeCluster, fix_type_of
from evoforge.synthesis.schema import SynthFormat, SyntheticExample
from evoforge.synthesis.seed import Seed
from evoforge.text import format_tools
from evoforge.trace.schema import TraceLineage

_SFT_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "ideal_response": {"type": "string"},
                },
                "required": ["instruction", "ideal_response"],
            },
        }
    },
    "required": ["examples"],
}

_SFT_SYSTEM = """You generate corrective training examples for an AI agent that
keeps failing in a specific way. Each example must demonstrate the CORRECT behavior
for the described failure mode.

The "ideal_response" is the model answer the agent SHOULD produce. It must fully
RESOLVE the user's request — it is the corrected end-state, not a narration of the
failure. Specifically:
  * Do the right thing the failure mode demands (e.g. for a fragile tool: retry,
    then CONFIRM the successful result with concrete details).
  * Do NOT merely say the tool errored or that you are "trying"/"not giving up"
    with no resolution — end by actually completing the task successfully.
Make instructions diverse and realistic."""

_DPO_SYSTEM = """You are correcting a specific agent failure. Given the user
request and the agent's FAILED response, write the response a correct agent
should have produced. Reply with ONLY the corrected response text (no preamble)."""


class ModeConditionedGenerator:
    """Generate :class:`SyntheticExample`s targeted at one failure cluster.

    Internally each cluster is lifted into a *failure seed* (component 1) so the
    same scenario backbone drives corrective-example generation.
    """

    def __init__(self, pool: Any, complexity: float = 0.6) -> None:
        self._pool = pool
        self._complexity = complexity

    def generate(
        self,
        cluster: FailureModeCluster,
        task_spec: str,
        tools: list[Any],
        system_prompt: str,
        n: int,
        fmt: SynthFormat,
    ) -> list[SyntheticExample]:
        seed = Seed.from_failure_cluster(cluster, complexity=self._complexity)
        if fmt == SynthFormat.DPO:
            return self._generate_dpo(cluster, seed, n)
        return self._generate_sft(cluster, seed, task_spec, tools, system_prompt, n, fmt)

    # ── SFT / tool-trace ──────────────────────────────────────────────

    def _generate_sft(
        self,
        cluster: FailureModeCluster,
        seed: Seed,
        task_spec: str,
        tools: list[Any],
        system_prompt: str,
        n: int,
        fmt: SynthFormat,
    ) -> list[SyntheticExample]:
        examples_text = "\n".join(
            f'- input: "{(e.trigger or "")[:100]}" failed with: "{e.response[:100]}"'
            for e in seed.failure_examples
        )
        conditions = "; ".join(seed.conditions) if seed.conditions else "(none)"
        prompt = (
            f"AGENT TASK: {task_spec}\n"
            f"TOOLS: {format_tools(tools)}\n"
            f"SYSTEM PROMPT: {system_prompt or '(generic)'}\n\n"
            f"FAILURE MODE: {cluster.mode.value}\n"
            f"CAPABILITY: {seed.capability}\n"
            f"SCENARIO: {seed.scenario}\n"
            f"GOAL: {seed.goal}\n"
            f"DIFFICULTY: {seed.complexity_band()}\n"
            f"HARDENING CONDITIONS: {conditions}\n"
            f"REAL FAILURES:\n{examples_text or '(none)'}\n\n"
            f"Generate {n} corrective examples that teach the agent to avoid this failure. "
            "For each, the ideal_response must fully RESOLVE the request (the corrected "
            "end-state that completes the task successfully), not narrate the error."
        )
        raw = generate_structured(
            self._pool, prompt, _SFT_SCHEMA, system=_SFT_SYSTEM,
            temperature=0.7, max_tokens=1500,
        )
        items = coerce_records(raw, key="examples")

        out: list[SyntheticExample] = []
        for item in items[:n]:
            instruction = str(item.get("instruction", "")).strip()
            response = str(item.get("ideal_response", item.get("response", ""))).strip()
            if not instruction or not response:
                continue
            tool_calls = item.get("tool_calls") or []
            out.append(
                SyntheticExample(
                    target_mode=cluster.mode,
                    target_cluster_id=cluster.cluster_id,
                    capability=cluster.capability,
                    format=fmt,
                    instruction=instruction,
                    ideal_response=response,
                    tool_calls=tool_calls if isinstance(tool_calls, list) else [],
                    lineage=self._lineage(cluster),
                )
            )
        return out

    # ── DPO (real failure as the negative) ────────────────────────────

    def _generate_dpo(
        self,
        cluster: FailureModeCluster,
        seed: Seed,
        n: int,
    ) -> list[SyntheticExample]:
        out: list[SyntheticExample] = []
        for ex in seed.failure_examples[:n]:
            instruction = (ex.trigger or "").strip()
            failing = (ex.response or "").strip()
            if not instruction:
                continue
            prompt = (
                f"AGENT TASK: {seed.goal or cluster.capability}\n"
                f"FAILURE MODE: {cluster.mode.value} ({seed.scenario})\n"
                f'USER REQUEST: "{instruction}"\n'
                f'FAILED RESPONSE: "{failing}"\n\n'
                "Write the corrected response."
            )
            chosen = str(self._call(prompt, _DPO_SYSTEM)).strip()
            if not chosen:
                continue
            out.append(
                SyntheticExample(
                    target_mode=cluster.mode,
                    target_cluster_id=cluster.cluster_id,
                    capability=cluster.capability,
                    format=SynthFormat.DPO,
                    instruction=instruction,
                    chosen=chosen,
                    rejected=failing,
                    lineage=self._lineage(cluster),
                )
            )
        return out

    # ── helpers ───────────────────────────────────────────────────────

    def _lineage(self, cluster: FailureModeCluster) -> TraceLineage:
        return TraceLineage(
            parent_trace_ids=list(cluster.trace_ids),
            failure_signature_id=cluster.signature_ids[0] if cluster.signature_ids else None,
            generation_method="synthesis",
            derived_from=cluster.cluster_id,
            tags=[cluster.mode.value, fix_type_of(cluster.mode)],
        )

    def _call(self, prompt: str, system: str) -> str:
        raw = self._pool.generate(prompt, system=system, temperature=0.7, max_tokens=1024)
        if inspect.isawaitable(raw):
            raise RuntimeError(
                "ModeConditionedGenerator received an async LLM pool in sync mode. "
                "Use a synchronous pool (for example, OllamaLLMPool)."
            )
        return str(raw)
