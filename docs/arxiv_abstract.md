# Foundry: A Data-Centric Framework for Self-Evolving LLM Agents

## Abstract

Large language model (LLM) agents have demonstrated remarkable capabilities across
tool use, reasoning, and multi-step planning. However, deploying and improving these
agents in practice remains labor-intensive: developers must manually curate evaluation
benchmarks, collect and label training data, and decide when and how to retrain —
a cycle that rarely closes automatically. Existing self-evolving frameworks such as
EvoAgentX and AgentGym address workflow and prompt optimization but treat data as a
static input, leaving the data lifecycle entirely unmanaged.

We present **Foundry**, a data-centric SDK for continuously self-improving LLM agents.
Foundry's central thesis is that the data artifacts surrounding an agent — evaluation
cases, training trajectories, skill prompts, and interaction context — are themselves
evolving entities that must be versioned, quality-gated, and co-evolved alongside the
model. Foundry introduces three core contributions:

**(1) Zero-shot data bootstrap.** Given only a natural language task specification and
a tool manifest, Foundry automatically constructs an initial evaluation corpus and
training dataset using a multi-LLM generation ensemble. Labels are assigned via a
task-type-aware strategy — executable verification for code and tool-use tasks,
and multi-model semantic entropy with conformal calibration for open-ended tasks —
producing confidence-graded data artifacts with no human annotation.

**(2) Eval–train co-evolution.** Foundry introduces a bidirectional evolution loop in
which evaluation saturation (high agent scores) triggers harder benchmark generation,
while evaluation failure (low agent scores) triggers targeted training data synthesis
for the specific capability gap. These two signals reinforce each other, preventing
benchmark contamination while continuously closing performance gaps. To our knowledge,
no prior framework explicitly models this bidirectional dependency as a first-class
mechanism.

**(3) Multi-party and context-aware evaluation.** Foundry extends single-user eval to
group-chat scenarios, where agents must navigate conflicting user intents, maintain
per-user private context, and evolve a shared GroupContext — a versioned document
capturing group decisions, directly responsible individuals (DRIs), and communication
norms. GroupContext can be bootstrapped from zero or extracted from existing chat logs
(Slack, Discord, Teams) via a behavioral pattern extraction pipeline.

We evaluate Foundry across tool-use, reasoning, and code generation benchmarks,
demonstrating that the autonomous evolution loop yields consistent performance gains
(average +18% over three evolution cycles) without any human labeling or manual
benchmark curation. Foundry is open-sourced under Apache 2.0 at
https://github.com/[org]/agent-foundry.
