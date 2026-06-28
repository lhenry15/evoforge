# Foundry Architecture

## North Star

> An agent that continuously improves itself from data — with zero human intervention —
> while maintaining trustworthy, never-stale evaluation standards.

The "data-centric" framing means: **data is the source of truth for every improvement
decision.** You don't manually tweak prompts or decide when to retrain. The data tells
you what's broken, what to fix, and when to promote.

---

## Design Principles

1. Zero-to-running in 3 lines for simple cases
2. Decorator-first instrumentation — no rewriting agent code
3. Sensible defaults everywhere — override only what you need
4. All subsystems composable independently
5. Explicit > magic for anything that touches data or models

---

## Six-Layer Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  AGENT RUNTIME / ENVIRONMENT                                    │
│  (LangGraph, AutoGen, EvoAgentX, Custom...)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │ @instrument hooks
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: TELEMETRY & COLLECTION                                │
│  trajectory logger · user feedback · cost + latency metadata    │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: DATA STORE & VERSIONING                               │
│  EvalRegistry · TrainRegistry · SkillRegistry                   │
│  GroupContextRegistry · PerUserContextStore                     │
│  git-like versioning · quality metadata · dedup indexes         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: DATA FACTORY                                          │
│  Generator (multi-LLM) → Labeler → UQ → Quality Gate           │
│  Modes: scratch · mutation · trajectory mining · self-play ·    │
│         distillation                                            │
│  Formats: SFT · DPO · PRM · tool traces                        │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: EVOLUTION ENGINE                                      │
│  SaturationDetector · FailureDetector · DriftDetector           │
│  EvalEvolver · TrainEvolver · SkillEvolver                      │
│  FineTuneTrigger · A/B Test · Promotion                         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: FINE-TUNE ORCHESTRATION                               │
│  LoRABackend · OpenAIFineTuneBackend · (extensible)             │
│  Training job launcher · model version manager                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6: OBSERVABILITY & GOVERNANCE  (V1.0)                    │
│  provenance dashboard · evolution audit log · PII filter        │
│  human review queue · privacy assertions                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Evolution Flywheel

```
Agent runs → trajectories collected → data curated
    ↑                                      ↓
Model promoted ← fine-tune triggered ← eval detects gaps
    ↑                                      ↓
A/B test passes ← skill registry updated ← targeted data generated
```

### Eval ↔ Train Co-Evolution (Key Innovation)

```
EVAL → TRAIN:  capability score < 0.6  → FailureDetector fires
               → targeted training data generated for that capability
               → fine-tune → score rises → then eval can expand

TRAIN → EVAL:  capability score > 0.85 → SaturationDetector fires
               → harder eval cases generated for that capability
               → benchmark stays discriminating as model improves
```

### Evolution Cycle Phases

```
Phase 1: OBSERVE    drift check · run eval · per-capability breakdown
Phase 2: DECIDE     classify: failing | healthy | saturating
Phase 3: EVOLVE     targeted data gen · eval expansion · fine-tune
Phase 4: REPORT     what changed · why · confidence · next cycle ETA
```

---

## Data Artifacts

### EvalLabel Schema (5 Components)
```
gold_answer           Optional[str]      reference answer (or None)
valid_answer_variants List[str]          semantically equivalent answers
scoring_rubric        RubricSpec         dimensions + weights + scale
scoring_function      Optional[Callable] auto-synthesized executable scorer
metadata              dict               difficulty · tags · confidence ·
                                         provenance · is_reference_free
```

### Eval Case Taxonomy
```
SingleTurnEvalCase     single user · single exchange
MultiTurnEvalCase      single user · multi-turn · UserSimulator
                       3-layer scoring: task + milestones + step quality
MultiPartyEvalCase     multi-user · GroupContext · MultiPartySimulator
                       conflict scenarios · privacy assertions ·
                       individual satisfaction targets (min not avg)
```

### Training Data Formats
```
SFT          (instruction, response)
DPO          (instruction, chosen, rejected)   current vs. teacher model
             margin filter: gap must be >= threshold
PRM          (trajectory, [step_scores])        turn-level process supervision
             one score per agent turn (user_reply OR tool_call = 1 turn)
ToolTraces   (instruction, tool_calls[], results[], final_response)
```

### Label Confidence Grades
```
Grade A  ★★★★★  executable verify / env feedback   → auto-approve eval + train
Grade B  ★★★★☆  multi-model ensemble consensus     → auto-approve train only
Grade C  ★★★☆☆  single model / low confidence      → HITL queue
```

---

## Task Type Routing

| Task Type | Label Strategy | Default Formats | Bootstrap Mode |
|---|---|---|---|
| Code / SQL | Execute + verify | SFT + DPO | Generate → run tests |
| Math / Reasoning | Symbolic checker | SFT + PRM | Generate → verify |
| Tool Use / API | Environment feedback | SFT + Tool Traces | Sandbox rollout |
| RAG / Knowledge | Retrieval + ensemble | SFT + DPO | Generate Q + verify |
| Planning / Multi-step | LLM judge consensus | SFT + PRM | Self-play |
| Conversation | Judge + reward model | DPO-heavy | Persona injection |

Composite types: use highest-trust strategy available.

---

## Multi-Party Architecture

```
Agent context stack per response:
  PerUserContext[user_A]   PRIVATE — never surfaces to other users
  PerUserContext[user_B]   personal prefs · private history · auth
  PerUserContext[user_C]

  GroupContext             SHARED — informs all responses
                           decisions · DRIs · norms · conflicts
                           active initiatives · communication prefs

  ConversationHistory      EPHEMERAL — bounded window
  EnvironmentState         LIVE — current world state
  ToolManifest             STATIC — available actions
```

### GroupContext Bootstrap Paths
```
Zero-shot:    task_spec + member roles → multi-LLM synthesizes GroupContext
Chat log:     Slack/Discord/Teams export → extract decisions · DRIs ·
              conflicts · norms via behavioral pattern mining + UQ
Merge:        chat log takes priority · zero-shot fills gaps
```

---

## Environment Layer

```
Mode 1: Built-in SyntheticEnv (zero config)
  Input:  tool manifest (signatures + descriptions)
  Auto-generates: entity model · response simulator · world state tracker
                  failure injector · goal state verifier

Mode 2: External Sandbox (pluggable via EnvironmentProtocol)
  DockerSandboxConnector · HTTPSandboxConnector · MCPConnector
  WebArenaConnector · OSWorldConnector · custom

Fidelity ladder:
  Stage 1: SyntheticEnv (bootstrap)
  Stage 2: SyntheticEnv + real response examples (grounded)
  Stage 3: Staging API (real responses, no prod risk)
  Stage 4: Production mirror (highest fidelity)
```

---

## Uncertainty Quantification

```
Tier 4: Executable verification      → confidence = 1.0, skip UQ
Tier 1: Semantic entropy             → cluster outputs, compute H
         (Farquhar et al. NeurIPS 2024)
Tier 2: Multi-model ensemble         → agreement score = 1 - σ/range
Tier 3: Conformal prediction wrapper → statistically valid intervals
         (eval data only — strict guarantees required)
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage default | Local filesystem + SQLite metadata | Zero config, pluggable to S3/GCS |
| Fine-tune backends | LoRA (HuggingFace PEFT) + OpenAI API | Local GPU + cloud both supported |
| Multi-LLM router | Copilot SDK (default) + LLMPool protocol | Multi-model bootstrap + swappable |
| License | Apache 2.0 | Enterprise-friendly + patent protection |
