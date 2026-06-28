# 🔥 EvoForge

**A data-centric SDK for self-evolving LLM agents.**

EvoForge automatically evaluates your agent, identifies capability gaps, and improves it through prompt evolution, skill generation, and LoRA fine-tuning — all with a single command.

```bash
evoforge run my_agent.py --cycles 4
```

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

---

## Why EvoForge?

Most agent frameworks help you **build** agents. EvoForge helps you **improve** them — continuously, autonomously, driven by data.

| Traditional approach | EvoForge approach |
|---|---|
| You write eval cases manually | EvoForge bootstraps them from your task spec |
| You guess what's wrong | EvoForge identifies exact capability gaps |
| You tweak prompts by hand | EvoForge evolves prompts, skills, and models |
| You hope it got better | EvoForge measures improvement with A/B tests |

---

## Quickstart

```bash
pip install evoforge
```

**1. Wrap your agent (any framework):**

```python
import evoforge

sdk = evoforge.init(
    task_spec="A flight booking assistant that searches and books flights.",
)

@sdk.agent(tools=["search_flights", "book_flight"])
def my_agent(messages):
    # Your agent logic here (any LLM, any framework)
    ...
```

**2. Run the evolution loop:**

```bash
evoforge run my_agent.py --cycles 4
```

That's it. EvoForge will:
- Bootstrap eval cases from your task spec + tools
- Evaluate your agent per capability
- Evolve prompts and skills to fix gaps
- Detect when prompt evolution hits a ceiling → switch to LoRA training
- Track the full evolution history

---

## CLI

```bash
evoforge bootstrap my_agent.py    # Generate eval cases
evoforge eval my_agent.py         # Score agent per capability
evoforge evolve my_agent.py       # Identify gaps + evolve
evoforge run my_agent.py          # Full autonomous loop
evoforge status                   # Show stored data
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     EvoForge Evolution Loop                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Bootstrap ──→ Eval ──→ Gaps? ──→ Evolve ──→ Re-eval ──→ ...  │
│                           │                                       │
│                     Saturating? ──→ Expand Eval (harder cases)    │
│                                                                   │
│   Evolution targets:                                              │
│     1. Prompts (instant, free)                                    │
│     2. Skills (versioned .md files)                               │
│     3. Architecture (CoT, debate, decompose)                      │
│     4. Model (LoRA fine-tuning)                                   │
│                                                                   │
│   Data strategies:                                                │
│     • Teacher distillation (gpt-4o-mini generates examples)       │
│     • Rejection sampling (generate N, keep top-K)                 │
│     • Self-play (model debates itself)                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Framework-Agnostic

EvoForge wraps **any** agent — smolagents, pydantic-ai, LangChain, or plain Python:

```python
# smolagents
@sdk.agent(tools=[search_flights, book_flight])
def my_agent(messages):
    return smol_agent.run(messages[-1].content)

# pydantic-ai
@sdk.agent(tools=["search", "book"])
def my_agent(messages):
    return pai_agent.run_sync(messages[-1].content).output

# Plain Python with OpenAI
@sdk.agent(tools=["search", "book"])
def my_agent(messages):
    return client.chat.completions.create(...).choices[0].message.content
```

---

## Local-First

EvoForge runs entirely on your machine:

- **Inference**: Ollama (qwen2.5:3b or any model)
- **Training**: MLX LoRA on Apple Silicon
- **Judge**: Local LLM (or gpt-4o-mini for higher accuracy)
- **Storage**: `.evoforge/` directory in your project

Zero cloud dependency. Zero cost. Full privacy.

---

## Dashboard

```bash
evoforge report
```

Generates a standalone HTML report showing your agent's evolution story:
- Score progression over time
- Per-capability breakdown (GAP / OK / SATURATING)
- Collapsible evolution steps with diffs and failure analysis
- Training data samples and skill history

---

## Key Concepts

| Concept | What it does |
|---|---|
| **Bootstrap** | Auto-generates eval cases from your task spec + tools |
| **Capabilities** | Broad skill areas inferred from your agent (3-4 per agent) |
| **LLM Judge** | Scores responses semantically (not just string matching) |
| **Skill Files** | Versioned instruction docs that evolve and can be retired |
| **Rejection Sampling** | Generate candidates → score → keep only the good ones |
| **Regression Guard** | A/B test before promoting a new model (no regressions allowed) |
| **Eval Expansion** | When agent saturates (>0.85), auto-generates harder cases |

---

## Proven Results

| Agent | Task | Improvement | Method |
|---|---|---|---|
| Flight booking | JSON format output | 20% → 100% | LoRA (12 examples, 200 iters) |
| Flight booking | Full pipeline | 0.156 → 0.278 | Prompt evolution (4 cycles) |
| Code review | Structured output | 11% → 83% | Prompt + skills (4 rounds) |
| Error handling | Architecture search | 0.0 → 0.5 | Chain-of-Thought wrapper |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local inference)
- Apple Silicon Mac (for MLX LoRA training) — or skip training and use prompt evolution only

---

## Installation

```bash
# From source
git clone https://github.com/lhenry15/evoforge.git
cd evoforge
pip install -e .

# Pull the model
ollama pull qwen2.5:3b
```

---

## Project Structure

```
src/evoforge/
├── core/           # SDK, config, types, agent history
├── eval/           # Eval runner, LLM judge, multi-turn, expander
├── evolution/      # Prompt evolver, skill registry, architecture search
├── factory/        # Data generation (teacher, rejection sampling, DPO)
├── training/       # MLX LoRA backend
├── bootstrap/      # Zero-shot eval case generation
├── llm/            # Ollama + GitHub Models pools
├── dashboard.py    # HTML report generator
└── cli.py          # Click CLI
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome — especially for:
- New data generation strategies
- Better LLM judge prompts
- Additional training backends (QLoRA, OpenAI FT)
- Multi-party eval

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{evoforge2025,
  title={EvoForge: A Data-Centric SDK for Self-Evolving LLM Agents},
  author={Henry, L.},
  year={2025},
  url={https://github.com/lhenry15/evoforge}
}
```
