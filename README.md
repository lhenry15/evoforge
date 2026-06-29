<div align="center">

# 🔥 EvoForge — Self-Evolving AI Agents, Data-Centric

**Open-source SDK that lets LLM agents test, fix, and future-proof themselves — automatically.**

EvoForge turns your agent's traces into evaluation data, mines failure modes, generates targeted fixes, and **forecasts failures before they ship** — a data-centric flywheel for continuously self-improving AI agents.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)](#)
[![Local-first](https://img.shields.io/badge/local--first-Ollama%20%7C%20MLX-orange.svg)](#local-first)

```bash
evoforge run my_agent.py --cycles 4
```

*self-evolving agents · agent evaluation · data-centric AI · LLM agent reliability · synthetic eval data · failure forecasting · agent observability*

</div>

---

## What is EvoForge?

Most agent frameworks help you **build** agents. EvoForge helps your agent **improve itself** — continuously, autonomously, from data.

It closes the loop every agent developer does by hand: write evals → find failures → make data → retrain → hope it's better. EvoForge automates that loop and adds the missing piece — **predicting failures before users hit them**.

> Reactive testing tells you what already broke. **EvoForge moves agent development from reactive to preventive.**

---

## Why EvoForge is different

Existing self-evolving frameworks are **algorithm-centric** (they search prompts/workflows). EvoForge is **data-centric**: it treats the data around your agent — eval cases, traces, training examples, skills — as the thing that evolves.

| Traditional agent dev | EvoForge |
|---|---|
| Hand-write eval cases | Bootstraps + expands them from your task spec |
| Guess what's wrong | Mines real failure modes from traces |
| Hand-curate fix data | Synthesizes targeted training data per failure |
| Ship and find out | Forecasts failure risk **before** deploy |
| Hope it improved | Measures every change, gated against regressions |

---

## Quickstart

```bash
pip install evoforge
```

**1. Wrap any agent (any framework, any LLM):**

```python
import evoforge

sdk = evoforge.init(task_spec="A flight booking assistant that searches and books flights.")

@sdk.agent(tools=["search_flights", "book_flight"])
def my_agent(messages):
    ...  # your existing agent code — unchanged
```

**2. Let it evolve:**

```bash
evoforge run my_agent.py --cycles 4
```

EvoForge will bootstrap evals, run your agent, mine failure modes, generate targeted data, expand test coverage, optionally fine-tune, and track the whole story — zero hand-labeling.

---

## The predictive flywheel

```
            ┌──────────────────────────────────────────────────────┐
            ▼                                                      │
   run agent → traces ─→ mine failure modes ─→ synthesize fixes ─→ │
        ▲                      │                                    │
        │                      ▼                                    ▼
   forecast risk ◀── expand eval coverage ◀── evolve prompts/skills/model
   (pre-deploy)        (close blind spots)        (gated, no regressions)
```

1. **Trace** — every run is normalized into a trace with a failure signature + lineage
2. **Mine** — cluster real failures into ranked, root-caused modes
3. **Synthesize** — generate targeted training data (real failures become DPO negatives)
4. **Cover** — auto-expand evals to close blind spots, so the benchmark never goes stale
5. **Forecast** — predict failure risk of a new request *before* it runs (honest cross-validation)
6. **Fix** — evolve prompts, skills, architecture, or fine-tune — A/B-gated

---

## Features

- 🔁 **Autonomous loop** — `evoforge run` does bootstrap → eval → evolve → re-eval
- 🧩 **Failure-mode mining** — turns messy failures into actionable, ranked root causes
- 🧪 **Synthetic eval + train data** — schema-constrained generation that works even on a 3B local model
- 🗺️ **Adaptive coverage** — capability × failure-mode heatmap; auto-generates cases for blind spots
- 🔮 **Failure forecasting** — risk scoring with calibration + drift monitoring
- 🧠 **Skill + prompt evolution** — versioned, dedup'd, retire-able
- 🛡️ **Regression-gated promotion** — A/B test before any model swap
- 📊 **One-command dashboard** — evolution history + failure intelligence in a single HTML report
- 🧰 **Framework-agnostic** — smolagents, pydantic-ai, LangChain, or plain Python
- 🔒 **Local-first** — runs fully offline on Ollama + MLX, zero cloud cost

---

## CLI

```bash
evoforge bootstrap my_agent.py   # design eval cases from task spec + tools
evoforge eval my_agent.py        # score per capability (LLM judge)
evoforge evolve my_agent.py      # mine gaps + evolve
evoforge run my_agent.py         # full autonomous loop
evoforge forecast my_agent.py "cancel my booking ASAP"   # risk before running
evoforge insights my_agent.py    # failure-intelligence dashboard
evoforge report                  # evolution + intelligence report
```

---

## Framework-agnostic

```python
# smolagents / pydantic-ai / LangChain / plain OpenAI — wrap them all the same way
@sdk.agent(tools=[search_flights, book_flight])
def my_agent(messages):
    return your_agent.run(messages[-1].content)
```

---

## Local-First

Runs entirely on your machine — zero cloud dependency, zero cost, full privacy:

- **Inference & judge:** Ollama (`qwen2.5:3b` or any model)
- **Structured generation:** schema-constrained decoding for reliable data even from small models
- **Training:** MLX LoRA on Apple Silicon
- **Storage:** local `.foundry/` directory

---

## What it does in practice

EvoForge is validated end-to-end against a live local agent (no seeded data):

- Designs eval cases, runs the real agent, mines failures, and **closes coverage blind spots cycle-over-cycle** (0% → 100% on the demo agent)
- Generates expansion cases the agent actually fails — i.e. genuinely discriminating, not filler
- Reports forecasting quality with **honest cross-validation** (so it never over-claims on tiny data)

*Numbers vary by agent/model; metrics use held-out evaluation, not training-set scores.*

---

## Install

```bash
git clone https://github.com/lhenry15/evoforge.git
cd evoforge && pip install -e .
ollama pull qwen2.5:3b
```

Requirements: Python 3.10+, [Ollama](https://ollama.ai/) for local inference, Apple Silicon for MLX LoRA (training optional).

---

## Project structure

```
src/evoforge/
├── core/        # SDK, config, types, agent history
├── trace/       # normalized traces, failure signatures, lineage
├── mining/      # failure-mode clustering + root-cause labeling
├── synthesis/   # mode-conditioned training-data generation
├── coverage/    # adaptive eval expansion (blind-spot closing)
├── forecast/    # pre-deploy failure forecasting + calibration
├── eval/        # eval runner, LLM judge, multi-turn
├── evolution/   # prompt/skill/architecture evolution, A/B promotion
├── training/    # MLX LoRA backend
├── bootstrap/   # zero-shot eval generation
├── llm/         # Ollama (structured outputs) + GitHub Models pools
└── cli.py
```

---

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Great first areas: new synthesis strategies, better judge prompts, more training backends (QLoRA, OpenAI FT), multi-party eval.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{evoforge2025,
  title  = {EvoForge: A Data-Centric SDK for Self-Evolving LLM Agents},
  author = {Henry, L.},
  year   = {2025},
  url    = {https://github.com/lhenry15/evoforge}
}
```

---

<div align="center">
<sub>Keywords: self-evolving agents · self-improving LLM agents · agent development · agentic AI · LLM evaluation · agent reliability · data-centric AI · synthetic data generation · failure forecasting · prompt optimization · LoRA fine-tuning · AI agent testing · agent observability</sub>
</div>
