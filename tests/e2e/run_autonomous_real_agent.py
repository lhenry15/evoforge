"""Fully autonomous run against the REAL Ollama flight agent.

Nothing here is hand-authored test data: the SDK infers capabilities, designs
eval scenarios, runs the real agent (live LLM + tool calls), scores with an LLM
judge, records traces, mines failures, builds coverage, auto-expands eval on
blind spots, re-tests, forecasts, and writes the unified report.
"""

import importlib.util
import shutil
import sys
import time
from pathlib import Path

# Repo root = two levels up from tests/e2e/, so the script runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import foundry
from foundry.core.config import SDKConfig, StorageConfig
from foundry.llm.ollama import OllamaLLMPool

STORE = Path.home() / "agent-foundry" / ".foundry" / "real_demo"
if STORE.exists():
    shutil.rmtree(STORE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# 1. Load the REAL agent (live Ollama + tools) from the example file.
spec = importlib.util.spec_from_file_location(
    "flight_ex", str(REPO_ROOT / "examples" / "flight_agent.py")
)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
agent = ex.flight_agent
agent_name = agent.__name__

# 2. Fresh SDK we control, pointed at a clean store, same task spec.
sdk = foundry.FoundrySDK(SDKConfig(
    task_spec=ex.sdk.config.task_spec,
    storage=StorageConfig(path=STORE),
))
pool = OllamaLLMPool()

log(f"Agent: {agent_name}")
log(f"Task: {sdk.config.task_spec}")

# 3. SDK designs the eval scenarios (no hand-written cases).
log("BOOTSTRAP: SDK inferring capabilities + designing eval cases...")
t0 = time.time()
boot = sdk.bootstrap(agent=agent, num_eval_cases=12)
log(f"  -> {len(boot.capabilities)} capabilities: {[c['name'] for c in boot.capabilities]}")
log(f"  -> {len(boot.eval_cases)} eval cases designed ({time.time()-t0:.0f}s)")
cases = boot.eval_cases

# 4. SDK runs the REAL agent against its own cases + scores with LLM judge.
log("EVAL #1: running real agent on designed cases...")
t0 = time.time()
r1 = sdk.eval.run(agent=agent, cases=cases, parallelism=2)
log(f"  -> overall={r1.overall_score:.3f} passed={r1.n_passed}/{r1.n_total} ({time.time()-t0:.0f}s)")
for cap, s in sorted(r1.capability_scores.items()):
    log(f"     {cap}: {s:.2f}")

# 5. Record normalized traces (mining/coverage/forecast inputs).
sdk.trace.record_eval_run(r1, cases=cases)
log(f"TRACE: recorded {len(sdk.trace.load(agent_name))} traces, "
    f"{len(sdk.trace.failures(agent_name))} failures")

# 6. Mine failure modes from the REAL failures (LLM re-classifies vague ones).
log("MINE: discovering failure modes (with LLM re-classification)...")
mining = sdk.mine.run(agent_name, pool=pool, persist=True)
log(f"  re-classified {mining.metadata.get('n_reclassified', 0)} vague failures")
for c in mining.clusters[:5]:
    log(f"  [{c.cluster_id}] {c.mode.value} (cap={c.capability}) "
        f"size={c.size} impact={c.impact_score:.2f} fix={c.suggested_fix_type}")

# 7. Coverage + autonomous eval expansion on blind spots.
spots = sdk.coverage.blindspots(agent_name, pool=pool)
log(f"COVERAGE before: ratio={sdk.coverage.map(agent_name, pool=pool).coverage_ratio():.0%}, "
    f"{len(spots)} blind spots")
new_cases = []
if spots:
    log("EXPAND: SDK generating targeted eval cases for blind spots...")
    t0 = time.time()
    new_cases = sdk.coverage.expand(agent_name, pool=pool, persist=True, cases_per_blindspot=2)
    log(f"  -> +{len(new_cases)} new adversarial cases ({time.time()-t0:.0f}s)")
    # Measure closure right after expansion (before re-eval records new traces).
    after = sdk.coverage.map(agent_name, pool=pool)
    log(f"COVERAGE after:  ratio={after.coverage_ratio():.0%}, "
        f"{len(after.blindspots())} blind spots  (closed {len(spots) - len(after.blindspots())})")

# 8. Re-test on the expanded suite (real agent again) — are the new cases hard?
if new_cases:
    log("EVAL #2: re-testing real agent on expanded (adversarial) suite...")
    t0 = time.time()
    r2 = sdk.eval.run(agent=agent, cases=new_cases, parallelism=2)
    sdk.trace.record_eval_run(r2, cases=new_cases)
    log(f"  -> on generated cases: overall={r2.overall_score:.3f} "
        f"passed={r2.n_passed}/{r2.n_total} ({time.time()-t0:.0f}s)")
    # Re-classify the new traces so the offline report shows real modes.
    sdk.mine.run(agent_name, pool=pool, persist=True)

# 9. Forecast (honest cross-validation) if we have enough labeled traces.
traces = sdk.trace.load(agent_name)
log(f"FORECAST: {len(traces)} labeled traces total")
try:
    ev = sdk.forecast.cross_validate(agent_name, k=4)
    log(f"  -> honest CV: brier={ev.model_brier} vs capability={ev.capability_brier} "
        f"| acc={ev.model_accuracy} auc={ev.model_auc} honest={ev.honest}")
except Exception as e:
    log(f"  -> forecast skipped: {e}")

# 10. Unified report.
from foundry.dashboard import DashboardGenerator
report = DashboardGenerator(storage_path=str(STORE)).generate(str(STORE / "report.html"))
log(f"REPORT: {report}")
print("REPORT_PATH=" + report)
