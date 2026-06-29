"""Integration tests for failure forecasting (Phase 4).

Deterministic tests (no LLM): the risk model learns an input-token signal that
beats naive baselines, predicts higher risk for risky inputs, classifies the
likely mode, and the calibration / drift / namespace / CLI wiring all work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
from evoforge.cli import cli
from evoforge.core.config import SDKConfig, StorageConfig
from evoforge.core.types import EvalCase, EvalCaseResult, Message, ScoringMethod
from evoforge.forecast import DriftMonitor, RiskForecaster, RiskLevel
from evoforge.trace import TraceNormalizer
from evoforge.trace.schema import TraceRecord


def _trace(cid: str, capability: str, text: str, fail: bool) -> TraceRecord:
    norm = TraceNormalizer()
    case = EvalCase(
        id=cid, capability=capability,
        messages=[Message(role="user", content=text)],
        expected="ok", scoring_method=ScoringMethod.CONTAINS,
    )
    response = "[ERROR: invalid request]" if fail else "Here are your results."
    result = EvalCaseResult(
        case_id=cid, capability=capability, agent_response=response,
        score=0.0 if fail else 1.0, passed=not fail,
    )
    return norm.from_eval_result(result, case=case, agent_name="agent")


def _dataset() -> list[TraceRecord]:
    """Within 'booking', the token 'invalid' separates failures from successes."""
    traces = []
    cid = 0
    for i in range(12):
        traces.append(_trace(f"s{cid}", "booking", f"book flight {i} for a passenger", False))
        cid += 1
        traces.append(_trace(f"f{cid}", "booking", f"book invalid flight {i} now", True))
        cid += 1
    for i in range(8):
        traces.append(_trace(f"o{cid}", "search", f"find flights to city {i}", False))
        cid += 1
    return traces


# ── Model quality ───────────────────────────────────────────────────────────────


def test_forecaster_beats_naive_baselines():
    traces = _dataset()
    fc = RiskForecaster().fit(traces)
    ev = fc.evaluate(traces)
    assert ev.beats_majority is True
    assert ev.beats_capability is True
    assert ev.model_brier < ev.capability_brier
    assert ev.model_accuracy >= 0.8
    assert ev.model_auc is not None and ev.model_auc >= 0.8
    assert ev.method == "resubstitution"
    assert ev.honest is False


def _noisy_dataset() -> list[TraceRecord]:
    """Signal ('invalid' -> fail) plus deterministic label noise every 6th item."""
    traces = []
    for i in range(36):
        invalid = (i % 2 == 0)
        fail = invalid
        if i % 6 == 0:                       # inject noise: flip ~1 in 6
            fail = not fail
        text = f"book invalid flight {i}" if invalid else f"book flight {i} for John"
        traces.append(_trace(f"n{i}", "booking", text, fail))
    return traces


def test_cross_validation_is_honest_and_still_beats_baseline():
    fc = RiskForecaster()
    ev = fc.cross_validate(_noisy_dataset(), k=5)
    # Honest, held-out evaluation.
    assert ev.honest is True
    assert ev.method == "cross_validation"
    assert ev.n_folds == 5
    # On noisy data the model is no longer perfect...
    assert ev.model_brier > 0.0
    # ...but still beats the capability baseline (it reads the 'invalid' token).
    assert ev.model_brier < ev.capability_brier
    assert ev.beats_capability is True


def test_cross_validate_leaves_model_fitted_for_serving():
    fc = RiskForecaster()
    fc.cross_validate(_dataset(), k=5)
    assert fc.fitted is True
    f = fc.forecast("book invalid flight now", capability="booking")
    assert f.p_failure > 0.5


def test_cross_validation_falls_back_on_tiny_data():
    tiny = [
        _trace("a", "booking", "book invalid flight", True),
        _trace("b", "booking", "book flight for john", False),
    ]
    fc = RiskForecaster()
    ev = fc.cross_validate(tiny, k=5)
    assert ev.honest is False
    assert ev.method == "resubstitution"


def test_forecaster_predicts_higher_risk_for_risky_input():
    fc = RiskForecaster().fit(_dataset())
    risky = fc.forecast("book invalid flight 99 now", capability="booking")
    safe = fc.forecast("book flight 3 for a passenger", capability="booking")
    assert risky.p_failure > safe.p_failure
    assert risky.p_failure > 0.5


def test_forecast_predicts_likely_mode():
    fc = RiskForecaster().fit(_dataset())
    f = fc.forecast("book invalid flight now", capability="booking")
    # All failures were ERROR responses -> environment_fragility signature.
    assert f.likely_mode == "environment_fragility"
    assert f.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)


def test_novelty_higher_for_unseen_tokens():
    fc = RiskForecaster().fit(_dataset())
    seen = fc.forecast("book flight 1 for a passenger", capability="booking")
    novel = fc.forecast("zzqq xytron wibble frobnicate", capability="booking")
    assert novel.novelty > seen.novelty
    # Out-of-distribution -> wider confidence interval.
    assert (novel.confidence_interval[1] - novel.confidence_interval[0]) >= (
        seen.confidence_interval[1] - seen.confidence_interval[0]
    )


def test_fit_requires_labeled_traces():
    fc = RiskForecaster()
    try:
        fc.fit([])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── Calibration ─────────────────────────────────────────────────────────────────


def test_calibration_report_shape():
    fc = RiskForecaster().fit(_dataset())
    report = fc.calibration(_dataset())
    assert report.n > 0
    assert 0.0 <= report.brier <= 1.0
    assert isinstance(report.within_tolerance, bool)


# ── Drift ───────────────────────────────────────────────────────────────────────


def test_drift_detected_on_capability_shift():
    reference = _dataset()  # booking-heavy
    recent = [_trace(f"r{i}", "refunds", f"process refund {i} urgently", True) for i in range(10)]
    monitor = DriftMonitor().fit(reference)
    report = monitor.compare(recent)
    assert report.drifted is True
    assert report.capability_js_divergence > 0.0


def test_no_drift_on_similar_window():
    reference = _dataset()
    monitor = DriftMonitor().fit(reference)
    report = monitor.compare(_dataset())
    assert report.drifted is False
    assert report.drift_score <= 0.15


# ── Namespace + CLI ──────────────────────────────────────────────────────────────


def test_sdk_forecast_namespace(tmp_path: Path):
    config = SDKConfig(task_spec="A flight booking assistant.", storage=StorageConfig(path=tmp_path))
    sdk = evoforge.FoundrySDK(config)
    sdk.trace.store.save_many(_dataset())

    sdk.forecast.fit("agent")
    f = sdk.forecast.predict("agent", "book invalid flight now", capability="booking")
    assert f.p_failure > 0.5

    ev = sdk.forecast.evaluate("agent")
    assert ev.beats_capability is True


def test_forecast_cli_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["forecast", "--help"])
    assert result.exit_code == 0
    assert "Forecast failure risk" in result.output


def test_forecast_types_exposed_on_public_api():
    assert evoforge.RiskForecaster is not None
    assert evoforge.ForecastRequest is not None
    assert evoforge.DriftMonitor is not None
