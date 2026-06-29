"""Integration test for the intelligence dashboard (no LLM)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import foundry
from foundry.cli import cli
from foundry.core.config import SDKConfig, StorageConfig
from foundry.core.types import EvalCase, EvalCaseResult, Message, ScoringMethod
from foundry.intelligence_dashboard import IntelligenceDashboard
from foundry.trace import TraceNormalizer


def _seed(sdk):
    norm = TraceNormalizer()
    recs = []
    cid = 0
    for i in range(8):
        for text, fail in [
            (f"book flight {i} for passenger", False),
            (f"book invalid flight {i} now", True),
        ]:
            case = EvalCase(
                id=f"c{cid}", capability="booking",
                messages=[Message(role="user", content=text)],
                expected="ok", scoring_method=ScoringMethod.CONTAINS,
            )
            r = EvalCaseResult(
                case_id=f"c{cid}", capability="booking",
                agent_response="[ERROR: invalid]" if fail else "ok",
                score=0.0 if fail else 1.0, passed=not fail,
            )
            recs.append(norm.from_eval_result(r, case=case, agent_name="agent"))
            cid += 1
    sdk.trace.store.save_many(recs)


def test_intelligence_dashboard_renders_all_panels(tmp_path: Path):
    sdk = foundry.FoundrySDK(SDKConfig(task_spec="A flight agent.", storage=StorageConfig(path=tmp_path)))
    _seed(sdk)

    out = tmp_path / "insights.html"
    path = IntelligenceDashboard().generate(sdk, "agent", str(out))

    assert Path(path).exists()
    text = Path(path).read_text()
    assert "Failure Intelligence" in text
    assert "Failure Modes" in text          # mining panel
    assert "Coverage Heatmap" in text       # coverage panel
    assert "Forecasting" in text            # forecast panel
    assert "AUC" in text                    # forecaster fitted (>=6 labeled traces)


def test_insights_cli_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["insights", "--help"])
    assert result.exit_code == 0
    assert "intelligence dashboard" in result.output.lower()


def test_collect_returns_expected_keys(tmp_path: Path):
    sdk = foundry.FoundrySDK(SDKConfig(task_spec="A flight agent.", storage=StorageConfig(path=tmp_path)))
    _seed(sdk)
    data = IntelligenceDashboard.collect(sdk, "agent")
    assert data["n_traces"] == 16
    assert data["n_failures"] == 8
    assert "mining" in data and "coverage" in data
    assert data["forecast"] is not None


def test_unified_report_includes_intelligence_panels(tmp_path: Path):
    """evoforge report (DashboardGenerator) folds in the intelligence panels."""
    from foundry.dashboard import DashboardGenerator

    sdk = foundry.FoundrySDK(SDKConfig(task_spec="A flight agent.", storage=StorageConfig(path=tmp_path)))
    _seed(sdk)

    out = tmp_path / "report.html"
    path = DashboardGenerator(storage_path=str(tmp_path)).generate(str(out))
    text = Path(path).read_text()

    # Trace-only agent (no evolution history) still appears with intelligence panels.
    assert "Failure Intelligence" in text
    assert "Failure Modes" in text
    assert "Coverage Heatmap" in text
    assert "Forecasting" in text


def test_collect_from_storage_without_sdk(tmp_path: Path):
    sdk = foundry.FoundrySDK(SDKConfig(task_spec="A flight agent.", storage=StorageConfig(path=tmp_path)))
    _seed(sdk)
    data = IntelligenceDashboard.collect_from_storage(str(tmp_path), "agent")
    assert data["n_traces"] == 16
    assert data["forecast"] is not None
