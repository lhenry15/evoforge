"""Intelligence dashboard — visualize the predictive loop's outputs.

Renders a single self-contained HTML page with three panels:
  1. Failure-mode mining (ranked modes, impact/stability, mode distribution)
  2. Coverage heatmap (capability x failure-mode, blind spots highlighted)
  3. Forecasting (model vs baselines, calibration)

It reads only persisted traces via the SDK namespaces, so it works offline.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

from foundry.coverage.report import CoverageReport
from foundry.mining.report import FailureModeReport
from foundry.trace.schema import TraceOutcome


class IntelligenceDashboard:
    """Build the failure-intelligence dashboard from recorded traces."""

    # ── data collection ───────────────────────────────────────────────

    @staticmethod
    def collect(sdk: Any, agent_name: str) -> dict[str, Any]:
        """Collect dashboard data via an SDK instance."""
        return IntelligenceDashboard.collect_from_storage(
            str(sdk.config.storage.path), agent_name
        )

    @staticmethod
    def collect_from_storage(storage_path: str, agent_name: str) -> dict[str, Any]:
        """Collect dashboard data directly from persisted traces (no SDK needed)."""
        from foundry.coverage.mapper import CoverageMapper
        from foundry.forecast.forecaster import RiskForecaster
        from foundry.mining.miner import FailureModeMiner
        from foundry.trace.store import TraceStore

        store = TraceStore(storage_path)
        traces = store.load(agent_name)
        failures = [t for t in traces if t.is_failure]
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]

        mining = FailureModeMiner().mine(traces, agent_name=agent_name)
        eval_cases = IntelligenceDashboard._load_eval_cases(storage_path)
        coverage = CoverageMapper().build(eval_cases, mining, agent_name=agent_name)

        data: dict[str, Any] = {
            "agent_name": agent_name,
            "n_traces": len(traces),
            "n_failures": len(failures),
            "n_labeled": len(labeled),
            "recurrence_rate": store.recurrence_rate(agent_name),
            "mining": FailureModeReport(mining).to_dict(),
            "coverage": CoverageReport(coverage).to_dict(),
            "forecast": None,
        }

        if len(labeled) >= 6:
            try:
                fc = RiskForecaster()
                evaluation, calibration = fc.cross_validate_full(traces, k=5)
                data["forecast"] = {
                    "evaluation": evaluation.model_dump(),
                    "calibration": calibration.model_dump(),
                }
            except Exception:
                data["forecast"] = None
        return data

    @staticmethod
    def _load_eval_cases(storage_path: str) -> list:
        """Load all persisted eval cases (any tag) to measure coverage supply."""
        import json as _json

        from foundry.core.types import EvalCase

        cases: list = []
        eval_dir = Path(storage_path) / "eval_cases"
        if not eval_dir.exists():
            return cases
        for f in eval_dir.glob("*.json"):
            try:
                for item in _json.loads(f.read_text()):
                    cases.append(EvalCase(**item))
            except Exception:
                continue
        return cases

    def generate(self, sdk: Any, agent_name: str, output_path: str) -> str:
        data = self.collect(sdk, agent_name)
        return self.write(data, output_path)

    def write(self, data: dict[str, Any], output_path: str) -> str:
        html_text = self.render(data)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_text)
        return str(path.resolve())

    # ── rendering ─────────────────────────────────────────────────────

    def render(self, data: dict[str, Any]) -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EvoForge — Failure Intelligence</title>
<style>{self._css()}</style></head><body>
<header>
  <h1>🔮 EvoForge — Failure Intelligence</h1>
  <p class="subtitle">Agent: <code>{html.escape(str(data.get('agent_name', '')))}</code></p>
</header>
{self.panels(data)}
</body></html>"""

    def panels(self, data: dict[str, Any]) -> str:
        """Render just the intelligence panels (no <html> wrapper) for embedding."""
        return f"""{self._overview(data)}
{self._mining_panel(data.get('mining', {}))}
{self._coverage_panel(data.get('coverage', {}))}
{self._forecast_panel(data.get('forecast'))}"""

    def _overview(self, data: dict[str, Any]) -> str:
        cards = [
            ("Traces", data.get("n_traces", 0), ""),
            ("Failures", data.get("n_failures", 0), "bad"),
            ("Recurrence", f"{data.get('recurrence_rate', 0):.0%}", "bad"),
            ("Coverage", f"{data.get('coverage', {}).get('coverage_ratio', 0):.0%}", "good"),
        ]
        fc = data.get("forecast")
        if fc:
            auc = fc["evaluation"].get("model_auc")
            cards.append(("Forecast AUC", f"{auc:.2f}" if auc is not None else "—", "good"))
        cells = "".join(
            f'<div class="stat {cls}"><span class="stat-value">{val}</span>'
            f'<span class="stat-label">{label}</span></div>'
            for label, val, cls in cards
        )
        return f'<div class="card"><div class="agent-stats">{cells}</div></div>'

    def _mining_panel(self, mining: dict[str, Any]) -> str:
        clusters = mining.get("clusters", [])
        if not clusters:
            return '<div class="card"><h2>Failure Modes</h2><p class="dim">No failures recorded yet.</p></div>'

        dist = mining.get("mode_distribution", {})
        bars = ""
        max_count = max(dist.values()) if dist else 1
        for mode, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = int(100 * count / max_count)
            bars += (
                f'<div class="cap-row"><span class="cap-name">{html.escape(mode)}</span>'
                f'<div class="cap-bar-bg"><div class="cap-bar" style="width:{pct}%;background:#f0883e"></div></div>'
                f'<span class="cap-score">{count}</span></div>'
            )

        rows = ""
        for c in clusters[:10]:
            fix = c.get("suggested_fix") or ""
            rows += (
                "<tr>"
                f'<td><span class="pill">{html.escape(c["mode"])}</span></td>'
                f'<td>{html.escape(str(c.get("capability") or "n/a"))}</td>'
                f'<td>{c["size"]}</td>'
                f'<td>{c["impact_score"]:.2f}</td>'
                f'<td>{self._bar(c.get("stability", 0))}</td>'
                f'<td><code>{html.escape(c["suggested_fix_type"])}</code></td>'
                f'<td class="label">{html.escape(c.get("label", ""))}'
                f'{("<br><span class=dim>→ " + html.escape(fix) + "</span>") if fix else ""}</td>'
                "</tr>"
            )

        return f"""<div class="card">
  <h2>🧩 Failure Modes <span class="dim">(top {min(10, len(clusters))} by impact, {mining.get('n_failures', 0)} failures)</span></h2>
  <h3>Mode distribution</h3>{bars}
  <h3>Ranked failure modes</h3>
  <table><thead><tr><th>Mode</th><th>Capability</th><th>Size</th><th>Impact</th><th>Stability</th><th>Fix</th><th>Root cause</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>"""

    def _coverage_panel(self, coverage: dict[str, Any]) -> str:
        caps = coverage.get("capabilities", [])
        modes = coverage.get("modes", [])
        matrix = coverage.get("matrix", {})
        blindspots = coverage.get("blindspots", [])
        if not caps or not modes:
            return '<div class="card"><h2>Coverage</h2><p class="dim">No coverage data yet.</p></div>'

        header = "".join(f"<th>{html.escape(m)}</th>" for m in modes)
        body = ""
        for cap in caps:
            cells = ""
            for mode in modes:
                info = matrix.get(cap, {}).get(mode)
                if not info:
                    cells += '<td class="cell empty">·</td>'
                    continue
                cls = "blind" if info["blindspot"] else ("under" if info.get("undercovered") else "ok")
                cells += (
                    f'<td class="cell {cls}" title="observed={info["observed"]} eval={info["eval_cases"]}">'
                    f'{info["observed"]}/{info["eval_cases"]}</td>'
                )
            body += f"<tr><th class='rowhead'>{html.escape(cap)}</th>{cells}</tr>"

        spots = ""
        for b in blindspots:
            spots += (
                f'<li><b>{html.escape(b["capability"])} / {html.escape(b["mode"])}</b>: '
                f'{b["observed_failures"]} real failures, 0 eval cases '
                f'(impact {b["impact"]:.2f} — suggest {b["suggested_cases"]} cases)</li>'
            )
        spots_html = f"<h3>Blind spots</h3><ul class='blindspots'>{spots}</ul>" if spots else \
            "<p class='dim'>No blind spots — every observed failure mode is probed.</p>"

        return f"""<div class="card">
  <h2>🗺️ Coverage Heatmap <span class="dim">(observed failures / targeted eval cases)</span></h2>
  <p class="dim">coverage_ratio = {coverage.get('coverage_ratio', 0):.0%} &nbsp;·&nbsp;
     <span class="legend ok">covered</span>
     <span class="legend under">under-covered</span>
     <span class="legend blind">blind spot</span></p>
  <table class="heatmap"><thead><tr><th></th>{header}</tr></thead><tbody>{body}</tbody></table>
  {spots_html}
</div>"""

    def _forecast_panel(self, forecast: Optional[dict[str, Any]]) -> str:
        if not forecast:
            return ('<div class="card"><h2>Forecasting</h2>'
                    '<p class="dim">Need ≥6 labeled traces to fit a forecaster. Run more evals.</p></div>')

        ev = forecast["evaluation"]
        cal = forecast["calibration"]
        # Lower Brier is better — invert for bar width.
        def briar_bar(v, color):
            width = int(max(0, min(100, (1 - v) * 100)))
            return f'<div class="cap-bar-bg"><div class="cap-bar" style="width:{width}%;background:{color}"></div></div>'

        beats = "✅" if ev.get("beats_capability") else "⚠️"
        rows = [
            ("Model", ev["model_brier"], "#4ade80"),
            ("Capability baseline", ev["capability_brier"], "#f0883e"),
            ("Majority baseline", ev["majority_brier"], "#8b949e"),
        ]
        bars = ""
        for label, v, color in rows:
            bars += (
                f'<div class="cap-row"><span class="cap-name">{label}</span>'
                f'{briar_bar(v, color)}<span class="cap-score">{v:.3f}</span></div>'
            )

        bins = cal.get("bins", [])
        rel = ""
        for b in bins:
            rel += (
                f'<div class="rel-row"><span class="dim">[{b["lower"]:.1f}–{b["upper"]:.1f}]</span> '
                f'pred {b["mean_predicted"]:.2f} vs obs {b["observed_rate"]:.2f} (n={b["n"]})</div>'
            )

        if ev.get("honest"):
            method_label = (
                f'<span class="legend ok">✓ {ev.get("n_folds", "k")}-fold cross-validated '
                f'(held-out, honest)</span>'
            )
        else:
            method_label = (
                '<span class="legend blind">⚠ resubstitution '
                '(evaluated on training data — optimistic)</span>'
            )

        return f"""<div class="card">
  <h2>🔮 Forecasting <span class="dim">(P(failure) before running)</span></h2>
  <p class="dim">Evaluation: {method_label} &nbsp;·&nbsp; n={ev.get('n', 0)}</p>
  <div class="agent-stats">
    <div class="stat good"><span class="stat-value">{ev.get('model_auc') if ev.get('model_auc') is not None else '—'}</span><span class="stat-label">AUC</span></div>
    <div class="stat"><span class="stat-value">{ev['model_accuracy']:.2f}</span><span class="stat-label">Accuracy</span></div>
    <div class="stat"><span class="stat-value">{ev['base_rate']:.2f}</span><span class="stat-label">Base rate</span></div>
    <div class="stat {'good' if cal.get('within_tolerance') else 'bad'}"><span class="stat-value">{cal['ece']:.3f}</span><span class="stat-label">Calib. ECE</span></div>
  </div>
  <h3>Brier score vs baselines {beats} <span class="dim">(longer bar = better)</span></h3>{bars}
  <h3>Reliability</h3>{rel or "<p class='dim'>(insufficient bins)</p>"}
</div>"""

    @staticmethod
    def _bar(value: float) -> str:
        pct = int(max(0.0, min(1.0, value)) * 100)
        return (f'<div class="mini-bar-bg"><div class="mini-bar" style="width:{pct}%"></div></div>'
                f'<span class="dim">{value:.2f}</span>')

    def css(self) -> str:
        """Public accessor for the panel CSS (used when embedding in other reports)."""
        return self._css()

    def _css(self) -> str:
        return """
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:980px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}
header{border-bottom:1px solid #21262d;margin-bottom:20px;padding-bottom:12px}
h1{color:#a371f7;margin:0;font-size:24px}
h2{color:#f0f6fc;margin:0 0 12px;font-size:18px}
h3{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:18px 0 8px}
.subtitle{color:#484f58;font-size:13px;margin:4px 0 0}
.card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:24px;margin:18px 0}
.agent-stats{display:flex;gap:28px;flex-wrap:wrap}
.stat{text-align:center;min-width:70px}
.stat-value{display:block;font-size:26px;font-weight:bold;color:#58a6ff}
.stat.bad .stat-value{color:#f87171}
.stat.good .stat-value{color:#4ade80}
.stat-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.3px}
.cap-row{display:flex;align-items:center;gap:10px;padding:5px 0}
.cap-name{min-width:170px;font-size:13px}
.cap-bar-bg{flex:1;height:16px;background:#21262d;border-radius:4px;overflow:hidden}
.cap-bar{height:100%;border-radius:4px}
.cap-score{font-size:13px;font-weight:600;min-width:42px;text-align:right}
.dim{color:#6e7681}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase}
td.label{font-size:12px}
.pill{background:#1f2937;color:#f0883e;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
code{background:#21262d;padding:2px 6px;border-radius:3px;font-size:12px;color:#79c0ff}
.mini-bar-bg{display:inline-block;width:60px;height:8px;background:#21262d;border-radius:3px;overflow:hidden;vertical-align:middle;margin-right:6px}
.mini-bar{height:100%;background:#58a6ff}
.heatmap td.cell{text-align:center;font-weight:600;border:1px solid #0d1117}
.heatmap .rowhead{color:#c9d1d9}
.cell.ok{background:#16331f;color:#4ade80}
.cell.under{background:#3a2a14;color:#f0b429}
.cell.blind{background:#3d1f1f;color:#f87171;outline:1px solid #f87171}
.cell.empty{background:#11161d;color:#30363d}
.legend{padding:1px 8px;border-radius:8px;font-size:11px;margin-left:6px}
.legend.ok{background:#16331f;color:#4ade80}
.legend.under{background:#3a2a14;color:#f0b429}
.legend.blind{background:#3d1f1f;color:#f87171}
.blindspots{margin:6px 0 0;padding-left:18px;font-size:13px}
.blindspots li{margin:4px 0;color:#f0a0a0}
.rel-row{font-size:12px;padding:2px 0}
"""
