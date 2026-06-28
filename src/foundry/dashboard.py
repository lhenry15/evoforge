"""Dashboard — single-agent evolution story with collapsible steps."""

from __future__ import annotations

import json
from pathlib import Path


class DashboardGenerator:
    """
    Generate a focused dashboard showing ONE agent's evolution journey.

    Structure:
      - Agent header (name, current score, model, skills count)
      - Collapsible evolution steps — click to expand each step showing:
        • What happened (eval/prompt change/skill/training)
        • Scores at that point
        • Failures at that point
        • Data used (eval cases, training examples)
      - Current state summary at the top
    
    Only shows agents with AgentEvolutionHistory (real lifecycle tracking).
    """

    def __init__(self, storage_path: str) -> None:
        self._path = Path(storage_path)

    def generate(self, output_path: str) -> str:
        histories = self._load_histories()
        html = self._render(histories)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html)
        return str(Path(output_path).resolve())

    def _load_histories(self) -> dict[str, dict]:
        histories = {}
        agents_dir = self._path / "agents"
        if not agents_dir.exists():
            return histories
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            history_file = agent_dir / "history.json"
            if history_file.exists():
                try:
                    histories[agent_dir.name] = json.loads(history_file.read_text())
                except Exception:
                    pass
        return histories

    def _render(self, histories: dict[str, dict]) -> str:
        if not histories:
            return self._empty_page()

        agents_html = ""
        for agent_name, h in histories.items():
            agents_html += self._render_agent(agent_name, h)

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Foundry — Agent Evolution</title>
<style>{self._css()}</style></head><body>
<header>
<h1>🔥 Foundry — Agent Evolution Dashboard</h1>
<p class="subtitle">Only showing agents with tracked evolution history</p>
</header>
{agents_html}
<script>{self._js()}</script>
</body></html>"""

    def _render_agent(self, agent_name: str, h: dict) -> str:
        events = h.get("events", [])
        snapshots = h.get("snapshots", [])

        # Current state
        last_eval = None
        for e in reversed(events):
            if e["event_type"] == "eval":
                last_eval = e["detail"]
                break
        current_score = last_eval.get("score", 0) if last_eval else 0
        current_caps = last_eval.get("capability_scores", {}) if last_eval else {}
        latest_snap = snapshots[-1] if snapshots else {}
        n_skills = len(latest_snap.get("skill_prompts", {}))
        model = latest_snap.get("model_id", "?")

        # Score trend
        scores = [e["detail"]["score"] for e in events if e["event_type"] == "eval"]
        n_gaps = sum(1 for s in current_caps.values() if s < 0.6)
        n_sat = sum(1 for s in current_caps.values() if s >= 0.85)

        # Current capabilities
        cap_bars = ""
        for cap, score in sorted(current_caps.items(), key=lambda x: x[1]):
            pct = int(score * 100)
            color = "#22c55e" if score >= 0.85 else "#3b82f6" if score >= 0.6 else "#ef4444"
            label = "SATURATING" if score >= 0.85 else "OK" if score >= 0.6 else "GAP"
            cap_bars += f"""<div class="cap-row">
<span class="cap-name">{cap}</span>
<div class="cap-bar-bg"><div class="cap-bar" style="width:{pct}%;background:{color}"></div></div>
<span class="cap-score">{score:.2f}</span>
<span class="badge {'sat' if label=='SATURATING' else 'ok' if label=='OK' else 'gap'}">{label}</span>
</div>"""

        # Build evolution steps (collapsible)
        steps_html = self._render_steps(events, snapshots)

        return f"""
<div class="agent-card">
<div class="agent-header">
  <h2>🤖 {agent_name}</h2>
  <div class="agent-meta">
    <span>Model: <code>{model}</code></span>
    <span>Skills: <b>{n_skills}</b></span>
    <span>Versions: <b>{len(snapshots)}</b></span>
  </div>
</div>
<div class="agent-stats">
  <div class="stat"><span class="stat-value">{current_score:.3f}</span><span class="stat-label">Current Score</span></div>
  <div class="stat bad"><span class="stat-value">{n_gaps}</span><span class="stat-label">Gaps</span></div>
  <div class="stat good"><span class="stat-value">{n_sat}</span><span class="stat-label">Saturating</span></div>
  <div class="stat"><span class="stat-value">{len(scores)}</span><span class="stat-label">Eval Runs</span></div>
</div>
<div class="score-trend">
  <h3>Score Progression</h3>
  <div class="trend-line">{"".join(f'<span class="trend-point" style="color:{self._score_color(s)}">{s:.3f}</span>' for s in scores)}</div>
</div>
<div class="capabilities">
  <h3>Current Capabilities</h3>
  {cap_bars}
</div>
<div class="evolution">
  <h3>Evolution History <span class="dim">(click to expand each step)</span></h3>
  {steps_html}
</div>
</div>"""

    def _render_steps(self, events: list[dict], snapshots: list[dict]) -> str:
        """Render each event as a collapsible step with details."""
        html = ""
        step_num = 0
        for i, e in enumerate(events):
            step_num += 1
            etype = e["event_type"]
            detail = e.get("detail", {})
            timestamp = e.get("timestamp", "")[:16]

            icon = {"eval": "📊", "prompt_change": "✏️", "skill_added": "🧠",
                    "skill_refined": "🔄", "training_complete": "🏋️",
                    "ab_test_passed": "✅", "model_promoted": "🚀",
                    "eval_expanded": "📈", "bootstrap": "🌱",
                    "skill_retired": "💤", "ab_test_failed": "❌"}.get(etype, "•")

            # Title
            title = self._step_title(etype, detail)

            # Expanded content
            content = self._step_content(etype, detail)

            uid = f"step-{i}"
            html += f"""<div class="step">
<div class="step-header" onclick="toggle('{uid}')">
  <span class="step-num">{step_num}</span>
  <span class="step-icon">{icon}</span>
  <span class="step-title">{title}</span>
  <span class="step-time">{timestamp}</span>
  <span class="step-chevron">▸</span>
</div>
<div class="step-body" id="{uid}" style="display:none">{content}</div>
</div>"""
        return html

    def _step_title(self, etype: str, detail: dict) -> str:
        if etype == "bootstrap":
            return f"Bootstrapped {detail.get('n_cases', 0)} eval cases ({', '.join(detail.get('capabilities', []))})"
        if etype == "eval":
            score = detail.get("score", 0)
            n = detail.get("n_total", 0)
            failures = detail.get("failures", [])
            return f"Evaluated: <b>{score:.3f}</b> ({detail.get('n_passed',0)}/{n} passed, {len(failures)} failures)"
        if etype == "prompt_change":
            return f"Prompt evolved: {detail.get('reason', '')[:50]}"
        if etype == "skill_added":
            return f"Skill added: <b>{detail.get('name', '')}</b> (for {detail.get('capability', '')})"
        if etype == "skill_refined":
            return f"Skill refined: <b>{detail.get('name', '')}</b> — {detail.get('reason', '')[:40]}"
        if etype == "eval_expanded":
            return f"Eval expanded: {detail.get('n_old', 0)} → {detail.get('n_new', 0)} cases"
        if etype == "training_complete":
            return f"Training complete: {detail.get('n_examples', 0)} examples, loss={detail.get('train_loss', 0):.4f}"
        if etype == "model_promoted":
            return f"Model promoted: {detail.get('new_model', '')}"
        return etype

    def _step_content(self, etype: str, detail: dict) -> str:
        if etype == "eval":
            # Show capability scores + failures
            caps = detail.get("capability_scores", {})
            failures = detail.get("failures", [])

            caps_html = ""
            for cap, score in sorted(caps.items(), key=lambda x: x[1]):
                color = "#22c55e" if score >= 0.6 else "#ef4444"
                caps_html += f'<div class="mini-cap"><span style="color:{color}">{"✓" if score >= 0.6 else "✗"}</span> {cap}: {score:.2f}</div>'

            failures_html = ""
            if failures:
                failures_html = "<h4>Failures:</h4>"
                for f in failures[:5]:
                    failures_html += f"""<div class="failure-mini">
<span class="dim">[{f.get('case_id', '?')}]</span> <code>{f.get('response', '')[:100]}</code>
<br><span class="dim">{f.get('reasoning', '')}</span></div>"""

            return f'<div class="step-detail"><h4>Capabilities:</h4>{caps_html}{failures_html}</div>'

        if etype == "prompt_change":
            return f"""<div class="step-detail">
<h4>Before:</h4><pre>{detail.get('old', '')}</pre>
<h4>After:</h4><pre>{detail.get('new', '')}</pre>
<p class="dim">Reason: {detail.get('reason', '')}</p></div>"""

        if etype == "skill_added":
            return f"""<div class="step-detail">
<h4>{detail.get('name', '')} (capability: {detail.get('capability', '')})</h4>
<pre>{detail.get('content', '')}</pre></div>"""

        if etype == "skill_refined":
            return f"""<div class="step-detail">
<h4>Old:</h4><pre>{detail.get('old', '')}</pre>
<h4>New:</h4><pre>{detail.get('new', '')}</pre>
<p class="dim">Reason: {detail.get('reason', '')}</p></div>"""

        if etype == "eval_expanded":
            return f"""<div class="step-detail">
<p>Cases: {detail.get('n_old', 0)} → {detail.get('n_new', 0)}</p>
<p>New capabilities tested: {detail.get('capabilities', [])}</p></div>"""

        if etype == "training_complete":
            return f"""<div class="step-detail">
<p>Job: <code>{detail.get('job_id', '')}</code></p>
<p>Examples: {detail.get('n_examples', 0)}</p>
<p>Train loss: {detail.get('train_loss', 0):.4f} | Val loss: {detail.get('val_loss', 0):.4f}</p>
<p>Adapter: <code>{detail.get('adapter_path', '')}</code></p></div>"""

        if etype == "bootstrap":
            return f"""<div class="step-detail">
<p>Generated {detail.get('n_cases', 0)} eval cases</p>
<p>Capabilities: {', '.join(detail.get('capabilities', []))}</p></div>"""

        return f'<div class="step-detail"><pre>{json.dumps(detail, indent=2)[:500]}</pre></div>'

    def _score_color(self, s: float) -> str:
        if s >= 0.85:
            return "#4ade80"
        if s >= 0.6:
            return "#60a5fa"
        if s >= 0.3:
            return "#fbbf24"
        return "#f87171"

    def _empty_page(self) -> str:
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Foundry</title>
<style>{self._css()}</style></head><body>
<header><h1>🔥 Foundry — Agent Evolution Dashboard</h1></header>
<div class="agent-card"><p>No agents with evolution history found.</p>
<p>Run: <code>python tests/e2e/test_single_agent_lifecycle.py</code> to create one.</p></div>
</body></html>"""

    def _js(self) -> str:
        return """
function toggle(id) {
  var el = document.getElementById(id);
  var parent = el.parentElement;
  if (el.style.display === 'none') {
    el.style.display = 'block';
    parent.classList.add('expanded');
  } else {
    el.style.display = 'none';
    parent.classList.remove('expanded');
  }
}"""

    def _css(self) -> str:
        return """
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}
header{border-bottom:1px solid #21262d;margin-bottom:24px;padding-bottom:12px}
h1{color:#58a6ff;margin:0;font-size:24px}
h2{color:#f0f6fc;margin:0 0 8px}
h3{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin:20px 0 8px}
h4{color:#c9d1d9;font-size:13px;margin:10px 0 4px}
.subtitle{color:#484f58;font-size:13px;margin:4px 0 0}
.agent-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:28px;margin:20px 0}
.agent-header{margin-bottom:16px}
.agent-meta{display:flex;gap:16px;font-size:13px;color:#8b949e;margin-top:6px}
.agent-meta code{font-size:12px}
.agent-stats{display:flex;gap:28px;margin:16px 0;flex-wrap:wrap}
.stat{text-align:center;min-width:60px}
.stat-value{display:block;font-size:28px;font-weight:bold;color:#58a6ff}
.stat.bad .stat-value{color:#f87171}
.stat.good .stat-value{color:#4ade80}
.stat-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.3px}
.score-trend{margin:16px 0}
.trend-line{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.trend-point{font-size:16px;font-weight:600}
.trend-point::after{content:' → ';color:#484f58;font-weight:normal;font-size:12px}
.trend-point:last-child::after{content:''}
.capabilities{margin:16px 0}
.cap-row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #21262d}
.cap-name{min-width:140px;font-size:13px}
.cap-bar-bg{flex:1;height:20px;background:#21262d;border-radius:4px;overflow:hidden}
.cap-bar{height:100%;border-radius:4px;transition:width .3s}
.cap-score{font-size:13px;font-weight:600;min-width:36px;text-align:right}
.badge{padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.badge.gap{background:#3d1f1f;color:#f87171}
.badge.ok{background:#1a3a2a;color:#4ade80}
.badge.sat{background:#1a3a1a;color:#86efac}
.dim{color:#484f58}
.evolution{margin-top:20px}
.step{border:1px solid #21262d;border-radius:6px;margin:6px 0;overflow:hidden}
.step.expanded{border-color:#30363d}
.step-header{display:flex;align-items:center;gap:8px;padding:10px 14px;cursor:pointer;background:#0d1117}
.step-header:hover{background:#161b22}
.step-num{font-size:11px;color:#484f58;min-width:20px}
.step-icon{font-size:14px}
.step-title{flex:1;font-size:13px}
.step-time{font-size:11px;color:#484f58}
.step-chevron{color:#484f58;transition:transform .2s}
.step.expanded .step-chevron{transform:rotate(90deg)}
.step-body{padding:14px;background:#161b22;border-top:1px solid #21262d}
.step-detail{font-size:13px}
.mini-cap{padding:3px 0}
.failure-mini{background:#1a1a2e;border-left:2px solid #f87171;padding:8px;margin:6px 0;border-radius:3px;font-size:12px}
pre{background:#0d1117;padding:10px;border-radius:4px;font-size:12px;overflow-x:auto;white-space:pre-wrap;margin:4px 0;color:#79c0ff}
code{background:#21262d;padding:2px 5px;border-radius:3px;font-size:12px;color:#79c0ff}
"""
