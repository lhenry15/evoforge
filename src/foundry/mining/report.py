"""FailureModeReport — render a mining result for humans and the dashboard."""

from __future__ import annotations

from typing import Any

from foundry.mining.schema import MiningResult


class FailureModeReport:
    """Render a :class:`MiningResult` as text or a dashboard-ready dict."""

    def __init__(self, result: MiningResult) -> None:
        self._result = result

    def to_dict(self) -> dict[str, Any]:
        r = self._result
        return {
            "agent_name": r.agent_name,
            "n_failures": r.n_failures,
            "n_clusters": r.n_clusters,
            "coverage_top10": r.coverage_top10,
            "mode_distribution": r.mode_distribution,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "mode": c.mode.value,
                    "capability": c.capability,
                    "label": c.label,
                    "size": c.size,
                    "impact_score": c.impact_score,
                    "stability": c.stability,
                    "cohesion": c.cohesion,
                    "reproducibility": c.reproducibility,
                    "confidence": c.confidence,
                    "suggested_fix_type": c.suggested_fix_type,
                    "suggested_fix": c.metadata.get("suggested_fix"),
                    "signature_ids": c.signature_ids,
                    "examples": [e.model_dump() for e in c.examples],
                }
                for c in r.clusters
            ],
        }

    def to_text(self, top: int = 10) -> str:
        r = self._result
        if r.n_failures == 0:
            return f"No failures mined for '{r.agent_name}'."

        lines = [
            f"Failure-mode report for '{r.agent_name}'",
            f"  failures={r.n_failures}  clusters={r.n_clusters}  "
            f"top10_coverage={r.coverage_top10:.0%}",
            "",
            "  Mode distribution:",
        ]
        for mode, count in sorted(r.mode_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"    {mode:24s} {count}")
        lines.append("")
        lines.append(f"  Top {min(top, len(r.clusters))} failure modes (by impact):")
        for c in r.clusters[:top]:
            lines.append(
                f"    [{c.cluster_id}] {c.mode.value} "
                f"(cap={c.capability or 'n/a'}) "
                f"size={c.size} impact={c.impact_score:.2f} "
                f"stability={c.stability:.2f} fix={c.suggested_fix_type}"
            )
            lines.append(f"        {c.label}")
            fix = c.metadata.get("suggested_fix")
            if fix:
                lines.append(f"        → {fix}")
        return "\n".join(lines)
