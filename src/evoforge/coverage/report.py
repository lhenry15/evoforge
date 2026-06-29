"""CoverageReport — render a coverage map as a heatmap (text + dashboard dict)."""

from __future__ import annotations

from typing import Any

from evoforge.coverage.schema import CoverageMap


class CoverageReport:
    """Render a :class:`CoverageMap` for humans and the dashboard."""

    def __init__(self, coverage_map: CoverageMap) -> None:
        self._map = coverage_map

    def to_dict(self) -> dict[str, Any]:
        m = self._map
        return {
            "agent_name": m.agent_name,
            "n_eval_cases": m.n_eval_cases,
            "n_tagged_cases": m.n_tagged_cases,
            "coverage_ratio": m.coverage_ratio(),
            "capabilities": m.capabilities,
            "modes": m.modes,
            "matrix": m.matrix(),
            "blindspots": [b.model_dump() for b in m.blindspots()],
        }

    def to_text(self) -> str:
        m = self._map
        blindspots = m.blindspots()
        lines = [
            f"Coverage map for '{m.agent_name}'",
            f"  eval_cases={m.n_eval_cases} tagged={m.n_tagged_cases} "
            f"coverage_ratio={m.coverage_ratio():.0%}",
            "",
            "  Heatmap (observed failures / eval cases; * = blind spot):",
        ]
        lines.append(self._heatmap_text())
        lines.append("")
        if blindspots:
            lines.append(f"  Blind spots ({len(blindspots)}, by impact):")
            for b in blindspots:
                lines.append(
                    f"    ! {b.capability} / {b.mode}: "
                    f"{b.observed_failures} real failures, 0 eval cases "
                    f"(impact={b.impact:.2f}, suggest {b.suggested_cases} cases)"
                )
        else:
            lines.append("  No blind spots — every observed failure mode is probed.")
        return "\n".join(lines)

    def _heatmap_text(self) -> str:
        m = self._map
        if not m.capabilities or not m.modes:
            return "    (no data)"
        matrix = m.matrix()
        mode_width = max((len(mode) for mode in m.modes), default=4)
        cap_width = max((len(cap) for cap in m.capabilities), default=10)

        header = " " * (cap_width + 6) + "  ".join(mode.ljust(mode_width) for mode in m.modes)
        rows = [header]
        for cap in m.capabilities:
            cells = []
            for mode in m.modes:
                info = matrix.get(cap, {}).get(mode)
                if not info:
                    cells.append("-".ljust(mode_width))
                    continue
                mark = "*" if info["blindspot"] else " "
                cells.append(f"{info['observed']}/{info['eval_cases']}{mark}".ljust(mode_width))
            rows.append("    " + cap.ljust(cap_width + 2) + "  ".join(cells))
        return "\n".join(rows)
