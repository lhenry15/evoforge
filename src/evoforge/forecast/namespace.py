"""ForecastNamespace — sdk.forecast interface.

Fit a risk forecaster on an agent's recorded traces and predict failure risk for
new requests before they run. Forecasters are cached per agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from evoforge.core.sdk import FoundrySDK

from evoforge.forecast.drift import DriftMonitor
from evoforge.forecast.forecaster import RiskForecaster
from evoforge.forecast.schema import (
    CalibrationReport,
    DriftReport,
    Forecast,
    ForecastEvaluation,
    ForecastRequest,
)
from evoforge.trace.schema import TraceRecord


class ForecastNamespace:
    """sdk.forecast — predict failures before they happen.

    Usage::

        sdk.forecast.fit("my_agent")
        f = sdk.forecast.predict("my_agent", "cancel my booking ASAP, this is broken")
        print(f.p_failure, f.likely_mode, f.risk_level)

        sdk.forecast.evaluate("my_agent")     # vs naive baselines
        sdk.forecast.drift("my_agent", recent_traces)
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk
        self._models: dict[str, RiskForecaster] = {}

    # ── training ──────────────────────────────────────────────────────

    def fit(self, agent_name: str, traces: Optional[list[TraceRecord]] = None) -> RiskForecaster:
        traces = traces if traces is not None else self._sdk.trace.load(agent_name)
        forecaster = RiskForecaster().fit(traces)
        self._models[agent_name] = forecaster
        return forecaster

    def _get(self, agent_name: str) -> RiskForecaster:
        if agent_name not in self._models:
            self.fit(agent_name)
        return self._models[agent_name]

    # ── prediction ────────────────────────────────────────────────────

    def predict(
        self,
        agent_name: str,
        request: Union[ForecastRequest, str],
        capability: Optional[str] = None,
    ) -> Forecast:
        return self._get(agent_name).forecast(request, capability=capability)

    # ── diagnostics ───────────────────────────────────────────────────

    def evaluate(
        self, agent_name: str, traces: Optional[list[TraceRecord]] = None
    ) -> ForecastEvaluation:
        traces = traces if traces is not None else self._sdk.trace.load(agent_name)
        return self._get(agent_name).evaluate(traces)

    def cross_validate(
        self, agent_name: str, traces: Optional[list[TraceRecord]] = None, k: int = 5
    ) -> ForecastEvaluation:
        """Honest k-fold evaluation; (re)fits and caches the full model."""
        traces = traces if traces is not None else self._sdk.trace.load(agent_name)
        forecaster = RiskForecaster()
        ev = forecaster.cross_validate(traces, k=k)
        self._models[agent_name] = forecaster
        return ev

    def calibration(
        self, agent_name: str, traces: Optional[list[TraceRecord]] = None, tolerance: float = 0.1
    ) -> CalibrationReport:
        traces = traces if traces is not None else self._sdk.trace.load(agent_name)
        return self._get(agent_name).calibration(traces, tolerance=tolerance)

    def drift(
        self,
        agent_name: str,
        recent_traces: list[TraceRecord],
        reference_traces: Optional[list[TraceRecord]] = None,
    ) -> DriftReport:
        reference = reference_traces if reference_traces is not None else self._sdk.trace.load(agent_name)
        monitor = DriftMonitor().fit(reference)
        return monitor.compare(recent_traces)
