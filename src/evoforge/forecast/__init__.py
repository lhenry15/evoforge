"""Failure forecasting.

Phase 4 of the predictive loop: predict whether a *new* request will fail —
before the user is affected — including the likely failure mode, a calibrated
probability, and an uncertainty interval. Also provides baseline comparison,
calibration, and drift monitoring.

Public surface::

    from evoforge.forecast import (
        RiskForecaster, ForecastRequest, Forecast, RiskLevel,
        ForecastEvaluation, CalibrationReport, DriftReport, DriftMonitor,
        TraceFeaturizer, LogisticRisk, ModeClassifier,
    )
"""

from evoforge.forecast.schema import (
    CalibrationReport,
    DriftReport,
    Forecast,
    ForecastEvaluation,
    ForecastRequest,
    RiskLevel,
)
from evoforge.forecast.features import TraceFeaturizer
from evoforge.forecast.model import LogisticRisk, ModeClassifier
from evoforge.forecast.drift import DriftMonitor
from evoforge.forecast.forecaster import RiskForecaster

__all__ = [
    "CalibrationReport",
    "DriftReport",
    "Forecast",
    "ForecastEvaluation",
    "ForecastRequest",
    "RiskLevel",
    "TraceFeaturizer",
    "LogisticRisk",
    "ModeClassifier",
    "DriftMonitor",
    "RiskForecaster",
]
