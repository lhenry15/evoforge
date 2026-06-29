"""RiskForecaster — predict failure risk + likely mode for an incoming request.

Ties together the featurizer, the logistic risk model, the Naive-Bayes mode
classifier, optional Platt calibration, and a baseline comparison so we can
prove the model "beats naive baselines" (a Phase 4 exit criterion).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from foundry.forecast.calibration import PlattScaler, brier_score, calibration_report
from foundry.forecast.features import TraceFeaturizer
from foundry.forecast.model import LogisticRisk, ModeClassifier
from foundry.forecast.schema import (
    CalibrationReport,
    Forecast,
    ForecastEvaluation,
    ForecastRequest,
    RiskLevel,
)
from foundry.trace.schema import TraceOutcome, TraceRecord


class RiskForecaster:
    """Train on labeled traces; forecast risk for new requests."""

    def __init__(
        self,
        dim: int = 256,
        calibrate: bool = True,
        medium_threshold: float = 0.33,
        high_threshold: float = 0.66,
    ) -> None:
        self._feat = TraceFeaturizer(dim=dim)
        self._risk = LogisticRisk()
        self._mode = ModeClassifier()
        self._platt: Optional[PlattScaler] = None
        self._calibrate = calibrate
        self._medium = medium_threshold
        self._high = high_threshold
        self._train_size = 0
        self._base_rate = 0.0
        self.fitted = False

    # ── training ──────────────────────────────────────────────────────

    def fit(self, traces: list[TraceRecord]) -> "RiskForecaster":
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]
        if not labeled:
            raise ValueError("RiskForecaster.fit needs labeled traces (eval-derived).")

        self._feat.fit(labeled)
        X = np.vstack([self._feat.transform_trace(t) for t in labeled])
        y = np.array([1.0 if t.is_failure else 0.0 for t in labeled])

        self._risk.fit(X, y)
        self._mode.fit([t for t in labeled if t.is_failure])

        if self._calibrate and len(labeled) >= 4:
            raw = self._risk.predict_proba(X)
            self._platt = PlattScaler().fit(raw, y)

        self._train_size = len(labeled)
        self._base_rate = float(np.mean(y))
        self.fitted = True
        return self

    # ── prediction ────────────────────────────────────────────────────

    def forecast(self, request: Union[ForecastRequest, str], capability: Optional[str] = None) -> Forecast:
        if isinstance(request, str):
            request = ForecastRequest.from_text(request, capability=capability)
        text = request.text
        cap = request.capability or capability

        x = self._feat.transform_request(
            text,
            capability=cap,
            n_tools=len(request.tool_invocations),
            n_failed=sum(1 for t in request.tool_invocations if not t.get("success", True)),
        )
        raw_p = self._risk.predict_one(x)
        p = self._platt.transform_one(raw_p) if self._platt else raw_p

        novelty = self._feat.novelty(text)
        likely_mode, mode_probs = self._mode.predict(text)
        ci = self._confidence_interval(p, novelty)

        return Forecast(
            p_failure=round(float(p), 4),
            risk_level=self._risk_level(p),
            likely_mode=likely_mode,
            mode_probabilities=mode_probs,
            capability=cap,
            confidence_interval=ci,
            novelty=novelty,
            rationale=self._rationale(text, cap, novelty, p),
            metadata={"raw_p": round(float(raw_p), 4), "train_size": self._train_size},
        )

    # ── evaluation vs baselines ───────────────────────────────────────

    def evaluate(self, traces: list[TraceRecord]) -> ForecastEvaluation:
        """Resubstitution evaluation of the already-fitted model (NOT held out).

        This grades the model on data it may have trained on, so the numbers are
        optimistic. Prefer :meth:`cross_validate` for an honest estimate.
        """
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]
        y = np.array([1.0 if t.is_failure else 0.0 for t in labeled])
        if len(y) == 0:
            raise ValueError("No labeled traces to evaluate on.")

        model_p = np.array([self.forecast(self._as_request(t)).p_failure for t in labeled])
        majority_p = np.full(len(y), self._base_rate)
        cap_p = np.array([self._feat.capability_failrate(t.capability) for t in labeled])
        return self._eval_from(model_p, y, majority_p, cap_p, method="resubstitution",
                               n_folds=None, honest=False)

    def cross_validate(self, traces: list[TraceRecord], k: int = 5) -> ForecastEvaluation:
        """Honest stratified k-fold evaluation (each fold predicted out-of-sample).

        Leaves ``self`` fitted on all labeled traces afterward (for serving).
        """
        return self.cross_validate_full(traces, k=k)[0]

    def cross_validate_full(
        self, traces: list[TraceRecord], k: int = 5
    ) -> tuple[ForecastEvaluation, CalibrationReport]:
        """Cross-validated evaluation + calibration computed on the same OOF predictions."""
        p, y, maj, cap, kused, honest, method = self._cv_predictions(traces, k)
        if not p:
            raise ValueError("No labeled traces to evaluate on.")
        ev = self._eval_from(
            np.array(p), np.array(y), np.array(maj), np.array(cap),
            method=method, n_folds=(kused or None), honest=honest,
        )
        cal = calibration_report(np.array(p), np.array(y))
        # Refit on all labeled data so the served model uses everything.
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]
        if labeled:
            self.fit(labeled)
        return ev, cal

    def calibration(
        self,
        traces: list[TraceRecord],
        tolerance: float = 0.1,
        method: str = "resubstitution",
        k: int = 5,
    ) -> CalibrationReport:
        if method == "cv":
            return self.cross_validate_full(traces, k=k)[1]
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]
        y = np.array([1.0 if t.is_failure else 0.0 for t in labeled])
        p = np.array([self.forecast(self._as_request(t)).p_failure for t in labeled])
        return calibration_report(p, y, tolerance=tolerance)

    # ── cross-validation internals ────────────────────────────────────

    def _cv_predictions(
        self, traces: list[TraceRecord], k: int
    ) -> tuple[list[float], list[float], list[float], list[float], int, bool, str]:
        """Out-of-fold predictions. Falls back to resubstitution on tiny data."""
        labeled = [t for t in traces if t.outcome != TraceOutcome.UNKNOWN]
        pos = [t for t in labeled if t.is_failure]
        neg = [t for t in labeled if not t.is_failure]

        kused = min(k, len(pos), len(neg))
        if len(labeled) < 4 or kused < 2:
            # Not enough data / one class too small for honest folds.
            self.fit(labeled)
            p = [self.forecast(self._as_request(t)).p_failure for t in labeled]
            y = [1.0 if t.is_failure else 0.0 for t in labeled]
            maj = [self._base_rate] * len(labeled)
            cap = [self._feat.capability_failrate(t.capability) for t in labeled]
            return p, y, maj, cap, 0, False, "resubstitution"

        # Stratified round-robin folds (deterministic).
        folds: list[list[TraceRecord]] = [[] for _ in range(kused)]
        for group in (pos, neg):
            for i, t in enumerate(group):
                folds[i % kused].append(t)

        p: list[float] = []
        y: list[float] = []
        maj: list[float] = []
        cap: list[float] = []
        for i in range(kused):
            test = folds[i]
            train = [t for j in range(kused) if j != i for t in folds[j]]
            if not train or not test:
                continue
            sub = RiskForecaster(
                dim=self._feat.dim, calibrate=False,
                medium_threshold=self._medium, high_threshold=self._high,
            )
            sub.fit(train)
            for t in test:
                p.append(sub.forecast(sub._as_request(t)).p_failure)
                y.append(1.0 if t.is_failure else 0.0)
                maj.append(sub._base_rate)
                cap.append(sub._feat.capability_failrate(t.capability))
        return p, y, maj, cap, kused, True, "cross_validation"

    def _eval_from(
        self,
        model_p: np.ndarray,
        y: np.ndarray,
        majority_p: np.ndarray,
        cap_p: np.ndarray,
        method: str,
        n_folds: Optional[int],
        honest: bool,
    ) -> ForecastEvaluation:
        model_brier = brier_score(model_p, y)
        majority_brier = brier_score(majority_p, y)
        capability_brier = brier_score(cap_p, y)
        accuracy = float(np.mean((model_p >= 0.5) == (y >= 0.5))) if len(y) else 0.0
        return ForecastEvaluation(
            n=len(y),
            base_rate=round(float(np.mean(y)) if len(y) else 0.0, 4),
            model_brier=round(model_brier, 4),
            majority_brier=round(majority_brier, 4),
            capability_brier=round(capability_brier, 4),
            model_accuracy=round(accuracy, 4),
            model_auc=self._auc(model_p, y),
            beats_majority=model_brier <= majority_brier,
            beats_capability=model_brier <= capability_brier,
            method=method,
            n_folds=n_folds,
            honest=honest,
        )

    # ── internals ─────────────────────────────────────────────────────

    def _as_request(self, trace: TraceRecord) -> ForecastRequest:
        return ForecastRequest(
            messages=[m for m in trace.input_messages if m.role == "user"],
            capability=trace.capability,
        )

    def _risk_level(self, p: float) -> RiskLevel:
        if p >= self._high:
            return RiskLevel.HIGH
        if p >= self._medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _confidence_interval(self, p: float, novelty: float) -> tuple[float, float]:
        # Effective support shrinks with novelty -> wider interval out-of-distribution.
        n_eff = max(1.0, self._train_size * (1.0 - novelty))
        se = (p * (1.0 - p) / n_eff) ** 0.5
        widen = 1.0 + novelty  # inflate for out-of-distribution inputs
        lo = max(0.0, p - 1.96 * se * widen)
        hi = min(1.0, p + 1.96 * se * widen)
        return (round(lo, 4), round(hi, 4))

    def _rationale(self, text: str, capability: Optional[str], novelty: float, p: float) -> list[str]:
        reasons: list[str] = []
        cap_rate = self._feat.capability_failrate(capability)
        if cap_rate > 0.5:
            reasons.append(f"capability '{capability or 'unknown'}' fails often ({cap_rate:.0%})")
        if self._feat.risk_keywords_present(text):
            reasons.append("input contains risk keywords")
        if novelty > 0.5:
            reasons.append(f"input is novel vs training data (novelty {novelty:.2f})")
        if not reasons:
            reasons.append("no strong risk signals" if p < 0.5 else "elevated risk from learned token patterns")
        return reasons

    @staticmethod
    def _auc(probs: np.ndarray, labels: np.ndarray) -> Optional[float]:
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        # Mann-Whitney U statistic / (|pos|*|neg|).
        wins = 0.0
        for pp in pos:
            wins += np.sum(pp > neg) + 0.5 * np.sum(pp == neg)
        return round(float(wins / (len(pos) * len(neg))), 4)
