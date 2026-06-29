"""Calibration utilities — Brier score, ECE, reliability bins, Platt scaling."""

from __future__ import annotations

import numpy as np

from evoforge.forecast.schema import CalibrationBin, CalibrationReport


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[float, list[CalibrationBin]]:
    """ECE plus per-bin reliability stats."""
    if len(probs) == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_pred = float(np.mean(probs[mask]))
        observed = float(np.mean(labels[mask]))
        ece += (count / n) * abs(mean_pred - observed)
        bins.append(
            CalibrationBin(
                lower=round(float(lo), 4),
                upper=round(float(hi), 4),
                n=count,
                mean_predicted=round(mean_pred, 4),
                observed_rate=round(observed, 4),
            )
        )
    return round(ece, 4), bins


def calibration_report(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, tolerance: float = 0.1
) -> CalibrationReport:
    ece, bins = expected_calibration_error(probs, labels, n_bins)
    return CalibrationReport(
        n=len(probs),
        brier=round(brier_score(probs, labels), 4),
        ece=ece,
        bins=bins,
        within_tolerance=ece <= tolerance,
    )


class PlattScaler:
    """1-D logistic calibration mapping raw probabilities to calibrated ones."""

    def __init__(self, lr: float = 0.1, iters: int = 500) -> None:
        self.a = 1.0
        self.b = 0.0
        self.lr = lr
        self.iters = iters
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        # Operate in logit space for stability.
        eps = 1e-6
        logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
        a, b = 1.0, 0.0
        for _ in range(self.iters):
            z = a * logits + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = p - labels
            grad_a = float(np.mean(err * logits))
            grad_b = float(np.mean(err))
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a, self.b = a, b
        self._fitted = True
        return self

    def transform_one(self, prob: float) -> float:
        if not self._fitted:
            return prob
        eps = 1e-6
        logit = np.log(min(max(prob, eps), 1 - eps) / min(max(1 - prob, eps), 1 - eps))
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))
