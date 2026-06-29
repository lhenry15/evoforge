"""Forecasting models.

- LogisticRisk: a compact numpy logistic-regression for P(failure). Deterministic
  (zero init, fixed iterations), no sklearn dependency at runtime.
- ModeClassifier: multinomial Naive Bayes over input tokens, trained only on
  failing traces, to predict the *likely* failure mode given a new input.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from foundry.forecast.features import tokenize
from foundry.trace.schema import TraceRecord


class LogisticRisk:
    """Binary logistic regression trained with full-batch gradient descent."""

    def __init__(self, lr: float = 0.5, iters: int = 400, l2: float = 1e-3) -> None:
        self.lr = lr
        self.iters = iters
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRisk":
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        for _ in range(self.iters):
            z = X @ self.w + self.b
            p = _sigmoid(z)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = float(np.mean(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.full(X.shape[0], 0.5)
        return _sigmoid(X @ self.w + self.b)

    def predict_one(self, x: np.ndarray) -> float:
        if self.w is None:
            return 0.5
        return float(_sigmoid(float(x @ self.w + self.b)))


class ModeClassifier:
    """Naive Bayes over tokens to predict the likely failure mode of an input."""

    def __init__(self) -> None:
        self._mode_tokens: dict[str, Counter] = defaultdict(Counter)
        self._mode_count: Counter = Counter()
        self._vocab: set[str] = set()
        self.modes: list[str] = []

    def fit(self, failure_traces: list[TraceRecord]) -> "ModeClassifier":
        for t in failure_traces:
            if t.failure_signature is None:
                continue
            mode = t.failure_signature.mode.value
            toks = tokenize(self._text(t))
            self._mode_tokens[mode].update(toks)
            self._mode_count[mode] += 1
            self._vocab.update(toks)
        self.modes = list(self._mode_count.keys())
        return self

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        if not self.modes:
            return "unknown", {}
        toks = tokenize(text)
        vocab_size = len(self._vocab) or 1
        total = sum(self._mode_count.values())

        logp: dict[str, float] = {}
        for mode in self.modes:
            lp = math.log(self._mode_count[mode] / total)
            denom = sum(self._mode_tokens[mode].values()) + vocab_size
            for tok in toks:
                lp += math.log((self._mode_tokens[mode][tok] + 1) / denom)
            logp[mode] = lp

        mx = max(logp.values())
        exps = {m: math.exp(v - mx) for m, v in logp.items()}
        s = sum(exps.values()) or 1.0
        probs = {m: round(exps[m] / s, 4) for m in exps}
        best = max(probs, key=probs.get)
        return best, probs

    @staticmethod
    def _text(trace: TraceRecord) -> str:
        return "\n".join(
            m.content for m in trace.input_messages if getattr(m, "role", "") == "user"
        )


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
