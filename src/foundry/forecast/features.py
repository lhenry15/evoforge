"""TraceFeaturizer — turn traces/requests into fixed-width numeric features.

Deterministic and dependency-light: hashed token counts plus a few structured
signals (length, risk keywords, tool usage, capability historical failure rate).
Novelty (fraction of out-of-vocabulary tokens) is computed separately and feeds
the forecast's uncertainty, not the model input, to avoid train/serve skew.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from foundry.text import tokenize
from foundry.trace.schema import TraceRecord

_RISK_KEYWORDS = (
    "invalid", "error", "cancel", "refund", "urgent", "asap", "wrong",
    "broken", "fail", "complaint", "immediately", "angry",
)
_N_STRUCTURED = 5


class TraceFeaturizer:
    """Featurize traces and forecast requests into numpy vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._vocab: set[str] = set()
        self._cap_failrate: dict[str, float] = {}
        self._global_failrate: float = 0.0
        self.fitted = False

    @property
    def width(self) -> int:
        return self.dim + _N_STRUCTURED

    def fit(self, traces: list[TraceRecord]) -> "TraceFeaturizer":
        cap_total: Counter = Counter()
        cap_fail: Counter = Counter()
        total = fail = 0
        for t in traces:
            self._vocab.update(tokenize(self._text(t)))
            cap = t.capability or "unknown"
            cap_total[cap] += 1
            total += 1
            if t.is_failure:
                cap_fail[cap] += 1
                fail += 1
        self._global_failrate = fail / total if total else 0.0
        self._cap_failrate = {
            c: cap_fail[c] / cap_total[c] for c in cap_total if cap_total[c]
        }
        self.fitted = True
        return self

    # ── transform ─────────────────────────────────────────────────────

    def transform_trace(self, trace: TraceRecord) -> np.ndarray:
        text = self._text(trace)
        n_tools = len(trace.tool_invocations)
        n_failed = sum(1 for t in trace.tool_invocations if not t.succeeded)
        return self._vectorize(text, trace.capability, n_tools, n_failed)

    def transform_request(
        self, text: str, capability: Any = None, n_tools: int = 0, n_failed: int = 0
    ) -> np.ndarray:
        return self._vectorize(text, capability, n_tools, n_failed)

    def novelty(self, text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 1.0
        unseen = sum(1 for t in toks if t not in self._vocab)
        return round(unseen / len(toks), 4)

    def capability_failrate(self, capability: Any) -> float:
        return self._cap_failrate.get(capability or "unknown", self._global_failrate)

    # ── internals ─────────────────────────────────────────────────────

    def _vectorize(self, text: str, capability: Any, n_tools: int, n_failed: int) -> np.ndarray:
        toks = tokenize(text)
        vec = np.zeros(self.width, dtype=np.float64)
        for tok in toks:
            idx = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        joined = " ".join(toks)
        vec[self.dim + 0] = min(len(toks) / 50.0, 2.0)
        vec[self.dim + 1] = 1.0 if any(q in toks for q in ("what", "how", "why", "when", "can")) else 0.0
        vec[self.dim + 2] = 1.0 if any(k in joined for k in _RISK_KEYWORDS) else 0.0
        vec[self.dim + 3] = min(n_tools / 5.0, 2.0) + min(n_failed / 3.0, 2.0)
        vec[self.dim + 4] = self.capability_failrate(capability)
        return vec

    @staticmethod
    def _text(trace: TraceRecord) -> str:
        return "\n".join(
            m.content for m in trace.input_messages if getattr(m, "role", "") == "user"
        )

    def risk_keywords_present(self, text: str) -> bool:
        joined = " ".join(tokenize(text))
        return any(k in joined for k in _RISK_KEYWORDS)
