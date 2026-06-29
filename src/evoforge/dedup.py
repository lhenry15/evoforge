"""Shared novelty + near-duplicate primitives for generated-data quality gates.

Used by both the synthesis QualityGate (training examples) and the coverage
EvalCaseQualityGate (eval messages). Each gate keeps its own policy (judge
consistency, request-shape, thresholds); only the similarity core is shared.
"""

from __future__ import annotations

from typing import Optional

from evoforge.text import jaccard, token_set


class CorpusNovelty:
    """Measure how novel a text is versus a fixed reference corpus."""

    def __init__(self, corpus: list[str] | None = None) -> None:
        self._sets = [token_set(c) for c in (corpus or []) if c]

    def max_similarity(self, text: str) -> float:
        toks = token_set(text)
        if not toks or not self._sets:
            return 0.0
        return max(jaccard(toks, c) for c in self._sets)

    def novelty(self, text: str) -> float:
        """1.0 = entirely novel, 0.0 = identical to a corpus item."""
        return round(1.0 - self.max_similarity(text), 4)


def near_duplicate_index(
    text: str, existing: list[str], threshold: float
) -> Optional[int]:
    """Index of the first near-duplicate in ``existing`` (Jaccard >= threshold), else None."""
    toks = token_set(text)
    for i, e in enumerate(existing):
        if jaccard(toks, token_set(e)) >= threshold:
            return i
    return None
