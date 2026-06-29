"""FailureClusterer — group failing traces into coherent clusters.

Strategy (deterministic, dependency-light by default):
  1. Group failures by their deterministic ``signature_id`` (stable base clusters).
  2. Merge base clusters that share a :class:`FailureMode` and have highly
     similar symptom/trigger text, so the same root cause does not fragment.

An optional ``embedder`` (callable: list[str] -> list[vector]) can be injected
for semantic similarity; otherwise a lightweight token-Jaccard similarity is
used so the package works with zero heavy dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Optional

from foundry.text import jaccard as _jaccard
from foundry.text import token_set as _tokens
from foundry.trace.schema import TraceRecord


def _cosine(a: Any, b: Any) -> float:
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
    except TypeError:
        return 0.0


class FailureClusterer:
    """Cluster failing traces into groups representing the same failure mode."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        embedder: Optional[Callable[[list[str]], list[Any]]] = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._embedder = embedder

    def cluster(self, failures: list[TraceRecord]) -> list[list[TraceRecord]]:
        """Return clusters (lists of traces). Deterministic given the same input."""
        signed = [f for f in failures if f.failure_signature is not None]
        if not signed:
            return []

        # 1. Base clusters by signature_id (deterministic order by id).
        by_sig: dict[str, list[TraceRecord]] = defaultdict(list)
        for f in signed:
            by_sig[f.failure_signature.signature_id].append(f)
        base_groups = [by_sig[k] for k in sorted(by_sig.keys())]

        # 2. Greedy merge of base groups that share a mode and similar text.
        return self._merge(base_groups)

    def _merge(self, groups: list[list[TraceRecord]]) -> list[list[TraceRecord]]:
        texts = [self._group_text(g) for g in groups]
        embeddings = self._maybe_embed(texts)
        token_sets = [_tokens(t) for t in texts]

        clusters: list[list[TraceRecord]] = []
        cluster_mode: list[str] = []
        cluster_tokens: list[set[str]] = []
        cluster_rep: list[int] = []  # representative group index (for embedding sim)

        for idx, group in enumerate(groups):
            mode = group[0].failure_signature.mode.value

            best_i, best_sim = -1, 0.0
            for i in range(len(clusters)):
                if cluster_mode[i] != mode:
                    continue
                if embeddings is not None:
                    sim = _cosine(embeddings[idx], embeddings[cluster_rep[i]])
                else:
                    sim = _jaccard(token_sets[idx], cluster_tokens[i])
                if sim > best_sim:
                    best_sim, best_i = sim, i

            if best_i >= 0 and best_sim >= self._threshold:
                clusters[best_i].extend(group)
                cluster_tokens[best_i] |= token_sets[idx]
            else:
                clusters.append(list(group))
                cluster_mode.append(mode)
                cluster_tokens.append(set(token_sets[idx]))
                cluster_rep.append(idx)

        return clusters

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _group_text(group: list[TraceRecord]) -> str:
        sig = group[0].failure_signature
        parts = [sig.symptom or "", sig.trigger or ""]
        parts.extend(e for e in sig.evidence[:2])
        return " ".join(parts)

    def _maybe_embed(self, texts: list[str]) -> Optional[list[Any]]:
        if self._embedder is None:
            return None
        try:
            return list(self._embedder(texts))
        except Exception:
            return None
