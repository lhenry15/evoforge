"""FailureModeMiner — turn failing traces into ranked, labeled failure modes."""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from typing import Any, Optional

from foundry.trace.schema import FailureMode, TraceRecord
from foundry.mining.clusterer import FailureClusterer
from foundry.mining.schema import (
    FailureExample,
    FailureModeCluster,
    MiningResult,
    fix_type_of,
    severity_of,
)

_LABEL_SYSTEM = """You are a reliability engineer analyzing AI agent failures.
Given a cluster of similar failures, write a concise root-cause label and a one
line suggested fix. Reply with ONLY JSON:
{"root_cause": "<= 12 words", "suggested_fix": "<= 16 words"}"""


class FailureModeMiner:
    """Cluster failures, score them, and (optionally) LLM-label root causes.

    Without an LLM ``pool`` the miner is fully deterministic and uses heuristic
    labels. With a pool, the top clusters get human-readable root-cause labels.
    """

    def __init__(
        self,
        pool: Any = None,
        clusterer: Optional[FailureClusterer] = None,
        max_labeled_clusters: int = 8,
        reclassify: bool = True,
        reclassify_confidence: float = 0.4,
        max_classify_calls: int = 24,
    ) -> None:
        self._pool = pool
        self._clusterer = clusterer or FailureClusterer()
        self._max_labeled = max_labeled_clusters
        self._reclassify = reclassify
        self._reclassify_confidence = reclassify_confidence
        self._max_classify_calls = max_classify_calls
        self.n_reclassified = 0

    def mine(self, traces: list[TraceRecord], agent_name: str = "") -> MiningResult:
        failures = [t for t in traces if t.is_failure and t.failure_signature is not None]
        if not failures:
            return MiningResult(agent_name=agent_name, n_failures=0, n_clusters=0)

        # Re-label vague failures (unknown / low confidence) before clustering so
        # they group by their true mode instead of the stale 'unknown' bucket.
        if self._pool is not None and self._reclassify:
            self._reclassify_failures(failures)

        groups = self._clusterer.cluster(failures)
        clusters = [self._build_cluster(i, g) for i, g in enumerate(groups)]
        clusters.sort(key=lambda c: (c.impact_score, c.size), reverse=True)

        if self._pool is not None:
            for c in clusters[: self._max_labeled]:
                self._label_cluster(c)

        n = len(failures)
        coverage = round(sum(c.size for c in clusters[:10]) / n, 4) if n else 0.0
        mode_dist = Counter(f.failure_signature.mode.value for f in failures)

        return MiningResult(
            agent_name=agent_name,
            n_failures=n,
            n_clusters=len(clusters),
            coverage_top10=coverage,
            mode_distribution=dict(mode_dist),
            clusters=clusters,
            metadata={"n_reclassified": self.n_reclassified},
        )

    def _reclassify_failures(self, failures: list[TraceRecord]) -> None:
        """LLM-classify failures whose mode is unknown or low-confidence."""
        from foundry.mining.llm_classifier import LLMModeClassifier, apply_classification

        classifier = LLMModeClassifier(self._pool, max_calls=self._max_classify_calls)
        for t in failures:
            sig = t.failure_signature
            if sig is None:
                continue
            needs = sig.mode == FailureMode.UNKNOWN or sig.confidence < self._reclassify_confidence
            if not needs:
                continue
            result = classifier.classify(t)
            if result is None:
                continue
            mode, symptom, confidence = result
            apply_classification(t, mode, symptom, confidence)
            self.n_reclassified += 1

    # ── cluster construction ──────────────────────────────────────────

    def _build_cluster(self, idx: int, group: list[TraceRecord]) -> FailureModeCluster:
        size = len(group)
        sig_ids = [t.failure_signature.signature_id for t in group]
        modes = [t.failure_signature.mode for t in group]
        dominant_mode = Counter(modes).most_common(1)[0][0]
        dominant_sig, dominant_sig_count = Counter(sig_ids).most_common(1)[0]

        capability = self._dominant(
            [t.capability for t in group if t.capability], default=None
        )
        symptom = self._dominant(
            [t.failure_signature.symptom for t in group if t.failure_signature.symptom],
            default="",
        )

        cohesion = round(dominant_sig_count / size, 4) if size else 0.0
        reproducibility = self._reproducibility(group)
        stability = round(0.6 * cohesion + 0.4 * reproducibility, 4)
        severity = severity_of(dominant_mode)
        impact = round(size * severity, 4)
        confidence = round(
            sum(t.failure_signature.confidence for t in group) / size, 4
        ) if size else 0.0

        return FailureModeCluster(
            cluster_id=f"fm-{idx:03d}",
            mode=dominant_mode,
            capability=capability,
            label=f"{dominant_mode.value}: {symptom}" if symptom else dominant_mode.value,
            symptom_summary=symptom,
            suggested_fix_type=fix_type_of(dominant_mode),
            signature_ids=sorted(set(sig_ids)),
            trace_ids=[t.trace_id for t in group],
            size=size,
            severity=severity,
            impact_score=impact,
            cohesion=cohesion,
            reproducibility=reproducibility,
            stability=stability,
            confidence=confidence,
            examples=self._examples(group),
        )

    @staticmethod
    def _reproducibility(group: list[TraceRecord]) -> float:
        """Fraction of traces whose input (context_hash) repeats within the cluster."""
        hashes = [t.context_hash for t in group if t.context_hash]
        if not hashes:
            return 0.0
        counts = Counter(hashes)
        repeated = sum(c for c in counts.values() if c > 1)
        return round(repeated / len(hashes), 4)

    @staticmethod
    def _dominant(values: list[Any], default: Any) -> Any:
        if not values:
            return default
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _examples(group: list[TraceRecord], n: int = 3) -> list[FailureExample]:
        examples = []
        for t in group[:n]:
            examples.append(
                FailureExample(
                    trace_id=t.trace_id,
                    trigger=t.failure_signature.trigger,
                    response=(t.final_response or "")[:200],
                    signature_id=t.failure_signature.signature_id,
                )
            )
        return examples

    # ── optional LLM labeling ─────────────────────────────────────────

    def _label_cluster(self, cluster: FailureModeCluster) -> None:
        examples = "\n".join(
            f'- input: "{(e.trigger or "")[:100]}" | response: "{e.response[:100]}"'
            for e in cluster.examples
        )
        prompt = (
            f"Failure mode: {cluster.mode.value}\n"
            f"Capability: {cluster.capability or 'n/a'}\n"
            f"Observed symptom: {cluster.symptom_summary}\n"
            f"Examples:\n{examples}\n\n"
            "Write the root cause and suggested fix."
        )
        try:
            raw = self._pool.generate(prompt, system=_LABEL_SYSTEM, temperature=0, max_tokens=120)
        except Exception:
            return
        if inspect.isawaitable(raw):
            raise RuntimeError(
                "FailureModeMiner received an async LLM pool in sync mode. "
                "Use a synchronous pool (for example, OllamaLLMPool)."
            )
        data = self._parse_json(str(raw))
        if not data:
            return
        root_cause = str(data.get("root_cause", "")).strip()
        suggested_fix = str(data.get("suggested_fix", "")).strip()
        if root_cause:
            cluster.label = root_cause
        if suggested_fix:
            cluster.metadata["suggested_fix"] = suggested_fix

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict[str, Any]]:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else None
        except (json.JSONDecodeError, AttributeError):
            return None
