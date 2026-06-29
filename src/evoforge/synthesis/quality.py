"""QualityGate — filter synthetic examples for novelty, dedup, and correctness.

Quality gates are what keep synthesis from polluting the training/eval corpus.
An example must be (a) not a duplicate of an already-accepted example, (b)
sufficiently novel versus the existing corpus (avoids benchmark contamination),
and (c) judged consistent/correct when a judge pool is available.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional

from evoforge.dedup import CorpusNovelty, near_duplicate_index
from evoforge.synthesis.schema import QualityReport, SyntheticExample


_JUDGE_SYSTEM = """You verify whether a training example correctly fixes a known
agent failure mode. Reply with ONLY a number 0.0-1.0: 1.0 = the response is
correct and clearly avoids the failure mode, 0.0 = incorrect or still failing."""


class QualityGate:
    """Evaluate synthetic examples against novelty, dedup, and consistency gates."""

    def __init__(
        self,
        corpus_instructions: Optional[list[str]] = None,
        judge_pool: Any = None,
        near_dup_threshold: float = 0.85,
        min_novelty: float = 0.2,
        min_consistency: float = 0.5,
        judge_votes: int = 3,
    ) -> None:
        self._novelty = CorpusNovelty(corpus_instructions)
        self._judge_pool = judge_pool
        self._near_dup_threshold = near_dup_threshold
        self._min_novelty = min_novelty
        self._min_consistency = min_consistency
        self._judge_votes = max(1, int(judge_votes))

    def evaluate(
        self,
        example: SyntheticExample,
        accepted: list[SyntheticExample],
    ) -> QualityReport:
        toks_text = example.instruction
        reasons: list[str] = []

        # 1. Dedup vs already-accepted examples.
        accepted_texts = [acc.instruction for acc in accepted]
        dup_idx = near_duplicate_index(toks_text, accepted_texts, self._near_dup_threshold)
        duplicate_of = accepted[dup_idx].id if dup_idx is not None else None

        # 2. Novelty vs existing corpus AND the already-accepted set.
        max_sim_corpus = self._novelty.max_similarity(toks_text)
        max_sim_accepted = CorpusNovelty(accepted_texts).max_similarity(toks_text)
        novelty = round(1.0 - max(max_sim_corpus, max_sim_accepted), 4)

        # 3. Consistency (judge), only when a pool is provided.
        consistency = self._judge(example) if self._judge_pool is not None else 0.7

        accepted_flag = True
        if duplicate_of is not None:
            accepted_flag = False
            reasons.append(f"duplicate of {duplicate_of}")
        if novelty < self._min_novelty:
            accepted_flag = False
            reasons.append(f"low novelty ({novelty:.2f})")
        if self._judge_pool is not None and consistency < self._min_consistency:
            accepted_flag = False
            reasons.append(f"low consistency ({consistency:.2f})")

        confidence = round(0.5 * novelty + 0.5 * consistency, 4)
        return QualityReport(
            accepted=accepted_flag,
            confidence=confidence,
            novelty=novelty,
            consistency=round(consistency, 4),
            duplicate_of=duplicate_of,
            reasons=reasons,
        )

    def _judge(self, example: SyntheticExample) -> float:
        response = example.ideal_response or example.chosen or ""
        prompt = (
            f"FAILURE MODE: {example.target_mode.value}\n"
            f'USER REQUEST: "{example.instruction[:160]}"\n'
            f'CANDIDATE RESPONSE: "{response[:200]}"\n'
            "Does the response correctly handle the request and avoid the failure mode? "
            "End your reply with a single number from 0.0 to 1.0 on its own."
        )
        # A small judge ensemble (median) beats single-shot nondeterminism — some
        # judge endpoints emit no/garbled content under a tiny token budget, so we
        # give room and read the LAST number (reasoning models conclude at the end).
        scores: list[float] = []
        for _ in range(self._judge_votes):
            raw = self._judge_pool.generate(prompt, system=_JUDGE_SYSTEM, temperature=0, max_tokens=64)
            if inspect.isawaitable(raw):
                raise RuntimeError(
                    "QualityGate received an async judge pool in sync mode. "
                    "Use a synchronous pool (for example, OllamaLLMPool)."
                )
            score = self._parse_score(str(raw))
            if score is not None:
                scores.append(score)
        if not scores:
            return 0.5
        scores.sort()
        median = scores[len(scores) // 2]
        return median

    @staticmethod
    def _parse_score(raw: str) -> Optional[float]:
        """Read the LAST 0..1 number in the judge output (verdict comes last)."""
        matches = re.findall(r"[0-9]*\.?[0-9]+", raw)
        if not matches:
            return None
        try:
            return min(1.0, max(0.0, float(matches[-1])))
        except ValueError:
            return None
