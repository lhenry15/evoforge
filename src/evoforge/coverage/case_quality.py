"""Quality gate for generated eval cases.

Ensures expansion data is *good*, not just valid: non-empty, sufficiently long,
genuinely a user request, de-duplicated (exact + near), and novel versus the
existing benchmark (so expansion doesn't just re-add cases the agent already sees).
"""

from __future__ import annotations

from evoforge.dedup import CorpusNovelty, near_duplicate_index

# Markers that a string is an actual user request, not a meta-description.
_REQUEST_MARKERS = (
    "?", "book", "find", "search", "cancel", "change", "need", "want", "can you",
    "could you", "please", "help", "i'd like", "i would like", "show", "get me",
    "look", "reserve", "check", "require", "trying to", "i'm looking", "would like",
    "set up", "arrange", "schedule",
)


class EvalCaseQualityGate:
    """Filter generated user messages for quality, dedup, and novelty."""

    def __init__(
        self,
        existing_messages: list[str] | None = None,
        min_chars: int = 15,
        near_dup_threshold: float = 0.8,
        min_novelty: float = 0.25,
    ) -> None:
        self._novelty = CorpusNovelty(existing_messages)
        self._min_chars = min_chars
        self._near_dup_threshold = near_dup_threshold
        self._min_novelty = min_novelty

    def accept(self, message: str, accepted: list[str]) -> tuple[bool, str]:
        """Return (accepted, reason). Reason is empty when accepted."""
        msg = (message or "").strip()
        if len(msg) < self._min_chars:
            return False, "too short"
        if not self._looks_like_request(msg):
            return False, "not a user request"

        # Dedup against already-accepted messages in this batch.
        if near_duplicate_index(msg, accepted, self._near_dup_threshold) is not None:
            return False, "near-duplicate of accepted case"

        # Novelty vs the existing benchmark corpus.
        novelty = self._novelty.novelty(msg)
        if novelty < self._min_novelty:
            return False, f"low novelty vs existing ({novelty:.2f})"

        return True, ""

    def filter(self, messages: list[str]) -> list[str]:
        accepted: list[str] = []
        for m in messages:
            ok, _ = self.accept(m, accepted)
            if ok:
                accepted.append(m.strip())
        return accepted

    @staticmethod
    def _looks_like_request(message: str) -> bool:
        low = message.lower()
        return any(marker in low for marker in _REQUEST_MARKERS)
