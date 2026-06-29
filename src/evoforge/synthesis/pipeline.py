"""The ``seed -> conversation -> label`` pipeline as one control surface.

:class:`ScenarioSynthesizer` ties the three synthesis components into the flow you
actually run:

1. **author seeds** — from a natural-language brief
   (:meth:`~ScenarioSynthesizer.seeds_from_brief`) or from mined failure clusters
   (:meth:`~ScenarioSynthesizer.seeds_from_failures`, for corrective conversations);
2. **generate a conversation** per seed (single- or multi-party, decided by the seed); and
3. **honestly label** every turn with a chosen, self-certifiable labeler set.

It returns a :class:`LabeledDataset` you can serialize, convert to eval cases, or feed
to training. This is the surface the ``live_synthesis`` example *and* the evolution
loop drive: **change the seeds and the labelers, get a different dataset** — no
edits to the components themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from evoforge.core.types import EvalCase
from evoforge.mining.schema import FailureModeCluster
from evoforge.synthesis.conversation import ConversationGenerator, SimTranscript
from evoforge.synthesis.labeler import REGISTRY, Labeler, label_transcript
from evoforge.synthesis.seed import Seed, SeedGenerator, SimParticipant


class LabeledDataset(BaseModel):
    """A set of labeled conversations — the unit a synthesis run emits.

    Each conversation is a :class:`~evoforge.synthesis.conversation.SimTranscript`
    whose turns carry whatever labels the chosen labelers attached. Serialize with
    :meth:`write_jsonl` or lower to eval cases with :meth:`to_eval_cases`.
    """

    conversations: list[SimTranscript] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.conversations)

    def __add__(self, other: "LabeledDataset") -> "LabeledDataset":
        merged = LabeledDataset(
            conversations=list(self.conversations) + list(other.conversations),
            metadata={**self.metadata, **other.metadata},
        )
        return merged

    def label_names(self) -> list[str]:
        """All distinct label names present across the dataset's turns."""
        names: set[str] = set()
        for transcript in self.conversations:
            for turn in transcript.turns:
                names.update(turn.labels.keys())
        return sorted(names)

    def to_eval_cases(self) -> list[EvalCase]:
        """Lower every conversation to an :class:`~evoforge.core.types.EvalCase`."""
        cases = [transcript.to_eval_case() for transcript in self.conversations]
        return cases

    def write_jsonl(self, path: Union[str, Path]) -> int:
        """Write one conversation per line (JSONL); returns the number written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for transcript in self.conversations:
                handle.write(transcript.model_dump_json())
                handle.write("\n")
        return len(self.conversations)

    @classmethod
    def read_jsonl(cls, path: Union[str, Path]) -> "LabeledDataset":
        """Read a JSONL file of conversations back into a dataset."""
        transcripts: list[SimTranscript] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                transcripts.append(SimTranscript.model_validate(json.loads(line)))
        dataset = cls(conversations=transcripts)
        return dataset


class ScenarioSynthesizer:
    """High-level ``seed -> conversation -> label`` pipeline.

    Construct once with a (sync) LLM pool, then author seeds and build datasets.
    Labeler selection is open: pass labeler *names* registered in
    :data:`~evoforge.synthesis.labeler.REGISTRY` (author your own schema and
    ``REGISTRY.register_schema(...)`` to extend the set). When ``labelers`` is left
    ``None`` a sensible default is chosen per seed (``avoids_failure`` for a failure
    seed, ``target_presence`` otherwise).
    """

    def __init__(self, pool: Any, votes: int = 3, name: str = "native") -> None:
        self._pool = pool
        self._seedgen = SeedGenerator(pool)
        self._convgen = ConversationGenerator(pool, name=name)
        self._votes = max(1, int(votes))
        self._name = name

    # ── (1) author seeds — the control surface ────────────────────────────────
    def seeds_from_brief(self, brief: str, **kwargs: Any) -> list[Seed]:
        """Author ordinary seeds from a natural-language brief (see ``SeedGenerator.generate``)."""
        seeds = self._seedgen.generate(brief, **kwargs)
        return seeds

    def seeds_from_failures(self, cluster: FailureModeCluster, **kwargs: Any) -> list[Seed]:
        """Author *failure* seeds from a mined cluster — the basis of corrective conversations."""
        seeds = self._seedgen.from_failures(cluster, **kwargs)
        return seeds

    # ── (2)+(3) seed -> conversation -> label ─────────────────────────────────
    def simulate(
        self,
        seed: Seed,
        labelers: Optional[list[str]] = None,
        votes: Optional[int] = None,
        retries: int = 1,
    ) -> SimTranscript:
        """Generate one conversation for ``seed`` and label every turn.

        Retries generation up to ``retries`` times if the model returns too few
        turns (a transient parse/Generation hiccup), then applies the resolved
        labeler set.
        """
        transcript = self._convgen.generate(seed)
        attempt = 0
        while len(transcript.turns) < 2 and attempt < retries:
            transcript = self._convgen.generate(seed)
            attempt += 1
        chosen = self._resolve_labelers(seed, labelers, votes)
        if chosen and transcript.turns:
            label_transcript(transcript, chosen, seed)
        return transcript

    def simulate_many(
        self,
        seeds: list[Seed],
        labelers: Optional[list[str]] = None,
        votes: Optional[int] = None,
        min_turns: int = 2,
    ) -> list[SimTranscript]:
        """Simulate+label many seeds, dropping any that produced fewer than ``min_turns`` turns."""
        transcripts: list[SimTranscript] = []
        for seed in seeds:
            transcript = self.simulate(seed, labelers=labelers, votes=votes)
            if len(transcript.turns) >= min_turns:
                transcripts.append(transcript)
        return transcripts

    def build_dataset(
        self,
        seeds: list[Seed],
        labelers: Optional[list[str]] = None,
        votes: Optional[int] = None,
        min_turns: int = 2,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LabeledDataset:
        """End-to-end: simulate+label ``seeds`` and collect them into a dataset."""
        conversations = self.simulate_many(seeds, labelers=labelers, votes=votes, min_turns=min_turns)
        dataset = LabeledDataset(conversations=conversations, metadata=metadata or {})
        return dataset

    # ── labeler resolution (open set) ─────────────────────────────────────────
    def _resolve_labelers(
        self, seed: Seed, labelers: Optional[list[str]], votes: Optional[int]
    ) -> list[Labeler]:
        names = labelers
        if names is None:
            names = ["avoids_failure"] if seed.is_failure_seed() else ["target_presence"]
        vote_count = votes if votes is not None else self._votes
        resolved = [REGISTRY.create(name, self._pool, votes=vote_count) for name in names]
        return resolved


__all__ = ["LabeledDataset", "ScenarioSynthesizer", "SimParticipant"]
