"""Native multi-party simulator — the eval-seam adapter over the synthesis pipeline.

This is the EvoForge-native conversation simulator: it produces labeled, single- **or**
multi-party transcripts purely from EvoForge's own synthesis components, with no
optional dependency. It is a thin adapter that delegates the actual
``seed -> conversation -> label`` work to
:class:`~evoforge.synthesis.pipeline.ScenarioSynthesizer` (the single source of
truth) and exposes it through the eval seam's
:class:`~evoforge.eval.simulator.ConversationSimulator` protocol, so it is a drop-in
for :func:`~evoforge.eval.simulator.simulate_many` and the MULTI_PARTY eval path.
"""

from __future__ import annotations

from typing import Any, Optional

from evoforge.synthesis.conversation import SimTranscript
from evoforge.synthesis.pipeline import ScenarioSynthesizer
from evoforge.synthesis.seed import Seed


class NativeMultiPartySimulator:
    """Generate + label a conversation from a seed using only EvoForge internals.

    ``labelers`` selects which registered labelers annotate the transcript. By
    default a *failure seed* is labeled with ``avoids_failure`` (did the agent avoid
    the targeted failure?) and any other seed with ``target_presence``. Pass explicit
    names to expand the set. All generation/labeling is delegated to
    :class:`~evoforge.synthesis.pipeline.ScenarioSynthesizer`.
    """

    def __init__(
        self,
        pool: Any,
        labelers: Optional[list[str]] = None,
        votes: int = 3,
        name: str = "native",
    ) -> None:
        self._synthesizer = ScenarioSynthesizer(pool, votes=votes, name=name)
        self._labeler_names = labelers
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def simulate(self, scenario: Seed, agent: Optional[Any] = None) -> SimTranscript:
        transcript = self._synthesizer.simulate(scenario, labelers=self._labeler_names)
        return transcript


__all__ = ["NativeMultiPartySimulator"]
