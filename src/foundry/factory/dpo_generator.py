"""DPO data generation — mine trajectories for chosen/rejected pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from foundry.core.types import Trajectory


class DPOPair(BaseModel):
    """A single DPO training pair."""
    instruction: str           # the user prompt
    chosen: str                # high-quality response (from good trajectory)
    rejected: str              # low-quality response (from bad trajectory)
    capability: str = ""
    margin: float = 0.0        # score difference (chosen_score - rejected_score)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DPOConfig(BaseModel):
    """Configuration for DPO pair generation."""
    min_margin: float = 0.2          # minimum score gap to form a pair
    max_pairs: int = 500             # cap total pairs
    deduplicate: bool = True         # remove near-duplicate instructions


class DPOGenerator:
    """
    Generate DPO (Direct Preference Optimization) training pairs from trajectories.

    Strategy:
      1. Load all trajectories for an agent
      2. Group by similar instructions (same eval case or similar prompt)
      3. Within each group, pair high-score vs low-score responses
      4. Filter by minimum margin (gap must be meaningful)

    Usage::

        gen = DPOGenerator(config=DPOConfig(min_margin=0.2))
        pairs = gen.from_eval_results(
            trajectories=all_trajectories,
            eval_results=historical_results,
        )
        gen.save_jsonl(pairs, "dpo_train.jsonl")
    """

    def __init__(self, config: Optional[DPOConfig] = None) -> None:
        self._config = config or DPOConfig()

    def from_eval_results(
        self,
        trajectories: list[Trajectory],
        eval_scores: dict[str, float],  # trajectory_id → score
    ) -> list[DPOPair]:
        """
        Generate DPO pairs by comparing trajectories with different scores.

        Args:
            trajectories: All recorded trajectories for an agent.
            eval_scores:  Mapping of trajectory_id → score (0-1).
        """
        # Group trajectories by instruction similarity
        groups: dict[str, list[tuple[Trajectory, float]]] = {}
        for traj in trajectories:
            if not traj.messages:
                continue
            instruction = traj.messages[-1].content if traj.messages else ""
            # Simple grouping key: first 50 chars of instruction
            key = instruction[:50].lower().strip()
            score = eval_scores.get(traj.id, 0.5)
            groups.setdefault(key, []).append((traj, score))

        # Generate pairs within each group
        pairs = []
        for key, items in groups.items():
            if len(items) < 2:
                continue
            # Sort by score
            items.sort(key=lambda x: x[1], reverse=True)
            # Pair best with worst
            for i in range(len(items) // 2):
                chosen_traj, chosen_score = items[i]
                rejected_traj, rejected_score = items[-(i + 1)]
                margin = chosen_score - rejected_score

                if margin >= self._config.min_margin:
                    instruction = chosen_traj.messages[-1].content if chosen_traj.messages else ""
                    pairs.append(DPOPair(
                        instruction=instruction,
                        chosen=chosen_traj.response,
                        rejected=rejected_traj.response,
                        margin=margin,
                        metadata={
                            "chosen_id": chosen_traj.id,
                            "rejected_id": rejected_traj.id,
                            "chosen_score": chosen_score,
                            "rejected_score": rejected_score,
                        },
                    ))

        # Apply limits
        pairs.sort(key=lambda p: p.margin, reverse=True)
        return pairs[:self._config.max_pairs]

    def from_trajectories_with_llm_judge(
        self,
        trajectories: list[Trajectory],
        pool: Any,
        system_prompt: str = "",
    ) -> list[DPOPair]:
        """
        Generate DPO pairs using LLM to judge which responses are better.

        Groups trajectories by instruction, then uses LLM to rank responses.
        """
        # Group by instruction
        groups: dict[str, list[Trajectory]] = {}
        for traj in trajectories:
            if not traj.messages or not traj.response:
                continue
            key = traj.messages[-1].content[:50].lower()
            groups.setdefault(key, []).append(traj)

        pairs = []
        for key, trajs in groups.items():
            if len(trajs) < 2:
                continue
            # Use LLM to rank responses
            scored = self._llm_rank(trajs, pool, system_prompt)
            scored.sort(key=lambda x: x[1], reverse=True)

            for i in range(len(scored) // 2):
                chosen_traj, chosen_s = scored[i]
                rejected_traj, rejected_s = scored[-(i + 1)]
                margin = chosen_s - rejected_s
                if margin >= self._config.min_margin:
                    pairs.append(DPOPair(
                        instruction=chosen_traj.messages[-1].content,
                        chosen=chosen_traj.response,
                        rejected=rejected_traj.response,
                        margin=margin,
                    ))

        return pairs[:self._config.max_pairs]

    def _llm_rank(
        self, trajs: list[Trajectory], pool: Any, system_prompt: str
    ) -> list[tuple[Trajectory, float]]:
        """Score each trajectory response using LLM (synchronous)."""
        import inspect
        import re
        results = []
        for traj in trajs[:10]:  # cap at 10 per group
            prompt = f"""Rate this agent response quality (0.0-1.0).
Task: respond to user's request.
User: "{traj.messages[-1].content[:100]}"
Agent: "{traj.response[:200]}"
Reply with ONLY a number between 0.0 and 1.0:"""
            raw = pool.generate(prompt, temperature=0, max_tokens=10)
            if inspect.isawaitable(raw):
                raise RuntimeError(
                    "DPOGenerator received an async LLM pool in sync mode. "
                    "Use a synchronous pool (for example, OllamaLLMPool)."
                )
            try:
                score = float(re.search(r'[0-9.]+', str(raw)).group())
                score = min(1.0, max(0.0, score))
            except (AttributeError, ValueError):
                score = 0.5
            results.append((traj, score))
        return results

    def save_jsonl(self, pairs: list[DPOPair], output_path: str, system_prompt: str = "") -> str:
        """
        Save DPO pairs as JSONL in standard format.

        Format (compatible with TRL DPOTrainer):
            {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for pair in pairs:
                obj = {
                    "prompt": pair.instruction,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
                if system_prompt:
                    obj["system"] = system_prompt
                f.write(json.dumps(obj) + "\n")

        return str(path)
