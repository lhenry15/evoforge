"""Data generation strategies — pluggable approaches for creating training data."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from foundry.core.types import CapabilityGap
from foundry.factory.data_factory import TrainingExample


class DataStrategy(ABC):
    """
    Base class for data generation strategies.

    Subclass this to implement new approaches for generating training data.
    The DataFactory delegates to a strategy — swap strategies without changing
    the pipeline.

    Built-in strategies:
      - TeacherStrategy:        Strong model generates examples (default)
      - RejectionSampling:      Self-generate N, score, keep top-K
      - STaRStrategy:           Self-Taught Reasoner (generate + verify)
      - SPINStrategy:           Self-Play (model debates itself)
      - DistillationStrategy:   Teacher provides chain-of-thought rationale
    """

    @abstractmethod
    def generate(
        self,
        gaps: list[CapabilityGap],
        task_spec: str,
        tools: list[Any],
        system_prompt: str,
        n_per_gap: int,
    ) -> list[TrainingExample]:
        """Generate training examples for the given capability gaps."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/lineage."""
        ...


class TeacherStrategy(DataStrategy):
    """
    Strong teacher model generates training examples.

    Best when: you have access to a model stronger than your target agent.
    Pro: highest quality data.
    Con: requires API access to a better model (cost).
    """

    def __init__(self, pool: Any, temperature: float = 0.8) -> None:
        self._pool = pool
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "teacher"

    def generate(self, gaps, task_spec, tools, system_prompt, n_per_gap) -> list[TrainingExample]:
        from foundry.factory.data_factory import DataFactory, DataFactoryConfig
        factory = DataFactory(pool=self._pool, config=DataFactoryConfig(examples_per_gap=n_per_gap))
        return factory.generate_for_gaps(gaps, task_spec, tools, system_prompt)


class RejectionSampling(DataStrategy):
    """
    Self-generate N responses, score each with a judge, keep top-K.

    No teacher needed — the model generates candidates, a judge (LLM or rule)
    filters for quality. Only high-scoring outputs become training data.

    Algorithm:
      1. For each capability gap, generate diverse prompts
      2. For each prompt, generate N candidate responses (high temperature)
      3. Score each candidate with the judge
      4. Keep only candidates scoring above threshold
      5. These become training examples

    Best when: no strong teacher available, but you have a reliable judge.
    Pro: works with self-generated data, no external API needed.
    Con: inefficient (generates many, keeps few), judge must be reliable.
    """

    def __init__(
        self,
        generator_pool: Any,            # model that generates candidates
        judge_pool: Optional[Any] = None,  # model that judges (defaults to generator)
        n_candidates: int = 8,          # candidates per prompt
        keep_threshold: float = 0.7,    # minimum score to keep
        temperature: float = 0.9,       # high temp for diversity
    ) -> None:
        self._generator = generator_pool
        self._judge = judge_pool or generator_pool
        self._n_candidates = n_candidates
        self._keep_threshold = keep_threshold
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "rejection_sampling"

    def generate(self, gaps, task_spec, tools, system_prompt, n_per_gap) -> list[TrainingExample]:
        all_examples = []
        for gap in gaps:
            examples = self._sample_for_gap(gap, task_spec, tools, system_prompt, n_per_gap)
            all_examples.extend(examples)
        return all_examples

    def _sample_for_gap(self, gap, task_spec, tools, system_prompt, n_target):
        # Step 1: Generate diverse prompts for this capability
        prompts = self._generate_prompts(gap, task_spec, tools, n_target * 2)

        # Step 2: For each prompt, generate N candidate responses
        accepted = []
        for prompt in prompts:
            if len(accepted) >= n_target:
                break

            candidates = self._generator.generate_ensemble(
                f"You are a helpful assistant. Respond fully and helpfully to this user request.\n\nUser: {prompt}\n\nAssistant:",
                n=self._n_candidates,
                temperature=self._temperature,
            )

            # Step 3: Score each candidate
            scored = self._score_candidates(prompt, candidates, gap.capability, task_spec)

            # Step 4: Keep above threshold
            for response, score in scored:
                if score >= self._keep_threshold and len(accepted) < n_target:
                    accepted.append(TrainingExample(
                        capability=gap.capability,
                        instruction=prompt,
                        ideal_response=response,
                        metadata={"score": score, "strategy": "rejection_sampling"},
                    ))

        return accepted

    def _generate_prompts(self, gap, task_spec, tools, n):
        """Generate diverse user prompts for a capability."""
        tool_str = ", ".join(str(t) for t in tools) if tools else "none"
        prompt = f"""Generate {n} diverse, realistic user messages for a {task_spec}.
These should test the '{gap.capability}' capability.
Tools available: {tool_str}

Rules:
- Each message should be something a REAL user would type
- Keep messages short and natural (1-2 sentences)
- Vary the specific details (different cities, names, dates)
- Do NOT include expected responses, just user messages

Reply with ONLY a JSON array of strings:
["message 1", "message 2", ...]"""

        raw = self._generator.generate(prompt, temperature=0.8, max_tokens=2048)
        try:
            # Extract JSON array
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                items = json.loads(match.group())
                return [str(i) for i in items if isinstance(i, str)][:n]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback
        return [f"Help me with {gap.capability}" for _ in range(min(3, n))]

    def _score_candidates(self, prompt, candidates, capability, task_spec):
        """Score each candidate response using the judge."""
        scored = []
        for candidate in candidates:
            if not candidate.strip():
                continue
            judge_prompt = f"""Score this response (0.0-1.0) for the '{capability}' capability.
Task: {task_spec}
User asked: "{prompt[:100]}"
Response: "{candidate[:200]}"

Score criteria: Is the response helpful, correct, and relevant?
Reply with ONLY a number (0.0-1.0):"""

            raw = self._judge.generate(judge_prompt, temperature=0, max_tokens=10)
            try:
                score = float(re.search(r'[0-9.]+', raw).group())
                score = min(1.0, max(0.0, score))
            except (AttributeError, ValueError):
                score = 0.5
            scored.append((candidate, score))

        return scored


class STaRStrategy(DataStrategy):
    """
    Self-Taught Reasoner — generate response + rationale, verify via execution.

    Algorithm:
      1. Generate (response, rationale) pairs
      2. Verify response correctness via environment/tool execution
      3. Keep only verified-correct examples
      4. Include rationale in training data (teaches reasoning)

    Best when: you have an execution environment that can verify correctness.
    Pro: ground truth verification, no judge model needed.
    Con: requires executable environment.
    """

    def __init__(self, pool: Any, verifier: Any = None) -> None:
        self._pool = pool
        self._verifier = verifier  # callable(instruction, response) -> bool

    @property
    def name(self) -> str:
        return "star"

    def generate(self, gaps, task_spec, tools, system_prompt, n_per_gap) -> list[TrainingExample]:
        # Stub — implement when environment connectors are ready
        raise NotImplementedError(
            "STaR requires an execution environment to verify responses. "
            "Use TeacherStrategy or RejectionSampling for now."
        )


class SPINStrategy(DataStrategy):
    """
    Self-Play Improvement — model generates competing responses, best wins.

    Algorithm:
      1. Generate prompt
      2. Generate response A and response B (different temperatures/prompts)
      3. Judge which is better
      4. Winner = chosen, loser = rejected (DPO format)

    Best when: you want DPO pairs without a teacher.
    Pro: generates preference data autonomously.
    Con: quality limited by model's own capabilities.
    """

    def __init__(self, pool: Any, judge_pool: Any = None) -> None:
        self._pool = pool
        self._judge = judge_pool or pool

    @property
    def name(self) -> str:
        return "spin"

    def generate(self, gaps, task_spec, tools, system_prompt, n_per_gap) -> list[TrainingExample]:
        # Returns SFT examples from the winner side
        examples = []
        for gap in gaps:
            for _ in range(n_per_gap):
                prompt = f"As a {task_spec}, help with: {gap.capability}"
                # Generate two competing responses
                responses = self._pool.generate_ensemble(prompt, n=2, temperature=0.9)
                if len(responses) < 2:
                    continue
                # Judge
                judge_prompt = f"""Which response is better for '{gap.capability}'?
A: "{responses[0][:150]}"
B: "{responses[1][:150]}"
Reply with ONLY 'A' or 'B':"""
                winner = self._judge.generate(judge_prompt, temperature=0, max_tokens=5)
                chosen = responses[0] if 'A' in winner.upper() else responses[1]
                examples.append(TrainingExample(
                    capability=gap.capability,
                    instruction=prompt,
                    ideal_response=chosen,
                    metadata={"strategy": "spin"},
                ))
        return examples


class DistillationStrategy(DataStrategy):
    """
    Knowledge distillation — teacher provides chain-of-thought rationale.

    Algorithm:
      1. Teacher generates (response + step-by-step reasoning)
      2. Student trains on the full chain-of-thought
      3. Student learns the reasoning process, not just the answer

    Best when: you want the student to learn HOW to think, not just WHAT to say.
    Pro: teaches reasoning patterns.
    Con: requires strong teacher.
    """

    def __init__(self, teacher_pool: Any) -> None:
        self._teacher = teacher_pool

    @property
    def name(self) -> str:
        return "distillation"

    def generate(self, gaps, task_spec, tools, system_prompt, n_per_gap) -> list[TrainingExample]:
        examples = []
        for gap in gaps:
            for _ in range(n_per_gap):
                prompt = f"""You are an expert. A user needs help with '{gap.capability}'.
Task context: {task_spec}
Tools available: {', '.join(str(t) for t in tools)}

Generate a user question AND your response WITH step-by-step reasoning.
Format:
USER: <question>
REASONING: <your thought process>
RESPONSE: <final answer>"""

                raw = self._teacher.generate(prompt, temperature=0.7, max_tokens=1024)

                # Parse
                user_match = re.search(r'USER:\s*(.+?)(?=REASONING:|$)', raw, re.DOTALL)
                reasoning_match = re.search(r'REASONING:\s*(.+?)(?=RESPONSE:|$)', raw, re.DOTALL)
                response_match = re.search(r'RESPONSE:\s*(.+?)$', raw, re.DOTALL)

                if user_match and response_match:
                    instruction = user_match.group(1).strip()
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                    response = response_match.group(1).strip()
                    # Include reasoning in the ideal response (teaches thinking)
                    ideal = f"Let me think through this:\n{reasoning}\n\n{response}" if reasoning else response
                    examples.append(TrainingExample(
                        capability=gap.capability,
                        instruction=instruction,
                        ideal_response=ideal,
                        metadata={"strategy": "distillation", "has_reasoning": bool(reasoning)},
                    ))
        return examples
