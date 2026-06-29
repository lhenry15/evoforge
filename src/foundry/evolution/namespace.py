"""EvolveNamespace — sdk.evolve interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK

from foundry.core.types import (
    CapabilityGap,
    EvolutionAction,
    EvolutionDecision,
    EvalRunResult,
    SaturationSignal,
)
from foundry.evolution.engine import EvolutionTriggers
from foundry.training.backends.protocol import FineTuneBackend, TrainingJob


class CycleResult:
    """Result of a full evolution cycle execution."""

    def __init__(self) -> None:
        self.decision: Optional[EvolutionDecision] = None
        # Prompt evolution (fast, free)
        self.prompt_patches: list[Any] = []       # PromptPatch objects
        self.new_skills: list[Any] = []           # SkillFile objects
        # Training data evolution (slow, costly)
        self.training_examples_generated: int = 0
        self.training_job: Optional[TrainingJob] = None
        self.validation_response: Optional[str] = None
        # Eval expansion
        self.expanded_eval_cases: list[Any] = []
        self.errors: list[str] = []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.decision is not None

    def __repr__(self) -> str:
        parts = [f"CycleResult(success={self.success}"]
        if self.decision:
            parts.append(f"actions={[a.value for a in self.decision.actions]}")
        if self.training_examples_generated:
            parts.append(f"examples={self.training_examples_generated}")
        if self.training_job:
            parts.append(f"job={self.training_job.status}")
        if self.errors:
            parts.append(f"errors={self.errors}")
        return ", ".join(parts) + ")"


class EvolveNamespace:
    """
    sdk.evolve — analyse eval results and decide what to evolve.

    Usage::

        # Just decide (no execution)
        decision = sdk.evolve.run_cycle(agent=my_agent, eval_result=result)

        # Decide AND execute (generate data, train)
        cycle = sdk.evolve.execute_cycle(
            agent=my_agent,
            eval_result=result,
            llm_pool=pool,
            training_backend=backend,
        )
    """

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk
        self._triggers = EvolutionTriggers()

    def run_cycle(
        self,
        agent: Callable,
        eval_result: EvalRunResult,
    ) -> EvolutionDecision:
        """
        Analyse an EvalRunResult and return evolution actions.

        This is decision-only — does not execute any actions.
        Use execute_cycle() to also act on the decisions.
        """
        gaps: list[CapabilityGap] = []
        saturations: list[SaturationSignal] = []
        actions: list[EvolutionAction] = []

        for cap, score in eval_result.capability_scores.items():
            if score < self._triggers.train_failure_threshold:
                gaps.append(CapabilityGap(
                    capability=cap,
                    score=score,
                    threshold=self._triggers.train_failure_threshold,
                    suggested_n_examples=max(50, int((self._triggers.train_failure_threshold - score) * 500)),
                ))
            elif score > self._triggers.eval_saturation_threshold:
                saturations.append(SaturationSignal(
                    capability=cap,
                    score=score,
                    threshold=self._triggers.eval_saturation_threshold,
                    suggested_expansion=int(self._triggers.eval_expansion_factor * 10),
                ))

        if gaps:
            actions.append(EvolutionAction.GENERATE_TRAIN_DATA)
        if saturations:
            actions.append(EvolutionAction.EXPAND_EVAL)

        # Check if fine-tune is warranted
        from foundry.evolution.engine import _can_fine_tune, _can_auto_promote
        n_trajectories = len(
            self._sdk.data.load_trajectories(agent.__name__)
            if hasattr(self._sdk, "_data") and self._sdk._data is not None
            else []
        )
        if gaps and n_trajectories >= self._triggers.train_data_threshold:
            can_ft, _ = _can_fine_tune(agent)
            if can_ft:
                can_promo, _ = _can_auto_promote(agent)
                actions.append(
                    EvolutionAction.TRIGGER_FINE_TUNE if can_promo
                    else EvolutionAction.EMIT_PROMOTION_EVENT
                )

        if not actions:
            actions.append(EvolutionAction.NO_ACTION)

        summary_parts = [
            f"overall_score={eval_result.overall_score:.3f}",
            f"gaps={[g.capability for g in gaps]}",
            f"saturations={[s.capability for s in saturations]}",
            f"actions={[a.value for a in actions]}",
        ]

        return EvolutionDecision(
            agent_name=agent.__name__,
            actions=actions,
            capability_gaps=gaps,
            saturation_signals=saturations,
            summary=" | ".join(summary_parts),
        )

    def execute_cycle(
        self,
        agent: Callable,
        eval_result: EvalRunResult,
        llm_pool: Any,
        training_backend: Optional[FineTuneBackend] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[Any]] = None,
        examples_per_gap: int = 10,
    ) -> CycleResult:
        """
        Full evolution cycle: decide → generate data → train → validate.

        Steps:
          1. Analyse eval_result → EvolutionDecision
          2. If GENERATE_TRAIN_DATA: mine failure modes → synthesize targeted data
          3. If training_backend provided: launch fine-tune job
          4. If job succeeds: validate with a test prompt
          5. Expand eval coverage toward observed failure modes (blind spots)

        Args:
            agent:             Decorated agent function.
            eval_result:       Result from sdk.eval.run().
            llm_pool:          LLM pool for data generation (OllamaLLMPool etc.)
            training_backend:  FineTuneBackend for fine-tuning (None = skip training).
            system_prompt:     Override system prompt (or pulled from AgentConfig).
            tools:             Tool descriptions for data generation context.
            examples_per_gap:  Number of training examples per gap.

        Returns:
            CycleResult with decision, generated data count, and training job.
        """
        result = CycleResult()

        # Step 1: Decide
        decision = self.run_cycle(agent, eval_result)
        result.decision = decision

        if EvolutionAction.NO_ACTION in decision.actions:
            return result

        # Resolve system prompt from AgentConfig if not provided
        if system_prompt is None:
            from foundry.evolution.engine import _get_system_prompt
            system_prompt = _get_system_prompt(agent) or ""

        # Resolve tools from agent metadata
        if tools is None:
            tools = getattr(agent, "_foundry_tools", [])

        # Step 2a: Prompt/Skill evolution (fast, free — try first)
        if EvolutionAction.GENERATE_TRAIN_DATA in decision.actions and decision.capability_gaps:
            try:
                from foundry.evolution.prompt_evolver import PromptEvolver

                evolver = PromptEvolver(pool=llm_pool)
                prompt_result = evolver.evolve(
                    agent=agent,
                    eval_result=eval_result,
                    gaps=decision.capability_gaps,
                    task_spec=self._sdk.config.task_spec,
                )
                result.prompt_patches = prompt_result.patches
                result.new_skills = prompt_result.new_skills

                # Get existing skills from agent config
                config = getattr(agent, "_foundry_agent_config", None)
                existing_skills = dict(config.skill_prompts) if config and config.skill_prompts else {}

                # Apply skills: add new ones, refine existing ones (no duplicates)
                for skill in prompt_result.new_skills:
                    if skill.name in existing_skills:
                        # Skill already exists — only update if content is different
                        if skill.content != existing_skills[skill.name]:
                            existing_skills[skill.name] = skill.content
                    else:
                        existing_skills[skill.name] = skill.content

                if config:
                    config.skill_prompts = existing_skills

                # Apply prompt patch (only if substantially different)
                for patch in prompt_result.patches:
                    if config and patch.target == "system_prompt":
                        if patch.revised and patch.revised != config.system_prompt:
                            config.system_prompt = patch.revised

                # Save skills to disk
                if existing_skills:
                    skills_dir = str(
                        Path(self._sdk.config.storage.path) / "skills" / agent.__name__
                    )
                    evolver.apply_skills(prompt_result, skills_dir)

            except Exception as e:
                result.errors.append(f"Prompt evolution failed: {e}")

        # Step 2b: Generate training data for capability gaps (mode-conditioned synthesis)
        if EvolutionAction.GENERATE_TRAIN_DATA in decision.actions and decision.capability_gaps:
            try:
                from foundry.synthesis.synthesizer import DataSynthesizer

                # Mine real failure modes (re-classify + persist), then synthesize
                # targeted training data conditioned on those modes.
                mining = self._sdk.mine.run(agent.__name__, pool=llm_pool, persist=True)
                corpus = [
                    c.messages[-1].content
                    for c in self._sdk.data.load_eval_cases(tag="bootstrap")
                    if c.messages
                ]
                synthesizer = DataSynthesizer(
                    pool=llm_pool, per_cluster=examples_per_gap, max_clusters=5
                )
                synth_result = synthesizer.synthesize(
                    mining_result=mining,
                    task_spec=self._sdk.config.task_spec,
                    tools=tools,
                    system_prompt=system_prompt,
                    corpus_instructions=corpus,
                )
                examples = synth_result.training_examples()
                result.training_examples_generated = len(examples)

                if not examples:
                    result.errors.append("Synthesis produced 0 training examples")
                    return result

                # Step 3: Launch training if backend provided
                if training_backend is not None and examples:
                    job = training_backend.launch_from_examples(
                        examples=examples,
                        system_prompt=system_prompt,
                    )
                    result.training_job = job

                    # Step 4: Validate if training succeeded
                    if job.succeeded:
                        try:
                            test_prompt = decision.capability_gaps[0].capability
                            response = training_backend.validate(
                                job.job_id,
                                prompt=f"Help me with: {test_prompt}",
                                system_prompt=system_prompt,
                            )
                            result.validation_response = response
                        except Exception as e:
                            result.errors.append(f"Validation failed: {e}")
                    elif job.status == "failed":
                        result.errors.append(
                            f"Training failed: {job.metadata.get('error', 'unknown')}"
                        )

            except Exception as e:
                result.errors.append(f"Data generation failed: {e}")

        # Step 5: Expand eval coverage toward real failure modes (blind spots)
        if decision.capability_gaps:
            try:
                new_cases = self._sdk.coverage.expand(
                    agent.__name__,
                    pool=llm_pool,
                    tools=tools,
                    system_prompt=system_prompt or "",
                    persist=True,
                )
                result.expanded_eval_cases = new_cases
            except Exception as e:
                result.errors.append(f"Eval expansion failed: {e}")

        return result
