"""Main FoundrySDK class — the top-level developer-facing object."""

from __future__ import annotations

from typing import Any, Callable
from evoforge.core.config import SDKConfig
from evoforge.core.agent_config import AgentConfig


class FoundrySDK:
    """
    Top-level Foundry SDK instance. Created via evoforge.init().

    Provides:
        .agent()          — decorator: instrument a single-user agent
        .group_agent()    — decorator: instrument a multi-party agent
        .instrument()     — decorator: add collection to existing agent
        .bootstrap()      — zero-shot or chat-log data bootstrap
        .eval             — EvalRunner namespace
        .evolve           — EvolutionEngine namespace
        .data             — DataRegistry namespace
        .trace            — Trace intelligence namespace
        .mine             — Failure-mode mining namespace
        .synth            — Targeted synthetic data namespace
        .coverage         — Adaptive eval coverage namespace
        .forecast         — Failure forecasting namespace
        .context          — Context management namespace
        .env              — Environment management namespace
        .register()       — override default components

    AgentConfig tiers
    -----------------
    Black box (no config):
        Works for telemetry, eval, and train data curation.
        SDK observes inputs/outputs only.

    With AgentConfig:
        system_prompt  → enables SkillRegistry versioning + prompt evolution
        model config   → enables FineTuneTrigger to select backend + base model
        swap_model fn  → enables auto-promotion after A/B test
                         if omitted, PromotionEvent is emitted instead
    """

    def __init__(self, config: SDKConfig) -> None:
        self.config = config
        self._eval: Any = None
        self._evolve: Any = None
        self._data: Any = None
        self._context: Any = None
        self._env: Any = None
        self._trace: Any = None
        self._mine: Any = None
        self._synth: Any = None
        self._coverage: Any = None
        self._forecast: Any = None

    # ── Decorators ────────────────────────────────────────────────────

    def agent(
        self,
        tools: list[Any] | None = None,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ) -> Callable:
        """
        Decorator: instrument a single-user agent for Foundry.

        Args:
            tools:  List of tool functions or tool names available to the agent.
            config: Optional AgentConfig exposing system_prompt, model identity,
                    and swap_model fn. Without it, Foundry works as a black box.

        Unlocked by AgentConfig fields:
            system_prompt  → SkillRegistry versioning + prompt evolution
            model.id/host  → auto fine-tune trigger + backend selection
            swap_model     → auto model promotion after A/B test passes

        Example (black box — works for all data collection + eval)::

            @sdk.agent(tools=[search_flights, book_flight])
            def my_agent(messages): ...

        Example (with AgentConfig — unlocks prompt evolution + fine-tuning)::

            @sdk.agent(
                tools=[search_flights, book_flight],
                config=AgentConfig(
                    system_prompt="You are a helpful flight booking assistant...",
                    model=ModelConfig(id="gpt-4o", host=ModelHost.OPENAI),
                    swap_model=lambda new_id: my_agent.set_model(new_id),
                )
            )
            def my_agent(messages): ...
        """
        def decorator(fn: Callable) -> Callable:
            fn._foundry_agent_config = config
            fn._foundry_tools = tools or []
            fn._foundry_task_spec = self.config.task_spec
            fn._foundry_sdk = self
            fn._foundry_multi_party = False
            self._warn_missing_config(config, fn.__name__)

            import functools
            import time
            import uuid

            @functools.wraps(fn)
            def instrumented(*args, **kwargs):
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000

                # Record trajectory to storage (best-effort, never raises)
                try:
                    from evoforge.core.types import Trajectory
                    messages = args[0] if args else kwargs.get("messages", [])
                    traj = Trajectory(
                        id=str(uuid.uuid4())[:8],
                        agent_name=fn.__name__,
                        messages=messages if isinstance(messages, list) else [],
                        response=str(result),
                        latency_ms=round(latency_ms, 1),
                    )
                    storage_path = str(self.config.storage.path)
                    from evoforge.data.storage.local import LocalStorageBackend
                    import json
                    store = LocalStorageBackend(storage_path)
                    store.write(
                        f"trajectories/{fn.__name__}/{traj.id}.json",
                        json.dumps(traj.model_dump(), indent=2).encode(),
                    )
                except Exception:
                    pass  # telemetry must never break the agent

                return result

            # Copy foundry metadata onto the wrapper
            instrumented._foundry_agent_config = config
            instrumented._foundry_tools = tools or []
            instrumented._foundry_task_spec = self.config.task_spec
            instrumented._foundry_sdk = self
            instrumented._foundry_multi_party = False
            return instrumented
        return decorator

    def group_agent(
        self,
        tools: list[Any] | None = None,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ) -> Callable:
        """
        Decorator: instrument a multi-party group agent for Foundry.

        Same as @agent but injects user_ctx and group_ctx into the function
        signature and enables multi-party eval + GroupContext management.

        The decorated function must accept (messages, user_id, user_ctx, group_ctx).
        """
        def decorator(fn: Callable) -> Callable:
            fn._foundry_agent_config = config
            fn._foundry_tools = tools or []
            fn._foundry_task_spec = self.config.task_spec
            fn._foundry_sdk = self
            fn._foundry_multi_party = True
            self._warn_missing_config(config, fn.__name__)
            return fn
        return decorator

    def instrument(
        self,
        collect_trajectories: bool = True,
        collect_feedback: bool = True,
        collect_cost: bool = True,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ) -> Callable:
        """
        Decorator: add Foundry telemetry to any existing agent without
        requiring it to be declared with @agent.

        Useful for wrapping third-party agents or agents defined elsewhere.
        """
        def decorator(fn: Callable) -> Callable:
            fn._foundry_instrumented = True
            fn._foundry_agent_config = config
            fn._foundry_collect = {
                "trajectories": collect_trajectories,
                "feedback": collect_feedback,
                "cost": collect_cost,
                **kwargs,
            }
            self._warn_missing_config(config, fn.__name__)
            return fn
        return decorator

    # ── Bootstrap ─────────────────────────────────────────────────────

    def bootstrap(self, agent: Callable, source: Any = None, **kwargs: Any) -> Any:
        """
        Bootstrap eval cases for an agent from task_spec + tools.

        Zero human input required. Infers capabilities from the agent's tools
        and task_spec, then generates diverse eval cases per capability.

        Args:
            agent:  The decorated agent function.
            source: Optional BootstrapConfig or ZeroShotSource for customization.

        Returns:
            BootstrapResult with capabilities and generated eval_cases.

        Example::

            result = sdk.bootstrap(agent=my_agent)
            print(f"Generated {result.n_generated} eval cases")
            print(f"Capabilities: {[c['name'] for c in result.capabilities]}")

            # Run eval immediately
            eval_result = sdk.eval.run(agent=my_agent, cases=result.eval_cases)
        """
        from evoforge.bootstrap.pipeline import BootstrapPipeline, BootstrapConfig
        from evoforge.llm.ollama import OllamaLLMPool

        # Determine config
        if isinstance(source, BootstrapConfig):
            config = source
        else:
            config = BootstrapConfig(**kwargs) if kwargs else BootstrapConfig()

        # Use Ollama by default (free, local)
        pool = OllamaLLMPool()

        # Extract agent metadata
        task_spec = self.config.task_spec
        tools = getattr(agent, "_foundry_tools", [])
        system_prompt = ""
        agent_config = getattr(agent, "_foundry_agent_config", None)
        if agent_config and agent_config.system_prompt:
            system_prompt = agent_config.system_prompt

        pipeline = BootstrapPipeline(pool=pool, config=config)
        result = pipeline.run(
            task_spec=task_spec,
            tools=tools,
            system_prompt=system_prompt,
            agent=agent,
        )

        # Auto-save generated eval cases
        if result.eval_cases:
            self.data.save_eval_cases(result.eval_cases, tag="bootstrap")

        return result

    # ── Subsystem namespaces (lazy) ────────────────────────────────────

    @property
    def eval(self) -> Any:
        if self._eval is None:
            from evoforge.eval.namespace import EvalNamespace
            self._eval = EvalNamespace(self)
        return self._eval

    @property
    def evolve(self) -> Any:
        if self._evolve is None:
            from evoforge.evolution.namespace import EvolveNamespace
            self._evolve = EvolveNamespace(self)
        return self._evolve

    @property
    def data(self) -> Any:
        if self._data is None:
            from evoforge.data.namespace import DataNamespace
            self._data = DataNamespace(self)
        return self._data

    @property
    def context(self) -> Any:
        if self._context is None:
            from evoforge.context.namespace import ContextNamespace
            self._context = ContextNamespace(self)
        return self._context

    @property
    def env(self) -> Any:
        if self._env is None:
            from evoforge.environment.namespace import EnvNamespace
            self._env = EnvNamespace(self)
        return self._env

    @property
    def trace(self) -> Any:
        if self._trace is None:
            from evoforge.trace.namespace import TraceNamespace
            self._trace = TraceNamespace(self)
        return self._trace

    @property
    def mine(self) -> Any:
        if self._mine is None:
            from evoforge.mining.namespace import MiningNamespace
            self._mine = MiningNamespace(self)
        return self._mine

    @property
    def synth(self) -> Any:
        if self._synth is None:
            from evoforge.synthesis.namespace import SynthesisNamespace
            self._synth = SynthesisNamespace(self)
        return self._synth

    @property
    def coverage(self) -> Any:
        if self._coverage is None:
            from evoforge.coverage.namespace import CoverageNamespace
            self._coverage = CoverageNamespace(self)
        return self._coverage

    @property
    def forecast(self) -> Any:
        if self._forecast is None:
            from evoforge.forecast.namespace import ForecastNamespace
            self._forecast = ForecastNamespace(self)
        return self._forecast

    def register(self, **components: Any) -> None:
        """Override default components: environment, label_strategy, evolver, etc."""
        for key, value in components.items():
            setattr(self, f"_override_{key}", value)

    # ── Internal helpers ───────────────────────────────────────────────

    def _warn_missing_config(
        self, config: AgentConfig | None, agent_name: str
    ) -> None:
        """Emit informational warnings when AgentConfig is missing fields."""
        if config is None:
            if self.config.verbose:
                print(
                    f"[foundry] {agent_name}: no AgentConfig provided. "
                    "Foundry will operate as black box (collection + eval only). "
                    "Add AgentConfig to enable prompt evolution and auto fine-tuning."
                )
            return
        if config.system_prompt is None and self.config.verbose:
            print(
                f"[foundry] {agent_name}: no system_prompt in AgentConfig. "
                "Prompt evolution will suggest but not auto-apply changes."
            )
        if config.model is None and self.config.verbose:
            print(
                f"[foundry] {agent_name}: no model config in AgentConfig. "
                "Training data will be curated but auto fine-tune is disabled."
            )
        if config.swap_model is None and self.config.verbose:
            print(
                f"[foundry] {agent_name}: no swap_model fn in AgentConfig. "
                "After A/B test, a PromotionEvent will be emitted instead "
                "of auto-promotion."
            )

