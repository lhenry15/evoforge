"""EvoForge CLI — terminal interface for the data-centric agent evolution SDK."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _resolve_store_path() -> Path:
    """Resolve storage path, preferring a project-local .foundry directory."""
    project_store = Path.cwd() / ".foundry"
    if project_store.exists():
        return project_store
    return Path.home() / "agent-foundry" / ".foundry"


def _load_agent_from_file(agent_path: str):
    """Import a Python file and find the @sdk.agent()-decorated function."""
    path = Path(agent_path).resolve()
    if not path.exists():
        console.print(f"[red]Error:[/] File not found: {agent_path}")
        sys.exit(1)

    # Add parent dir to path so imports work
    sys.path.insert(0, str(path.parent))
    # Also add src/ if it exists
    src_dir = Path.cwd() / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    spec = importlib.util.spec_from_file_location("_agent_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find decorated agents
    agents = []
    sdk_instance = None
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_foundry_sdk"):
            agents.append((name, obj))
            sdk_instance = obj._foundry_sdk

    if not agents:
        console.print(f"[red]Error:[/] No @sdk.agent() decorated function found in {agent_path}")
        sys.exit(1)

    return agents, sdk_instance


@click.group()
@click.version_option(version="0.1.0", prog_name="evoforge")
def cli():
    """EvoForge — data-centric self-evolving agent SDK."""
    pass


@cli.command()
@click.argument("agent_file")
@click.option("--cases", "-n", default=20, help="Number of eval cases to generate")
@click.option("--min-per-cap", default=3, help="Minimum cases per capability")
def bootstrap(agent_file: str, cases: int, min_per_cap: int):
    """Auto-generate eval cases from agent's task_spec + tools."""
    console.print(Panel("[bold]EvoForge Bootstrap[/]", style="blue"))

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, agent_fn = agents[0]

    console.print(f"  Agent: [cyan]{agent_name}[/]")
    console.print(f"  Task:  {sdk.config.task_spec[:70]}...")
    console.print()

    with console.status("[bold green]Inferring capabilities & generating eval cases..."):
        result = sdk.bootstrap(
            agent=agent_fn,
            num_eval_cases=cases,
            min_per_capability=min_per_cap,
        )

    # Display results
    table = Table(title=f"Bootstrapped Eval Cases ({result.n_generated} total)")
    table.add_column("Capability", style="cyan")
    table.add_column("Cases", justify="center")
    table.add_column("Difficulty Mix", style="dim")

    by_cap = {}
    for c in result.eval_cases:
        by_cap.setdefault(c.capability, []).append(c)

    for cap, cap_cases in by_cap.items():
        diffs = [c.metadata.get("difficulty", "?") for c in cap_cases]
        diff_str = ", ".join(f"{d}" for d in diffs)
        table.add_row(cap, str(len(cap_cases)), diff_str)

    console.print(table)
    console.print("\n  [green]✓[/] Saved to .foundry/eval_cases/bootstrap.json")
    console.print(f"  Next: [bold]evoforge eval {agent_file}[/]")


@cli.command()
@click.argument("agent_file")
@click.option("--tag", "-t", default="bootstrap", help="Eval case tag to load")
@click.option("--parallelism", "-p", default=1, help="Concurrent eval workers")
def eval(agent_file: str, tag: str, parallelism: int):
    """Run eval cases against an agent and score results."""
    console.print(Panel("[bold]EvoForge Eval[/]", style="green"))

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, agent_fn = agents[0]

    # Load cases
    cases = sdk.data.load_eval_cases(tag=tag)
    if not cases:
        console.print(f"[red]No eval cases found for tag '{tag}'.[/]")
        console.print(f"Run [bold]evoforge bootstrap {agent_file}[/] first.")
        sys.exit(1)

    console.print(f"  Agent: [cyan]{agent_name}[/]")
    console.print(f"  Cases: {len(cases)} (tag={tag})")
    console.print()

    with console.status(f"[bold green]Evaluating {len(cases)} cases..."):
        result = sdk.eval.run(agent=agent_fn, cases=cases, parallelism=parallelism)

    # Display results
    console.print()
    score_color = "green" if result.overall_score >= 0.6 else "yellow" if result.overall_score >= 0.3 else "red"
    console.print(f"  Overall Score: [{score_color} bold]{result.overall_score:.3f}[/]  ({result.n_passed}/{result.n_total} passed)")
    console.print()

    table = Table(title="Capability Scores")
    table.add_column("Capability", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Bar", justify="left")
    table.add_column("Status", justify="center")

    for cap, score in sorted(result.capability_scores.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "[green]✓ pass[/]" if score >= 0.6 else "[red]✗ fail[/]"
        table.add_row(cap, f"{score:.2f}", bar, status)

    console.print(table)

    # Per-case details
    console.print()
    for r in result.case_results:
        icon = "[green]✓[/]" if r.passed else "[red]✗[/]"
        console.print(f"  {icon} [{r.case_id}] {r.capability}: {r.score:.2f}  [dim]{r.judge_reasoning or ''}[/]")

    # Save result + record normalized traces (feeds mining / coverage / forecasting)
    sdk.data.save_eval_result(result)
    try:
        sdk.trace.record_eval_run(result, cases=cases)
    except Exception:
        pass
    console.print(f"\n  [green]✓[/] Results saved to .foundry/eval_results/{agent_name}/")
    console.print(f"  Next: [bold]evoforge evolve {agent_file}[/]")


@cli.command()
@click.argument("agent_file")
@click.option("--tag", "-t", default="bootstrap", help="Eval case tag")
@click.option("--train/--no-train", default=False, help="Also run training if gaps found")
@click.option("--iters", default=100, help="LoRA training iterations")
def evolve(agent_file: str, tag: str, train: bool, iters: int):
    """Analyze eval results and evolve the agent."""
    console.print(Panel("[bold]EvoForge Evolve[/]", style="magenta"))

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, agent_fn = agents[0]

    # Load and run eval first
    cases = sdk.data.load_eval_cases(tag=tag)
    if not cases:
        console.print("[red]No eval cases. Run bootstrap first.[/]")
        sys.exit(1)

    with console.status("[bold green]Running eval..."):
        eval_result = sdk.eval.run(agent=agent_fn, cases=cases, parallelism=1)

    console.print(f"  Agent: [cyan]{agent_name}[/]  Score: {eval_result.overall_score:.3f}")
    console.print()

    # Run evolution cycle
    if train:
        from foundry.llm.ollama import OllamaLLMPool
        from foundry.training.backends.mlx_lora import MLXLoRABackend, MLXLoRAConfig

        pool = OllamaLLMPool()
        backend = MLXLoRABackend(config=MLXLoRAConfig(iters=iters))

        with console.status("[bold magenta]Generating data + training..."):
            cycle = sdk.evolve.execute_cycle(
                agent=agent_fn,
                eval_result=eval_result,
                llm_pool=pool,
                training_backend=backend,
            )

        console.print(f"  Examples generated: {cycle.training_examples_generated}")
        if cycle.training_job:
            j = cycle.training_job
            status_color = "green" if j.succeeded else "red"
            console.print(f"  Training: [{status_color}]{j.status}[/]  loss: {j.train_loss:.4f}")
            if j.model_id:
                console.print(f"  Adapter:  {j.model_id}")
    else:
        decision = sdk.evolve.run_cycle(agent=agent_fn, eval_result=eval_result)

        # Display decision
        table = Table(title="Evolution Decision")
        table.add_column("Actions", style="magenta")
        for a in decision.actions:
            table.add_row(a.value)
        console.print(table)

        if decision.capability_gaps:
            console.print("\n  [yellow]Capability Gaps (need training):[/]")
            for g in decision.capability_gaps:
                console.print(f"    ⚠ {g.capability}: {g.score:.2f} → generate {g.suggested_n_examples} examples")

        if decision.saturation_signals:
            console.print("\n  [green]Saturating (expand eval):[/]")
            for s in decision.saturation_signals:
                console.print(f"    ✓ {s.capability}: {s.score:.2f} → add {s.suggested_expansion} harder cases")

    console.print(f"\n  To train: [bold]evoforge evolve {agent_file} --train[/]")


@cli.command()
@click.argument("agent_file")
@click.option("--cases", "-n", default=15, help="Bootstrap eval case count")
@click.option("--cycles", "-c", default=3, help="Number of evolution cycles")
@click.option("--train/--no-train", default=True, help="Allow LoRA training when prompt ceiling hit")
@click.option("--iters", default=100, help="LoRA training iterations")
def run(agent_file: str, cases: int, cycles: int, train: bool, iters: int):
    """Full autonomous evolution loop: bootstrap → eval → evolve → repeat."""
    console.print(Panel("[bold]EvoForge Run — Autonomous Evolution[/]", style="blue bold"))
    console.print()

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, agent_fn = agents[0]

    console.print(f"  Agent: [cyan]{agent_name}[/]")
    console.print(f"  Task:  {sdk.config.task_spec[:60]}...")
    console.print()

    from foundry.core.history import AgentEvolutionHistory
    from foundry.llm.ollama import OllamaLLMPool
    from foundry.core.types import EvolutionAction

    history = AgentEvolutionHistory(agent_name=agent_name, storage_path=str(sdk.config.storage.path))
    pool = OllamaLLMPool()

    # 1. Bootstrap
    console.rule("[bold blue]Phase 1: Bootstrap")
    with console.status("Generating eval cases + multi-turn scenarios..."):
        bootstrap_result = sdk.bootstrap(agent=agent_fn, num_eval_cases=cases)
    eval_cases = bootstrap_result.eval_cases
    mt_scenarios = bootstrap_result.multi_turn_scenarios
    console.print(f"  → {len(bootstrap_result.capabilities)} capabilities")
    console.print(f"  → {len(eval_cases)} single-turn cases, {len(mt_scenarios)} multi-turn scenarios")
    history.record_bootstrap(n_cases=len(eval_cases) + len(mt_scenarios),
                            capabilities=[c['name'] for c in bootstrap_result.capabilities])
    history.snapshot(
        system_prompt=getattr(getattr(agent_fn, '_foundry_agent_config', None), 'system_prompt', '') or '',
        model_id='initial', trigger='bootstrap',
    )

    prev_score = 0.0
    declining_count = 0
    agent_skills = {}

    for cycle_num in range(1, cycles + 1):
        console.rule(f"[bold]Cycle {cycle_num}/{cycles}")

        # 2. Eval (single + multi-turn)
        with console.status(f"Evaluating ({len(eval_cases)} cases + {len(mt_scenarios)} scenarios)..."):
            eval_result = sdk.eval.run_full(agent=agent_fn, cases=eval_cases, scenarios=mt_scenarios, parallelism=1)

        # Record normalized traces for mining / coverage / forecasting.
        try:
            sdk.trace.record_eval_run(eval_result, cases=eval_cases)
        except Exception:
            pass

        score = eval_result.overall_score
        delta = score - prev_score if cycle_num > 1 else 0
        score_color = "green" if score >= 0.6 else "yellow" if score >= 0.3 else "red"
        delta_str = f" (Δ {delta:+.3f} {'📈' if delta > 0 else '📉' if delta < 0 else '—'})" if cycle_num > 1 else ""

        console.print(f"  Score: [{score_color} bold]{score:.3f}[/]{delta_str}  ({eval_result.n_passed}/{eval_result.n_total})")
        for cap, s in sorted(eval_result.capability_scores.items()):
            console.print(f"    {'✓' if s >= 0.6 else '✗'} {cap}: {s:.2f}")

        history.record_eval(score=score, capability_scores=eval_result.capability_scores,
                           n_passed=eval_result.n_passed, n_total=eval_result.n_total,
                           failures=[{'case_id': c.case_id, 'response': c.agent_response[:80]}
                                     for c in eval_result.case_results if not c.passed])

        # 3. Check for declining scores (prompt ceiling)
        if delta < 0:
            declining_count += 1
        else:
            declining_count = 0

        # 4. Decide
        decision = sdk.evolve.run_cycle(agent=agent_fn, eval_result=eval_result)
        gaps = decision.capability_gaps

        if EvolutionAction.NO_ACTION in decision.actions:
            console.print("  [green]✓ All capabilities passing — no action needed[/]")
            history.snapshot(system_prompt=agent_fn._foundry_agent_config.system_prompt if agent_fn._foundry_agent_config else '',
                           skill_prompts=agent_skills, model_id='qwen2.5:3b', eval_score=score,
                           capability_scores=eval_result.capability_scores, trigger=f'cycle_{cycle_num}_passing')
            prev_score = score
            continue

        # 5. Evolve
        if gaps:
            if declining_count >= 2 and train:
                # Prompt ceiling — switch to LoRA
                console.print(f"  [yellow]⚠ Score declining for {declining_count} cycles — switching to LoRA training[/]")
                from foundry.training.backends.mlx_lora import MLXLoRABackend, MLXLoRAConfig
                backend = MLXLoRABackend(config=MLXLoRAConfig(iters=iters))
                with console.status("Generating data + LoRA training..."):
                    cycle_result = sdk.evolve.execute_cycle(
                        agent=agent_fn, eval_result=eval_result,
                        llm_pool=pool, training_backend=backend,
                    )
                if cycle_result.training_job and cycle_result.training_job.succeeded:
                    console.print(f"  → Training complete: loss={cycle_result.training_job.train_loss:.4f}")
                    history.record_training(job_id=cycle_result.training_job.job_id,
                        n_examples=cycle_result.training_examples_generated, result='complete',
                        train_loss=cycle_result.training_job.train_loss, val_loss=cycle_result.training_job.val_loss,
                        adapter_path=cycle_result.training_job.model_id or '')
            else:
                # Prompt evolution only (fast, free — skip data generation)
                console.print(f"  Evolving prompts for: {[g.capability for g in gaps]}")
                from foundry.evolution.prompt_evolver import PromptEvolver
                with console.status("Prompt evolution..."):
                    evolver = PromptEvolver(pool=pool)
                    prompt_result = evolver.evolve(
                        agent=agent_fn,
                        eval_result=eval_result,
                        gaps=gaps,
                        task_spec=sdk.config.task_spec,
                    )

                # Apply prompt patches
                if prompt_result.patches:
                    for p in prompt_result.patches:
                        console.print(f"    ✏️  Prompt: {p.reasoning[:50]}")
                        history.record_prompt_change(p.original[:100], p.revised[:100], p.reasoning)
                        if agent_fn._foundry_agent_config:
                            agent_fn._foundry_agent_config.system_prompt = p.revised

                # Apply skills (dedup)
                config = getattr(agent_fn, "_foundry_agent_config", None)
                for skill in prompt_result.new_skills:
                    if skill.name not in agent_skills:
                        agent_skills[skill.name] = skill.content
                        console.print(f"    🧠 Skill: {skill.name}")
                        history.record_skill_added(skill.name, skill.content[:100], skill.capability_targeted)
                if config:
                    config.skill_prompts = agent_skills.copy()

        # 6. Expand eval coverage toward observed failure modes (blind spots)
        with console.status("Expanding eval coverage on blind spots..."):
            sys_prompt = getattr(getattr(agent_fn, '_foundry_agent_config', None), 'system_prompt', '') or ''
            new_cases = sdk.coverage.expand(
                agent_name,
                pool=pool,
                persist=True,
                tools=getattr(agent_fn, '_foundry_tools', []),
                system_prompt=sys_prompt,
            )
        if new_cases:
            n_old = len(eval_cases)
            eval_cases = eval_cases + new_cases
            console.print(f"  📈 Eval expanded: +{len(new_cases)} targeted cases (blind spots)")
            history.record_eval_expanded(n_old=n_old, n_new=len(eval_cases),
                                        capabilities=sorted({c.capability for c in new_cases}))

        history.snapshot(
            system_prompt=agent_fn._foundry_agent_config.system_prompt if agent_fn._foundry_agent_config else '',
            skill_prompts=agent_skills, model_id='qwen2.5:3b', eval_score=score,
            capability_scores=eval_result.capability_scores, trigger=f'cycle_{cycle_num}',
        )
        prev_score = score

    # Summary
    console.print()
    console.rule("[bold]Evolution Complete")
    trend = history.get_score_trend()
    console.print(f"  Score trend: {' → '.join(f'{s:.3f}' for s in trend)}")
    console.print(f"  Start: {trend[0]:.3f} → End: {trend[-1]:.3f} (Δ {trend[-1]-trend[0]:+.3f})")
    console.print(f"  Versions: {history.n_versions} | Events: {len(history.events)}")
    console.print("  Dashboard: [bold]evoforge report[/]")


@cli.command()
def status():
    """Show current EvoForge data status."""
    console.print(Panel("[bold]EvoForge Status[/]", style="cyan"))

    store = _resolve_store_path()
    if not store.exists():
        console.print("  [dim]No .foundry/ directory found. Run `evoforge bootstrap` first.[/]")
        return

    # Eval cases
    eval_dir = store / "eval_cases"
    if eval_dir.exists():
        tags = [f.stem for f in eval_dir.glob("*.json")]
        console.print(f"  Eval case tags: {tags}")

    # Results
    results_dir = store / "eval_results"
    if results_dir.exists():
        agents = [d.name for d in results_dir.iterdir() if d.is_dir()]
        console.print(f"  Eval results for: {agents}")

    # Trajectories
    traj_dir = store / "trajectories"
    if traj_dir.exists():
        for agent_dir in sorted(traj_dir.iterdir()):
            if agent_dir.is_dir():
                count = len(list(agent_dir.glob("*.json")))
                if count > 0:
                    console.print(f"  Trajectories: {agent_dir.name} ({count})")

    # Training
    train_dir = store / "training"
    if train_dir.exists():
        jobs = [d.name for d in train_dir.iterdir() if d.is_dir() and d.name != "staged"]
        console.print(f"  Training jobs: {len(jobs)}")


@cli.command()
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output HTML file path (default: ~/agent-foundry/.foundry/report.html)",
)
def report(output: str | None):
    """Generate the unified HTML report (evolution history + failure intelligence)."""
    console.print(Panel("[bold]EvoForge Report[/]", style="cyan"))

    store = _resolve_store_path()
    if not store.exists():
        console.print("  [dim]No .foundry/ directory found. Run `evoforge run` first.[/]")
        return

    from foundry.dashboard import DashboardGenerator

    out_path = Path(output).expanduser() if output else (store / "report.html")
    generator = DashboardGenerator(storage_path=str(store))
    report_path = generator.generate(str(out_path))

    console.print(f"  [green]✓[/] Report generated: [bold]{report_path}[/]")


@cli.command()
@click.argument("agent_file")
@click.argument("message")
@click.option("--capability", "-c", default=None, help="Capability hint for the request")
def forecast(agent_file: str, message: str, capability: str | None):
    """Forecast failure risk for a request before running the agent."""
    console.print(Panel("[bold]EvoForge Forecast[/]", style="yellow"))

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, _ = agents[0]

    traces = sdk.trace.load(agent_name)
    if len(traces) < 4:
        console.print(
            f"  [yellow]Not enough recorded traces for '{agent_name}' "
            f"({len(traces)} found, need ≥4).[/]"
        )
        console.print(f"  Run [bold]evoforge eval {agent_file}[/] first to record traces.")
        return

    with console.status("Fitting risk forecaster..."):
        sdk.forecast.fit(agent_name)
    result = sdk.forecast.predict(agent_name, message, capability=capability)

    color = {"low": "green", "medium": "yellow", "high": "red"}[result.risk_level.value]
    lo, hi = result.confidence_interval
    console.print(f"  Request: [dim]{message[:70]}[/]")
    console.print(
        f"  Risk: [{color} bold]{result.risk_level.value.upper()}[/]  "
        f"P(failure)={result.p_failure:.2f}  CI=[{lo:.2f}, {hi:.2f}]"
    )
    console.print(f"  Likely failure mode: [cyan]{result.likely_mode}[/]")
    console.print(f"  Novelty: {result.novelty:.2f}")
    console.print(f"  Rationale: {'; '.join(result.rationale)}")


@cli.command()
@click.argument("agent_file")
@click.option("--output", "-o", default=None, help="Output HTML path (default: <store>/insights.html)")
def insights(agent_file: str, output: str | None):
    """Generate the failure-intelligence dashboard (mining + coverage + forecasting)."""
    console.print(Panel("[bold]EvoForge Insights[/]", style="magenta"))

    agents, sdk = _load_agent_from_file(agent_file)
    agent_name, _ = agents[0]

    traces = sdk.trace.load(agent_name)
    if not traces:
        console.print(
            f"  [yellow]No recorded traces for '{agent_name}'.[/] "
            f"Run [bold]evoforge eval {agent_file}[/] first."
        )
        return

    from foundry.intelligence_dashboard import IntelligenceDashboard

    store = _resolve_store_path()
    out_path = Path(output).expanduser() if output else (store / "insights.html")
    with console.status("Building intelligence dashboard..."):
        path = IntelligenceDashboard().generate(sdk, agent_name, str(out_path))

    console.print(f"  [green]✓[/] Insights dashboard: [bold]{path}[/]")
    console.print("  Open it in your browser to explore failure modes, coverage, and forecasts.")


def main():
    cli()


if __name__ == "__main__":
    main()
