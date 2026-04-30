"""CLI command for rendering a full trajectory report."""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.log_builder import LogBuilder
from episodiq.analytics.path_frequency import PathFrequencyTagger, PathFrequencyThresholds
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.cli.env import load_config
from episodiq.cli.rendering import (
    LogRenderer,
    OutputFormat,
    RenderContext,
    RenderMode,
    TrajectoryStats,
)

console = Console(stderr=True)


def _detect_format(format_arg: str) -> OutputFormat:
    if format_arg == "json":
        return OutputFormat.JSON
    if format_arg == "pretty":
        return OutputFormat.PRETTY
    return OutputFormat.PRETTY if sys.stdout.isatty() else OutputFormat.JSON


def report(
    trajectory_id: str = typer.Argument(..., help="Trajectory UUID"),
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    format: str = typer.Option("auto", "--format", "-f", help="pretty|json|auto"),
) -> None:
    """Render a full trajectory report with analytics signals."""
    # Validate UUID
    try:
        tid = UUID(trajectory_id)
    except ValueError:
        console.print(f"[red]Invalid trajectory ID: {trajectory_id}[/red]")
        raise typer.Exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(env)
    engine = create_async_engine(config.get_database_url(), poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    output_format = _detect_format(format)

    async def _run() -> None:
        from episodiq.storage.postgres.models import Trajectory
        from episodiq.storage.postgres.repository import TrajectoryPathRepository

        async with session_factory() as session:
            # Load trajectory
            trajectory = await session.get(Trajectory, tid)
            if trajectory is None:
                console.print(f"[red]Trajectory {tid} not found[/red]")
                raise typer.Exit(1)

            if trajectory.status == "active":
                console.print(
                    f"[yellow]Warning: trajectory {tid} is still active, "
                    f"report may be incomplete[/yellow]"
                )

            # Load paths
            path_repo = TrajectoryPathRepository(session)
            paths = await path_repo.get_trajectory_paths(tid)

            if not paths:
                console.print(f"[red]No completed paths for trajectory {tid}[/red]")
                raise typer.Exit(1)

            # Build analyzer, tagger, predictor, builder
            analyzer = TransitionAnalyzer(path_repo=path_repo, config=config.analytics)
            tagger = PathFrequencyTagger(
                PathFrequencyThresholds(
                    config.analytics.low_entropy,
                    config.analytics.high_entropy,
                ),
            )
            from episodiq.analytics.dead_end.inference import DeadEndPredictor
            predictor = DeadEndPredictor(
                model_path=Path(config.analytics.dead_end_model),
                threshold=config.analytics.dead_end_threshold,
            )
            predictor.load()
            builder = LogBuilder(
                path_frequency_tagger=tagger,
                dead_end_predictor=predictor if predictor.is_available else None,
            )

            # Analyze all paths in parallel
            analytics_list = await asyncio.gather(
                *[analyzer.analyze(p) for p in paths]
            )

            # Build all entries
            entry_pairs = []
            dead_end_flagged = False
            for path, analytics in zip(paths, analytics_list):
                entries, dead_end_flagged = builder.build(path, analytics, dead_end_flagged)
                entry_pairs.append((entries[0], entries[1]))

            # Compute stats
            unannotated = sum(
                1 for obs, act in entry_pairs
                if "annotation" not in obs or "annotation" not in act
            )
            dead_end_step = next(
                (i for i, (obs, _) in enumerate(entry_pairs) if obs.get("dead_end_flagged")),
                None,
            )
            duration_s = (
                trajectory.updated_at - trajectory.created_at
            ).total_seconds()

            last_path = paths[-1] if paths else None
            stats = TrajectoryStats(
                trajectory_id=str(tid),
                started_at=trajectory.created_at,
                ended_at=trajectory.updated_at,
                duration_s=duration_s,
                step_count=len(paths),
                status=trajectory.status,
                fail_risk_action_count=last_path.fail_risk_action_count if last_path else 0,
                fail_risk_transition_count=last_path.fail_risk_transition_count if last_path else 0,
                success_signal_action_count=last_path.success_signal_action_count if last_path else 0,
                success_signal_transition_count=last_path.success_signal_transition_count if last_path else 0,
                loop_count=last_path.loop_count if last_path else 0,
                dead_end_first_step=dead_end_step,
                unannotated_step_count=unannotated,
            )

            # Render
            out_console = Console() if output_format == OutputFormat.PRETTY else Console(stderr=True)
            renderer = LogRenderer(out_console)
            ctx = RenderContext(mode=RenderMode.REPORT, format=output_format)

            renderer.render_trajectory_header(stats, ctx)
            for obs, act in entry_pairs:
                renderer.render_entry_pair(obs, act, ctx)

    asyncio.run(_run())
