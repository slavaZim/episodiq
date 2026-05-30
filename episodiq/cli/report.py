"""CLI command for rendering a full trajectory report. Thin wrapper over
:class:`~episodiq.analytics.trajectory_report.TrajectoryReportBuilder`.
"""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.trajectory_report import TrajectoryReportBuilder
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


def _parse_steps_range(value: str) -> tuple[int | None, int | None]:
    lo, _, hi = value.partition("-")
    return (
        int(lo) if lo else None,
        int(hi) if hi else None,
    )


def report(
    trajectory_id: str = typer.Argument(..., help="Trajectory UUID"),
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    format: str = typer.Option("auto", "--format", "-f", help="pretty|json|auto"),
    analytics: bool = typer.Option(
        False, "--analytics", "-a",
        help="Run retrieval + transition analysis and annotate entries with loop / fail-frac / variance signals.",
    ),
    steps: str = typer.Option(
        None, "--steps",
        help="Render only a slice of steps, inclusive (e.g. '47-75' or '60-' for from-step-60-onward).",
    ),
) -> None:
    """Render a full trajectory report. Use ``-a`` to include analytics signals."""
    try:
        tid = UUID(trajectory_id)
    except ValueError:
        console.print(f"[red]Invalid trajectory ID: {trajectory_id}[/red]")
        raise typer.Exit(1)

    step_lo, step_hi = _parse_steps_range(steps) if steps else (None, None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(env)
    engine = create_async_engine(config.get_database_url(), poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    output_format = _detect_format(format)

    async def _run() -> None:
        async with session_factory() as session:
            report_builder = TrajectoryReportBuilder(
                session,
                analytics_config=config.analytics,
                retrieval_config=config.retrieval,
            )
            try:
                result = await report_builder.build(tid, analytics=analytics)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1)

            if result is None:
                console.print(f"[red]Trajectory {tid} not found[/red]")
                raise typer.Exit(1)

            trajectory = result.trajectory
            if trajectory.status == "active":
                console.print(
                    f"[yellow]Warning: trajectory {tid} is still active, "
                    f"report may be incomplete[/yellow]"
                )

            stats = TrajectoryStats(
                trajectory_id=str(tid),
                started_at=trajectory.created_at,
                ended_at=trajectory.updated_at,
                duration_s=(
                    trajectory.updated_at - trajectory.created_at
                ).total_seconds(),
                step_count=len(result.entry_pairs),
                status=trajectory.status,
                loop_count=result.loop_count,
                unclassified_step_count=result.unclassified_step_count,
                peak_fail_frac=result.peak_fail_frac,
                variance_high_count=result.variance_high_count,
                variance_low_count=result.variance_low_count,
            )

            # Render. With skip_initial_observation the trajectory's initial
            # observation (the task prompt) is omitted from the report output.
            out_console = (
                Console() if output_format == OutputFormat.PRETTY
                else Console(stderr=True)
            )
            renderer = LogRenderer(out_console)
            ctx = RenderContext(mode=RenderMode.REPORT, format=output_format)

            renderer.render_trajectory_header(stats, ctx)
            skip_initial = config.skip_initial_observation
            for i, (obs, act) in enumerate(result.entry_pairs):
                if step_lo is not None and i < step_lo:
                    continue
                if step_hi is not None and i > step_hi:
                    break
                renderer.render_entry_pair(
                    obs, act, ctx,
                    skip_observation=(i == 0 and skip_initial),
                )

    asyncio.run(_run())
