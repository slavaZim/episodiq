"""CLI commands for parameter tuning."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.tune.signal_tuner import (
    CONCURRENCY as SIGNAL_CONCURRENCY,
    DEFAULT_MAX_RATE,
    DEFAULT_MIN_RATE,
    DEFAULT_SAMPLE_TRAJECTORIES,
    SignalTunerResult,
)
from episodiq.analytics.tune.path_frequency import (
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    DEFAULT_SAMPLE_SIZE as PATH_FREQ_SAMPLE_SIZE,
    PathFrequencyResult,
)
from episodiq.analytics.tune.prefetch_topk import (
    CONCURRENCY,
    DEFAULT_PREFETCH_GRID,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TOLERANCE,
    DEFAULT_TOPK_GRID,
    PrefetchTopkResult,
)
from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config

app = typer.Typer()
console = Console()


def _make_session_factory(database_url: str) -> async_sessionmaker:
    engine = create_async_engine(database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _grid_table(result: PrefetchTopkResult) -> Table:
    prefetch_values = sorted({g.prefetch_n for g in result.grid})
    topk_values = sorted({g.top_k for g in result.grid})

    table = Table(title=f"hit@5 grid (n={result.n_sampled}, margin={result.margin:.3f})")
    table.add_column("top_k \\ prefetch", style="cyan", justify="right")
    for pn in prefetch_values:
        table.add_column(str(pn), justify="right")

    lookup = {(g.prefetch_n, g.top_k): g for g in result.grid}
    for tk in topk_values:
        row = [str(tk)]
        for pn in prefetch_values:
            g = lookup.get((pn, tk))
            if g is None:
                row.append("—")
                continue
            val = f"{g.hit_at_5:.1%}"
            if pn == result.suggested_prefetch and tk == result.suggested_top_k:
                val = f"[bold green]{val} <-[/bold green]"
            row.append(val)
        table.add_row(*row)

    return table


@app.command(name="prefetch-topk")
def prefetch_topk(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    prefetch: str = typer.Option(",".join(str(x) for x in DEFAULT_PREFETCH_GRID), "--prefetch", help="Comma-separated prefetch_n values"),
    topk: str = typer.Option(",".join(str(x) for x in DEFAULT_TOPK_GRID), "--topk", help="Comma-separated top_k values"),
    sample_size: int = typer.Option(DEFAULT_SAMPLE_SIZE, "--sample", "-n", help="Paths to sample"),
    concurrency: int = typer.Option(CONCURRENCY, "--concurrency", "-w", help="Concurrent DB queries"),
    tolerance: float = typer.Option(DEFAULT_TOLERANCE, "--tolerance", "-t", help="Hit@5 tolerance for suggestion (e.g. 0.05 = 5%)"),
) -> None:
    """Grid-search prefetch_n x top_k, suggest minimal values for stable hit@5."""
    from episodiq.analytics.tune.prefetch_topk import PrefetchTopkTuner

    prefetch_grid = _parse_int_list(prefetch)
    topk_grid = _parse_int_list(topk)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> PrefetchTopkResult:
        async with session_factory() as session:
            from episodiq.storage.postgres.repository import TrajectoryPathRepository

            path_repo = TrajectoryPathRepository(session)
            tuner = PrefetchTopkTuner(path_repo, session_factory=session_factory)
            return await tuner.run(
                prefetch_grid=prefetch_grid,
                topk_grid=topk_grid,
                sample_size=sample_size,
                concurrency=concurrency,
                tolerance=tolerance,
            )

    result = asyncio.run(_run())

    if result.n_sampled == 0:
        console.print("[red]No paths found. Check that paths have embeddings.[/red]")
        raise typer.Exit(1)

    console.print(f"\nSampled: {result.n_sampled} paths\n")
    console.print(_grid_table(result))
    console.print(
        f"\n[bold]Suggested: EPISODIQ_PREFETCH_N={result.suggested_prefetch}  "
        f"EPISODIQ_TOP_K={result.suggested_top_k}[/bold]"
    )
    console.print(f"  (margin={result.margin:.3f}, 95% binomial CI)")


def _signal_sweep_table(
    title: str,
    thresholds: list,
    suggested: object | None,
) -> Table:
    table = Table(title=title)
    table.add_column("Threshold >=", justify="right", style="cyan")
    table.add_column("Rate%", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("95% CI", justify="right")

    for row in thresholds:
        is_best = suggested and row.threshold == suggested.threshold
        style = "bold green" if is_best else ""
        marker = " <-" if is_best else ""
        table.add_row(
            f"{row.threshold:.2f}",
            f"{row.signal_rate:.1%}",
            f"[{style}]{row.auc:.3f}{marker}[/{style}]" if style else f"{row.auc:.3f}",
            f"{row.auc_ci_lower:.3f}-{row.auc_ci_upper:.3f}",
        )

    return table


def _print_signal_suggestion(label: str, env_var: str, suggested, direction: str) -> None:
    s = suggested
    console.print(f"\n[bold]{label}: {env_var}={s.threshold:.2f}[/bold]")
    console.print(f"  AUC: {s.auc:.3f} (95% CI: {s.auc_ci_lower:.3f}-{s.auc_ci_upper:.3f})")
    console.print(f"  Mean signal rate: {s.signal_rate:.1%}")

    if direction == "fail" and s.auc >= 0.6:
        console.print("  Signal: rate predicts failure")
    elif direction == "success" and s.auc <= 0.4:
        console.print("  Signal: rate predicts success (inverse)")
    else:
        console.print("  Weak signal")


@app.command(name="signal-thresholds")
def signal_thresholds(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    sample_size: int = typer.Option(DEFAULT_SAMPLE_TRAJECTORIES, "--sample", "-n", help="Trajectories to sample (50/50 stratified)"),
    min_rate: float = typer.Option(DEFAULT_MIN_RATE, "--min-rate", help="Min mean signal rate to consider"),
    max_rate: float = typer.Option(DEFAULT_MAX_RATE, "--max-rate", help="Max mean signal rate to consider"),
    concurrency: int = typer.Option(SIGNAL_CONCURRENCY, "--concurrency", "-w", help="Concurrent path analysis"),
) -> None:
    """Sweep fail_similarity thresholds for fail-risk and success-signal action signals."""
    from episodiq.analytics.tune.signal_tuner import SignalTuner
    from episodiq.storage.postgres.repository import TrajectoryRepository

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> SignalTunerResult:
        async with session_factory() as session:
            traj_repo = TrajectoryRepository(session)
            tuner = SignalTuner(traj_repo, session_factory)
            return await tuner.run(
                sample_size=sample_size,
                min_rate=min_rate,
                max_rate=max_rate,
                concurrency=concurrency,
            )

    result = asyncio.run(_run())

    if result.n_trajectories == 0:
        console.print("[red]No trajectories found.[/red]")
        raise typer.Exit(1)

    console.print(f"\nTrajectories: {result.n_trajectories} ({result.n_success} success, {result.n_failure} failure)")
    console.print(f"Paths analyzed: {result.n_paths}")

    # Fail-risk action
    if result.fail_risk_thresholds:
        console.print()
        console.print(_signal_sweep_table(
            "Fail-risk action (max AUC)", result.fail_risk_thresholds, result.fail_risk_suggested,
        ))
        if result.fail_risk_suggested:
            _print_signal_suggestion(
                "Suggested", "EPISODIQ_FAIL_RISK_ACTION_THRESHOLD",
                result.fail_risk_suggested, "fail",
            )
    else:
        console.print("\n[yellow]No fail-risk thresholds in rate range.[/yellow]")

    # Success-signal action
    if result.success_signal_thresholds:
        console.print()
        console.print(_signal_sweep_table(
            "Success-signal action (min AUC)", result.success_signal_thresholds, result.success_signal_suggested,
        ))
        if result.success_signal_suggested:
            _print_signal_suggestion(
                "Suggested", "EPISODIQ_SUCCESS_SIGNAL_ACTION_THRESHOLD",
                result.success_signal_suggested, "success",
            )
    else:
        console.print("\n[yellow]No success-signal thresholds in rate range.[/yellow]")


# ---------------------------------------------------------------------------
# path-freq
# ---------------------------------------------------------------------------

def _entropy_stats_table(result: PathFrequencyResult) -> Table:
    table = Table(title="Entropy Distribution")
    table.add_column("min", justify="right")
    table.add_column("p25", justify="right")
    table.add_column("p50", justify="right", style="bold")
    table.add_column("p75", justify="right")
    table.add_column("max", justify="right")

    s = result.entropy_stats
    table.add_row(*[f"{v:.2f}" for v in (s.min, s.p25, s.p50, s.p75, s.max)])

    return table


def _variance_table(result: PathFrequencyResult) -> Table:
    t = result.thresholds
    table = Table(title=f"Action Variance (low ≤ {t.low_entropy:.2f}, high ≥ {t.high_entropy:.2f})")
    table.add_column("Flag", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Description")

    descriptions = {
        "low": "entropy ≤ low threshold — few likely actions, very predictable",
        "normal": "between thresholds — typical variance",
        "high": "entropy ≥ high threshold — many options, unpredictable",
    }

    for key in ("low", "normal", "high"):
        n = result.variance_counts.get(key, 0)
        pct = n / result.n_valid * 100 if result.n_valid > 0 else 0
        table.add_row(key, str(n), f"{pct:.1f}%", descriptions[key])

    return table


@app.command(name="path-freq")
def path_freq(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    low_pct: float = typer.Option(DEFAULT_LOW_PERCENTILE, "--low-pct", "-l", help="Percentile for low-variance threshold"),
    high_pct: float = typer.Option(DEFAULT_HIGH_PERCENTILE, "--high-pct", "-h", help="Percentile for high-variance threshold"),
    sample_size: int = typer.Option(PATH_FREQ_SAMPLE_SIZE, "--sample", "-n", help="Paths to sample"),
) -> None:
    """Analyze trajectory paths and suggest action-variance thresholds."""
    from episodiq.analytics.tune.path_frequency import PathFrequencyTuner
    from episodiq.storage.postgres.repository import TrajectoryPathRepository

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> PathFrequencyResult:
        async with session_factory() as session:
            path_repo = TrajectoryPathRepository(session)
            tuner = PathFrequencyTuner(
                path_repo,
                low_percentile=low_pct,
                high_percentile=high_pct,
            )
            return await tuner.run(sample_size=sample_size)

    result = asyncio.run(_run())

    if result.thresholds is None:
        console.print(f"[red]Too few valid signals ({result.n_valid}). Need more data.[/red]")
        raise typer.Exit(1)

    console.print(f"\nPaths: {result.n_sampled} sampled, {result.n_valid} valid\n")
    console.print(_entropy_stats_table(result))
    console.print()
    console.print(_variance_table(result))
    console.print()
    console.print(f"[bold]Suggested thresholds (p{low_pct:.0f} / p{high_pct:.0f}):[/bold]")
    console.print(f"  EPISODIQ_LOW_ENTROPY={result.thresholds.low_entropy:.2f}  EPISODIQ_HIGH_ENTROPY={result.thresholds.high_entropy:.2f}")


