"""CLI commands for parameter tuning."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.tune.path_frequency import (
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    DEFAULT_SAMPLE_SIZE as PATH_FREQ_SAMPLE_SIZE,
    PathFrequencyResult,
)
from episodiq.analytics.tune.top_k import (
    CONCURRENCY as TOPK_CONCURRENCY,
    DEFAULT_SAMPLE_SIZE as TOPK_SAMPLE_SIZE,
    DEFAULT_TOLERANCE,
    DEFAULT_TOPK_GRID,
    TopKResult,
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


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------

def _topk_table(result: TopKResult) -> Table:
    table = Table(title="fail_frac AUC by top_k")
    table.add_column("top_k", justify="right", style="cyan")
    table.add_column("AUC", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")

    for point in result.grid:
        is_best = point.top_k == result.suggested_top_k
        style = "bold green" if is_best else ""
        marker = " <-" if is_best else ""
        auc_cell = (
            f"[{style}]{point.auc:.3f}{marker}[/{style}]" if style
            else f"{point.auc:.3f}"
        )
        table.add_row(
            str(point.top_k),
            auc_cell,
            f"{point.auc_ci_lower:.3f}-{point.auc_ci_upper:.3f}",
            str(point.n_trajectories),
        )

    return table


@app.command(name="top-k")
def top_k(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    topk: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_TOPK_GRID),
        "--topk", help="Comma-separated top_k values",
    ),
    sample_size: int = typer.Option(
        TOPK_SAMPLE_SIZE, "--sample", "-n", help="Trajectories to sample",
    ),
    concurrency: int = typer.Option(
        TOPK_CONCURRENCY, "--concurrency", "-w", help="Concurrent retrievals",
    ),
    tolerance: float = typer.Option(
        DEFAULT_TOLERANCE, "--tolerance", "-t",
        help="AUC tolerance for suggesting the smallest top_k",
    ),
) -> None:
    """Sweep top_k, suggest the smallest value within tolerance of the best fail_frac AUC."""
    from episodiq.analytics.tune.top_k import TopKTuner
    from episodiq.storage.postgres.repository import TrajectoryRepository

    topk_grid = _parse_int_list(topk)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> TopKResult:
        async with session_factory() as session:
            traj_repo = TrajectoryRepository(session)
            tuner = TopKTuner(traj_repo, session_factory)
            return await tuner.run(
                topk_grid=topk_grid,
                sample_size=sample_size,
                concurrency=concurrency,
                tolerance=tolerance,
            )

    result = asyncio.run(_run())

    if not result.grid:
        console.print(
            "[red]No usable samples. Check that paths have profiles.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"\nTrajectories: {result.n_trajectories}  Paths: {result.n_paths}\n"
    )
    console.print(_topk_table(result))
    console.print(f"\n[bold]Suggested: EPISODIQ_RETRIEVAL_TOP_K={result.suggested_top_k}[/bold]")


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


@app.command(name="retrieval-sweep")
def retrieval_sweep(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    top_k: str = typer.Option(None, "--top-k", help="Comma-separated top_k values"),
    sim: str = typer.Option(None, "--sim", help="Comma-separated similarity thresholds"),
    limit: int = typer.Option(None, "--limit", help="Tune-query trajectory count"),
    offset: int = typer.Option(0, "--offset", help="Tune-query trajectory offset"),
    order_file: Path = typer.Option(
        None, "--order-file",
        help="JSON list of trajectory UUIDs; overrides default UUID ordering "
             "for --offset/--limit slicing. Use for stratified tune/eval splits.",
    ),
    save_output: Path = typer.Option(None, "--save-output", help="Save full sweep as CSV"),
) -> None:
    """Sweep (top_k, similarity_threshold) for MinHash retrieval.

    Tune queries: deterministic slice over completed trajectories. Default
    order is by trajectory UUID; pass ``--order-file`` to use an externally
    computed (e.g. stratified) ordering. Each query path runs leave-one-out
    against the rest of the corpus.
    """
    import csv as _csv
    import json as _json
    from uuid import UUID
    from episodiq.retrieval.tune.sweep import (
        DEFAULT_SIM_GRID, DEFAULT_TOPK_GRID, DEFAULT_TUNE_LIMIT, RetrievalSweep,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    topk_grid = _parse_int_list(top_k) if top_k else list(DEFAULT_TOPK_GRID)
    sim_grid = (
        [float(x.strip()) for x in sim.split(",") if x.strip()]
        if sim else list(DEFAULT_SIM_GRID)
    )
    tune_limit = limit if limit is not None else DEFAULT_TUNE_LIMIT

    ordered_ids: list[UUID] | None = None
    if order_file is not None:
        with open(order_file) as f:
            ordered_ids = [UUID(s) for s in _json.load(f)]
        logger = logging.getLogger(__name__)
        logger.info("ordering: %s (%d ids)", order_file, len(ordered_ids))

    async def _run():
        sweep = RetrievalSweep(
            session_factory, limit=tune_limit, offset=offset,
            ordered_traj_ids=ordered_ids,
        )
        return await sweep.run(topk_grid, sim_grid)

    report = asyncio.run(_run())

    table = Table(title="Retrieval sweep (AUC@s60 + coverage)")
    for col, just in [
        ("top_k", "right"), ("sim", "right"),
        ("cov@s60", "right"), ("AUC@s60", "right"),
        ("n_snaps", "right"),
    ]:
        table.add_column(col, justify=just)
    best = report.best

    def _fmt_auc(v):
        return f"{v:.3f}" if v is not None else "n/a"

    def _fmt_cov(c):
        return f"{c * 100:.1f}%" if c is not None else "n/a"

    for p in sorted(
        report.points,
        key=lambda x: -(x.auc_step60_current or -1.0),
    ):
        is_best = best is not None and p == best
        s60 = _fmt_auc(p.auc_step60_current)
        s60_cell = f"[bold green]{s60}[/bold green]" if is_best else s60
        table.add_row(
            str(p.top_k), f"{p.similarity_threshold:.2f}",
            _fmt_cov(p.coverage_step60), s60_cell,
            str(p.n_snapshots),
        )
    console.print(table)

    if save_output:
        with open(save_output, "w", newline="") as f:
            w = _csv.DictWriter(
                f, fieldnames=[
                    "top_k", "similarity_threshold",
                    "coverage_step60", "auc_step60_current", "n_snapshots",
                ],
            )
            w.writeheader()
            for p in report.points:
                w.writerow({
                    "top_k": p.top_k,
                    "similarity_threshold": f"{p.similarity_threshold:.4f}",
                    "coverage_step60": (
                        f"{p.coverage_step60:.4f}"
                        if p.coverage_step60 is not None else ""
                    ),
                    "auc_step60_current": (
                        f"{p.auc_step60_current:.4f}"
                        if p.auc_step60_current is not None else ""
                    ),
                    "n_snapshots": p.n_snapshots,
                })
        console.print(f"Saved sweep CSV to {save_output}")


@app.command(name="path-freq")
def path_freq(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    low_pct: float = typer.Option(DEFAULT_LOW_PERCENTILE, "--low-pct", "-l", help="Percentile for low-variance threshold"),
    high_pct: float = typer.Option(DEFAULT_HIGH_PERCENTILE, "--high-pct", "-h", help="Percentile for high-variance threshold"),
    sample_size: int = typer.Option(PATH_FREQ_SAMPLE_SIZE, "--sample", "-n", help="Paths to sample"),
) -> None:
    """Analyze trajectory paths and suggest action-variance thresholds."""
    from episodiq.analytics.tune.path_frequency import PathFrequencyTuner
    from episodiq.retrieval.retrieval import Retrieval
    from episodiq.storage.postgres.repository import TrajectoryPathRepository

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> PathFrequencyResult:
        async with session_factory() as session:
            path_repo = TrajectoryPathRepository(session)
            retrieval = Retrieval(path_repo, config.retrieval)
            tuner = PathFrequencyTuner(
                path_repo, retrieval,
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
