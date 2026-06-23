"""CLI commands for parameter tuning."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from episodiq.analytics.tune.path_frequency import (
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    DEFAULT_SAMPLE_SIZE as PATH_FREQ_SAMPLE_SIZE,
    PathFrequencyResult,
)
from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config
from episodiq.retrieval.tune.constants import (
    DEFAULT_AGGREGATION_GRID,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EVAL_MIN_STEP,
    DEFAULT_MULTIVARIATE,
    DEFAULT_N_JOBS,
    DEFAULT_N_TRIALS,
    DEFAULT_N_WORKERS,
    DEFAULT_OPTUNA_SEED,
    DEFAULT_WINDOW_GRID,
)

app = typer.Typer()
console = Console()


def _make_session_factory(
    database_url: str, *, pool_size: int = 5, max_overflow: int = 10,
) -> async_sessionmaker:
    """Default to a sized connection pool — the sweep opens one session
    per snapshot per trial under high concurrency, so reusing
    connections is much cheaper than ``NullPool``'s
    "new-connection-per-checkout".
    """
    engine = create_async_engine(
        database_url, pool_size=pool_size, max_overflow=max_overflow,
    )
    return async_sessionmaker(engine, expire_on_commit=False)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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
    w_grid: str = typer.Option(
        ",".join(str(w) for w in DEFAULT_WINDOW_GRID), "--w-grid",
        help="Comma-separated EVEN window sizes W = 2w (e.g. 10,14 = "
             "w∈{5,7}). Each value triggers a fresh LSH alt-table rebuild "
             "from existing trace_tokens.",
    ),
    n_trials: int = typer.Option(
        DEFAULT_N_TRIALS, "--n-trials",
        help="Optuna trials per W per metric study.",
    ),
    limit: int = typer.Option(
        None, "--limit", help="Tune-slice trajectory count (default: all)",
    ),
    offset: int = typer.Option(
        0, "--offset", help="Tune-slice trajectory offset",
    ),
    eval_min_step: int = typer.Option(
        DEFAULT_EVAL_MIN_STEP, "--eval-min-step",
        help="Lower bound on snapshot step for the per-step AUC window.",
    ),
    seed: int = typer.Option(
        DEFAULT_OPTUNA_SEED, "--seed", help="Optuna sampler seed",
    ),
    alt_table_suffix: str = typer.Option(
        "", "--alt-table-suffix",
        help="Suffix appended to alt LSH table names so parallel runs of "
             "this command don't collide. Defaults to empty.",
    ),
    agg_grid: str = typer.Option(
        ",".join(DEFAULT_AGGREGATION_GRID), "--agg-grid",
        help="Comma-separated neighborhood aggregations to sweep over: "
             "'mean,min'. Each one is its own outer slot alongside W "
             "with a fresh cache.",
    ),
    multivariate: bool = typer.Option(
        DEFAULT_MULTIVARIATE, "--multivariate/--no-multivariate",
        help="Use TPE's multivariate Parzen estimator (joint over continuous "
             "params). Catches correlations univariate TPE misses.",
    ),
    n_jobs: int = typer.Option(
        DEFAULT_N_JOBS, "--n-jobs",
        help="Optuna threaded trials per metric study (outer concurrency). "
             "Total peak DB sessions ≈ n_jobs × n_workers.",
    ),
    n_workers: int = typer.Option(
        DEFAULT_N_WORKERS, "--n-workers",
        help="Concurrent retrieval.search() calls inside one trial (inner "
             "concurrency, asyncio.Semaphore-bounded).",
    ),
    early_stop_patience: int = typer.Option(
        DEFAULT_EARLY_STOP_PATIENCE, "--early-stop-patience",
        help="Stop one (W, agg) study once this many consecutive trials "
             "fail to improve best_seen. 0 disables.",
    ),
    save_trials: Path = typer.Option(
        None, "--save-trials",
        help="Write all trials (params + per-metric AUC) as CSV",
    ),
    shuffle_seed: int = typer.Option(
        None, "--shuffle-seed",
        help="Deterministic Random(seed).shuffle of trajectories before "
             "offset/limit slicing. Use the same seed in basic.py for a "
             "parallel tune/eval split.",
    ),
    shuffle_field: str = typer.Option(
        None, "--shuffle-field",
        help="Dotted path selecting the pre-shuffle sort key (e.g. 'id' "
             "or 'meta.instance_id'). Two pipelines that share the same "
             "field + seed pick identical tune/eval slices. None = use "
             "the DB-provided UUID order.",
    ),
    stratify_field: str = typer.Option(
        None, "--stratify-field",
        help="Dotted path used to stratify the shuffled list (e.g. "
             "'status'). Every tune/eval prefix inherits the population's "
             "class distribution along that key, neutralising seed-driven "
             "imbalance. None = no stratification.",
    ),
    objective_metric: str = typer.Option(
        None, "--objective-metric",
        help="Restrict TPE's metric-as-categorical sampler to a single "
             "metric (e.g. 'cummean'). When set, every trial optimises "
             "that metric's weighted AUC. Default samples all three.",
    ),
) -> None:
    """Optuna sweep of the retrieval cascade across W + cheap knobs.

    Outer grid: W. Each W rebuilds its alt LSH table from existing
    trace_tokens (no retokenization). Inner: three Optuna TPE studies
    (one per metric — cummax, cummean, cummeanmax) over
    prefetch_n_uniq / jaccard_n_uniq / top_k / penalty_shape / lam /
    gap_open / gap_extend. Each trial fans out ``retrieval.search()``
    across snapshots via asyncio.gather; concurrency = n_jobs × n_workers.
    """
    import csv as _csv
    from episodiq.analytics.metrics import SIMILARITY_METRICS as METRICS
    from episodiq.retrieval.tune.sweep import (
        RetrievalSweep, RetrievalSweepConfig,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv(env)
    config = get_config()

    # Size the pool to comfortably fit n_workers × n_jobs concurrent
    # sessions per W study, plus a margin for the warm-up phase.
    pool_target = max(n_workers * n_jobs, 16)
    session_factory = _make_session_factory(
        config.get_database_url(),
        pool_size=pool_target, max_overflow=pool_target,
    )
    grid = tuple(_parse_int_list(w_grid))
    agg_grid_t = tuple(s.strip() for s in agg_grid.split(",") if s.strip())
    from episodiq.analytics.metrics import SIMILARITY_METRICS as _METS
    metric_choices = (
        (objective_metric,) if objective_metric else _METS
    )
    sweep_cfg = RetrievalSweepConfig(
        window_grid=grid,
        n_trials_per_window=n_trials,
        eval_min_step=eval_min_step,
        optuna_seed=seed,
        alt_table_suffix=alt_table_suffix,
        aggregation_grid=agg_grid_t,
        metric_choices=metric_choices,
        multivariate=multivariate,
        n_jobs=n_jobs,
        n_workers=n_workers,
        early_stop_patience=early_stop_patience,
    )

    async def _run():
        sweep = RetrievalSweep(
            session_factory, sweep_cfg, limit=limit, offset=offset,
            shuffle_seed=shuffle_seed, shuffle_field=shuffle_field,
            stratify_field=stratify_field,
        )
        return await sweep.run()

    report = asyncio.run(_run())

    if not report.trials:
        console.print("[red]No trials produced any signal. Check --limit / --eval-min-step.[/red]")
        raise typer.Exit(1)

    overall_best = report.best
    for m in METRICS:
        table = Table(title=f"Retrieval sweep — {m}")
        table.add_column("W", justify="right", style="cyan")
        table.add_column("agg", justify="left")
        table.add_column("AUC", justify="right")
        table.add_column("params", overflow="fold")
        for W in grid:
            for agg in agg_grid_t:
                t = report.by_window_agg_per_metric(W, agg).get(m)
                if t is None:
                    continue
                is_overall = overall_best is not None and t is overall_best
                auc_cell = f"{t.target_auc:.4f}"
                if is_overall:
                    auc_cell = f"[bold green]{auc_cell}[/bold green]"
                params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
                table.add_row(str(W), agg, auc_cell, params_str)
        console.print(table)
        console.print()

    if save_trials:
        param_keys = sorted(report.trials[0].params.keys())
        fieldnames = [
            "window", "aggregation", "target_metric", *param_keys, "target_auc",
        ]
        fieldnames += [f"auc_{m}" for m in METRICS]
        with open(save_trials, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for t in report.trials:
                row = {
                    "window": t.window,
                    "aggregation": t.aggregation,
                    "target_metric": t.target_metric,
                    "target_auc": f"{t.target_auc:.4f}",
                }
                for k, v in t.params.items():
                    row[k] = (
                        f"{v:.4f}" if isinstance(v, float) else str(v)
                    )
                for m in METRICS:
                    v = t.auc_per_metric.get(m)
                    row[f"auc_{m}"] = f"{v:.4f}" if v is not None else ""
                w.writerow(row)
        console.print(f"Saved trials CSV to {save_trials}")

    # Per-step AUC curves live in eval-time tooling
    # (benchmarks/demo_eval/eval_cascade.py) — the sweep keeps trial
    # output to scalar weighted_aucs to stay leakage-cheap.


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
    from episodiq.storage.postgres.repository import (
        TrajectoryPathRepository,
        TrajectoryWindowLSHRepository,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> PathFrequencyResult:
        async with session_factory() as session:
            path_repo = TrajectoryPathRepository(session)
            lsh_repo = TrajectoryWindowLSHRepository(session)
            retrieval = Retrieval(
                path_repo, lsh_repo,
                minhash_config=config.minhash,
                retrieval_config=config.retrieval,
                scoring_config=config.scoring,
            )
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


if __name__ == "__main__":
    app()
