"""CLI commands for clustering: run and grid-search."""

import asyncio
import csv
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.cli.env import _load_dotenv
from episodiq.clustering.grid_search import GridJobSpec, GridSearchEntry, GridSearchReport
from episodiq.clustering.constants import DEFAULT_PARAMS, Params
from episodiq.clustering.manager import CategoryResult, JobSpec
from episodiq.analytics.path_state import PathStateCalculator
from episodiq.clustering.path_updater import WORKERS, TrajectoryPathUpdater
from episodiq.clustering.pipeline import ClusteringPipeline, GridSearchClusteringPipeline
from episodiq.config import get_config

app = typer.Typer()
console = Console()

_TYPE_ALIASES: dict[str, str] = {"a": "action", "o": "observation"}


def _parse_types(raw: list[str]) -> list[str]:
    return [_TYPE_ALIASES.get(t, t) for t in raw]


def _build_params(**overrides) -> Params:
    return Params(**{k: v for k, v in overrides.items() if v is not None})


def _build_specs(types: list[str], categories: list[str], params: Params = DEFAULT_PARAMS) -> list[JobSpec]:
    if not types and not categories:
        return []
    types = types or ["action", "observation"]
    categories = categories or ["text", "tool"]
    return [JobSpec(type=t, category=c, params=params) for t in types for c in categories]


def _build_custom_grid(**axes: str | None) -> list[Params] | None:
    """Build grid from comma-separated axis values. Returns None if no axes specified."""
    from itertools import product

    parsers = {
        "min_cluster_size": int, "min_samples": int,
        "umap_dims": int, "umap_n_neighbors": int,
        "cluster_selection_method": str, "cluster_selection_epsilon": float,
    }
    parsed: dict[str, list] = {}
    any_specified = False
    for name, parse in parsers.items():
        raw = axes.get(name)
        if raw:
            parsed[name] = [parse(v.strip()) for v in raw.split(",")]
            any_specified = True
        else:
            parsed[name] = [getattr(DEFAULT_PARAMS, name)]

    if not any_specified:
        return None

    return [Params(**dict(zip(parsed.keys(), combo))) for combo in product(*parsed.values())]


def _build_grid_specs(
    types: list[str], categories: list[str],
    grid: list[Params] | None = None,
) -> list[GridJobSpec]:
    if not types and not categories:
        return []
    types = types or ["action", "observation"]
    categories = categories or ["text", "tool"]
    if grid:
        return [GridJobSpec(type=t, category=c, params_list=grid) for t in types for c in categories]
    return [GridJobSpec(type=t, category=c) for t in types for c in categories]


def _results_table(results: list[CategoryResult]) -> Table:
    table = Table(title="Clustering Results")
    table.add_column("Type", style="cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Messages", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column("Noise%", justify="right")
    table.add_column("DBCV", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("Score", justify="right")

    for r in results:
        table.add_row(
            r.type,
            r.category,
            str(len(r.message_ids)),
            str(r.n_clusters),
            str(r.noise_count),
            f"{r.noise_ratio:.1%}",
            f"{r.dbcv:.3f}",
            f"{r.entropy:.3f}",
            f"{r.score:.3f}",
        )
    return table


def _grid_top_table(key: str, entries: list[GridSearchEntry], top_n: int = 5) -> Table:
    if not entries:
        return Table(title=f"{key} (no results)")

    winner = entries[0]
    table = Table(title=f"{key} (winner: min_cs={winner.params.min_cluster_size}, min_s={winner.params.min_samples})")
    table.add_column("#", justify="right")
    table.add_column("min_cs", justify="right")
    table.add_column("min_s", justify="right")
    table.add_column("umap_d", justify="right")
    table.add_column("umap_nn", justify="right")
    table.add_column("method", justify="right")
    table.add_column("eps", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Noise%", justify="right")
    table.add_column("DBCV", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for i, e in enumerate(entries[:top_n], 1):
        table.add_row(
            str(i),
            str(e.params.min_cluster_size),
            str(e.params.min_samples),
            str(e.params.umap_dims),
            str(e.params.umap_n_neighbors),
            e.params.cluster_selection_method,
            f"{e.params.cluster_selection_epsilon:.1f}",
            str(e.n_clusters),
            f"{e.noise_ratio:.1%}",
            f"{e.dbcv:.3f}",
            f"{e.entropy:.3f}",
            f"{e.score:.3f}",
        )
    return table


def _save_csv(report: GridSearchReport, path: Path) -> None:
    fieldnames = [
        "type", "category", "min_cluster_size", "min_samples",
        "umap_dims", "umap_n_neighbors",
        "cluster_selection_method", "cluster_selection_epsilon",
        "n_clusters", "noise_count", "noise_ratio", "dbcv", "entropy", "score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, entries in report.entries.items():
            type_, category = key.split(":", 1)
            for e in entries:
                writer.writerow({
                    "type": type_,
                    "category": category,
                    "min_cluster_size": e.params.min_cluster_size,
                    "min_samples": e.params.min_samples,
                    "umap_dims": e.params.umap_dims,
                    "umap_n_neighbors": e.params.umap_n_neighbors,
                    "cluster_selection_method": e.params.cluster_selection_method,
                    "cluster_selection_epsilon": e.params.cluster_selection_epsilon,
                    "n_clusters": e.n_clusters,
                    "noise_count": e.noise_count,
                    "noise_ratio": f"{e.noise_ratio:.4f}",
                    "dbcv": f"{e.dbcv:.4f}",
                    "entropy": f"{e.entropy:.4f}",
                    "score": f"{e.score:.4f}",
                })
    console.print(f"Saved grid search report to {path}")


def _make_session_factory(database_url: str) -> async_sessionmaker:
    engine = create_async_engine(database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@app.command()
def run(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    types: Optional[list[str]] = typer.Option(None, "--type", "-t", help="action/observation/a/o"),
    categories: Optional[list[str]] = typer.Option(None, "--category", "-c", help="text, tool (all tools), or specific tool e.g. str_replace_editor"),
    min_cluster_size: Optional[int] = typer.Option(None, "--min-cs", help="HDBSCAN min_cluster_size"),
    min_samples: Optional[int] = typer.Option(None, "--min-s", help="HDBSCAN min_samples"),
    umap_dims: Optional[int] = typer.Option(None, "--umap-dims", "-d", help="UMAP dimensions"),
    umap_n_neighbors: Optional[int] = typer.Option(None, "--umap-nn", help="UMAP n_neighbors"),
    selection_method: Optional[str] = typer.Option(None, "--selection-method", help="HDBSCAN cluster_selection_method (eom/leaf)"),
    selection_epsilon: Optional[float] = typer.Option(None, "--selection-epsilon", help="HDBSCAN cluster_selection_epsilon"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip DB writes"),
) -> None:
    """Run clustering pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    params = _build_params(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_dims=umap_dims,
        umap_n_neighbors=umap_n_neighbors,
        cluster_selection_method=selection_method,
        cluster_selection_epsilon=selection_epsilon,
    )
    parsed_types = _parse_types(types or [])
    specs = _build_specs(parsed_types, categories or [], params)

    async def _run():
        pipeline = ClusteringPipeline(session_factory, specs)
        return await pipeline.run(dry_run=dry_run)

    results = asyncio.run(_run())

    console.print(_results_table(results))


@app.command(name="build-paths")
def build_paths(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    fill_signals: bool = typer.Option(False, "--fill-signals", help="Populate signal counts, improves dead end signal. Might take some time"),
    workers: int = typer.Option(WORKERS, "--workers", "-w", help="Concurrency for path rebuilding"),
) -> None:
    """Rebuild all trajectory paths from message cluster labels."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run():
        async with session_factory() as session:
            from episodiq.storage.postgres.repository import MessageRepository, TrajectoryPathRepository
            msg_repo = MessageRepository(session)
            path_repo = TrajectoryPathRepository(session)
            updater = TrajectoryPathUpdater(
                msg_repo, path_repo, PathStateCalculator(),
                session_factory=session_factory,
                workers=workers,
            )
            total = await updater.update(fill_signals=fill_signals)
            await session.commit()
            return total

    total = asyncio.run(_run())
    console.print(f"Rebuilt {total} trajectory paths")


@app.command(name="grid-search")
def grid_search(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
    types: Optional[list[str]] = typer.Option(None, "--type", "-t", help="action/observation/a/o"),
    categories: Optional[list[str]] = typer.Option(None, "--category", "-c", help="text, tool (all tools), or specific tool e.g. str_replace_editor"),
    min_cluster_size: Optional[str] = typer.Option(None, "--min-cs", help="Comma-separated min_cluster_size values"),
    min_samples: Optional[str] = typer.Option(None, "--min-s", help="Comma-separated min_samples values"),
    umap_dims: Optional[str] = typer.Option(None, "--umap-dims", "-d", help="Comma-separated UMAP dimensions"),
    umap_n_neighbors: Optional[str] = typer.Option(None, "--umap-nn", help="Comma-separated UMAP n_neighbors"),
    selection_method: Optional[str] = typer.Option(None, "--selection-method", help="Comma-separated methods (eom,leaf)"),
    selection_epsilon: Optional[str] = typer.Option(None, "--selection-epsilon", help="Comma-separated epsilon values"),
    save_output: Optional[Path] = typer.Option(None, "--save-output", help="Save full report as CSV"),
) -> None:
    """Run grid-search clustering (report only, no DB writes).

    Without axis flags: uses DEFAULT_GRID.
    With axis flags: builds cartesian product of all specified values.
    Unspecified axes use their default value.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    custom_grid = _build_custom_grid(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_dims=umap_dims,
        umap_n_neighbors=umap_n_neighbors,
        cluster_selection_method=selection_method,
        cluster_selection_epsilon=selection_epsilon,
    )
    if custom_grid:
        console.print(f"Custom grid: {len(custom_grid)} parameter combinations")

    parsed_types = _parse_types(types or [])
    specs = _build_grid_specs(parsed_types, categories or [], grid=custom_grid)

    pipeline = GridSearchClusteringPipeline(session_factory, specs)
    report = asyncio.run(pipeline.run())

    for key, entries in report.entries.items():
        console.print(_grid_top_table(key, entries))
        console.print()

    if save_output:
        _save_csv(report, save_output)
