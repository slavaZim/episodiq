"""CLI commands for indexing trajectory paths with act_obs-level tokens + MinHash signatures."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.cli.env import _load_dotenv
from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.clustering.tokenizer.path_updater import TrajectoryPathTokenUpdater
from episodiq.config import get_config
from episodiq.retrieval.path_state import PathStateCalculator
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    MessageRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


def _make_session_factory(database_url: str) -> async_sessionmaker:
    engine = create_async_engine(database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@app.command()
def build(
    env: Path = typer.Option(Path(".env"), "--env", help="Path to .env file"),
) -> None:
    """Backfill trace_tokens + per-window LSH bands on all completed
    trajectory_paths.

    Run after `cluster tokenize`: resolves each path's (a_cluster_id,
    o_cluster_id) to a token ordinal via TokenAssigner and writes the
    cumulative token sequence plus one row per LSH band per window into
    trajectory_window_lsh (powers the retrieval cascade Stage-1 lookup).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> int:
        TokenAssigner.invalidate()
        async with session_factory() as session:
            assigner = TokenAssigner(
                TokenMappingRepository(session),
                TokenClusterRepository(session),
                ClusterRepository(session),
            )
            updater = TrajectoryPathTokenUpdater(
                MessageRepository(session),
                TrajectoryPathRepository(session),
                TrajectoryWindowLSHRepository(session),
                PathStateCalculator(assigner),
            )
            total = await updater.update()
            await session.commit()
            return total

    total = asyncio.run(_run())
    console.print(f"index build: updated {total} trajectory paths")
