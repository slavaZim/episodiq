"""CLI commands for indexing trajectory paths with act_obs-level tokens + MinHash signatures."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.cli.env import _load_dotenv
from episodiq.clustering.tokenizer.path_updater import TrajectoryPathTokenUpdater
from episodiq.config import get_config
from episodiq.storage.postgres.repository import (
    MessageRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
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
    """Backfill trace_tokens + minhash_sig on all completed trajectory_paths.

    Run after `cluster tokenize`: looks up each path's (a_cluster_id,
    o_cluster_id) in token_mapping and writes the cumulative token sequence
    plus the MinHash signature used by the retrieval prefilter.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(env)
    config = get_config()
    session_factory = _make_session_factory(config.get_database_url())

    async def _run() -> int:
        async with session_factory() as session:
            updater = TrajectoryPathTokenUpdater(
                MessageRepository(session),
                TrajectoryPathRepository(session),
                TokenMappingRepository(session),
                TokenClusterRepository(session),
                PathStateCalculator(),
            )
            total = await updater.update()
            await session.commit()
            return total

    total = asyncio.run(_run())
    console.print(f"index build: updated {total} trajectory paths")
