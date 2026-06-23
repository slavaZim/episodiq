"""Sweep-only LSH rebuild: take the current trace_tokens from every completed
trajectory's latest path, compute per-window LSH bands for a given W, and
write into an alt table. trace_tokens themselves are W-agnostic so the
production table + production rows are untouched.

The alt table's schema mirrors ``TrajectoryWindowLSH`` (composite PK +
band-step composite index) so the production
``TrajectoryWindowLSHRepository`` works against it unchanged when
constructed with ``table=make_window_lsh_table(name)``.
"""

from __future__ import annotations

import logging
from uuid import UUID

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from episodiq.config.retrieval_config import WindowMinHashConfig
from episodiq.retrieval.window_minhash import WindowMinHasher
from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository,
    TrajectoryRepository,
    TrajectoryWindowLSHRepository,
    make_window_lsh_table,
)

logger = logging.getLogger(__name__)


async def create_alt_lsh_table(
    session: AsyncSession, table_name: str,
) -> None:
    """Drop + recreate the alt LSH table with the same schema + composite
    index as the production ``trajectory_window_lsh``.
    """
    await session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    await session.execute(text(f"""
        CREATE TABLE {table_name} (
            trajectory_id UUID NOT NULL,
            window_center INTEGER NOT NULL,
            band_index SMALLINT NOT NULL,
            band_hash BIGINT NOT NULL,
            PRIMARY KEY (trajectory_id, window_center, band_index)
        )
    """))
    await session.execute(text(
        f"CREATE INDEX ix_{table_name}_band_step "
        f"ON {table_name} (band_index, band_hash, window_center)",
    ))
    await session.commit()


async def drop_alt_lsh_table(
    session: AsyncSession, table_name: str,
) -> None:
    await session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    await session.commit()


async def rebuild_lsh_into(
    session: AsyncSession,
    mh_cfg: WindowMinHashConfig,
    table_name: str,
    *,
    statuses: tuple[str, ...] = ("success", "failure"),
    flush_every: int = 50_000,
) -> tuple[int, int]:
    """Rebuild LSH rows into ``table_name`` from existing trace_tokens.

    For each completed trajectory: take its latest path's trace_tokens,
    compute all per-window MinHash bands for ``mh_cfg.window`` (stride=1),
    emit one row per ``(trajectory_id, window_center, band_index, band_hash)``
    where ``window_center = first_token + half_window``.

    Raises ``ValueError`` if ``table_name`` already exists — concurrent
    sweeps with colliding names would corrupt each other's indexes.
    Caller must drop the table explicitly (``drop_alt_lsh_table``) or
    pass an ``alt_table_suffix`` that doesn't collide.

    Returns ``(n_trajectories_indexed, n_rows_written)``.
    """
    exists = await session.execute(text(
        "SELECT 1 FROM information_schema.tables WHERE table_name = :n",
    ), {"n": table_name})
    if exists.first() is not None:
        raise ValueError(
            f"alt LSH table {table_name!r} already exists — drop it "
            f"first (drop_alt_lsh_table) or pick another suffix.",
        )
    await create_alt_lsh_table(session, table_name)
    alt_table = make_window_lsh_table(table_name)
    lsh_repo = TrajectoryWindowLSHRepository(session, table=alt_table)
    path_repo = TrajectoryPathRepository(session)
    traj_repo = TrajectoryRepository(session)
    hasher = WindowMinHasher(mh_cfg)
    W = mh_cfg.window
    w = mh_cfg.half_window
    num_bands = mh_cfg.num_bands

    trajs = await traj_repo.get_with_completed_paths(list(statuses))
    traj_ids = [t.id for t in trajs]
    latest = await path_repo.get_latest_trace_tokens_for_trajectories(traj_ids)

    n_trajs = 0
    n_rows = 0
    buffered: list[tuple[UUID, int, int, int]] = []
    for tid, (_, _, tokens) in latest.items():
        if not tokens or len(tokens) < W:
            continue
        arr = np.asarray(tokens, dtype=np.int64)
        _sigs, starts, bands = hasher.signatures_and_bands(arr)
        for wi in range(bands.shape[0]):
            center = int(starts[wi]) + w
            for bi in range(num_bands):
                buffered.append((tid, center, bi, int(bands[wi, bi])))
        n_trajs += 1
        n_rows += int(bands.shape[0]) * num_bands
        if len(buffered) >= flush_every:
            await lsh_repo.bulk_insert(buffered)
            buffered.clear()
    if buffered:
        await lsh_repo.bulk_insert(buffered)
    await session.commit()
    logger.info(
        "lsh_rebuild_done table=%s W=%d trajs=%d rows=%d",
        table_name, W, n_trajs, n_rows,
    )
    return n_trajs, n_rows
