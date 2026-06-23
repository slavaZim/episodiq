"""DB-backed integration tests for sweep building blocks:

  - ``rebuild_lsh_into``: happy path (rows actually land in the alt
    table) + strict guard (raises when the alt table already exists).
  - Warm-up via ``_run_searches`` with the sweep's
    ``prefetch_n_uniq=MAX, jaccard_n_uniq=1, top_k=1`` shape: the
    shared ``RetrievalCache`` must come back populated across LSH,
    jaccard, and tokens slots so subsequent trials hit it.

The end-to-end ``RetrievalSweep.run()`` is intentionally NOT here —
it pulls Optuna into the loop and is best exercised via the
``08_tune.sh`` integration; unit-style sweep coverage lives in
``test_sweep.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from episodiq.config.retrieval_config import (
    RetrievalConfig, WindowMinHashConfig,
)
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.cache import RetrievalCache
from episodiq.retrieval.tune.lsh_rebuild import (
    create_alt_lsh_table, drop_alt_lsh_table, rebuild_lsh_into,
)
from episodiq.retrieval.tune.sweep import TuneSnapshot, _run_searches
from episodiq.storage.postgres.models import (
    Message, Trajectory,
)
from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository, make_window_lsh_table,
)


# ----------------------------------------------------------------------
# Seeding helpers — keep tests focused on the sweep code under test
# ----------------------------------------------------------------------


async def _seed_traj_with_tokens(
    session, *, tokens: list[int], status: str = "failure",
) -> tuple[UUID, UUID]:
    """Insert one completed trajectory with the given trace_tokens on
    its (sole) path. Returns ``(trajectory_id, path_id)``. Caller is
    responsible for committing if the session needs cross-connection
    visibility (sweep warm-up uses session_factory, which won't see
    rows still living in another session's savepoint).
    """
    tid = uuid4()
    session.add(Trajectory(id=tid, status=status))
    obs0 = Message(trajectory_id=tid, role="user", content=[], index=0)
    act = Message(trajectory_id=tid, role="assistant", content=[], index=1)
    obs1 = Message(trajectory_id=tid, role="user", content=[], index=2)
    session.add_all([obs0, act, obs1])
    await session.flush()
    repo = TrajectoryPathRepository(session)
    path = await repo.create(
        trajectory_id=tid,
        from_observation_id=obs0.id,
        action_message_id=act.id,
        to_observation_id=obs1.id,
        trace_tokens=tokens,
        trajectory_status=status,
    )
    await session.flush()
    return tid, path.id


async def _delete_traj_cascade(session, tids: list[UUID]) -> None:
    """Clean up trajectories committed by the warm-up test — db_session
    savepoints can't roll these back."""
    await session.execute(text(
        "DELETE FROM trajectory_window_lsh WHERE trajectory_id = ANY(:t)",
    ), {"t": tids})
    await session.execute(text(
        "DELETE FROM trajectory_paths WHERE trajectory_id = ANY(:t)",
    ), {"t": tids})
    await session.execute(text(
        "DELETE FROM messages WHERE trajectory_id = ANY(:t)",
    ), {"t": tids})
    await session.execute(text(
        "DELETE FROM trajectories WHERE id = ANY(:t)",
    ), {"t": tids})
    await session.commit()


# ----------------------------------------------------------------------
# rebuild_lsh_into
# ----------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
class TestRebuildLshInto:
    """The sweep's per-W LSH alt-table builder."""

    async def test_inserts_rows_for_completed_trajectories(self, db_session):
        # Tokens long enough for the default W=10 to form at least one
        # window (n - W + 1 windows; here 21 windows × 32 bands = 672).
        tokens = list(range(30))
        await _seed_traj_with_tokens(db_session, tokens=tokens)

        mh = WindowMinHashConfig(window=10)
        n_trajs, n_rows = await rebuild_lsh_into(
            db_session, mh, "test_alt_lsh_basic",
        )
        try:
            assert n_trajs == 1
            # 30 tokens, W=10 → 21 window positions; 32 bands each.
            assert n_rows == 21 * mh.num_bands

            # Round-trip via the production repository pointed at the
            # alt table to prove rows are actually queryable.
            from episodiq.storage.postgres.repository import (
                TrajectoryWindowLSHRepository,
            )
            alt_table = make_window_lsh_table("test_alt_lsh_basic")
            repo = TrajectoryWindowLSHRepository(
                db_session, table=alt_table,
            )
            # Pull any row's band → confirm at least one match exists.
            existing = await db_session.execute(text(
                "SELECT band_index, band_hash FROM test_alt_lsh_basic LIMIT 1"
            ))
            bi, bh = existing.first()
            hits = await repo.lookup(
                [(int(bi), int(bh))],
                step_min=0, step_max=1000, top_uniq=5,
            )
            assert len(hits) == 1
        finally:
            await drop_alt_lsh_table(db_session, "test_alt_lsh_basic")

    async def test_raises_when_table_already_exists(self, db_session):
        # Pre-create the alt table — simulates a stale table left by a
        # previous crashed sweep run.
        await create_alt_lsh_table(db_session, "test_alt_lsh_clash")
        try:
            mh = WindowMinHashConfig(window=10)
            with pytest.raises(ValueError, match="already exists"):
                await rebuild_lsh_into(
                    db_session, mh, "test_alt_lsh_clash",
                )
        finally:
            await drop_alt_lsh_table(db_session, "test_alt_lsh_clash")

    async def test_skips_trajectories_shorter_than_window(self, db_session):
        # 5 tokens, W=10 → no windows fit → trajectory contributes 0 rows.
        await _seed_traj_with_tokens(db_session, tokens=[1, 2, 3, 4, 5])
        mh = WindowMinHashConfig(window=10)
        n_trajs, n_rows = await rebuild_lsh_into(
            db_session, mh, "test_alt_lsh_short",
        )
        try:
            assert n_trajs == 0
            assert n_rows == 0
        finally:
            await drop_alt_lsh_table(db_session, "test_alt_lsh_short")


# ----------------------------------------------------------------------
# _run_searches as a sweep warm-up: cache populates across all slots
# ----------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
class TestWarmUpPopulatesCache:
    """The sweep calls ``_run_searches`` with ``prefetch_n_uniq=MAX``,
    ``jaccard_n_uniq=1``, ``top_k=1`` before the Optuna study so the
    expensive LSH lookups + dense jaccard computations land in the
    shared ``RetrievalCache``. Subsequent trials in the same
    ``(W, agg)`` slot then hit the cache instead of recomputing. These
    tests pin that contract: after warm-up the cache MUST have entries
    in every slot.
    """

    async def _seed_corpus(self, session_factory):
        """Two completed trajectories — both long enough to form
        windows under W=10 — committed through ``session_factory`` so
        the sweep's own session (a different connection) can see them.
        """
        async with session_factory() as session:
            tid_a, path_a = await _seed_traj_with_tokens(
                session, tokens=list(range(30)), status="failure",
            )
            tid_b, path_b = await _seed_traj_with_tokens(
                session, tokens=list(range(5, 35)), status="success",
            )
            await session.commit()
        return [(tid_a, path_a), (tid_b, path_b)]

    async def test_warmup_populates_lsh_jaccard_tokens_slots(
        self, session_factory,
    ):
        seeded = await self._seed_corpus(session_factory)
        seeded_tids = [tid for tid, _ in seeded]
        try:
            mh = WindowMinHashConfig(window=10)
            ms = AggShiftConfig(window=10)
            # Sweep's warm-up cascade config — widest pool, 1-deep
            # jaccard/min-shift to keep the JIT cheap.
            cas = RetrievalConfig(
                aggregation="mean",
                prefetch_n_uniq=200,
                jaccard_n_uniq=1,
                top_k=1,
            )

            # Build the alt LSH the warm-up will query.
            async with session_factory() as session:
                await rebuild_lsh_into(session, mh, "test_alt_lsh_warmup")

            # One snapshot per seeded trajectory — the warm-up runs all
            # of them so every (path_id, anchor) cache key gets seeded.
            snapshots = []
            for tid, path_id in seeded:
                snapshots.append(TuneSnapshot(
                    trajectory_id=tid,
                    step=0,
                    tokens=list(range(30)),
                    path_id=path_id,
                ))
            status = {seeded[0][0]: "failure", seeded[1][0]: "success"}
            cache = RetrievalCache()

            await _run_searches(
                snapshots, status, session_factory,
                "test_alt_lsh_warmup",
                mh_cfg=mh, cas_cfg=cas, ms_cfg=ms,
                n_workers=2, cache=cache,
            )

            # All four cache slots must have populated. LSH and
            # jaccard hold per (path, anchor, ..., agg) keys; tokens
            # hold per-trajectory entries; aggshift holds per-pair.
            assert cache.lsh._data, "LSH cache empty after warm-up"
            assert cache.jaccard._data, "jaccard cache empty after warm-up"
            assert cache.tokens._data, "tokens cache empty after warm-up"
            assert cache.aggshift._data, "agg-shift cache empty after warm-up"

            # Every queried trajectory should appear in the tokens
            # cache so later trials don't re-fetch trace_tokens.
            for tid, _ in seeded:
                assert tid in cache.tokens._data

            # LSH and jaccard cache keys must carry the agg the
            # warm-up ran under — agg flip between slots opens fresh
            # entries; this is what makes per-(W, agg) caches sound.
            assert all(k[-1] == "mean" for k in cache.lsh._data)
            assert all(k[-1] == "mean" for k in cache.jaccard._data)
        finally:
            async with session_factory() as session:
                await drop_alt_lsh_table(session, "test_alt_lsh_warmup")
                await _delete_traj_cascade(session, seeded_tids)
