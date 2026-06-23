"""End-to-end DB integration test for ``TrajectoryPathTokenUpdater``.

Seeds a small but real fixture in Postgres — Trajectory + Messages
linked to message-level Cluster rows, plus TokenCluster + TokenMapping
that the real ``TokenAssigner`` resolves at runtime — then runs the
updater with the production ``PathStateCalculator`` and asserts:

  - ``trace_tokens`` lands on every path (cumulative append matches
    the assigner-resolved ordinals);
  - ``trajectory_window_lsh`` has rows once the trajectory accumulates
    ``W`` tokens (no rows otherwise).

Mocked-orchestration tests live in
``tests/clustering/test_tokenizer_path_updater.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text

from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.clustering.tokenizer.path_updater import (
    TrajectoryPathTokenUpdater,
)
from episodiq.config.retrieval_config import WindowMinHashConfig
from episodiq.retrieval.path_state import PathStateCalculator
from episodiq.retrieval.window_minhash import WindowMinHasher
from episodiq.storage.postgres.models import (
    Message, TokenCluster, TokenMapping, Trajectory,
    TrajectoryPath,
)
from episodiq.storage.postgres.repository import (
    ClusterRepository, MessageRepository, TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)


def _zero_centroid(dims: int) -> list[float]:
    """Centroid bytes don't matter for this test — TokenAssigner falls
    back to the exact mapping table when an entry exists; centroids
    are only used for the cosine-NN miss path."""
    return [0.0] * dims


async def _seed_token_pool(
    session, *, a_label: str, o_label: str,
) -> tuple[UUID, UUID, UUID, int]:
    """Insert one act/observation Cluster, one TokenCluster (ordinal
    7), and the TokenMapping linking them so ``TokenAssigner.assign``
    returns 7 for that act_obs.

    Returns ``(action_cluster_id, observation_cluster_id,
    token_cluster_uuid, ordinal)``.
    """
    from episodiq.config import get_config
    dims = get_config().message_dims

    cluster_repo = ClusterRepository(session)
    a_cluster = await cluster_repo.create(
        type="action", category="exec", label=a_label,
    )
    o_cluster = await cluster_repo.create(
        type="observation", category="text", label=o_label,
    )

    token_cluster = TokenCluster(
        cluster_id=7, centroid=_zero_centroid(2 * dims),
    )
    session.add(token_cluster)
    await session.flush()

    mapping = TokenMapping(
        action_label=a_label, observation_label=o_label,
        action_cluster_id=a_cluster.id,
        observation_cluster_id=o_cluster.id,
        token_cluster_id=token_cluster.id,
    )
    session.add(mapping)
    await session.flush()
    return a_cluster.id, o_cluster.id, token_cluster.id, 7


async def _seed_trajectory_with_paths(
    session,
    *,
    n_paths: int,
    a_cluster_id: UUID,
    o_cluster_id: UUID,
) -> UUID:
    """One Trajectory + ``n_paths`` sequential TrajectoryPath rows
    where every action message points at ``a_cluster_id`` and every
    to_observation at ``o_cluster_id`` — so the assigner resolves the
    same token ordinal ``n_paths`` times.
    """
    tid = uuid4()
    session.add(Trajectory(id=tid, status="failure"))
    await session.flush()

    repo = TrajectoryPathRepository(session)
    # One leading observation, then alternating action / obs per path.
    obs_msgs = [Message(
        trajectory_id=tid, role="user", content=[], index=0,
        cluster_id=o_cluster_id, cluster_type="observation",
        category="text",
    )]
    session.add(obs_msgs[0])
    await session.flush()

    for i in range(n_paths):
        act = Message(
            trajectory_id=tid, role="assistant", content=[],
            index=2 * i + 1,
            cluster_id=a_cluster_id, cluster_type="action",
            category="exec",
        )
        next_obs = Message(
            trajectory_id=tid, role="user", content=[],
            index=2 * i + 2,
            cluster_id=o_cluster_id, cluster_type="observation",
            category="text",
        )
        session.add_all([act, next_obs])
        await session.flush()
        await repo.create(
            trajectory_id=tid,
            from_observation_id=obs_msgs[-1].id,
            action_message_id=act.id,
            to_observation_id=next_obs.id,
            trajectory_status="failure",
        )
        obs_msgs.append(next_obs)
    return tid


@pytest.mark.asyncio(loop_scope="session")
class TestTokenizerPathUpdaterDB:
    """End-to-end: real Postgres + real ``TokenAssigner`` + real
    ``PathStateCalculator``. Asserts the updater wires every layer
    together correctly."""

    async def test_real_pipeline_writes_tokens_and_lsh_rows(
        self, db_session,
    ):
        # Reset assigner class cache so previous test state doesn't
        # bleed in (the assigner caches mappings at the class level).
        TokenAssigner.invalidate()
        try:
            a_cid, o_cid, _tc_uuid, ordinal = await _seed_token_pool(
                db_session, a_label="a:exec:1", o_label="o:text:1",
            )
            # W=4 → window forms exactly when the 4th token lands.
            n_paths = 6
            tid = await _seed_trajectory_with_paths(
                db_session, n_paths=n_paths,
                a_cluster_id=a_cid, o_cluster_id=o_cid,
            )

            from episodiq.storage.postgres.repository import (
                TokenClusterRepository, TokenMappingRepository,
            )
            mh_cfg = WindowMinHashConfig(window=4)
            calc = PathStateCalculator(
                assigner=TokenAssigner(
                    token_mapping_repo=TokenMappingRepository(db_session),
                    token_cluster_repo=TokenClusterRepository(db_session),
                    cluster_repo=ClusterRepository(db_session),
                ),
                hasher=WindowMinHasher(mh_cfg),
            )

            msg_repo = MessageRepository(db_session)
            path_repo = TrajectoryPathRepository(db_session)
            lsh_repo = TrajectoryWindowLSHRepository(db_session)
            updater = TrajectoryPathTokenUpdater(
                msg_repo, path_repo, lsh_repo, calc,
            )
            count = await updater.update()
            await db_session.flush()

            # All paths got updated.
            assert count == n_paths

            # Every path's trace_tokens equals [ordinal] * (index+1).
            paths = (await db_session.execute(
                select(TrajectoryPath)
                .where(TrajectoryPath.trajectory_id == tid)
                .order_by(TrajectoryPath.index)
            )).scalars().all()
            for i, p in enumerate(paths):
                assert p.trace_tokens == [ordinal] * (i + 1), (
                    f"path {i}: got {p.trace_tokens}, "
                    f"want {[ordinal] * (i + 1)}"
                )

            # LSH rows must exist for this trajectory once the window
            # forms (n_paths >= W=4). The first window opens at
            # window_center = w = 2; the last opens at
            # window_center = (n_paths - 1) - w = n_paths - 3.
            lsh_centers = (await db_session.execute(text(
                "SELECT DISTINCT window_center FROM trajectory_window_lsh "
                "WHERE trajectory_id = :tid ORDER BY window_center",
            ), {"tid": tid})).all()
            centers = [row[0] for row in lsh_centers]
            assert centers == list(range(2, n_paths - 1))

            # Each window has all `num_bands` band rows.
            row_count = (await db_session.execute(text(
                "SELECT COUNT(*) FROM trajectory_window_lsh "
                "WHERE trajectory_id = :tid",
            ), {"tid": tid})).scalar()
            assert row_count == len(centers) * mh_cfg.num_bands

        finally:
            TokenAssigner.invalidate()

    async def test_short_trajectory_writes_tokens_but_no_lsh_rows(
        self, db_session,
    ):
        """When the trajectory has fewer than ``W`` paths, ``token_step``
        appends tokens but no window forms → no LSH rows."""
        TokenAssigner.invalidate()
        try:
            a_cid, o_cid, _tc, ordinal = await _seed_token_pool(
                db_session, a_label="a:exec:2", o_label="o:text:2",
            )
            mh_cfg = WindowMinHashConfig(window=4)
            tid = await _seed_trajectory_with_paths(
                db_session, n_paths=2,  # < W
                a_cluster_id=a_cid, o_cluster_id=o_cid,
            )
            from episodiq.storage.postgres.repository import (
                TokenClusterRepository, TokenMappingRepository,
            )
            calc = PathStateCalculator(
                assigner=TokenAssigner(
                    token_mapping_repo=TokenMappingRepository(db_session),
                    token_cluster_repo=TokenClusterRepository(db_session),
                    cluster_repo=ClusterRepository(db_session),
                ),
                hasher=WindowMinHasher(mh_cfg),
            )
            updater = TrajectoryPathTokenUpdater(
                MessageRepository(db_session),
                TrajectoryPathRepository(db_session),
                TrajectoryWindowLSHRepository(db_session),
                calc,
            )
            count = await updater.update()
            await db_session.flush()

            assert count == 2
            paths = (await db_session.execute(
                select(TrajectoryPath)
                .where(TrajectoryPath.trajectory_id == tid)
                .order_by(TrajectoryPath.index)
            )).scalars().all()
            assert paths[0].trace_tokens == [ordinal]
            assert paths[1].trace_tokens == [ordinal, ordinal]

            row_count = (await db_session.execute(text(
                "SELECT COUNT(*) FROM trajectory_window_lsh "
                "WHERE trajectory_id = :tid",
            ), {"tid": tid})).scalar()
            assert row_count == 0
        finally:
            TokenAssigner.invalidate()
