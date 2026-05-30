"""Tests for TrajectoryPathUpdater (rebuilds trajectory paths from
clustered messages)."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.clustering.path_updater import TrajectoryPathUpdater

from tests.in_memory_repos import (
    Cluster,
    InMemoryMessageRepository,
    InMemoryTrajectoryPathRepository,
    Message,
)


def _add_msg(
    repo: InMemoryMessageRepository,
    *,
    trajectory_id: UUID,
    index: int,
    role: str,
    label: str,
) -> Message:
    cluster = Cluster(id=uuid4(), type=role, category="text", label=label)
    repo.add_cluster(cluster)
    msg = Message(
        id=uuid4(),
        trajectory_id=trajectory_id,
        role=role,
        content=[{"text": label}],
        index=index,
        cluster_id=cluster.id,
        cluster=cluster,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    repo.add_message(msg)
    return msg


class TestTrajectoryPathUpdater:
    @pytest.mark.asyncio
    async def test_empty_repo_produces_no_paths(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)
        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator())
        total = await updater.update()
        assert total == 0
        assert path_repo._paths == []

    @pytest.mark.asyncio
    async def test_single_observation_creates_one_pending_path(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)
        tid = uuid4()
        _add_msg(
            msg_repo, trajectory_id=tid, index=0, role="user", label="o:text:hello",
        )

        updater = TrajectoryPathUpdater(
            msg_repo, path_repo, PathStateCalculator(),
        )
        total = await updater.update()

        assert total == 1
        assert len(path_repo._paths) == 1
        path = path_repo._paths[0]
        assert path.trajectory_id == tid
        assert path.action_message_id is None
        assert path.to_observation_id is None
        assert path.trace == ["o:text:hello"]

    @pytest.mark.asyncio
    async def test_obs_action_obs_yields_completed_and_pending(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)
        tid = uuid4()
        m_first_obs = _add_msg(
            msg_repo, trajectory_id=tid, index=0, role="user", label="o:text:0",
        )
        m_action = _add_msg(
            msg_repo, trajectory_id=tid, index=1, role="assistant", label="a:text:0",
        )
        m_second_obs = _add_msg(
            msg_repo, trajectory_id=tid, index=2, role="tool", label="o:text:1",
        )

        await TrajectoryPathUpdater(
            msg_repo, path_repo, PathStateCalculator(),
        ).update()

        paths = sorted(path_repo._paths, key=lambda p: p.index)
        assert len(paths) == 2

        completed = paths[0]
        assert completed.from_observation_id == m_first_obs.id
        assert completed.action_message_id == m_action.id
        assert completed.to_observation_id == m_second_obs.id
        # Trace lags by one step — first row carries only the leading observation.
        assert completed.trace == ["o:text:0"]

        pending = paths[1]
        assert pending.from_observation_id == m_second_obs.id
        assert pending.action_message_id is None
        assert pending.trace == ["o:text:0", "a:text:0", "o:text:1"]
