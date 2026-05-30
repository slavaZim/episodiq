"""TrajectoryPathUpdater: rebuilds trajectory paths from message cluster labels.

Matches online BuildPathStep behavior: one row per observation, with the trace
built incrementally.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.storage.postgres.repository import MessageRepository, TrajectoryPathRepository

logger = logging.getLogger(__name__)

WORKERS = 10


class TrajectoryPathUpdater:
    """Drop all trajectory paths, rebuild from message cluster labels.

    Creates one row per observation (matching online BuildPathStep), populating
    the alternating obs/action label trace.
    """

    def __init__(
        self,
        msg_repo: MessageRepository,
        path_repo: TrajectoryPathRepository,
        calc: PathStateCalculator,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        workers: int = WORKERS,
    ):
        self._msg_repo = msg_repo
        self._path_repo = path_repo
        self._calc = calc
        self._session_factory = session_factory
        self._workers = workers

    async def update(self) -> int:
        """Rebuild all trajectory paths. Returns total rows created."""
        await self._path_repo.delete_all()

        traj_ids = await self._msg_repo.get_distinct_trajectory_ids()
        logger.info("build_paths_start trajectories=%d", len(traj_ids))

        total = 0
        for i, tid in enumerate(traj_ids, 1):
            total += await self._build_trajectory(tid)
            if i % 100 == 0:
                logger.info("build_paths_progress %d/%d trajectories paths=%d", i, len(traj_ids), total)

        logger.info("build_paths_done trajectories=%d paths=%d", len(traj_ids), total)

        await self._path_repo.sync_trajectory_status()

        return total

    async def _build_trajectory(self, trajectory_id) -> int:
        """Build paths for a single trajectory. Returns rows created."""
        rows = await self._msg_repo.get_trajectory_with_clusters(trajectory_id)
        msgs = [m for m in rows if m.role != "system"]
        if not msgs:
            return 0

        prev_path = None
        count = 0

        for i in range(2, len(msgs), 2):
            trace = self._calc.granular_step(prev_path, msgs[i - 2].cluster_label)
            prev_path = await self._path_repo.create(
                trajectory_id=trajectory_id,
                from_observation_id=msgs[i - 2].id,
                action_message_id=msgs[i - 1].id,
                to_observation_id=msgs[i].id,
                trace=trace,
            )
            count += 1

        # Trailing observation — only if trajectory ends on observation (no dangling action)
        if len(msgs) % 2 == 1:
            last_obs = msgs[-1]
            trace = self._calc.granular_step(prev_path, last_obs.cluster_label)
            await self._path_repo.create(
                trajectory_id=trajectory_id,
                from_observation_id=last_obs.id,
                action_message_id=None,
                trace=trace,
            )
            count += 1

        return count
