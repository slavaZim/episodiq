"""TrajectoryPathTokenUpdater: backfills trace_tokens + per-window LSH
bands on existing completed trajectory_paths using the current
token_mapping / token_clusters. Mirrors clustering/path_updater.py.
"""

import logging

from episodiq.retrieval.path_state import ActObs, PathStateCalculator
from episodiq.storage.postgres.repository import (
    MessageRepository,
    TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)

logger = logging.getLogger(__name__)


class TrajectoryPathTokenUpdater:
    """Walk all completed trajectory_paths; populate trace_tokens + the
    per-window LSH band index used by the retrieval cascade.
    """

    def __init__(
        self,
        msg_repo: MessageRepository,
        path_repo: TrajectoryPathRepository,
        lsh_repo: TrajectoryWindowLSHRepository,
        calc: PathStateCalculator,
    ):
        self._msg_repo = msg_repo
        self._path_repo = path_repo
        self._lsh_repo = lsh_repo
        self._calc = calc

    async def update(self) -> int:
        """Backfill trace_tokens + LSH bands for every completed path.
        Returns total paths updated.
        """
        traj_ids = await self._msg_repo.get_distinct_trajectory_ids()
        await self._lsh_repo.delete_for_trajectories(traj_ids)
        logger.info("update_tokens_start trajectories=%d", len(traj_ids))

        total = 0
        for i, tid in enumerate(traj_ids, 1):
            total += await self._update_trajectory(tid)
            if i % 100 == 0:
                logger.info(
                    "update_tokens_progress %d/%d paths=%d",
                    i, len(traj_ids), total,
                )

        logger.info(
            "update_tokens_done trajectories=%d paths=%d",
            len(traj_ids), total,
        )
        return total

    async def _update_trajectory(self, trajectory_id) -> int:
        """Walk paths, grouping consecutive runs with the same
        ``parallel_group``. Sequential paths pass through one ActObs;
        parallel groups pass through a list so ``token_step`` resolves
        all ordinals, sorts them ASC, and appends in canonical order —
        making ``trace_tokens`` invariant to the model's tool_call order.
        All paths in a parallel group end up with the same post-batch
        ``trace_tokens``; each new window in the batch is inserted once.
        """
        paths = await self._path_repo.get_trajectory_paths(trajectory_id)
        prev_path = None
        count = 0
        lsh_rows: list[tuple] = []

        i = 0
        while i < len(paths):
            group = paths[i].parallel_group
            if group is None:
                run = [paths[i]]
                i += 1
            else:
                j = i
                while j < len(paths) and paths[j].parallel_group == group:
                    j += 1
                run = paths[i:j]
                i = j

            act_obs = [ActObs.from_path(p) for p in run]
            tokens, wins = await self._calc.token_step(
                prev_path, act_obs[0] if len(act_obs) == 1 else act_obs,
            )
            for p in run:
                await self._path_repo.update(p.id, trace_tokens=tokens)
                p.trace_tokens = tokens
                count += 1
            for win in wins:
                for band_idx, band_hash in enumerate(win.bands):
                    lsh_rows.append(
                        (trajectory_id, win.step, band_idx, band_hash),
                    )
            prev_path = run[-1]

        if lsh_rows:
            await self._lsh_repo.bulk_insert(lsh_rows)
        return count
