"""TrajectoryPathTokenUpdater: backfills trace_tokens + minhash_sig on
existing completed trajectory_paths using current token_mapping /
token_clusters. Mirrors clustering/path_updater.py pattern.
"""

import logging

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.storage.postgres.repository import (
    MessageRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
)

logger = logging.getLogger(__name__)


class TrajectoryPathTokenUpdater:
    """Walk all completed trajectory_paths; populate trace_tokens + minhash_sig."""

    def __init__(
        self,
        msg_repo: MessageRepository,
        path_repo: TrajectoryPathRepository,
        mapping_repo: TokenMappingRepository,
        token_cluster_repo: TokenClusterRepository,
        calc: PathStateCalculator,
    ):
        self._msg_repo = msg_repo
        self._path_repo = path_repo
        self._mapping_repo = mapping_repo
        self._tc_repo = token_cluster_repo
        self._calc = calc

    async def update(self) -> int:
        """Backfill trace_tokens + minhash_sig for every completed path.
        Returns total paths updated.
        """
        act_obs_to_ordinal = await self._build_act_obs_to_ordinal()

        traj_ids = await self._msg_repo.get_distinct_trajectory_ids()
        logger.info("update_tokens_start trajectories=%d", len(traj_ids))

        total = 0
        for i, tid in enumerate(traj_ids, 1):
            total += await self._update_trajectory(tid, act_obs_to_ordinal)
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

    async def _build_act_obs_to_ordinal(self) -> dict[tuple, int]:
        """Pre-load (a_cluster_id, o_cluster_id) → token ordinal int.
        Token pool is small (O(100s) tokens), the whole mapping fits in memory.
        """
        mappings = await self._mapping_repo.find_by()
        tc_rows = await self._tc_repo.get_centroids()
        uuid_to_ordinal = {row_id: int(cid) for row_id, cid, _c in tc_rows}
        return {
            (m.action_cluster_id, m.observation_cluster_id): uuid_to_ordinal[m.token_cluster_id]
            for m in mappings
            if m.action_cluster_id is not None
            and m.observation_cluster_id is not None
            and m.token_cluster_id is not None
            and m.token_cluster_id in uuid_to_ordinal
        }

    async def _update_trajectory(
        self, trajectory_id, act_obs_to_ordinal: dict[tuple, int],
    ) -> int:
        paths = await self._path_repo.get_trajectory_paths(trajectory_id)
        prev_path = None
        count = 0
        for path in paths:
            a_cid = path.action_message.cluster_id if path.action_message else None
            o_cid = path.to_observation.cluster_id if path.to_observation else None
            ordinal = (
                act_obs_to_ordinal.get((a_cid, o_cid))
                if a_cid is not None and o_cid is not None
                else None
            )
            if ordinal is not None:
                tokens, minhash_sig = self._calc.token_step(prev_path, ordinal)
            else:
                # Unmapped pair (a_cid or o_cid is None, or pair absent from
                # token_mapping) — carry prev_path's accumulated state forward.
                tokens = (prev_path.trace_tokens if prev_path else None) or []
                minhash_sig = prev_path.minhash_sig if prev_path else None
            await self._path_repo.update(
                path.id, trace_tokens=tokens, minhash_sig=minhash_sig,
            )
            path.trace_tokens = tokens
            path.minhash_sig = minhash_sig
            prev_path = path
            count += 1
        return count
