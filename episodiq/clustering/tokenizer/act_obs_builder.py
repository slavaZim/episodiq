"""ActObsBuilder: collect distinct (action_cluster_id, observation_cluster_id)
act_obs entries from trajectory_paths and compute concatenated act_obs
embeddings.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from episodiq.storage.postgres.repository import (
    ActObsCluster,
    ClusterRepository,
    TrajectoryPathRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class ActObsPool:
    """Aligned rows and concat embeddings. uuid_to_label is carried for the
    saver to populate token_mapping.action_label/observation_label.
    """
    rows: list[ActObsCluster]
    embs: np.ndarray
    uuid_to_label: dict[UUID, str]


class ActObsBuilder:
    """Build act_obs pool from completed trajectory_paths and cluster centroids."""

    def __init__(
        self,
        path_repo: TrajectoryPathRepository,
        cluster_repo: ClusterRepository,
    ):
        self._path_repo = path_repo
        self._cluster_repo = cluster_repo

    async def build(self) -> ActObsPool:
        # SQL excludes noise: action_message.cluster_id and
        # to_observation.cluster_id must both be set.
        rows = await self._path_repo.collect_act_obs()
        logger.info("collected %d distinct (a_cid, o_cid) act_obs", len(rows))
        if not rows:
            return ActObsPool([], np.zeros((0, 0), dtype=np.float32), {})

        cluster_ids = {r.a_cluster_id for r in rows} | {r.o_cluster_id for r in rows}
        centroids = await self._cluster_repo.get_centroids(cluster_ids)
        uuid_to_centroid = {
            cid: np.asarray(raw, dtype=np.float32)
            for cid, _label, raw in centroids if raw is not None
        }
        uuid_to_label = {cid: label for cid, label, _raw in centroids}

        embs = np.vstack([
            np.concatenate([uuid_to_centroid[r.a_cluster_id], uuid_to_centroid[r.o_cluster_id]])
            for r in rows
        ]).astype(np.float32)
        logger.info("built act_obs_embs: shape=%s", embs.shape)

        return ActObsPool(rows=rows, embs=embs, uuid_to_label=uuid_to_label)
