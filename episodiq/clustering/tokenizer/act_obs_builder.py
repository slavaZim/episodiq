"""ActObsBuilder: collect distinct (action_cluster_id, observation_cluster_id)
act_obs entries from trajectory_paths and compute concatenated act_obs
embeddings.

One-side-noise pairs (exactly one of (a, o) is HDBSCAN message-level noise,
i.e. ``cluster_id IS NULL``) participate in the clustering pool: the
per-category noise centroid is substituted for the missing side. Their
trace_tokens contribution is deliberately dropped downstream by the saver
(no token_mapping row), so retrieval carry-forwards on them.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sklearn.kernel_approximation import PolynomialCountSketch

from episodiq.storage.postgres.repository import (
    ActObsCluster,
    ClusterRepository,
    MessageRepository,
    TrajectoryPathRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class ActObsPool:
    """Aligned rows and concat embeddings. ``uuid_to_label`` is populated
    only for non-noise cluster_ids — saver uses it for token_mapping rows
    of the fully-clustered pairs; one-side-noise rows are skipped there.
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
        msg_repo: MessageRepository,
    ):
        self._path_repo = path_repo
        self._cluster_repo = cluster_repo
        self._msg_repo = msg_repo

    async def build(self) -> ActObsPool:
        rows = await self._path_repo.collect_act_obs()
        logger.info("collected %d distinct (a_cid, o_cid) act_obs", len(rows))
        if not rows:
            return ActObsPool([], np.zeros((0, 0), dtype=np.float32), {})

        cluster_ids: set[UUID] = set()
        for r in rows:
            if r.a_cluster_id is not None:
                cluster_ids.add(r.a_cluster_id)
            if r.o_cluster_id is not None:
                cluster_ids.add(r.o_cluster_id)
        cluster_centroids = await self._cluster_repo.get_centroids(cluster_ids)
        uuid_to_centroid: dict[UUID, np.ndarray] = {
            c.cluster_id: np.asarray(c.embedding, dtype=np.float32)
            for c in cluster_centroids
        }
        uuid_to_label: dict[UUID, str] = {
            c.cluster_id: c.label for c in cluster_centroids
        }

        noise_centroids = await self._msg_repo.get_category_centroids()
        noise_by_cat: dict[tuple[str, str], np.ndarray] = {
            (c.cluster_type, c.category): np.asarray(c.embedding, dtype=np.float32)
            for c in noise_centroids
        }
        logger.info("noise centroids: %d (cluster_type, category) entries", len(noise_by_cat))

        def side_emb(cid: UUID | None, category: str, side: str) -> np.ndarray:
            if cid is not None:
                return uuid_to_centroid[cid]
            return noise_by_cat[(side, category)]

        concat_rows = [
            np.concatenate([
                side_emb(r.a_cluster_id, r.a_category, "action"),
                side_emb(r.o_cluster_id, r.o_category, "observation"),
            ])
            for r in rows
        ]
        concat_embs = np.vstack(concat_rows).astype(np.float32)
        sketcher = PolynomialCountSketch(
            degree=2, n_components=concat_embs.shape[1], random_state=42,
        )
        embs = sketcher.fit_transform(concat_embs).astype(np.float32)
        logger.info("built act_obs_embs (polysketch deg=2): shape=%s", embs.shape)

        return ActObsPool(rows=rows, embs=embs, uuid_to_label=uuid_to_label)
