"""TokenSaver: persists token_clusters + token_mapping. Truncate-and-rebuild.

Each non-noise HDBSCAN cluster becomes a TokenCluster row with cluster_id =
label. All HDBSCAN-outlier (noise) act_obs entries collapse to one extra
TokenCluster row with cluster_id = -1 and centroid = mean of their concat
embeddings; their TokenMapping rows reference that row.
"""

import logging
from collections import defaultdict
from uuid import UUID

import numpy as np

from episodiq.clustering.clusterer import ClusterResult
from episodiq.clustering.tokenizer.act_obs_builder import ActObsPool
from episodiq.storage.postgres.repository import (
    TokenClusterRepository,
    TokenMappingRepository,
)

logger = logging.getLogger(__name__)

NOISE_CLUSTER_ID = -1


class TokenSaver:
    """Drop and re-create token_clusters + token_mapping from an ActObsPool +
    Clusterer result. Idempotent on full rebuild semantics.
    """

    def __init__(
        self,
        token_cluster_repo: TokenClusterRepository,
        token_mapping_repo: TokenMappingRepository,
    ):
        self._cluster_repo = token_cluster_repo
        self._mapping_repo = token_mapping_repo

    async def save(self, pool: ActObsPool, result: ClusterResult) -> int:
        """Persist tokens and mappings; return total TokenCluster rows written
        (real clusters + noise row).
        """
        cluster_accum: dict[int, list[np.ndarray]] = defaultdict(list)
        noise_accum: list[np.ndarray] = []
        for i, lbl in enumerate(result.labels):
            if lbl < 0:
                noise_accum.append(pool.embs[i])
            else:
                cluster_accum[int(lbl)].append(pool.embs[i])

        await self._cluster_repo.delete_all()
        cluster_id_to_uuid: dict[int, UUID] = {}
        for cid, vs in cluster_accum.items():
            centroid = np.mean(np.vstack(vs), axis=0)
            tc = await self._cluster_repo.create(
                cluster_id=cid, centroid=centroid.tolist(),
            )
            cluster_id_to_uuid[cid] = tc.id

        if noise_accum:
            noise_centroid = np.mean(np.vstack(noise_accum), axis=0)
            noise_row = await self._cluster_repo.create(
                cluster_id=NOISE_CLUSTER_ID, centroid=noise_centroid.tolist(),
            )
            cluster_id_to_uuid[NOISE_CLUSTER_ID] = noise_row.id
        logger.info(
            "persisted %d token_clusters (%d real + %d noise)",
            len(cluster_id_to_uuid), len(cluster_accum), 1 if noise_accum else 0,
        )

        await self._mapping_repo.delete_all()
        for row, lbl in zip(pool.rows, result.labels):
            label_int = int(lbl) if lbl >= 0 else NOISE_CLUSTER_ID
            token_cluster_id = cluster_id_to_uuid.get(label_int)
            await self._mapping_repo.create(
                action_label=pool.uuid_to_label[row.a_cluster_id],
                observation_label=pool.uuid_to_label[row.o_cluster_id],
                action_cluster_id=row.a_cluster_id,
                observation_cluster_id=row.o_cluster_id,
                token_cluster_id=token_cluster_id,
            )
        logger.info("persisted %d token_mapping rows", len(pool.rows))

        return len(cluster_id_to_uuid)
