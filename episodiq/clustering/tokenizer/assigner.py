"""TokenAssigner: online (action_cluster_id, observation_cluster_id) →
ordinal token id.

Direct lookup in token_mapping; if miss, brute-force nearest centroid in
the small token_clusters pool (centroids cached at class level — shared
across instances within the process).
"""

import logging
from typing import ClassVar
from uuid import UUID

import numpy as np

from episodiq.storage.postgres.repository import (
    ClusterRepository,
    TokenClusterRepository,
    TokenMappingRepository,
)

logger = logging.getLogger(__name__)

# Minimum cosine similarity for the nearest-centroid fallback to accept the
# match. Below this the query is too far from anything in the token pool;
# treat as unmapped and let the caller carry forward.
MIN_FALLBACK_COSINE = 0.9


class TokenAssigner:
    """Resolve (action_cluster_id, observation_cluster_id) → ordinal token id.

    Token pool is small by design (O(100s) clusters × ~2KB each) — class-level
    cache loaded on first use, reused across all subsequent calls in the
    process. Call `invalidate()` after re-indexing.
    """

    _uuids: ClassVar[list[UUID] | None] = None
    _ordinals: ClassVar[list[int] | None] = None
    _centroids: ClassVar[np.ndarray | None] = None
    _uuid_to_ordinal: ClassVar[dict[UUID, int] | None] = None

    def __init__(
        self,
        token_mapping_repo: TokenMappingRepository,
        token_cluster_repo: TokenClusterRepository,
        cluster_repo: ClusterRepository,
    ):
        self._tm_repo = token_mapping_repo
        self._tc_repo = token_cluster_repo
        self._cluster_repo = cluster_repo

    @classmethod
    def invalidate(cls) -> None:
        cls._uuids = None
        cls._ordinals = None
        cls._centroids = None
        cls._uuid_to_ordinal = None

    async def assign(
        self, action_cluster_id: UUID, observation_cluster_id: UUID,
    ) -> int | None:
        """Return token cluster ordinal (int; -1 = noise) for the act_obs.
        None when no mapping row exists and nearest-centroid fallback
        cannot meet the cosine similarity threshold (empty pool, missing
        cluster centroids, or best match below MIN_FALLBACK_COSINE).
        """
        await self._ensure_cache()

        # Direct mapping lookup.
        tcid = await self._tm_repo.find_by_cluster_ids(
            action_cluster_id, observation_cluster_id,
        )
        if tcid is not None:
            return TokenAssigner._uuid_to_ordinal.get(tcid)

        # Fallback: nearest token cluster centroid by cosine similarity.
        if not TokenAssigner._uuids:
            return None
        centroids = await self._cluster_repo.get_centroids(
            {action_cluster_id, observation_cluster_id}
        )
        by_id = {cid: np.asarray(raw, dtype=np.float32)
                 for cid, _label, raw in centroids if raw is not None}
        if action_cluster_id not in by_id or observation_cluster_id not in by_id:
            return None
        query = np.concatenate([by_id[action_cluster_id], by_id[observation_cluster_id]])

        q_norm = float(np.linalg.norm(query))
        c_norms = np.linalg.norm(TokenAssigner._centroids, axis=1)
        if q_norm == 0 or not np.any(c_norms > 0):
            return None
        sims = (TokenAssigner._centroids @ query) / (
            np.maximum(c_norms, 1e-9) * q_norm
        )
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < MIN_FALLBACK_COSINE:
            return None
        return TokenAssigner._ordinals[best_idx]

    async def _ensure_cache(self) -> None:
        if TokenAssigner._uuids is not None:
            return
        rows = await self._tc_repo.get_centroids()
        if not rows:
            TokenAssigner._uuids = []
            TokenAssigner._ordinals = []
            TokenAssigner._centroids = np.zeros((0, 0), dtype=np.float32)
            TokenAssigner._uuid_to_ordinal = {}
            return
        TokenAssigner._uuids = [row_id for row_id, _cid, _c in rows]
        TokenAssigner._ordinals = [int(cid) for _, cid, _ in rows]
        TokenAssigner._centroids = np.vstack([
            np.asarray(c, dtype=np.float32) for _, _, c in rows
        ])
        TokenAssigner._uuid_to_ordinal = dict(
            zip(TokenAssigner._uuids, TokenAssigner._ordinals)
        )
        logger.info("TokenAssigner cache loaded: %d clusters",
                    len(TokenAssigner._uuids))
