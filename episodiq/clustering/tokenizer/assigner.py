"""TokenAssigner: online (action_cluster_id, observation_cluster_id) →
ordinal token id.

Direct lookup in token_mapping; if miss, brute-force nearest centroid in
the small token_clusters pool (centroids cached at class level — shared
across instances within the process).

Noise tokens are encoded per action category: ``-1 - cat_idx`` where
``cat_idx`` is the position of the action's category in the sorted list
of distinct action categories. This keeps noise windows from different
categories from collapsing into a single ``-1`` token.
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


def encode_noise_token(action_cat_idx: int) -> int:
    """Per-category noise ordinal. ``-1`` for unknown category."""
    if action_cat_idx < 0:
        return -1
    return -1 - action_cat_idx


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
    _action_cat_to_idx: ClassVar[dict[str, int] | None] = None
    # (action_cluster_id, observation_cluster_id) → token_cluster_id. Loaded
    # once per process and reused across paths to avoid N DB roundtrips on
    # batch flows (path re-tokenize, index backfill).
    _mapping_cache: ClassVar[dict[tuple[UUID, UUID], UUID] | None] = None

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
        cls._action_cat_to_idx = None
        cls._mapping_cache = None

    def action_cat_idx(self, action_category: str | None) -> int:
        """Position of ``action_category`` in the sorted action-category list.

        Returns ``-1`` when the category is unknown to the cache (cache not
        loaded yet, or category appeared after the last ``invalidate()``).
        """
        if action_category is None or TokenAssigner._action_cat_to_idx is None:
            return -1
        return TokenAssigner._action_cat_to_idx.get(action_category, -1)

    async def assign(
        self,
        action_cluster_id: UUID,
        observation_cluster_id: UUID,
        action_category: str | None = None,
    ) -> int | None:
        """Return token cluster ordinal for the act_obs.

        Non-noise tokens are ``>= 0``. Noise tokens are
        ``encode_noise_token(action_cat_idx)`` (``-1`` when
        ``action_category`` is unknown). Returns ``None`` when no mapping
        row exists and nearest-centroid fallback cannot meet
        ``MIN_FALLBACK_COSINE``.
        """
        await self._ensure_cache()

        cat_idx = self.action_cat_idx(action_category)

        # Direct mapping lookup — uses class-level cache populated by
        # _ensure_cache(), so batch flows don't pay N DB roundtrips.
        tcid = TokenAssigner._mapping_cache.get(
            (action_cluster_id, observation_cluster_id),
        )
        if tcid is not None:
            ordinal = TokenAssigner._uuid_to_ordinal.get(tcid)
            if ordinal is None or ordinal < 0:
                return encode_noise_token(cat_idx)
            return ordinal

        # Fallback: nearest token cluster centroid by cosine similarity.
        if not TokenAssigner._uuids:
            return None
        centroids = await self._cluster_repo.get_centroids(
            {action_cluster_id, observation_cluster_id}
        )
        by_id = {
            c.cluster_id: np.asarray(c.embedding, dtype=np.float32)
            for c in centroids if c.embedding is not None
        }
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
        ordinal = TokenAssigner._ordinals[best_idx]
        if ordinal < 0:
            return encode_noise_token(cat_idx)
        return ordinal

    async def _ensure_cache(self) -> None:
        if TokenAssigner._uuids is not None:
            return
        action_cats = await self._cluster_repo.get_categories("action")
        TokenAssigner._action_cat_to_idx = {c: i for i, c in enumerate(action_cats)}

        mappings = await self._tm_repo.find_by()
        TokenAssigner._mapping_cache = {
            (m.action_cluster_id, m.observation_cluster_id): m.token_cluster_id
            for m in mappings
            if m.action_cluster_id is not None
            and m.observation_cluster_id is not None
            and m.token_cluster_id is not None
        }

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
        logger.info(
            "TokenAssigner cache loaded: %d clusters, %d mappings, %d action categories",
            len(TokenAssigner._uuids), len(TokenAssigner._mapping_cache), len(action_cats),
        )
