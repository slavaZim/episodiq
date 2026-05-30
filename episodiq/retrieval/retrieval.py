"""Retrieval: MinHash Jaccard prefilter on trace_token n-grams.

Per-query flow:
  1. MinHash similarity on the query path's signature → top_k candidate
     trajectories.
  2. Each candidate exposes the action_cluster_id of the trajectory's path
     with the highest MinHash similarity and the trajectory's status.

The retrieval corpus is loaded once per ``Retrieval`` instance and cached
in memory for subsequent searches.
"""

import logging
from uuid import UUID

from episodiq.config.retrieval_config import RetrievalConfig
from episodiq.retrieval.candidate import RetrievalCandidate
from episodiq.storage.postgres.models import TrajectoryPath
from episodiq.storage.postgres.repository import TrajectoryPathRepository

logger = logging.getLogger(__name__)


class Retrieval:
    """MinHash retrieval pipeline with an in-process corpus cache."""

    def __init__(
        self,
        path_repo: TrajectoryPathRepository,
        config: RetrievalConfig,
    ):
        self._path_repo = path_repo
        self._config = config
        self._corpus_cache: list[
            tuple[UUID, UUID | None, str, list[int]]
        ] | None = None

    async def _corpus(self) -> list[tuple[UUID, UUID | None, str, list[int]]]:
        if self._corpus_cache is None:
            self._corpus_cache = await self._path_repo.get_minhash_corpus()
        return self._corpus_cache

    def invalidate_cache(self) -> None:
        """Drop the cached retrieval corpus (call after writes that affect it)."""
        self._corpus_cache = None

    async def search(self, query: TrajectoryPath) -> list[RetrievalCandidate]:
        """Return top_k candidate trajectories by MinHash Jaccard estimate."""
        if not query.minhash_sig:
            return []
        corpus = await self._corpus()
        shortlist = await self._path_repo.minhash_prefilter(
            query.minhash_sig,
            query.trajectory_id,
            self._config.top_k,
            min_similarity=self._config.similarity_threshold,
            corpus=corpus,
        )
        return [
            RetrievalCandidate(
                trajectory_id=tid,
                score=sim,
                best_path_action_cluster_id=action_cid,
                trajectory_status=status,
            )
            for tid, sim, action_cid, status in shortlist
        ]
