"""Unit tests for Retrieval class (MinHash prefilter + corpus cache)."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from episodiq.config.retrieval_config import RetrievalConfig
from episodiq.retrieval.retrieval import Retrieval

from tests.in_memory_repos import (
    InMemoryMessageRepository,
    InMemoryPath,
    InMemoryTrajectoryPathRepository,
)


@dataclass
class FakeQueryPath:
    """Minimal stand-in for TrajectoryPath in retrieval.search()."""

    trajectory_id: UUID
    minhash_sig: list[int] | None
    lsh_buckets: list[int] | None = field(default=None)


def _cfg(top_k: int = 5, sim: float = 0.0, ngram_n: int = 3) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=top_k, similarity_threshold=sim, ngram_n=ngram_n,
        minhash_k=256, minhash_seed=0,
    )


def _add_path(repo, tid: UUID, sig: list[int], status: str = "success"):
    path = InMemoryPath(
        id=uuid4(),
        trajectory_id=tid,
        from_observation_id=uuid4(),
        to_observation_id=uuid4(),
        trajectory_status=status,
        minhash_sig=sig,
    )
    repo._paths.append(path)
    return path


class TestRetrieval:
    @pytest.mark.asyncio
    async def test_empty_signature_returns_empty(self):
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        r = Retrieval(repo, _cfg(top_k=5))
        query = FakeQueryPath(trajectory_id=uuid4(), minhash_sig=None)
        assert await r.search(query) == []

    @pytest.mark.asyncio
    async def test_excludes_query_trajectory(self):
        """Paths belonging to the query trajectory never appear in the shortlist."""
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        query_tid = uuid4()
        other_tid = uuid4()
        _add_path(repo, query_tid, [1, 2, 3, 4])
        _add_path(repo, other_tid, [1, 2, 3, 4])
        r = Retrieval(repo, _cfg(top_k=5))
        query = FakeQueryPath(trajectory_id=query_tid, minhash_sig=[1, 2, 3, 4])
        out = await r.search(query)
        assert {c.trajectory_id for c in out} == {other_tid}

    @pytest.mark.asyncio
    async def test_top_k_caps_result_size(self):
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        for _ in range(7):
            _add_path(repo, uuid4(), [1, 2, 3, 4])
        r = Retrieval(repo, _cfg(top_k=3))
        query = FakeQueryPath(trajectory_id=uuid4(), minhash_sig=[1, 2, 3, 4])
        out = await r.search(query)
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_similarity_threshold_filters_candidates(self):
        """Candidates whose MAX-pool similarity is below the threshold are dropped."""
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        near = uuid4()
        far = uuid4()
        _add_path(repo, near, [1, 2, 3, 4])     # 1.0 similarity
        _add_path(repo, far, [9, 9, 9, 9])      # 0.0 similarity
        r = Retrieval(repo, _cfg(top_k=5, sim=0.5))
        query = FakeQueryPath(trajectory_id=uuid4(), minhash_sig=[1, 2, 3, 4])
        out = await r.search(query)
        assert {c.trajectory_id for c in out} == {near}

    @pytest.mark.asyncio
    async def test_corpus_cache_reused_across_searches(self):
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        _add_path(repo, uuid4(), [1, 2, 3, 4])
        r = Retrieval(repo, _cfg(top_k=5))
        query = FakeQueryPath(trajectory_id=uuid4(), minhash_sig=[1, 2, 3, 4])

        await r.search(query)
        cached = r._corpus_cache
        await r.search(query)
        # Same in-memory list reused — no fresh fetch.
        assert r._corpus_cache is cached

    @pytest.mark.asyncio
    async def test_invalidate_cache_drops_corpus(self):
        repo = InMemoryTrajectoryPathRepository(InMemoryMessageRepository())
        _add_path(repo, uuid4(), [1, 2, 3, 4])
        r = Retrieval(repo, _cfg(top_k=5))
        query = FakeQueryPath(trajectory_id=uuid4(), minhash_sig=[1, 2, 3, 4])

        await r.search(query)
        assert r._corpus_cache is not None
        r.invalidate_cache()
        assert r._corpus_cache is None
