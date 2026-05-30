"""Tests for TokenAssigner (act_obs → ordinal token mapping)."""

from uuid import uuid4

import pytest

from episodiq.clustering.tokenizer.assigner import MIN_FALLBACK_COSINE, TokenAssigner

from tests.in_memory_repos import (
    Cluster,
    InMemoryClusterRepository,
    InMemoryMessageRepository,
    InMemoryTokenClusterRepository,
    InMemoryTokenMappingRepository,
    Message,
)


@pytest.fixture(autouse=True)
def reset_token_assigner_cache():
    """TokenAssigner caches at class level — reset before and after each test."""
    TokenAssigner.invalidate()
    yield
    TokenAssigner.invalidate()


def _build_cluster_repo_with_centroids(
    label_to_vec: dict[str, list[float]],
) -> tuple[InMemoryClusterRepository, InMemoryMessageRepository]:
    """Wire up a cluster repo whose get_centroids() yields the requested
    centroids (one message per cluster carrying the vector).
    """
    cluster_repo = InMemoryClusterRepository()
    msg_repo = InMemoryMessageRepository()
    traj_id = uuid4()
    for i, (label, vec) in enumerate(label_to_vec.items()):
        cluster = Cluster(
            id=uuid4(), type="observation", category="text", label=label,
        )
        cluster_repo._clusters.append(cluster)
        msg = Message(
            id=uuid4(), trajectory_id=traj_id, role="user", content=[],
            index=i, embedding=list(vec), cluster_id=cluster.id,
            cluster=cluster,
        )
        msg_repo.add_message(msg)
    return cluster_repo, msg_repo


class TestTokenAssigner:
    @pytest.mark.asyncio
    async def test_direct_mapping_returns_ordinal(self):
        token_cluster_repo = InMemoryTokenClusterRepository()
        token_mapping_repo = InMemoryTokenMappingRepository()
        cluster_repo = InMemoryClusterRepository()

        tc = token_cluster_repo.add(cluster_id=7, centroid=[0.0, 1.0])
        a_cid = uuid4()
        o_cid = uuid4()
        token_mapping_repo.add(
            action_label="a",
            observation_label="o",
            action_cluster_id=a_cid,
            observation_cluster_id=o_cid,
            token_cluster_id=tc.id,
        )

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        ordinal = await assigner.assign(a_cid, o_cid)
        assert ordinal == 7

    @pytest.mark.asyncio
    async def test_missing_mapping_with_no_centroids_returns_none(self):
        """No direct mapping AND no token clusters in the pool → None."""
        token_cluster_repo = InMemoryTokenClusterRepository()
        token_mapping_repo = InMemoryTokenMappingRepository()
        cluster_repo = InMemoryClusterRepository()

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        assert await assigner.assign(uuid4(), uuid4()) is None

    @pytest.mark.asyncio
    async def test_fallback_close_match_returns_ordinal(self):
        """When pair has no mapping, nearest-centroid fallback returns the
        ordinal as long as the best cosine ≥ MIN_FALLBACK_COSINE.
        """
        token_cluster_repo = InMemoryTokenClusterRepository()
        token_mapping_repo = InMemoryTokenMappingRepository()
        cluster_repo, msg_repo = _build_cluster_repo_with_centroids({
            "A": [1.0, 0.0],
            "O": [0.0, 1.0],
        })
        # Token centroid that exactly matches concat([A, O]) = [1, 0, 0, 1].
        token_cluster_repo.add(cluster_id=42, centroid=[1.0, 0.0, 0.0, 1.0])

        a_cid = cluster_repo.get_by_label("A").id
        o_cid = cluster_repo.get_by_label("O").id
        cluster_repo.link_messages(msg_repo)

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        assert await assigner.assign(a_cid, o_cid) == 42

    @pytest.mark.asyncio
    async def test_fallback_below_threshold_returns_none(self):
        """When the nearest token-cluster centroid is below the cosine
        threshold, fallback declines and returns None.
        """
        token_cluster_repo = InMemoryTokenClusterRepository()
        token_mapping_repo = InMemoryTokenMappingRepository()
        cluster_repo, msg_repo = _build_cluster_repo_with_centroids({
            "A": [1.0, 0.0],
            "O": [0.0, 1.0],
        })
        # Token centroid orthogonal to the query → cosine = 0, below 0.9.
        token_cluster_repo.add(cluster_id=99, centroid=[0.0, 1.0, 1.0, 0.0])

        a_cid = cluster_repo.get_by_label("A").id
        o_cid = cluster_repo.get_by_label("O").id
        cluster_repo.link_messages(msg_repo)

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        assert MIN_FALLBACK_COSINE > 0.0  # sanity
        assert await assigner.assign(a_cid, o_cid) is None

    @pytest.mark.asyncio
    async def test_invalidate_drops_cached_state(self):
        token_cluster_repo = InMemoryTokenClusterRepository()
        token_mapping_repo = InMemoryTokenMappingRepository()
        cluster_repo = InMemoryClusterRepository()

        tc = token_cluster_repo.add(cluster_id=1, centroid=[1.0, 0.0])
        token_mapping_repo.add(
            action_label="a",
            observation_label="o",
            action_cluster_id=uuid4(),
            observation_cluster_id=uuid4(),
            token_cluster_id=tc.id,
        )

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        # Populate the class-level cache.
        await assigner._ensure_cache()
        assert TokenAssigner._uuids
        TokenAssigner.invalidate()
        assert TokenAssigner._uuids is None
