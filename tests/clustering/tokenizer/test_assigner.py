"""Tests for TokenAssigner (act_obs → ordinal token mapping)."""

from uuid import uuid4

import pytest

from episodiq.clustering.tokenizer.assigner import (
    MIN_FALLBACK_COSINE, TokenAssigner, encode_noise_token,
)

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


class TestEncodeNoiseToken:
    """Per-category noise encoding: ``encode_noise_token(cat_idx)`` =
    ``-1 - cat_idx``. Each category gets a distinct negative ordinal
    so noise from different action categories doesn't merge into one
    token at retrieval time."""

    def test_each_category_gets_distinct_negative_id(self):
        # cat 0 collapses onto the legacy -1 sentinel; subsequent
        # categories cascade down.
        assert [encode_noise_token(i) for i in range(5)] == [-1, -2, -3, -4, -5]


class TestPerCategoryNoise:
    """Tokenization rules around HDBSCAN noise:

      (a) one-side-noise pairs (one cluster_id is NULL) are skipped at
          save time → no token_mapping row → assigner returns None.
      (b) both-side fully clustered but HDBSCAN-labelled noise → the
          mapping row points at the noise TokenCluster (cluster_id=-1);
          assigner emits ``encode_noise_token(cat_idx)`` so the noise
          ordinal is per-category, not a single ``-1`` for all noise.
      (c) the same action_category resolves to the same noise ordinal
          on every assigner instance, so retrieval signal stays stable
          across rebuilds / processes.
    """

    def _seed_action_categories(
        self, cluster_repo: InMemoryClusterRepository, *cats: str,
    ) -> None:
        for cat in cats:
            cluster_repo._clusters.append(Cluster(
                id=uuid4(), type="action", category=cat,
                label=f"a-{cat}",
            ))

    @pytest.mark.asyncio
    async def test_missing_mapping_returns_none(self):
        """No token_mapping row AND no fallback hit → assigner returns
        None. Callers (path_updater) carry-forward the previous token
        on None — they never write a noise ordinal for unknown pairs.
        """
        cluster_repo = InMemoryClusterRepository()
        self._seed_action_categories(cluster_repo, "exec")
        assigner = TokenAssigner(
            InMemoryTokenMappingRepository(),
            InMemoryTokenClusterRepository(),
            cluster_repo,
        )
        # Unknown pair; even with action_category set, no mapping +
        # no centroids → None.
        assert await assigner.assign(
            uuid4(), uuid4(), action_category="exec",
        ) is None

    @pytest.mark.asyncio
    async def test_known_mapping_to_noise_returns_per_category_noise(self):
        """Mapping row exists but points at the noise TokenCluster
        (negative cluster_id) → assigner returns
        ``encode_noise_token(cat_idx)``, NOT the raw negative ordinal.
        """
        cluster_repo = InMemoryClusterRepository()
        # Two categories so cat_idx for "zoo" is 1 (alpha-sorted).
        self._seed_action_categories(cluster_repo, "aaa", "zoo")

        token_cluster_repo = InMemoryTokenClusterRepository()
        noise_tc = token_cluster_repo.add(cluster_id=-1, centroid=[0.0])
        token_mapping_repo = InMemoryTokenMappingRepository()
        a_cid = uuid4()
        o_cid = uuid4()
        token_mapping_repo.add(
            action_label="a", observation_label="o",
            action_cluster_id=a_cid, observation_cluster_id=o_cid,
            token_cluster_id=noise_tc.id,
        )

        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        # cat_idx("zoo") = 1 (alpha: aaa=0, zoo=1) → noise = -1 - 1 = -2.
        assert await assigner.assign(
            a_cid, o_cid, action_category="zoo",
        ) == -2
        # cat_idx("aaa") = 0 → -1 - 0 = -1.
        a_cid2 = uuid4()
        o_cid2 = uuid4()
        token_mapping_repo.add(
            action_label="a2", observation_label="o2",
            action_cluster_id=a_cid2, observation_cluster_id=o_cid2,
            token_cluster_id=noise_tc.id,
        )
        # Re-load so the new mapping row is in the class cache.
        TokenAssigner.invalidate()
        assigner = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        assert await assigner.assign(
            a_cid2, o_cid2, action_category="aaa",
        ) == -1

    @pytest.mark.asyncio
    async def test_same_category_yields_same_noise_id_across_instances(self):
        """Two independently-built assigners over the same cluster
        repo must emit the SAME noise ordinal for the same action
        category — otherwise noise tokens drift between processes and
        retrieval signal collapses."""
        cluster_repo = InMemoryClusterRepository()
        self._seed_action_categories(cluster_repo, "alpha", "beta", "gamma")

        token_cluster_repo = InMemoryTokenClusterRepository()
        noise_tc = token_cluster_repo.add(cluster_id=-1, centroid=[0.0])
        token_mapping_repo = InMemoryTokenMappingRepository()
        a_cid = uuid4()
        o_cid = uuid4()
        token_mapping_repo.add(
            action_label="a", observation_label="o",
            action_cluster_id=a_cid, observation_cluster_id=o_cid,
            token_cluster_id=noise_tc.id,
        )

        a1 = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        n1 = await a1.assign(a_cid, o_cid, action_category="beta")

        # Rebuild from a clean cache to simulate a separate process.
        TokenAssigner.invalidate()
        a2 = TokenAssigner(
            token_mapping_repo, token_cluster_repo, cluster_repo,
        )
        n2 = await a2.assign(a_cid, o_cid, action_category="beta")

        # alpha=0, beta=1, gamma=2 → noise for "beta" must be -2 both times.
        assert n1 == n2 == -2
