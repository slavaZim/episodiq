"""Tests for UMAP + HDBSCAN clusterer."""

import numpy as np
import pytest

from episodiq.clustering.clusterer import ClusterResult, Clusterer
from episodiq.clustering.constants import Params


class TestClusterResult:
    def test_n_clusters_excludes_noise(self):
        r = ClusterResult(labels=np.array([0, 0, 1, 1, -1, -1, 2]), noise_count=2, dbcv=0.5)
        assert r.n_clusters == 3

    def test_n_clusters_all_noise(self):
        r = ClusterResult(labels=np.array([-1, -1, -1]), noise_count=3, dbcv=-1.0)
        assert r.n_clusters == 0

    def test_n_clusters_single_cluster(self):
        r = ClusterResult(labels=np.array([0, 0, 0, -1]), noise_count=1, dbcv=0.5)
        assert r.n_clusters == 1

    def test_entropy_uniform_two_clusters(self):
        """Equal-size clusters → entropy = 1.0."""
        r = ClusterResult(labels=np.array([0, 0, 1, 1]), noise_count=0, dbcv=0.5)
        assert r.entropy == pytest.approx(1.0)

    def test_entropy_single_cluster(self):
        """One cluster → entropy = 1.0 (degenerate case, k=1)."""
        r = ClusterResult(labels=np.array([0, 0, 0]), noise_count=0, dbcv=0.5)
        assert r.entropy == 1.0

    def test_entropy_skewed(self):
        """Highly skewed distribution → low entropy."""
        labels = np.array([0] * 100 + [1])
        r = ClusterResult(labels=labels, noise_count=0, dbcv=0.5)
        assert r.entropy < 0.2

    def test_entropy_ignores_noise(self):
        """Noise labels (-1) excluded from entropy calculation."""
        r = ClusterResult(labels=np.array([0, 0, 1, 1, -1, -1, -1]), noise_count=3, dbcv=0.5)
        assert r.entropy == pytest.approx(1.0)

    def test_entropy_all_noise(self):
        r = ClusterResult(labels=np.array([-1, -1]), noise_count=2, dbcv=-1.0)
        assert r.entropy == 0.0


class TestClusterer:
    @pytest.fixture
    def blobs(self):
        """Three well-separated gaussian blobs."""
        rng = np.random.RandomState(42)
        centers = [np.zeros(50), np.ones(50) * 5, np.ones(50) * -5]
        vecs = np.vstack([c + rng.randn(40, 50) * 0.3 for c in centers])
        return vecs  # 120 × 50

    def test_fit_finds_clusters(self, blobs):
        c = Clusterer(Params(min_cluster_size=10, min_samples=3, umap_dims=5, umap_n_neighbors=10))
        result = c.fit(blobs)
        assert result.n_clusters >= 2
        assert result.noise_count < len(blobs) // 2

    def test_fit_dbcv_positive_for_clear_clusters(self, blobs):
        c = Clusterer(Params(min_cluster_size=10, min_samples=3, umap_dims=5, umap_n_neighbors=10))
        result = c.fit(blobs)
        assert result.dbcv > 0

    def test_fit_high_min_cluster_size_fewer_clusters(self, blobs):
        small = Clusterer(Params(min_cluster_size=10, min_samples=3, umap_dims=5, umap_n_neighbors=10))
        large = Clusterer(Params(min_cluster_size=50, min_samples=3, umap_dims=5, umap_n_neighbors=10))
        r_small = small.fit(blobs)
        r_large = large.fit(blobs)
        assert r_large.n_clusters <= r_small.n_clusters

    def test_fit_effective_dims_capped(self):
        """When n_samples is small, umap_dims is capped to n_samples - 2."""
        rng = np.random.RandomState(42)
        vecs = rng.randn(15, 50)  # only 15 samples, umap_dims=30 would fail
        c = Clusterer(Params(min_cluster_size=5, min_samples=2, umap_dims=30, umap_n_neighbors=10))
        result = c.fit(vecs)
        assert isinstance(result, ClusterResult)

    def test_fit_labels_length_matches_input(self, blobs):
        c = Clusterer(Params(min_cluster_size=10, min_samples=3, umap_dims=5, umap_n_neighbors=10))
        result = c.fit(blobs)
        assert len(result.labels) == len(blobs)
