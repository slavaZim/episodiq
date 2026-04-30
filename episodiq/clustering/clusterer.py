"""UMAP + HDBSCAN clusterer for embedding vectors."""

import logging
import warnings
from dataclasses import dataclass

import hdbscan
from hdbscan import validity_index
import numpy as np

warnings.filterwarnings("ignore", message="n_jobs value.*overridden", module="umap")
from umap import UMAP  # noqa: E402

from episodiq.clustering.constants import DEFAULT_PARAMS, Params  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    labels: np.ndarray
    noise_count: int
    dbcv: float  # DBCV [-1, 1]

    @property
    def n_clusters(self) -> int:
        unique = set(self.labels)
        unique.discard(-1)
        return len(unique)

    @property
    def entropy(self) -> float:
        """Normalized entropy of cluster size distribution (0..1). Higher = more uniform."""
        cluster_labels = self.labels[self.labels != -1]
        if len(cluster_labels) == 0:
            return 0.0
        _, counts = np.unique(cluster_labels, return_counts=True)
        k = len(counts)
        if k <= 1:
            return 1.0
        probs = counts / counts.sum()
        h = -float((probs * np.log(probs)).sum())
        return h / np.log(k)


class Clusterer:
    """UMAP (cosine) + HDBSCAN for embedding vectors."""

    def __init__(self, params: Params = DEFAULT_PARAMS):
        self.params = params

    def fit(self, vectors: np.ndarray) -> ClusterResult:
        n_samples, n_features = vectors.shape
        p = self.params

        effective_dims = min(p.umap_dims, n_samples - 2)
        effective_neighbors = min(p.umap_n_neighbors, n_samples - 1)
        reducer = UMAP(
            n_components=effective_dims,
            n_neighbors=effective_neighbors,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(vectors)

        hdb = hdbscan.HDBSCAN(
            min_cluster_size=p.min_cluster_size,
            min_samples=p.min_samples,
            metric="euclidean",
            cluster_selection_method=p.cluster_selection_method,
            cluster_selection_epsilon=p.cluster_selection_epsilon,
        )
        labels = hdb.fit_predict(reduced)

        noise_count = int((labels == -1).sum())

        n_real = len(set(labels) - {-1})
        if n_real >= 1:
            dbcv = float(validity_index(reduced.astype(np.float64), labels))
        else:
            dbcv = -1.0

        return ClusterResult(labels=labels, noise_count=noise_count, dbcv=dbcv)
