"""Shared constants for the clustering module."""

from dataclasses import dataclass

PREFIXES = {"observation": "o", "action": "a"}

MAX_NOISE_RATIO = 0.20


@dataclass(frozen=True)
class Params:
    """HDBSCAN + UMAP parameters."""
    min_cluster_size: int = 10
    min_samples: int = 5
    umap_dims: int = 30
    umap_n_neighbors: int = 15
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0


DEFAULT_PARAMS = Params()

DEFAULT_GRID = [
    Params(min_cluster_size=cs, min_samples=ms, umap_dims=ud, umap_n_neighbors=un)
    for cs, ms in [(3, 2), (5, 3), (10, 5), (15, 7), (20, 10)]
    for ud in [40, 50, 60]
    for un in [5, 10, 15, 20, 25]
]
