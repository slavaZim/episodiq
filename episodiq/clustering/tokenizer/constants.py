"""Tokenizer clustering defaults: UMAP+HDBSCAN params for act_obs embeddings."""

from episodiq.clustering.constants import Params


DEFAULT_PARAMS = Params(
    min_cluster_size=10,
    min_samples=5,
    umap_dims=50,
    umap_n_neighbors=15,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
)


DEFAULT_GRID = [
    Params(
        min_cluster_size=cs,
        min_samples=ms,
        umap_dims=ud,
        umap_n_neighbors=un,
    )
    for cs, ms in [(5, 3), (10, 5), (15, 7), (20, 10)]
    for ud in [30, 50, 80]
    for un in [10, 15, 25]
]
