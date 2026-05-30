"""Tokenizer module: act_obs clustering → token_clusters + token_mapping.

Mirrors the episodiq/clustering/* structure (manager, saver, grid_search,
pipeline, assigner) but operates on act_obs embeddings (concatenated action
and observation centroids) instead of message-level embeddings.
"""
