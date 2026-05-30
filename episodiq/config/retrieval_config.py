"""Retrieval pipeline params."""

import os
from dataclasses import dataclass

from episodiq.retrieval.minhash import (
    DEFAULT_NGRAM_N,
    DEFAULT_SEED,
    DEFAULT_SIGNATURE_SIZE,
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Hyperparameters for the retrieval pipeline.

    top_k                : final number of candidate trajectories returned.
    similarity_threshold : MinHash Jaccard-estimate cutoff. Candidates with
                           per-trajectory MAX similarity below this are
                           excluded. Lower = bigger shortlists.
    ngram_n              : size of the n-gram window over trace_tokens used
                           to build MinHash signatures. With a small token
                           vocabulary (e.g. a few dozen act_obs clusters)
                           n=3 saturates fast — raise to 5–7 to keep
                           n-grams informative.
    minhash_k            : MinHash signature length (number of permutations).
                           Standard error of the Jaccard estimate is ~1/sqrt(K).
    minhash_seed         : seed for the MinHash universal-hash family.
                           Must match between index build and retrieval.
    """
    top_k: int
    similarity_threshold: float
    ngram_n: int
    minhash_k: int
    minhash_seed: int

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            top_k=int(os.getenv("EPISODIQ_RETRIEVAL_TOP_K", "25")),
            similarity_threshold=float(
                os.getenv("EPISODIQ_RETRIEVAL_SIMILARITY_THRESHOLD", "0.20"),
            ),
            ngram_n=int(os.getenv("EPISODIQ_NGRAM_N", str(DEFAULT_NGRAM_N))),
            minhash_k=int(
                os.getenv("EPISODIQ_MINHASH_K", str(DEFAULT_SIGNATURE_SIZE)),
            ),
            minhash_seed=int(
                os.getenv("EPISODIQ_MINHASH_SEED", str(DEFAULT_SEED)),
            ),
        )
