"""Incremental trace / token state calculator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from episodiq.config import get_config
from episodiq.retrieval.minhash import MinHasher, ngrams

if TYPE_CHECKING:
    from episodiq.storage.postgres.models import TrajectoryPath


class PathStateCalculator:
    """Builds the cluster-label trace, act_obs token sequence, and the
    n-gram MinHash signature used by the retrieval prefilter.

    ``ngram_n`` and the MinHasher default to ``RetrievalConfig`` values
    (env vars ``EPISODIQ_NGRAM_N`` / ``EPISODIQ_MINHASH_K`` /
    ``EPISODIQ_MINHASH_SEED``). Pass explicit values to override in tests
    or one-off scripts.
    """

    def __init__(
        self,
        minhasher: MinHasher | None = None,
        ngram_n: int | None = None,
    ) -> None:
        if minhasher is None or ngram_n is None:
            cfg = get_config().retrieval
            if minhasher is None:
                minhasher = MinHasher(k=cfg.minhash_k, seed=cfg.minhash_seed)
            if ngram_n is None:
                ngram_n = cfg.ngram_n
        self._minhasher = minhasher
        self._n = ngram_n

    def granular_step(
        self,
        prev_path: TrajectoryPath | None,
        obs_label: str,
    ) -> list[str]:
        """Trace of alternating obs/action cluster labels; lags one step
        behind the current observation.
        """
        if prev_path and prev_path.action_label:
            return list(prev_path.trace) + [prev_path.action_label, obs_label]
        return [obs_label]

    def token_step(
        self,
        prev_path: TrajectoryPath | None,
        token_ordinal: int,
    ) -> tuple[list[int], list[int] | None]:
        """Append a act_obs token; return (token_trace, minhash_signature).

        NOISE_TOKEN_ID participates in trace_tokens and n-grams like any
        other ordinal. The signature is None for paths shorter than the
        n-gram window.
        """
        prev = (prev_path.trace_tokens if prev_path else None) or []
        tokens = list(prev) + [token_ordinal]
        signature = self._minhasher.signature(ngrams(tokens, self._n))
        return tokens, signature
