"""MinHash signatures + Jaccard estimation for trace_tokens n-grams.

Each path's trace_tokens is hashed into a fixed-length signature of K
integers; the Jaccard similarity between two paths' n-gram sets is
approximated by the fraction of matching positions in their signatures.
Standard error of the estimate is ~1/sqrt(K).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TypeVar

import numpy as np

T = TypeVar("T")

# Mersenne prime — modulus for the universal hash family.
_HASH_MOD = (1 << 61) - 1

DEFAULT_SIGNATURE_SIZE = 256
DEFAULT_NGRAM_N = 3
DEFAULT_SEED = 0x4E50_4953  # "EPIS"


def ngrams(tokens: Iterable[int], n: int) -> set[int]:
    tokens = list(tokens)
    if len(tokens) < n:
        return set()
    B = 8192
    out: set[int] = set()
    x = 0
    for i in range(n):
        x = x * B + int(tokens[i])
    out.add(x)
    Bn = B ** n
    for i in range(1, len(tokens) - n + 1):
        x = (x * B + int(tokens[i + n - 1])) % Bn
        out.add(x)
    return out


class MinHasher:
    """Universal-hash MinHasher producing K-int signatures."""

    def __init__(
        self,
        k: int = DEFAULT_SIGNATURE_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _HASH_MOD) for _ in range(k)]
        self._b = [rng.randrange(0, _HASH_MOD) for _ in range(k)]
        self.k = k

    def signature(self, ngram_set: set[int]) -> list[int] | None:
        """Compute the K-int MinHash signature of an n-gram set.

        Returns None when the input set is empty so callers don't compare
        all-sentinel signatures (which would otherwise match each other
        100% and produce spurious hits).
        """
        if not ngram_set:
            return None
        sig = [_HASH_MOD] * self.k
        a = self._a
        b = self._b
        for ng in ngram_set:
            for i in range(self.k):
                h = (a[i] * ng + b[i]) % _HASH_MOD
                if h < sig[i]:
                    sig[i] = h
        return sig

    def signature_from_tokens(
        self, tokens: Iterable[int], n: int,
    ) -> list[int] | None:
        return self.signature(ngrams(tokens, n))


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Fraction of positions where the two signatures agree — converges to
    the true Jaccard similarity of the underlying n-gram sets.
    """
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def max_pool_jaccard_per_key(
    query_sig: list[int],
    corpus: list[tuple[T, list[int]]],
) -> dict[T, float]:
    """Compute MinHash Jaccard estimate between ``query_sig`` and every
    signature in ``corpus`` and MAX-pool per group key. Each corpus entry
    is ``(group_key, signature)``. Vectorised over the corpus via numpy.

    The corpus is taken as an argument (not loaded internally) so callers
    that issue many queries against the same corpus — like an offline
    tuning sweep — can fetch it once and reuse across queries.
    """
    k = len(query_sig)
    if k == 0 or not corpus:
        return {}
    keys: list[T] = []
    sigs: list[list[int]] = []
    for key, sig in corpus:
        if sig and len(sig) == k:
            keys.append(key)
            sigs.append(sig)
    if not sigs:
        return {}
    sig_matrix = np.asarray(sigs, dtype=np.int64)
    query_arr = np.asarray(query_sig, dtype=np.int64)
    sims = (sig_matrix == query_arr).sum(axis=1).astype(np.float64) / k
    out: dict[T, float] = {}
    for key, sim in zip(keys, sims):
        sim_f = float(sim)
        if sim_f > out.get(key, -1.0):
            out[key] = sim_f
    return out


