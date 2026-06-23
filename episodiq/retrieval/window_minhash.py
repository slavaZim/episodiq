"""Per-window MinHash signatures + LSH banding.

For each W-sized window of a trace_tokens sequence this module computes:
  * a fixed-length MinHash signature over the multiset of q-grams in the
    window;
  * its LSH banding — ``num_bands`` slices of ``rows_per_band`` rows each,
    each slice rolled into a single 63-bit ``band_hash`` for indexed lookup.

Two windows are LSH candidates iff at least one of their band hashes
matches. Approximate Jaccard is then estimated as
``matching_bands / num_bands`` (or via the full signature if exact rerank
is needed).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from episodiq.config.retrieval_config import WindowMinHashConfig

_MASK = (1 << 63) - 1       # signed 64-bit mask; fits Postgres BIGINT.
_BAND_MOD = (1 << 63) - 1
_BAND_POLY = 1000003
_QGRAM_BASE = 8192


@njit(cache=True, nogil=True)
def _compute_signatures_jit(
    tokens: np.ndarray,
    window: int,
    q_gram: int,
    stride: int,
    hash_seeds: np.ndarray,
    mask: int,
    qgram_base: int,
) -> np.ndarray:
    """Per-window MinHash signatures via seed-XOR-multiply hashing
    (bench's per-q-gram transform: ``((ng * (seed | 1)) ^ seed) & MASK``).
    """
    n = len(tokens)
    sig_size = len(hash_seeds)
    if n < window:
        return np.full((0, sig_size), mask, dtype=np.int64)
    n_wins = (n - window) // stride + 1
    sigs = np.full((n_wins, sig_size), mask, dtype=np.int64)
    qgram_mod = qgram_base ** q_gram
    qgrams_per_window = window - q_gram + 1
    if qgrams_per_window <= 0:
        return sigs
    for wi in range(n_wins):
        start = wi * stride
        ng = 0
        for k in range(q_gram):
            ng = ng * qgram_base + int(tokens[start + k])
        for hi in range(sig_size):
            seed = hash_seeds[hi]
            h = ((ng * (seed | 1)) ^ seed) & mask
            if h < sigs[wi, hi]:
                sigs[wi, hi] = h
        for qi in range(1, qgrams_per_window):
            ng = (ng * qgram_base + int(tokens[start + qi + q_gram - 1])) % qgram_mod
            for hi in range(sig_size):
                seed = hash_seeds[hi]
                h = ((ng * (seed | 1)) ^ seed) & mask
                if h < sigs[wi, hi]:
                    sigs[wi, hi] = h
    return sigs


@njit(cache=True, nogil=True)
def _bands_for_signatures(
    sigs: np.ndarray, num_bands: int, rows_per_band: int,
    poly: int, mod: int,
) -> np.ndarray:
    """Roll each signature's ``num_bands`` slices into single ints.

    Returns shape (n_windows, num_bands) of int64.
    """
    n_wins, sig_size = sigs.shape
    out = np.zeros((n_wins, num_bands), dtype=np.int64)
    for wi in range(n_wins):
        for b in range(num_bands):
            h = 0
            base = b * rows_per_band
            for r in range(rows_per_band):
                h = (h * poly + int(sigs[wi, base + r])) % mod
            out[wi, b] = h
    return out


class WindowMinHasher:
    """Universal-hash MinHasher producing per-window signatures + LSH bands."""

    def __init__(self, config: WindowMinHashConfig) -> None:
        self.config = config
        rng = np.random.default_rng(self.config.seed)
        self._hash_seeds = rng.integers(
            1, 1 << 63, size=self.config.signature_size, dtype=np.int64,
        )

    def signatures(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(signatures, starts)`` for every W-window in ``tokens``.

        signatures shape: ``(n_windows, signature_size)`` int64
        starts shape: ``(n_windows,)`` int64 — token index per window start
        """
        cfg = self.config
        tokens_arr = np.ascontiguousarray(tokens, dtype=np.int64)
        sigs = _compute_signatures_jit(
            tokens_arr, cfg.window, cfg.q_gram, 1,
            self._hash_seeds, _MASK, _QGRAM_BASE,
        )
        starts = np.arange(sigs.shape[0], dtype=np.int64)
        return sigs, starts

    def bands(self, signatures: np.ndarray) -> np.ndarray:
        """Roll signatures into LSH band hashes.

        Returns shape ``(n_windows, num_bands)`` of int64 band hashes.
        """
        cfg = self.config
        if signatures.size == 0:
            return np.zeros((0, cfg.num_bands), dtype=np.int64)
        return _bands_for_signatures(
            signatures, cfg.num_bands, cfg.rows_per_band, _BAND_POLY, _BAND_MOD,
        )

    def signatures_and_bands(
        self, tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: ``(signatures, starts, bands)`` for ``tokens``."""
        sigs, starts = self.signatures(tokens)
        return sigs, starts, self.bands(sigs)

    def bands_for_window(
        self, tokens: list[int] | np.ndarray, start: int,
    ) -> np.ndarray:
        """Compute LSH band hashes for the single window ``tokens[start:start+W]``.

        Returns shape ``(num_bands,)`` int64. Used incrementally as new
        windows form per token step (avoids recomputing all earlier
        windows).
        """
        sigs = self._signature_for_window_jit(tokens, start)
        if sigs.shape[0] == 0:
            return np.zeros(self.config.num_bands, dtype=np.int64)
        bands = _bands_for_signatures(
            sigs, self.config.num_bands, self.config.rows_per_band,
            _BAND_POLY, _BAND_MOD,
        )
        return bands[0]

    def signature_for_window(
        self, tokens: list[int] | np.ndarray, start: int,
    ) -> np.ndarray:
        """Compute the raw MinHash signature for the single window
        ``tokens[start:start+W]``. Returns shape ``(signature_size,)`` int64.
        Used by exact-jaccard rerank to compare query and candidate windows
        directly (band hashes throw away too much information for that).
        """
        sigs = self._signature_for_window_jit(tokens, start)
        if sigs.shape[0] == 0:
            return np.full(
                self.config.signature_size, _MASK, dtype=np.int64,
            )
        return sigs[0]

    def _signature_for_window_jit(
        self, tokens: list[int] | np.ndarray, start: int,
    ) -> np.ndarray:
        cfg = self.config
        end = start + cfg.window
        slice_tokens = np.ascontiguousarray(
            tokens[start:end] if isinstance(tokens, np.ndarray)
            else np.asarray(tokens[start:end], dtype=np.int64),
            dtype=np.int64,
        )
        if slice_tokens.shape[0] < cfg.window:
            return np.zeros((0, cfg.signature_size), dtype=np.int64)
        # stride=window so the jit kernel yields exactly one signature row.
        return _compute_signatures_jit(
            slice_tokens, cfg.window, cfg.q_gram, cfg.window,
            self._hash_seeds, _MASK, _QGRAM_BASE,
        )


def jaccard_estimate(
    sig_a: np.ndarray, sig_b: np.ndarray,
) -> float:
    """Per-position match fraction between two signatures (Jaccard estimate)."""
    if sig_a.size == 0 or sig_b.size == 0 or sig_a.size != sig_b.size:
        return 0.0
    return float((sig_a == sig_b).sum()) / sig_a.size
