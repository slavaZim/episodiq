"""Agg-shift scoring over a Lev cost grid.

The grid ``M[i, j]`` holds the Levenshtein distance between two windows
``a[i:i+W]`` and ``b[j:j+W]`` for cells inside the band. Uniform Lev is
used when ``gap_open == gap_extend == 1.0``; otherwise a Gotoh affine
kernel is invoked. Per query row the shift band ``d ∈ [-w, w]`` is
reduced by ``BandAgg`` (``min_distance`` = best-shift wins,
``mean`` = average) with a per-shape penalty on ``d``, and the result
is averaged over rows. Final ``sim ∈ (-∞, 1.0]`` (higher = more
similar).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numba import njit

from episodiq.config.scoring_config import AggShiftConfig


_INF = 1e18


class BandAgg(IntEnum):
    """Per-row reduction over the shift band. Lower-case member names
    match the cascade-wide ``aggregation`` string ("min_distance" /
    "mean") so ``BandAgg[aggregation]`` converts directly.
    """
    min_distance = 0
    mean = 1


class PenaltyShape(IntEnum):
    """Shift-penalty kernel selector for ``agg_shift_score``."""
    lin = 0
    const = 1
    quad = 2
    gauss = 3


@njit(cache=True, nogil=True)
def _window_edit_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Wagner-Fischer Levenshtein between two integer windows."""
    n = len(a)
    m = len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    prev = np.zeros(m + 1, dtype=np.float64)
    cur = np.zeros(m + 1, dtype=np.float64)
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0.0 if ai == b[j - 1] else 1.0)
            ins = cur[j - 1] + 1.0
            dele = prev[j] + 1.0
            best = sub
            if ins < best:
                best = ins
            if dele < best:
                best = dele
            cur[j] = best
        prev, cur = cur, prev
    return prev[m]


@njit(cache=True, nogil=True)
def _window_affine_distance(
    a: np.ndarray, b: np.ndarray, gap_open: float, gap_extend: float,
) -> float:
    """Gotoh affine-gap edit distance between two integer windows.

    Three DP layers: ``M`` (match/mismatch), ``X`` (gap in a), ``Y`` (gap
    in b). Substitution cost is 1 on mismatch, 0 on match.
    """
    n = len(a)
    m = len(b)
    if n == 0:
        return 0.0 if m == 0 else gap_open + (m - 1) * gap_extend
    if m == 0:
        return gap_open + (n - 1) * gap_extend

    M_prev = np.full(m + 1, _INF, dtype=np.float64)
    X_prev = np.full(m + 1, _INF, dtype=np.float64)
    Y_prev = np.full(m + 1, _INF, dtype=np.float64)
    M_cur = np.full(m + 1, _INF, dtype=np.float64)
    X_cur = np.full(m + 1, _INF, dtype=np.float64)
    Y_cur = np.full(m + 1, _INF, dtype=np.float64)

    M_prev[0] = 0.0
    for j in range(1, m + 1):
        Y_prev[j] = gap_open + (j - 1) * gap_extend
    for i in range(1, n + 1):
        M_cur[0] = _INF
        Y_cur[0] = _INF
        X_cur[0] = gap_open + (i - 1) * gap_extend
        ai = a[i - 1]
        for j in range(1, m + 1):
            sub_cost = 0.0 if ai == b[j - 1] else 1.0
            mp = M_prev[j - 1]
            xp = X_prev[j - 1]
            yp = Y_prev[j - 1]
            best_m = mp
            if xp < best_m:
                best_m = xp
            if yp < best_m:
                best_m = yp
            M_cur[j] = best_m + sub_cost

            x_open = M_prev[j] + gap_open
            x_ext = X_prev[j] + gap_extend
            X_cur[j] = x_open if x_open < x_ext else x_ext

            y_open = M_cur[j - 1] + gap_open
            y_ext = Y_cur[j - 1] + gap_extend
            Y_cur[j] = y_open if y_open < y_ext else y_ext

        for j in range(m + 1):
            M_prev[j] = M_cur[j]
            X_prev[j] = X_cur[j]
            Y_prev[j] = Y_cur[j]

    best = M_prev[m]
    if X_prev[m] < best:
        best = X_prev[m]
    if Y_prev[m] < best:
        best = Y_prev[m]
    return best


@njit(cache=True, nogil=True)
def compute_lev_grid(
    a: np.ndarray, b: np.ndarray, window: int, band: int,
) -> np.ndarray:
    """Pairwise window-cost grid with uniform Lev.

    ``M[i, j] = Lev(a[i:i+W], b[j:j+W])`` for ``|i - j| <= band``;
    cells outside the band are zero (never read by the scorer). Returns
    an empty grid when either side is shorter than ``window``.
    """
    nA = len(a)
    nB = len(b)
    n_pos_a = nA - window + 1
    n_pos_b = nB - window + 1
    if n_pos_a <= 0 or n_pos_b <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros((n_pos_a, n_pos_b), dtype=np.float64)
    for i in range(n_pos_a):
        j_lo = i - band
        if j_lo < 0:
            j_lo = 0
        j_hi = i + band + 1
        if j_hi > n_pos_b:
            j_hi = n_pos_b
        wa = a[i:i + window]
        for j in range(j_lo, j_hi):
            wb = b[j:j + window]
            out[i, j] = _window_edit_distance(wa, wb)
    return out


@njit(cache=True, nogil=True)
def compute_lev_grid_affine(
    a: np.ndarray, b: np.ndarray, window: int, band: int,
    gap_open: float, gap_extend: float,
) -> np.ndarray:
    """Pairwise window-cost grid with Gotoh affine gaps."""
    nA = len(a)
    nB = len(b)
    n_pos_a = nA - window + 1
    n_pos_b = nB - window + 1
    if n_pos_a <= 0 or n_pos_b <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros((n_pos_a, n_pos_b), dtype=np.float64)
    for i in range(n_pos_a):
        j_lo = i - band
        if j_lo < 0:
            j_lo = 0
        j_hi = i + band + 1
        if j_hi > n_pos_b:
            j_hi = n_pos_b
        wa = a[i:i + window]
        for j in range(j_lo, j_hi):
            wb = b[j:j + window]
            out[i, j] = _window_affine_distance(wa, wb, gap_open, gap_extend)
    return out


@njit(cache=True, nogil=True)
def agg_shift_score(
    M: np.ndarray,
    n_pos_q: int,
    n_pos_c: int,
    window: int,
    band: int,
    lam: float,
    shape_code: int,
    band_agg_code: int,
    sigma: float,
) -> float:
    """Agg-shift over ``M[:n_pos_q, :n_pos_c]`` with a shift-penalty
    kernel. Per query row ``i``, aggregates ``M[i, j] + pen(d)`` across
    ``d ∈ [-band, band]`` according to ``band_agg_code``
    (``BandAgg.min_distance`` or ``BandAgg.mean``). ``pen(d)`` is
    selected by ``shape_code`` (``PenaltyShape``):

      lin    ``lam · |d|``
      const  ``lam · (d != 0)``
      quad   ``lam · d²``
      gauss  ``lam · (1 - exp(-d²/(2σ²)))``

    ``sigma`` is only used by the Gaussian shape; the other shapes
    ignore it. Returns ``sim`` where 1.0 is perfect match; -1.0 on
    degenerate inputs. Int codes are passed in because Numba JIT
    cannot hold Python objects across the call boundary; ``IntEnum``
    values auto-convert.
    """
    inv_two_sigma_sq = 0.0
    if shape_code == 3:
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    n_use = n_pos_q if n_pos_q < n_pos_c else n_pos_c
    if n_use <= 0:
        return -1.0
    total = 0.0
    for i in range(n_use):
        best = 1e18
        band_total = 0.0
        band_count = 0
        for d in range(-band, band + 1):
            j = i + d
            if j < 0 or j >= n_pos_c:
                continue
            if shape_code == 0:
                ad = d if d >= 0 else -d
                pen = lam * ad
            elif shape_code == 1:
                pen = lam if d != 0 else 0.0
            elif shape_code == 2:
                pen = lam * (d * d)
            else:
                pen = lam * (1.0 - np.exp(-(d * d) * inv_two_sigma_sq))
            cost = M[i, j] + pen
            if cost < best:
                best = cost
            band_total += cost
            band_count += 1
        if band_count == 0:
            continue
        if band_agg_code == 1:
            total += band_total / band_count
        else:
            total += best
    return 1.0 - (total / n_use) / window


def score_pair(
    a: np.ndarray,
    b: np.ndarray,
    config: AggShiftConfig,
    a_prefix_len: int | None = None,
    band_aggregation: str = "min_distance",
) -> float:
    """Build the Lev grid for ``(a, b)`` and run ``agg_shift_score``.

    ``a_prefix_len`` restricts the query side to the first N tokens
    (snapshot mode) without rebuilding the grid. ``band_aggregation``
    is the cascade-wide ``aggregation`` string and selects how shifts
    within the band are reduced per query row — ``"min_distance"`` for
    classic min-shift (best-shift wins), ``"mean"`` for the average.
    """
    if config.is_uniform:
        M = compute_lev_grid(a, b, config.window, config.half_window)
    else:
        M = compute_lev_grid_affine(
            a, b, config.window, config.half_window,
            config.gap_open, config.gap_extend,
        )
    if M.size == 0:
        return -1.0
    n_pos_a, n_pos_b = M.shape
    n_pos_q = (a_prefix_len - config.window + 1) if a_prefix_len is not None else n_pos_a
    n_use = min(max(n_pos_q, 0), n_pos_a)
    if n_use <= 0 or n_pos_b <= 0:
        return -1.0
    return float(agg_shift_score(
        M, n_use, n_pos_b, config.window, config.half_window,
        config.lam, int(PenaltyShape[config.penalty_shape]),
        int(BandAgg[band_aggregation]), config.sigma,
    ))
