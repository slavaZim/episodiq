"""Shared aggregation + weighted-AUC computation for the three
similarity metrics (``cummax`` / ``cummean`` / ``cummeanmax``).

Single source of truth used by:
- ``retrieval.tune.sweep`` (per-trial weighted AUC, no per-step storage)
- ``benchmarks.demo_eval.eval_cascade`` (per-step curves for the eval report)
- ``benchmarks.basic`` (naive-RAG baseline, same metric vocabulary)

Two entry points share one internal step iterator:

- ``weighted_aucs(fail_sims, status, eval_min_step)`` returns
  ``{metric: float}`` — what the sweep needs per trial.
- ``compute_metric_curves(fail_sims, status, eval_min_step)`` returns
  ``{metric: MetricCurve}`` (per-step ``StepAUC`` list + weighted scalar)
  — what eval needs to dump per-step CSVs.

Input format (shared with both production cascade and the basic
baseline): ``fail_sims: {trajectory_key: [(step, fail_sim), ...]}``
plus ``status: {trajectory_key: "success" | "failure"}``. Keys can be
``UUID`` (DB pipeline) or ``int`` (basic.py snapshot indices); the
functions only require hashable equality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


SIMILARITY_METRICS: tuple[str, ...] = ("cummax", "cummean", "cummeanmax")


@dataclass(frozen=True)
class StepAUC:
    """One per-step ROC AUC: AUC across trajectories active at ``step``,
    weighted downstream by ``n_active``.
    """
    step: int
    auc: float
    n_active: int


@dataclass(frozen=True)
class MetricCurve:
    """Per-step AUC curve plus its weighted (by ``n_active``) average."""
    metric: str
    weighted_auc: float
    per_step: list[StepAUC]


def aggregate_ffs(ffs: np.ndarray) -> tuple[float, float, float]:
    """Fold an ordered ``fail_sim`` array into
    ``(cummax, cummean, cummeanmax)``.

    - ``cummax``     = max value
    - ``cummean``    = arithmetic mean
    - ``cummeanmax`` = peak of the running-mean curve (peak sustained avg)
    """
    cm = np.cumsum(ffs) / np.arange(1, ffs.size + 1)
    return float(ffs.max()), float(ffs.mean()), float(cm.max())


def _per_step_aucs(
    fail_sims, status, eval_min_step,
) -> dict[str, list[StepAUC]]:
    """Iterate steps ``[eval_min_step, max_step]`` and produce
    ``{metric: [StepAUC...]}``. Empty when fail_sims has no signal or
    the max step never crosses ``eval_min_step``.

    Each step uses only trajectories whose last observed step is
    ``>= S`` (we'd otherwise extrapolate past their lifetime). Steps
    with a single class (``len(set(y)) < 2``) are skipped since AUC
    is undefined.
    """
    if not fail_sims:
        return {}
    sorted_by_tid = {
        tid: sorted(snaps) for tid, snaps in fail_sims.items()
    }
    max_step_per_tid = {
        tid: max(s for s, _ in snaps)
        for tid, snaps in sorted_by_tid.items()
    }
    max_step_eval = (
        max(max_step_per_tid.values()) if max_step_per_tid else 0
    )
    if max_step_eval < eval_min_step:
        return {}

    aucs_per_metric: dict[str, list[StepAUC]] = {
        m: [] for m in SIMILARITY_METRICS
    }
    for S in range(eval_min_step, max_step_eval + 1):
        y: list[int] = []
        scores: dict[str, list[float]] = {
            m: [] for m in SIMILARITY_METRICS
        }
        for tid, snaps in sorted_by_tid.items():
            if tid not in status:
                continue
            if max_step_per_tid[tid] < S:
                continue
            ffs = np.asarray(
                [ff for s, ff in snaps if s <= S], dtype=np.float64,
            )
            if ffs.size == 0:
                continue
            cummax, cummean, cummeanmax = aggregate_ffs(ffs)
            y.append(1 if status[tid] == "failure" else 0)
            scores["cummax"].append(cummax)
            scores["cummean"].append(cummean)
            scores["cummeanmax"].append(cummeanmax)
        if len(set(y)) < 2:
            continue
        for m in SIMILARITY_METRICS:
            auc = float(roc_auc_score(y, scores[m]))
            aucs_per_metric[m].append(StepAUC(S, auc, len(y)))
    return aucs_per_metric


def weighted_aucs(
    fail_sims, status, eval_min_step: int,
) -> dict[str, float]:
    """Weighted AUC (by ``n_active``) per metric. Discards per-step
    detail — what the sweep stores per trial.
    """
    aucs = _per_step_aucs(fail_sims, status, eval_min_step)
    out: dict[str, float] = {}
    for m, step_aucs in aucs.items():
        if not step_aucs:
            continue
        total_w = sum(sa.n_active for sa in step_aucs)
        out[m] = sum(sa.auc * sa.n_active for sa in step_aucs) / total_w
    return out


def compute_metric_curves(
    fail_sims, status, eval_min_step: int,
) -> dict[str, MetricCurve]:
    """Full ``MetricCurve`` per metric (weighted AUC + per-step list).
    Used by eval to render per-step graphs.
    """
    aucs = _per_step_aucs(fail_sims, status, eval_min_step)
    out: dict[str, MetricCurve] = {}
    for m, step_aucs in aucs.items():
        if not step_aucs:
            continue
        total_w = sum(sa.n_active for sa in step_aucs)
        weighted = sum(sa.auc * sa.n_active for sa in step_aucs) / total_w
        out[m] = MetricCurve(
            metric=m, weighted_auc=weighted, per_step=step_aucs,
        )
    return out


@dataclass(frozen=True)
class AucCI:
    """Bootstrap confidence interval for one metric's weighted AUC."""
    lo: float
    hi: float
    mean: float


def bootstrap_aucs(
    fail_sims, status, eval_min_step: int,
    *, n_boot: int = 200, ci_level: float = 0.95, seed: int = 42,
) -> dict[str, AucCI]:
    """Per-metric weighted-AUC confidence interval via per-trajectory
    bootstrap (resample trajectories with replacement, recompute
    weighted AUC, report the percentile band).

    Args:
        n_boot: bootstrap draws. 200 stabilises the percentile within
            ~0.005 AUC on n≈100 trajs; raise for tighter CIs.
        ci_level: width of the central band (default 0.95).
        seed: NumPy RNG seed for reproducibility.

    Returns:
        ``{metric: AucCI}`` for every metric that produced a finite
        weighted AUC. Metrics that the point estimate dropped (no
        valid steps) are absent here too.
    """
    tids = list(fail_sims.keys())
    if not tids:
        return {}
    rng = np.random.default_rng(seed)
    boot_samples: dict[str, list[float]] = {
        m: [] for m in SIMILARITY_METRICS
    }
    for _ in range(n_boot):
        idxs = rng.integers(0, len(tids), size=len(tids))
        # Resampling the same trajectory N times needs distinct keys
        # so weighted_aucs sees them as separate trajectories.
        boot_fs: dict = {}
        boot_status: dict = {}
        for draw_idx, ti in enumerate(idxs):
            tid = tids[ti]
            key = (tid, draw_idx)
            boot_fs[key] = fail_sims[tid]
            if tid in status:
                boot_status[key] = status[tid]
        aucs = weighted_aucs(boot_fs, boot_status, eval_min_step)
        for m, auc in aucs.items():
            boot_samples[m].append(auc)

    alpha = (1.0 - ci_level) / 2.0
    out: dict[str, AucCI] = {}
    for m, samples in boot_samples.items():
        if not samples:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        out[m] = AucCI(
            lo=float(np.percentile(arr, 100 * alpha)),
            hi=float(np.percentile(arr, 100 * (1 - alpha))),
            mean=float(arr.mean()),
        )
    return out
