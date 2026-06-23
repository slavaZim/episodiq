"""Optuna-driven retrieval sweep.

Outer grid: window size W. Each W
  1. rebuilds an alt LSH table from existing trace_tokens (no
     retokenization — tokens are W-agnostic),
  2. constructs an empty ``RetrievalCache`` and runs a single warm-up
     pass — ``retrieval.search()`` over every snapshot with a
     ``RetrievalConfig`` that uses the maximum ``prefetch_n_uniq``,
     ``jaccard_n_uniq``, and ``top_k`` the sweep will ever sample. This
     populates the LSH, candidate-token, and jaccard slots; the result
     is discarded,
  3. runs three Optuna TPE studies sequentially — one per metric — in
     the main process. Each trial creates its own ``Retrieval`` bound
     to the trial's ``cas_cfg`` / ``ms_cfg`` and the shared cache, then
     fans out ``retrieval.search()`` across snapshots via
     ``asyncio.gather`` bounded by ``n_workers``. Optuna's ``n_jobs``
     limits concurrent trials inside one study.

Per trial: for every snapshot we get a ``fail_sim`` over the top-K
candidates and group by trajectory, then build three time series:

  - ``cummax``     = max fail_sim over snapshots ≤ S
  - ``cummean``    = mean fail_sim over snapshots ≤ S
  - ``cummeanmax`` = max of the running-mean curve (peak sustained avg)

Names match the analytics-side ``SIMILARITY_METRICS`` so a sweep
target maps directly to a runtime ``report --metric`` choice.

For each metric and step S in the eval window we compute a ROC AUC
across trajectories active at S, then weight per-step AUCs by the
count of active trajectories. Each metric's study optimises its own
weighted AUC.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
import optuna
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.analytics.metrics import SIMILARITY_METRICS, weighted_aucs
from episodiq.config.retrieval_config import (
    RetrievalConfig,
    WindowMinHashConfig,
    validate_retrieval_window,
)
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.cache import RetrievalCache
from episodiq.retrieval.retrieval import Retrieval, RetrievalQuery
from episodiq.retrieval.tune.constants import (
    DEFAULT_AGGREGATION_GRID,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EVAL_MIN_STEP,
    DEFAULT_GAP_EXTEND_RANGE,
    DEFAULT_GAP_OPEN_RANGE,
    DEFAULT_JACCARD_N_UNIQ_RANGE,
    DEFAULT_LAM_RANGE,
    DEFAULT_MULTIVARIATE,
    DEFAULT_N_JOBS,
    DEFAULT_N_TRIALS,
    DEFAULT_N_WORKERS,
    DEFAULT_OPTUNA_SEED,
    DEFAULT_PENALTY_SHAPE_CHOICES,
    DEFAULT_PREFETCH_N_UNIQ_RANGE,
    DEFAULT_SIGMA_RANGE,
    DEFAULT_TOP_K_RANGE,
    DEFAULT_WINDOW_GRID,
)
from episodiq.retrieval.tune.lsh_rebuild import (
    drop_alt_lsh_table,
    rebuild_lsh_into,
)
from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository,
    TrajectoryRepository,
    TrajectoryWindowLSHRepository,
    make_window_lsh_table,
)

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class TuneSnapshot:
    """One ``(trajectory, step)`` point used as a sweep query.

    ``path_id`` is the underlying ``TrajectoryPath.id`` and serves as
    the cache key when ``RetrievalCache`` is shared across trials.
    """
    trajectory_id: UUID
    step: int
    tokens: list[int]
    path_id: UUID


@dataclass(frozen=True)
class TrialResult:
    """One Optuna trial. ``target_metric`` is the metric Optuna
    sampled; ``target_auc`` is the weighted AUC under that metric.
    ``auc_per_metric`` keeps the weighted AUC for ALL three metrics
    so the CSV writer can compare side-by-side without re-running.
    Per-step curves live in eval-time tooling
    (``benchmarks/demo_eval/eval_cascade.py``).
    """
    window: int
    aggregation: str
    params: dict
    auc_per_metric: dict[str, float]
    target_metric: str
    target_auc: float


@dataclass
class SweepReport:
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def best(self) -> TrialResult | None:
        return max(self.trials, key=lambda t: t.target_auc) if self.trials else None

    def by_window(self, window: int) -> TrialResult | None:
        ts = [t for t in self.trials if t.window == window]
        return max(ts, key=lambda t: t.target_auc) if ts else None

    def by_window_per_metric(
        self, window: int,
    ) -> dict[str, TrialResult]:
        """Best trial PER metric in this window across all aggregations."""
        out: dict[str, TrialResult] = {}
        for t in self.trials:
            if t.window != window:
                continue
            prev = out.get(t.target_metric)
            if prev is None or t.target_auc > prev.target_auc:
                out[t.target_metric] = t
        return out

    def by_window_agg_per_metric(
        self, window: int, aggregation: str,
    ) -> dict[str, TrialResult]:
        """Best trial PER metric in one ``(W, agg)`` outer slot."""
        out: dict[str, TrialResult] = {}
        for t in self.trials:
            if t.window != window or t.aggregation != aggregation:
                continue
            prev = out.get(t.target_metric)
            if prev is None or t.target_auc > prev.target_auc:
                out[t.target_metric] = t
        return out


@dataclass(frozen=True)
class RetrievalSweepConfig:
    """Sweep knobs. Defaults live in
    ``episodiq.retrieval.tune.constants``; CLI flags reference the same
    constants so overrides stay readable.
    """
    window_grid: tuple[int, ...] = DEFAULT_WINDOW_GRID
    n_trials_per_window: int = DEFAULT_N_TRIALS
    eval_min_step: int = DEFAULT_EVAL_MIN_STEP
    # Aggregation is part of the outer grid alongside the window — LSH
    # lookup ranking + jaccard rerank both use it. ``min`` and ``mean``
    # get separate per-W cache instances; trials inside one
    # ``(W, agg)`` slot share one cache.
    aggregation_grid: tuple[str, ...] = DEFAULT_AGGREGATION_GRID
    prefetch_n_uniq_range: tuple[int, int] = DEFAULT_PREFETCH_N_UNIQ_RANGE
    jaccard_n_uniq_range: tuple[int, int] = DEFAULT_JACCARD_N_UNIQ_RANGE
    top_k_range: tuple[int, int] = DEFAULT_TOP_K_RANGE
    lam_range: tuple[float, float] = DEFAULT_LAM_RANGE
    gap_open_range: tuple[float, float] = DEFAULT_GAP_OPEN_RANGE
    gap_extend_range: tuple[float, float] = DEFAULT_GAP_EXTEND_RANGE
    sigma_range: tuple[float, float] = DEFAULT_SIGMA_RANGE
    penalty_shape_choices: tuple[str, ...] = DEFAULT_PENALTY_SHAPE_CHOICES
    # Subset of SIMILARITY_METRICS that TPE may sample as the trial's
    # objective. Defaults to all three; narrowing to a single metric
    # makes the sweep optimise exactly that AUC (no metric-by-metric
    # categorical drift).
    metric_choices: tuple[str, ...] = SIMILARITY_METRICS
    optuna_seed: int = DEFAULT_OPTUNA_SEED
    multivariate: bool = DEFAULT_MULTIVARIATE
    # Outer concurrency: concurrent trials within one study via
    # ``asyncio.Semaphore(n_jobs)`` and Optuna's ``ask/tell`` protocol.
    n_jobs: int = DEFAULT_N_JOBS
    # Inner concurrency: concurrent snapshot evaluations inside one
    # trial, bounded by an ``asyncio.Semaphore``.
    n_workers: int = DEFAULT_N_WORKERS
    # Early stop: stop one metric's study once this many consecutive
    # finished trials fail to improve best_seen. 0 disables.
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    # Alt-LSH table namespace suffix (e.g. PID-based) so concurrent
    # sweeps don't collide on the same physical table name.
    alt_table_suffix: str = ""


# ---------------------------------------------------------------------------
# Per-snapshot search runner — used by both warm-up and trial loops
# ---------------------------------------------------------------------------


async def _run_searches(
    snapshots: list[TuneSnapshot],
    status: dict[UUID, str],
    session_factory: async_sessionmaker[AsyncSession],
    table_name: str,
    mh_cfg: WindowMinHashConfig,
    cas_cfg: RetrievalConfig,
    ms_cfg: AggShiftConfig,
    n_workers: int,
    cache: RetrievalCache,
) -> dict[UUID, list[tuple[int, float]]]:
    """Run ``retrieval.search()`` for every snapshot under the given
    cascade + scoring configs, fanning out via ``asyncio.gather`` with a
    ``Semaphore(n_workers)`` cap. Returns ``{trajectory_id: [(step,
    fail_sim), ...]}`` — empty when no candidates survive.
    """
    alt_table = make_window_lsh_table(table_name)
    sem = asyncio.Semaphore(n_workers)

    async def _one(snap: TuneSnapshot):
        async with sem:
            async with session_factory() as session:
                path_repo = TrajectoryPathRepository(session)
                lsh_repo = TrajectoryWindowLSHRepository(
                    session, table=alt_table,
                )
                retrieval = Retrieval(
                    path_repo, lsh_repo,
                    minhash_config=mh_cfg,
                    retrieval_config=cas_cfg,
                    scoring_config=ms_cfg,
                    cache=cache,
                )
                q_tokens = np.asarray(snap.tokens, dtype=np.int64)
                query = RetrievalQuery(
                    tokens=q_tokens,
                    trajectory_id=snap.trajectory_id,
                    path_id=snap.path_id,
                )
                candidates = await retrieval.search(query)
                if not candidates:
                    return None
                n_fail = sum(
                    1 for c in candidates
                    if c.trajectory_status == "failure"
                )
                return (
                    snap.trajectory_id, snap.step,
                    n_fail / len(candidates),
                )

    results = await asyncio.gather(*[_one(s) for s in snapshots])
    fail_sims: dict[UUID, list[tuple[int, float]]] = defaultdict(list)
    for r in results:
        if r is None:
            continue
        tid, step, ff = r
        fail_sims[tid].append((step, ff))
    return fail_sims


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _resolve_shuffle_key(obj, path: str) -> str:
    """Resolve a dotted attribute / dict path against ``obj`` and
    return the value as a string (so heterogeneous types still sort).
    ``"meta.instance_id"`` reads ``obj.meta["instance_id"]``;
    ``"id"`` reads ``obj.id``. Missing values fall back to ``""``.
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return ""
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return "" if cur is None else str(cur)


def _stratified_interleave(items, stratify_field: str):
    """Interleave items proportionally to group size so any prefix
    preserves the population's class distribution.

    Every item gets a "ticket" equal to its position normalized to
    ``(0, 1]`` within its group. Sorting all tickets globally yields a
    list where, at prefix length L, ``count_g(L) ≈ L · size(g) / total``.
    """
    from collections import defaultdict as _dd
    groups: dict = _dd(list)
    for item in items:
        groups[_resolve_shuffle_key(item, stratify_field)].append(item)
    tickets = []
    for cls, group in groups.items():
        n = len(group)
        for i, item in enumerate(group):
            tickets.append(((i + 1) / n, cls, item))
    tickets.sort(key=lambda x: (x[0], x[1]))
    return [it for _, _, it in tickets]


class RetrievalSweep:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        config: RetrievalSweepConfig | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        shuffle_seed: int | None = None,
        shuffle_field: str | None = None,
        stratify_field: str | None = None,
    ) -> None:
        self._sf = session_factory
        self._cfg = config or RetrievalSweepConfig()
        self._limit = limit
        self._offset = offset
        # Seed that deterministically shuffles trajectories before
        # offset/limit slicing. ``None`` preserves DB order.
        # ``shuffle_field`` (dotted path) selects the pre-shuffle sort
        # key; same field+seed across pipelines (cascade + basic.py)
        # yields matching slices.
        # ``stratify_field`` groups the shuffled list by that key and
        # proportionally interleaves so any tune/eval prefix preserves
        # the population's class distribution (e.g. fail/success).
        self._shuffle_seed = shuffle_seed
        self._shuffle_field = shuffle_field
        self._stratify_field = stratify_field

    async def run(self) -> SweepReport:
        for w in self._cfg.window_grid:
            validate_retrieval_window(w)

        snapshots, status = await self._load_corpus()
        if not snapshots:
            logger.warning("no tune snapshots")
            return SweepReport()
        logger.info(
            "sweep tune slice: %d snapshots from %d trajs; status known for %d",
            len(snapshots), len({s.trajectory_id for s in snapshots}), len(status),
        )

        report = SweepReport()
        suffix = self._cfg.alt_table_suffix
        for W in self._cfg.window_grid:
            table_name = f"trajectory_window_lsh_sweep_w{W}{suffix}"
            mh_cfg = WindowMinHashConfig(window=W)
            async with self._sf() as session:
                n_trajs, n_rows = await rebuild_lsh_into(
                    session, mh_cfg, table_name,
                )
            logger.info(
                "W=%d alt LSH built: %d trajs %d rows -> %s",
                W, n_trajs, n_rows, table_name,
            )

            try:
                for ai, agg in enumerate(self._cfg.aggregation_grid):
                    cache = RetrievalCache()
                    logger.info(
                        "W=%d agg=%s cache warm-up start", W, agg,
                    )
                    # ``prefetch_n_uniq = MAX`` fills LSHCache + Jaccard
                    # (the latter loops over every LSH candidate).
                    # ``jaccard_n_uniq = top_k = 1`` keeps the un-cached
                    # min-shift kernel at one candidate per snapshot.
                    await _run_searches(
                        snapshots, status, self._sf, table_name,
                        mh_cfg=mh_cfg,
                        cas_cfg=RetrievalConfig(
                            aggregation=agg,
                            prefetch_n_uniq=self._cfg.prefetch_n_uniq_range[1],
                            jaccard_n_uniq=1,
                            top_k=1,
                        ),
                        ms_cfg=AggShiftConfig(window=W),
                        n_workers=self._cfg.n_workers, cache=cache,
                    )
                    logger.info(
                        "W=%d agg=%s cache warm-up done: "
                        "lsh=%d jaccard=%d tokens=%d",
                        W, agg, len(cache.lsh._data),
                        len(cache.jaccard._data),
                        len(cache.tokens._data),
                    )

                    # One Optuna study per (W, agg); the metric is itself
                    # a categorical knob the sampler tunes alongside the
                    # numeric params.
                    trials = await _run_study(
                        ai, W, agg, mh_cfg, self._cfg,
                        snapshots, status, self._sf, table_name, cache,
                    )
                    report.trials.extend(trials)
                    per_metric = report.by_window_agg_per_metric(W, agg)
                    for m, best_t in per_metric.items():
                        logger.info(
                            "W=%d agg=%s %s best AUC=%.4f params=%s",
                            W, agg, m, best_t.target_auc, best_t.params,
                        )
            finally:
                async with self._sf() as session:
                    await drop_alt_lsh_table(session, table_name)

        return report

    async def _load_corpus(
        self,
    ) -> tuple[list[TuneSnapshot], dict[UUID, str]]:
        async with self._sf() as session:
            traj_repo = TrajectoryRepository(session)
            trajs = await traj_repo.get_with_completed_paths(
                ["success", "failure"],
            )
        if not trajs:
            return [], {}
        status = {t.id: t.status for t in trajs}
        if self._shuffle_seed is not None:
            import random as _random
            trajs = list(trajs)
            if self._shuffle_field:
                trajs.sort(
                    key=lambda t: _resolve_shuffle_key(t, self._shuffle_field),
                )
            rng = _random.Random(self._shuffle_seed)
            rng.shuffle(trajs)
            if self._stratify_field:
                trajs = _stratified_interleave(trajs, self._stratify_field)
        end = (
            self._offset + self._limit if self._limit is not None else len(trajs)
        )
        tune_trajs = trajs[self._offset:end]
        snapshots: list[TuneSnapshot] = []
        for t in tune_trajs:
            for p in sorted(t.paths, key=lambda x: x.index or 0):
                if p.trace_tokens and p.index is not None:
                    snapshots.append(
                        TuneSnapshot(
                            trajectory_id=t.id,
                            step=int(p.index),
                            tokens=list(p.trace_tokens),
                            path_id=p.id,
                        ),
                    )
        return snapshots, status


# ---------------------------------------------------------------------------
# Per-metric study runner
# ---------------------------------------------------------------------------


def _sample_params(
    trial: optuna.Trial, W: int, agg: str, cfg: RetrievalSweepConfig,
) -> tuple[RetrievalConfig, AggShiftConfig, str, dict]:
    """Sample one trial's tunables. ``metric`` is itself a categorical
    knob — the sampler picks which curve to optimise alongside the
    numeric / shape params. ``agg`` is fixed by the outer ``(W, agg)``
    slot, not by the sampler.
    """
    params = {
        "prefetch_n_uniq": trial.suggest_int(
            "prefetch_n_uniq", *cfg.prefetch_n_uniq_range, step=10,
        ),
        "jaccard_n_uniq": trial.suggest_int(
            "jaccard_n_uniq", *cfg.jaccard_n_uniq_range, step=10,
        ),
        "top_k": trial.suggest_int(
            "top_k", *cfg.top_k_range,
        ),
        "penalty_shape": trial.suggest_categorical(
            "penalty_shape", list(cfg.penalty_shape_choices),
        ),
        # Discretise min-shift floats at 0.1 so trials in the same
        # neighbourhood hit the AggShiftCache instead of recomputing.
        # ``round(..., 1)`` collapses Optuna's floating-point drift
        # (``0.1 * 7 == 0.7000000000000001``) onto exact grid points
        # — without it adjacent trials would key the cache under
        # subtly different floats and re-run the Numba JIT for nothing.
        "lam": round(trial.suggest_float(
            "lam", *cfg.lam_range, step=0.1,
        ), 1),
        "gap_open": round(trial.suggest_float(
            "gap_open", *cfg.gap_open_range, step=0.1,
        ), 1),
        "gap_extend": round(trial.suggest_float(
            "gap_extend", *cfg.gap_extend_range, step=0.1,
        ), 1),
        # ``sigma`` only matters when ``penalty_shape == "gauss"``;
        # other shapes ignore it. Sampling it anyway keeps the search
        # space well-defined for the categorical.
        "sigma": round(trial.suggest_float(
            "sigma", *cfg.sigma_range, step=0.1,
        ), 1),
        "metric": trial.suggest_categorical("metric", list(cfg.metric_choices)),
    }
    cas_cfg = RetrievalConfig(
        aggregation=agg,
        prefetch_n_uniq=params["prefetch_n_uniq"],
        jaccard_n_uniq=params["jaccard_n_uniq"],
        top_k=params["top_k"],
    )
    ms_cfg = AggShiftConfig(
        window=W,
        lam=params["lam"],
        penalty_shape=params["penalty_shape"],
        gap_open=params["gap_open"],
        gap_extend=params["gap_extend"],
        sigma=params["sigma"],
    )
    return cas_cfg, ms_cfg, params["metric"], params


async def _run_study(
    ai: int,
    W: int,
    agg: str,
    mh_cfg: WindowMinHashConfig,
    cfg: RetrievalSweepConfig,
    snapshots: list[TuneSnapshot],
    status: dict[UUID, str],
    session_factory: async_sessionmaker[AsyncSession],
    table_name: str,
    cache: RetrievalCache,
) -> list[TrialResult]:
    """Run one TPE study for a fixed ``(W, agg)`` slot via Optuna's
    ask/tell. ``metric`` is sampled per trial as a categorical knob —
    the sampler is free to mix and match metrics with numeric params.
    Seed is offset per aggregation so the studies don't replay each
    other's TPE-startup random points.

    Each trial fans out its own ``retrieval.search()`` calls bounded by
    ``cfg.n_workers``; ``cfg.n_jobs`` caps in-flight trials via
    ``asyncio.Semaphore``.

    Early stop: once ``cfg.early_stop_patience`` consecutive completed
    trials fail to improve the running best, no further trials are
    launched. In-flight trials still finish.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg.optuna_seed + ai,
            multivariate=cfg.multivariate,
        ),
    )
    results: list[TrialResult] = []
    sem = asyncio.Semaphore(cfg.n_jobs)
    best_seen = 0.0
    no_improve = 0
    stop_event = asyncio.Event()

    async def _one_trial(ti: int) -> None:
        if stop_event.is_set():
            return
        async with sem:
            if stop_event.is_set():
                return
            trial = study.ask()
            cas_cfg, ms_cfg, metric, params = _sample_params(
                trial, W, agg, cfg,
            )
            fail_sims = await _run_searches(
                snapshots, status, session_factory, table_name,
                mh_cfg=mh_cfg, cas_cfg=cas_cfg, ms_cfg=ms_cfg,
                n_workers=cfg.n_workers, cache=cache,
            )
            auc_per_metric = weighted_aucs(
                fail_sims, status, cfg.eval_min_step,
            )
            auc = auc_per_metric.get(metric, 0.0)
            study.tell(trial, auc)
            nonlocal best_seen, no_improve
            if auc > best_seen:
                best_seen = auc
                no_improve = 0
            else:
                no_improve += 1
            if (
                cfg.early_stop_patience > 0
                and no_improve >= cfg.early_stop_patience
                and not stop_event.is_set()
            ):
                stop_event.set()
                logger.info(
                    "W=%d agg=%s early-stop after %d trials "
                    "(no improvement for %d) best=%.4f",
                    W, agg, len(study.trials),
                    cfg.early_stop_patience, best_seen,
                )
            if metric not in auc_per_metric:
                return
            results.append(TrialResult(
                window=W,
                aggregation=agg,
                params=params,
                auc_per_metric=auc_per_metric,
                target_metric=metric,
                target_auc=auc,
            ))
            logger.info(
                "W=%d agg=%s metric=%s trial %d/%d AUC=%.4f params=%s",
                W, agg, metric, ti + 1, cfg.n_trials_per_window,
                auc, params,
            )

    await asyncio.gather(*[
        _one_trial(i) for i in range(cfg.n_trials_per_window)
    ])
    return results


