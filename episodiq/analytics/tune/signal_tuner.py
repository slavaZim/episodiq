"""Sweep fail_similarity thresholds for both fail-risk and success-signal action signals."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.storage.postgres.repository import TrajectoryPathRepository, TrajectoryRepository
from episodiq.utils import bootstrap_auc_ci

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_TRAJECTORIES = 200
DEFAULT_MIN_RATE = 0.02
DEFAULT_MAX_RATE = 0.15
CONCURRENCY = 10
DEFAULT_PERCENTILES = [75, 80, 85, 90, 95]


@dataclass(frozen=True)
class ThresholdResult:
    """AUC result for one cosine threshold."""

    threshold: float
    signal_rate: float
    auc: float
    auc_ci_lower: float
    auc_ci_upper: float


@dataclass(frozen=True)
class SignalTunerResult:
    """Full result of signal threshold sweep (both directions)."""

    n_trajectories: int
    n_paths: int
    n_success: int
    n_failure: int
    fail_risk_thresholds: list[ThresholdResult]
    fail_risk_suggested: ThresholdResult | None
    success_signal_thresholds: list[ThresholdResult]
    success_signal_suggested: ThresholdResult | None


class SignalTuner:
    """Sweep fail_similarity thresholds for both fail-risk and success-signal directions.

    fail_risk_action: delta = fail_lev - succ_lev >= threshold → max AUC predictor of failure.
    success_signal_action: delta = succ_lev - fail_lev >= threshold → min AUC (inverted).
    """

    def __init__(
        self,
        traj_repo: TrajectoryRepository,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._traj_repo = traj_repo
        self._session_factory = session_factory

    async def run(
        self,
        sample_size: int = DEFAULT_SAMPLE_TRAJECTORIES,
        percentiles: list[int] = DEFAULT_PERCENTILES,
        min_rate: float = DEFAULT_MIN_RATE,
        max_rate: float = DEFAULT_MAX_RATE,
        concurrency: int = CONCURRENCY,
    ) -> SignalTunerResult:

        n_per_class = sample_size // 2

        success_trajs = await self._traj_repo.get_with_completed_paths(
            status="success", limit=n_per_class, require_embed=True,
        )
        failure_trajs = await self._traj_repo.get_with_completed_paths(
            status="failure", limit=n_per_class, require_embed=True,
        )
        all_trajs = success_trajs + failure_trajs

        traj_statuses: dict[UUID, str] = {t.id: t.status for t in all_trajs}
        traj_n_paths: dict[UUID, int] = {t.id: len(t.paths) for t in all_trajs}
        n_paths = sum(traj_n_paths.values())

        logger.info(
            "signal_tuner_start n_traj=%d n_paths=%d success=%d failure=%d",
            len(all_trajs), n_paths, len(success_trajs), len(failure_trajs),
        )

        # Analyze all paths → collect fail_similarity
        all_path_items: list[tuple[UUID, object]] = []
        for traj in all_trajs:
            for path in traj.paths:
                all_path_items.append((traj.id, path))

        sem = asyncio.Semaphore(concurrency)
        n_analyzed = 0

        async def analyze_async(path: object) -> float | None:
            nonlocal n_analyzed
            async with sem, self._session_factory() as session:
                analyzer = TransitionAnalyzer(path_repo=TrajectoryPathRepository(session))
                analytics = await analyzer.analyze(path)
            n_analyzed += 1
            if n_analyzed % 500 == 0:
                logger.info("progress %d/%d", n_analyzed, n_paths)
            return analytics.fail_similarity

        results = await asyncio.gather(*[analyze_async(p) for _, p in all_path_items])

        # Collect fail_similarity per trajectory
        traj_fail_signals: dict[UUID, list[float]] = {}
        traj_success_signals: dict[UUID, list[float]] = {}
        all_fail_sims: list[float] = []
        all_success_sims: list[float] = []

        for (tid, _), sim in zip(all_path_items, results):
            if sim is not None:
                traj_fail_signals.setdefault(tid, []).append(sim)
                all_fail_sims.append(sim)
                # success direction = -sim
                traj_success_signals.setdefault(tid, []).append(-sim)
                all_success_sims.append(-sim)

        # Auto-compute thresholds from percentiles
        fail_thresholds: list[float] = []
        success_thresholds: list[float] = []
        if all_fail_sims:
            arr = np.array(all_fail_sims)
            fail_thresholds = sorted({
                round(float(np.percentile(arr, p)), 2) for p in percentiles
            })
        if all_success_sims:
            arr = np.array(all_success_sims)
            success_thresholds = sorted({
                round(float(np.percentile(arr, p)), 2) for p in percentiles
            })

        # Sweep fail-risk thresholds (max AUC = predicts failure)
        fail_results = self._sweep(
            fail_thresholds, traj_statuses, traj_n_paths, traj_fail_signals,
            min_rate, max_rate,
        )
        fail_suggested = max(fail_results, key=lambda r: r.auc) if fail_results else None

        # Sweep success-signal thresholds (min AUC = predicts success)
        success_results = self._sweep(
            success_thresholds, traj_statuses, traj_n_paths, traj_success_signals,
            min_rate, max_rate,
        )
        success_suggested = min(success_results, key=lambda r: r.auc) if success_results else None

        return SignalTunerResult(
            n_trajectories=len(all_trajs),
            n_paths=n_paths,
            n_success=len(success_trajs),
            n_failure=len(failure_trajs),
            fail_risk_thresholds=fail_results,
            fail_risk_suggested=fail_suggested,
            success_signal_thresholds=success_results,
            success_signal_suggested=success_suggested,
        )

    @classmethod
    def _sweep(
        cls,
        thresholds: list[float],
        traj_statuses: dict[UUID, str],
        traj_n_paths: dict[UUID, int],
        traj_signals: dict[UUID, list[float]],
        min_rate: float,
        max_rate: float,
    ) -> list[ThresholdResult]:
        results: list[ThresholdResult] = []
        for t in thresholds:
            result = cls._eval_threshold(t, traj_statuses, traj_n_paths, traj_signals)
            if result is None:
                continue
            if result.signal_rate < min_rate or result.signal_rate > max_rate:
                continue
            results.append(result)
        return results

    @staticmethod
    def _eval_threshold(
        threshold: float,
        traj_statuses: dict[UUID, str],
        traj_n_paths: dict[UUID, int],
        traj_signals: dict[UUID, list[float]],
    ) -> ThresholdResult | None:
        """Compute AUC for one threshold. Returns None if AUC can't be computed."""
        y_true_list: list[int] = []
        y_score_list: list[float] = []

        for tid, status in traj_statuses.items():
            signals = traj_signals.get(tid, [])
            count = sum(1 for s in signals if s >= threshold)
            n_p = traj_n_paths[tid]
            rate = count / n_p if n_p > 0 else 0.0
            y_true_list.append(1 if status == "failure" else 0)
            y_score_list.append(rate)

        y_true = np.array(y_true_list)
        y_score = np.array(y_score_list)

        if len(set(y_score_list)) <= 1 or len(set(y_true_list)) < 2:
            return None

        auc = float(roc_auc_score(y_true, y_score))
        ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_score)
        mean_rate = float(np.mean(y_score))

        return ThresholdResult(
            threshold=threshold,
            signal_rate=mean_rate,
            auc=auc,
            auc_ci_lower=ci_lower,
            auc_ci_upper=ci_upper,
        )
