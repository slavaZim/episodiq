"""Sweep top_k, suggest the value that maximises fail_frac AUC."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository,
    TrajectoryRepository,
)
from episodiq.utils import bootstrap_auc_ci

logger = logging.getLogger(__name__)

DEFAULT_TOPK_GRID = [3, 5, 10, 25, 50]
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_TOLERANCE = 0.01
CONCURRENCY = 10


@dataclass(frozen=True)
class TopKPoint:
    """fail_frac AUC for one top_k value."""

    top_k: int
    auc: float
    auc_ci_lower: float
    auc_ci_upper: float
    n_trajectories: int


@dataclass(frozen=True)
class TopKResult:
    """Full result of a top_k sweep."""

    grid: list[TopKPoint]
    suggested_top_k: int
    n_trajectories: int
    n_paths: int


def _traj_cummax(path_statuses: list[list[str]], top_k: int) -> float | None:
    """Running traj_cummax of fail_frac across one trajectory's path snapshots.

    fail_frac = share of the top_k nearest distinct trajectories that failed;
    traj_cummax = max over the snapshots of the cumulative mean of fail_frac.
    Returns None if no snapshot had any neighbour.
    """
    total = 0.0
    n = 0
    cummax: float | None = None
    for statuses in path_statuses:
        candidates = statuses[:top_k]
        if not candidates:
            continue
        fail_frac = sum(s == "failure" for s in candidates) / len(candidates)
        total += fail_frac
        n += 1
        cum_mean = total / n
        cummax = cum_mean if cummax is None else max(cummax, cum_mean)
    return cummax


def _suggest(grid: list[TopKPoint], tolerance: float) -> int:
    """Smallest top_k whose AUC is within `tolerance` of the best."""
    if not grid:
        return 0
    best = max(grid, key=lambda g: g.auc)
    threshold = best.auc - tolerance
    for point in sorted(grid, key=lambda g: g.top_k):
        if point.auc >= threshold:
            return point.top_k
    return best.top_k


class TopKTuner:
    """Sweep top_k; pick the smallest value within tolerance of the best fail_frac AUC.

    One retrieval per sampled path at the largest top_k; smaller top_k values
    are evaluated by slicing that distance-ordered neighbour list. fail_frac is
    aggregated per trajectory by traj_cummax and scored (ROC AUC) against the
    trajectory's success/failure label.
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
        topk_grid: list[int] = DEFAULT_TOPK_GRID,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        concurrency: int = CONCURRENCY,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> TopKResult:
        trajectories = await self._traj_repo.get_with_completed_paths(
            status=["success", "failure"],
            limit=sample_size,
            require_embed=True,
        )

        labels = [1 if t.status == "failure" else 0 for t in trajectories]
        traj_ids = [t.id for t in trajectories]

        # Flatten paths into (traj_idx, profile), index-ordered within each trajectory.
        records: list[tuple[int, list[float]]] = []
        for ti, traj in enumerate(trajectories):
            ordered = sorted(
                traj.paths,
                key=lambda p: p.index if p.index is not None else 0,
            )
            for path in ordered:
                records.append((ti, path.profile))

        n_paths = len(records)
        logger.info(
            "top_k tuner: %d trajectories, %d paths", len(trajectories), n_paths,
        )
        if not records:
            raise RuntimeError(
                "No paths with profiles found. "
                "Run 'episodiq cluster build-paths' first."
            )

        max_top_k = max(topk_grid)
        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def prefetch(traj_idx: int, profile: list[float]) -> list[str]:
            nonlocal done
            async with sem, self._session_factory() as session:
                repo = TrajectoryPathRepository(session)
                neighbours = await repo.fetch_similar(
                    profile=profile,
                    exclude_trajectory_id=traj_ids[traj_idx],
                    limit=max_top_k,
                )
            done += 1
            if done % 200 == 0 or done == n_paths:
                logger.info("prefetch %d/%d", done, n_paths)
            return [n.trajectory_status for n in neighbours]

        statuses = await asyncio.gather(
            *[prefetch(ti, profile) for ti, profile in records]
        )

        # Regroup neighbour-status lists by trajectory, preserving path order.
        per_traj: list[list[list[str]]] = [[] for _ in trajectories]
        for (ti, _), neighbour_statuses in zip(records, statuses):
            per_traj[ti].append(neighbour_statuses)

        grid: list[TopKPoint] = []
        for top_k in sorted(topk_grid):
            y_true: list[int] = []
            y_score: list[float] = []
            for ti, path_statuses in enumerate(per_traj):
                cummax = _traj_cummax(path_statuses, top_k)
                if cummax is None:
                    continue
                y_true.append(labels[ti])
                y_score.append(cummax)

            if len(set(y_true)) < 2:
                logger.warning("top_k=%d: only one class present, skipped", top_k)
                continue

            yt = np.array(y_true)
            ys = np.array(y_score)
            auc = float(roc_auc_score(yt, ys))
            ci_lower, ci_upper = bootstrap_auc_ci(yt, ys)
            grid.append(TopKPoint(
                top_k=top_k,
                auc=auc,
                auc_ci_lower=ci_lower,
                auc_ci_upper=ci_upper,
                n_trajectories=len(y_true),
            ))
            logger.info("top_k=%d auc=%.3f n=%d", top_k, auc, len(y_true))

        return TopKResult(
            grid=grid,
            suggested_top_k=_suggest(grid, tolerance),
            n_trajectories=len(trajectories),
            n_paths=n_paths,
        )
