"""Grid search over prefetch_n and top_k to find minimal values for stable hit@5."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.config.config import AnalyticsConfig
from episodiq.storage.postgres.repository import TrajectoryPathRepository
from episodiq.utils import binomial_margin

logger = logging.getLogger(__name__)

DEFAULT_PREFETCH_GRID = [250, 500, 750, 1000]
DEFAULT_TOPK_GRID = [10, 25, 50, 100]
DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_TOLERANCE = 0.05
CONCURRENCY = 10


@dataclass(frozen=True)
class GridPoint:
    prefetch_n: int
    top_k: int
    hit_at_5: float
    n_evaluated: int


@dataclass(frozen=True)
class PrefetchTopkResult:
    grid: list[GridPoint]
    suggested_prefetch: int
    suggested_top_k: int
    n_sampled: int
    margin: float


class PrefetchTopkTuner:
    """Grid-search prefetch_n x top_k, pick minimal values for stable hit@5.

    Efficient: one HNSW query per sample path (using max prefetch_n),
    then all grid points evaluated via TransitionAnalyzer with prefetch= param.
    """

    def __init__(
        self,
        path_repo: TrajectoryPathRepository,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._path_repo = path_repo
        self._session_factory = session_factory

    async def run(
        self,
        prefetch_grid: list[int] = DEFAULT_PREFETCH_GRID,
        topk_grid: list[int] = DEFAULT_TOPK_GRID,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        concurrency: int = CONCURRENCY,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> PrefetchTopkResult:
        # 1. Sample completed paths with embeddings
        paths = await self._path_repo.get_completed(
            limit=sample_size, require_embed=True,
        )
        logger.info("sampled %d paths", len(paths))

        if not paths:
            raise RuntimeError(
                "No paths with embeddings found. "
                "Run 'episodiq cluster build-paths' first."
            )

        # 2. Prefetch once per path with max prefetch_n
        max_prefetch = max(prefetch_grid)
        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def prefetch_async(idx: int):
            nonlocal done
            p = paths[idx]
            if not p.action_label:
                return None
            async with sem:
                if self._session_factory:
                    async with self._session_factory() as session:
                        repo = TrajectoryPathRepository(session)
                        result = await repo.prefetch_similar(
                            profile_embed=p.profile_embed,
                            exclude_trajectory_id=p.trajectory_id,
                            limit=max_prefetch,
                        )
                else:
                    result = await self._path_repo.prefetch_similar(
                        profile_embed=p.profile_embed,
                        exclude_trajectory_id=p.trajectory_id,
                        limit=max_prefetch,
                    )
                done += 1
                if done % 100 == 0 or done == len(paths):
                    logger.info("prefetch %d/%d", done, len(paths))
                return result

        results = await asyncio.gather(
            *[prefetch_async(i) for i in range(len(paths))],
            return_exceptions=True,
        )

        # Pair paths with prefetch results, skip failures
        valid: list[tuple] = []  # (path, prefetched)
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("prefetch failed idx=%d: %s", idx, res)
                continue
            if res is None:
                continue
            valid.append((paths[idx], res))

        logger.info("valid samples: %d / %d", len(valid), len(paths))

        # 3. Evaluate grid — use TransitionAnalyzer for each (prefetch_n, top_k)
        grid: list[GridPoint] = []
        for pn in sorted(prefetch_grid):
            for tk in sorted(topk_grid):
                hit5, n_eval = await self._eval_grid_point(valid, pn, tk)
                grid.append(GridPoint(
                    prefetch_n=pn, top_k=tk,
                    hit_at_5=hit5, n_evaluated=n_eval,
                ))
                logger.info(
                    "prefetch=%d top_k=%d hit@5=%.3f n=%d",
                    pn, tk, hit5, n_eval,
                )

        # 4. Suggest minimal values
        suggested_prefetch, suggested_top_k, margin = self._suggest(
            grid, len(valid), tolerance=tolerance,
        )

        return PrefetchTopkResult(
            grid=grid,
            suggested_prefetch=suggested_prefetch,
            suggested_top_k=suggested_top_k,
            n_sampled=len(paths),
            margin=margin,
        )

    async def _eval_grid_point(
        self,
        valid: list[tuple],
        prefetch_n: int,
        top_k: int,
    ) -> tuple[float, int]:
        """Compute hit@5 for one grid point using TransitionAnalyzer."""
        cfg = AnalyticsConfig(
            prefetch_n=prefetch_n, top_k=top_k, min_voters=1,
            fail_risk_action_threshold=0.0, success_signal_action_threshold=0.0, loop_threshold=0,
            low_entropy=0.5, high_entropy=2.5,
            dead_end_model="dead_end.joblib", dead_end_threshold=0.85,
        )
        analyzer = TransitionAnalyzer(path_repo=self._path_repo, config=cfg)

        hits = 0
        evaluated = 0
        for path, prefetched in valid:
            pool = prefetched[:prefetch_n]
            analytics = await analyzer.analyze(path, prefetch=pool)
            if analytics.vote_distribution is None:
                continue

            evaluated += 1
            top5 = set(
                sorted(analytics.vote_distribution, key=analytics.vote_distribution.get, reverse=True)[:5]
            )
            if path.action_label in top5:
                hits += 1

        return hits / evaluated if evaluated else 0.0, evaluated

    @staticmethod
    def _suggest(
        grid: list[GridPoint],
        n: int,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> tuple[int, int, float]:
        """Find minimal (prefetch_n, top_k) within tolerance of the best hit@5.

        Uses max(tolerance, binomial_margin) so the margin is never
        narrower than the statistical uncertainty.
        """
        if not grid:
            return 0, 0, 0.0

        best = max(grid, key=lambda g: g.hit_at_5)
        margin = max(tolerance, binomial_margin(best.hit_at_5, n))
        threshold = best.hit_at_5 - margin

        # Sort by prefetch_n then top_k ascending — pick first above threshold
        candidates = sorted(grid, key=lambda g: (g.prefetch_n, g.top_k))
        for g in candidates:
            if g.hit_at_5 >= threshold:
                return g.prefetch_n, g.top_k, margin

        return best.prefetch_n, best.top_k, margin
