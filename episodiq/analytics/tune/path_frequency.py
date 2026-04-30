"""Sweep entropy percentiles, suggest action-variance thresholds."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.analytics.path_frequency import (
    PathFrequencyTagger,
    PathFrequencyThresholds,
)

if TYPE_CHECKING:
    from episodiq.storage.postgres.repository import TrajectoryPathRepository

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_LOW_PERCENTILE = 10.0
DEFAULT_HIGH_PERCENTILE = 90.0
MIN_VALID = 10


@dataclass(frozen=True)
class PercentileStats:
    """Percentile distribution for a single metric."""

    min: float
    p25: float
    p50: float
    p75: float
    max: float


@dataclass(frozen=True)
class PathFrequencyResult:
    """Result of action-variance threshold analysis."""

    n_sampled: int
    n_valid: int
    entropy_stats: PercentileStats | None = None
    thresholds: PathFrequencyThresholds | None = None
    variance_counts: dict[str, int] = field(default_factory=dict)


def _pstats(arr: np.ndarray) -> PercentileStats:
    """Compute min/p25/p50/p75/max from array."""
    pcts = np.percentile(arr, [0, 25, 50, 75, 100])
    return PercentileStats(*pcts.tolist())


class PathFrequencyTuner:
    """Collect entropy signals and suggest action-variance thresholds."""

    def __init__(
        self,
        path_repo: TrajectoryPathRepository,
        *,
        low_percentile: float = DEFAULT_LOW_PERCENTILE,
        high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    ) -> None:
        if high_percentile <= low_percentile:
            msg = f"high_percentile ({high_percentile}) must be > low_percentile ({low_percentile})"
            raise ValueError(msg)
        self._path_repo = path_repo
        self._low_percentile = low_percentile
        self._high_percentile = high_percentile

    async def run(
        self,
        sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    ) -> PathFrequencyResult:
        """Analyze completed paths and suggest entropy thresholds.

        Args:
            sample_size: Random sample size. None = all paths.
        """
        analyzer = TransitionAnalyzer(path_repo=self._path_repo)
        sampled = await self._path_repo.get_completed(limit=sample_size, require_embed=True)

        logger.info("path_freq_start n_sampled=%d", len(sampled))

        entropies: list[float] = []
        for i, path in enumerate(sampled):
            if i % 500 == 0 and i > 0:
                logger.info("progress %d/%d valid=%d", i, len(sampled), len(entropies))

            analytics = await analyzer.analyze(path)
            if analytics.vote_entropy is not None:
                entropies.append(analytics.vote_entropy)

        if len(entropies) < MIN_VALID:
            logger.warning("too few valid signals (%d < %d)", len(entropies), MIN_VALID)
            return PathFrequencyResult(n_sampled=len(sampled), n_valid=len(entropies))

        arr = np.array(entropies)
        low_e = float(np.percentile(arr, self._low_percentile))
        high_e = float(np.percentile(arr, self._high_percentile))

        if high_e <= low_e:
            logger.warning("degenerate distribution: p%s == p%s == %.4f",
                           self._low_percentile, self._high_percentile, low_e)
            return PathFrequencyResult(
                n_sampled=len(sampled),
                n_valid=len(entropies),
                entropy_stats=_pstats(arr),
            )

        thresholds = PathFrequencyThresholds(low_entropy=low_e, high_entropy=high_e)

        tagger = PathFrequencyTagger(thresholds)
        counts: Counter[str] = Counter()
        for e in entropies:
            v = tagger.tag(e)
            key = v.value if v else "normal"
            counts[key] += 1

        return PathFrequencyResult(
            n_sampled=len(sampled),
            n_valid=len(entropies),
            entropy_stats=_pstats(arr),
            thresholds=thresholds,
            variance_counts=dict(counts),
        )
