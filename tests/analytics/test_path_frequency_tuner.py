"""Tests for PathFrequencyTuner."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.tune.path_frequency import (
    PathFrequencyTuner,
    _pstats,
)
from episodiq.analytics.transition_types import TrajectoryAnalytics
from tests.in_memory_repos import InMemoryPath, InMemoryTrajectoryPathRepository


# --- _pstats ---

class TestPstats:
    def test_basic_distribution(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _pstats(arr)
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0

    def test_single_value(self):
        arr = np.array([42.0])
        stats = _pstats(arr)
        assert stats.min == stats.p25 == stats.p50 == stats.p75 == stats.max == 42.0


# --- PathFrequencyTuner ---

def _make_path() -> InMemoryPath:
    return InMemoryPath(
        id=uuid4(),
        trajectory_id=uuid4(),
        from_observation_id=uuid4(),
        to_observation_id=uuid4(),
        profile_embed=[0.1] * 10,
        trace=["o:text:a", "a:text:b", "o:text:c"],
    )


class TestPathFrequencyTuner:
    """Test the run() pipeline."""

    def test_rejects_invalid_percentiles(self):
        """high_percentile <= low_percentile raises ValueError."""
        repo = InMemoryTrajectoryPathRepository()
        with pytest.raises(ValueError, match="must be >"):
            PathFrequencyTuner(repo, low_percentile=50.0, high_percentile=50.0)

    @pytest.mark.asyncio
    async def test_empty_repo_returns_zero_stats(self):
        """No paths → n_valid=0."""
        repo = InMemoryTrajectoryPathRepository()
        tuner = PathFrequencyTuner(repo)
        result = await tuner.run()
        assert result.n_sampled == 0
        assert result.n_valid == 0

    @pytest.mark.asyncio
    async def test_too_few_valid_returns_early(self):
        """Fewer than MIN_VALID signals → early return."""
        repo = InMemoryTrajectoryPathRepository()
        repo._paths.append(_make_path())

        mock_ta = AsyncMock()
        mock_ta.analyze.return_value = TrajectoryAnalytics(vote_entropy=1.0)

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run()

        assert result.n_sampled == 1
        assert result.n_valid == 1
        assert result.variance_counts == {}

    @pytest.mark.asyncio
    async def test_sample_size_limits_analyze_calls(self):
        """20 paths in repo, sample_size=15 → analyze called 15 times."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(20):
            repo._paths.append(_make_path())

        idx = 0

        async def mock_analyze(path):
            nonlocal idx
            idx += 1
            return TrajectoryAnalytics(vote_entropy=float(idx))

        mock_ta = AsyncMock()
        mock_ta.analyze = mock_analyze

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run(sample_size=15)

        assert idx == 15
        assert result.n_sampled == 15

    @pytest.mark.asyncio
    async def test_skips_invalid_analytics(self):
        """Paths where analyze() returns no signals are excluded from n_valid."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(15):
            repo._paths.append(_make_path())

        call_count = 0

        async def mock_analyze(path):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return TrajectoryAnalytics()  # no signals
            return TrajectoryAnalytics(vote_entropy=float(call_count))

        mock_ta = AsyncMock()
        mock_ta.analyze = mock_analyze

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run()

        assert result.n_sampled == 15
        assert result.n_valid == 14

    @pytest.mark.asyncio
    async def test_variance_counts_populated(self):
        """Variance counts reflect how entropies fall relative to thresholds."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(20):
            repo._paths.append(_make_path())

        idx = 0

        async def mock_analyze(path):
            nonlocal idx
            idx += 1
            return TrajectoryAnalytics(vote_entropy=float(idx))

        mock_ta = AsyncMock()
        mock_ta.analyze = mock_analyze

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run()

        assert result.n_valid == 20
        assert len(result.variance_counts) > 0

    @pytest.mark.asyncio
    async def test_percentile_affects_thresholds(self):
        """Different percentile values produce different thresholds."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(20):
            repo._paths.append(_make_path())

        idx = 0

        async def mock_analyze(path):
            nonlocal idx
            idx += 1
            return TrajectoryAnalytics(vote_entropy=float(idx))

        mock_ta = AsyncMock()
        mock_ta.analyze = mock_analyze

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo, low_percentile=5.0, high_percentile=95.0)
            r_narrow = await tuner.run()

        idx = 0
        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo, low_percentile=25.0, high_percentile=75.0)
            r_wide = await tuner.run()

        assert r_narrow.thresholds.low_entropy < r_wide.thresholds.low_entropy
        assert r_narrow.thresholds.high_entropy > r_wide.thresholds.high_entropy

    @pytest.mark.asyncio
    async def test_stats_populated(self):
        """Result contains valid entropy stats."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(15):
            repo._paths.append(_make_path())

        mock_ta = AsyncMock()
        mock_ta.analyze.return_value = TrajectoryAnalytics(vote_entropy=2.0)

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run()

        assert result.entropy_stats.min == pytest.approx(2.0)
        assert result.entropy_stats.p50 == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_normal_distribution_returns_thresholds(self):
        """Varied entropies produce valid thresholds and variance counts."""
        repo = InMemoryTrajectoryPathRepository()
        for _ in range(20):
            repo._paths.append(_make_path())

        idx = 0

        async def mock_analyze(path):
            nonlocal idx
            idx += 1
            return TrajectoryAnalytics(vote_entropy=float(idx))

        mock_ta = AsyncMock()
        mock_ta.analyze = mock_analyze

        with patch(
            "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
            return_value=mock_ta,
        ):
            tuner = PathFrequencyTuner(repo)
            result = await tuner.run()

        assert result.thresholds is not None
        assert result.thresholds.low_entropy < result.thresholds.high_entropy
        assert result.thresholds.low_entropy == pytest.approx(np.percentile(np.arange(1, 21, dtype=float), 10))
        assert result.thresholds.high_entropy == pytest.approx(np.percentile(np.arange(1, 21, dtype=float), 90))
        assert "low" in result.variance_counts
        assert "high" in result.variance_counts
