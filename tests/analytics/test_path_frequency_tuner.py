"""Unit tests for ``PathFrequencyTuner`` — samples completed paths,
runs retrieval + ``TransitionAnalyzer`` per path, and folds the
collected entropies into low / high percentile thresholds.

Focus: orchestration. ``PathFrequencyTagger`` and ``TransitionAnalyzer``
are tested in their own modules; here we mock both so the test pins
ONLY the tuner's branching — empty / < MIN_VALID / degenerate /
happy-path — and the call shape into the retrieval + analyzer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from episodiq.analytics.transition_types import PathFrequencySignal
from episodiq.analytics.tune.path_frequency import (
    MIN_VALID, PathFrequencyTuner,
)


def _analytics(entropy: float | None):
    """Stand-in for ``TrajectoryAnalytics`` — the tuner only reads
    ``path_frequency_signal.entropy``."""
    a = MagicMock()
    a.path_frequency_signal = (
        PathFrequencySignal(entropy=entropy, n_matches=10)
        if entropy is not None else None
    )
    return a


def _make_tuner(*, paths, low=10.0, high=90.0):
    """Build a tuner whose ``path_repo.get_completed`` returns
    ``paths`` and whose ``retrieval.search`` is a no-op stub.
    """
    path_repo = MagicMock()
    path_repo.get_completed = AsyncMock(return_value=paths)
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[])
    tuner = PathFrequencyTuner(
        path_repo, retrieval, low_percentile=low, high_percentile=high,
    )
    return tuner, retrieval, path_repo


@pytest.fixture
def mock_analyzer():
    """Patch ``TransitionAnalyzer`` inside the tuner module so every
    ``analyze`` call returns the canned analytics from the test's
    ``side_effect`` list."""
    with patch(
        "episodiq.analytics.tune.path_frequency.TransitionAnalyzer",
    ) as analyzer_cls:
        yield analyzer_cls.return_value


class TestInit:
    def test_high_percentile_must_exceed_low(self):
        with pytest.raises(ValueError, match="must be > low_percentile"):
            PathFrequencyTuner(
                MagicMock(), MagicMock(),
                low_percentile=90.0, high_percentile=50.0,
            )

    def test_equal_percentiles_rejected(self):
        with pytest.raises(ValueError, match="must be > low_percentile"):
            PathFrequencyTuner(
                MagicMock(), MagicMock(),
                low_percentile=50.0, high_percentile=50.0,
            )


@pytest.mark.asyncio
class TestRun:
    """Branch coverage for ``run()``: empty sample, sub-MIN_VALID,
    degenerate distribution, happy path."""

    async def test_empty_corpus_returns_zero(self, mock_analyzer):
        tuner, *_ = _make_tuner(paths=[])
        result = await tuner.run(sample_size=10)
        assert result.n_sampled == 0
        assert result.n_valid == 0
        assert result.thresholds is None
        assert result.entropy_stats is None
        mock_analyzer.analyze.assert_not_called()

    async def test_below_min_valid_skips_thresholds(self, mock_analyzer):
        # MIN_VALID-1 valid signals → return without thresholds.
        n = MIN_VALID - 1
        paths = [MagicMock() for _ in range(n)]
        mock_analyzer.analyze.side_effect = [
            _analytics(0.5 + 0.01 * i) for i in range(n)
        ]
        tuner, *_ = _make_tuner(paths=paths)
        result = await tuner.run(sample_size=n)
        assert result.n_sampled == n
        assert result.n_valid == n
        assert result.thresholds is None
        assert result.entropy_stats is None

    async def test_degenerate_distribution_returns_stats_without_thresholds(
        self, mock_analyzer,
    ):
        """All entropies identical → p10 == p90 → degenerate. The
        tuner returns ``entropy_stats`` but no thresholds, leaving the
        operator to inspect the histogram before retrying."""
        n = MIN_VALID + 5
        paths = [MagicMock() for _ in range(n)]
        mock_analyzer.analyze.side_effect = [
            _analytics(0.5) for _ in range(n)
        ]
        tuner, *_ = _make_tuner(paths=paths)
        result = await tuner.run(sample_size=n)
        assert result.n_sampled == n
        assert result.n_valid == n
        assert result.thresholds is None
        assert result.entropy_stats is not None
        assert result.entropy_stats.p50 == pytest.approx(0.5)

    async def test_happy_path_produces_thresholds_and_counts(
        self, mock_analyzer,
    ):
        # Linear sweep so percentiles fall on known values.
        n = 100
        paths = [MagicMock() for _ in range(n)]
        entropies = [i / (n - 1) for i in range(n)]
        mock_analyzer.analyze.side_effect = [
            _analytics(e) for e in entropies
        ]
        tuner, *_ = _make_tuner(paths=paths, low=10.0, high=90.0)
        result = await tuner.run(sample_size=n)

        assert result.n_sampled == n
        assert result.n_valid == n
        assert result.thresholds is not None
        # p10 ≈ 0.099, p90 ≈ 0.891 from a 0..1 linear sweep.
        assert result.thresholds.low_entropy == pytest.approx(0.099, abs=0.01)
        assert result.thresholds.high_entropy == pytest.approx(0.891, abs=0.01)
        # tagger sorts every entropy into one of {low, normal, high};
        # counts must sum to n_valid.
        assert sum(result.variance_counts.values()) == n
        assert result.entropy_stats.min == pytest.approx(0.0)
        assert result.entropy_stats.max == pytest.approx(1.0)
        assert result.entropy_stats.p50 == pytest.approx(0.5, abs=0.01)

    async def test_signals_with_none_entropy_are_dropped(self, mock_analyzer):
        """``path_frequency_signal is None`` (no candidates returned by
        retrieval) → silently excluded; ``n_valid`` only counts those
        that did produce a signal."""
        paths = [MagicMock() for _ in range(MIN_VALID + 2)]
        mock_analyzer.analyze.side_effect = (
            [_analytics(None)] * MIN_VALID
            + [_analytics(0.5), _analytics(0.6)]
        )
        tuner, *_ = _make_tuner(paths=paths)
        result = await tuner.run(sample_size=len(paths))
        assert result.n_sampled == len(paths)
        assert result.n_valid == 2
        # 2 < MIN_VALID → no thresholds.
        assert result.thresholds is None

    async def test_calls_retrieval_search_once_per_path(self, mock_analyzer):
        paths = [MagicMock() for _ in range(5)]
        mock_analyzer.analyze.side_effect = [_analytics(0.5)] * 5
        tuner, retrieval, _ = _make_tuner(paths=paths)
        await tuner.run(sample_size=5)
        assert retrieval.search.await_count == 5
        # Each path is forwarded as the sole arg to ``search``.
        for call, p in zip(retrieval.search.call_args_list, paths):
            assert call.args[0] is p
