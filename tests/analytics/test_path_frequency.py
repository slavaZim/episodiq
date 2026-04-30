"""Tests for action variance tagger."""

from __future__ import annotations

import pytest

from episodiq.analytics.path_frequency import (
    ActionVariance,
    PathFrequencyTagger,
    PathFrequencyThresholds,
)
from episodiq.analytics.transition_types import TrajectoryAnalytics


# --- PathFrequencyThresholds ---

class TestPathFrequencyThresholds:

    def test_rejects_high_lte_low(self):
        with pytest.raises(ValueError, match="must be >"):
            PathFrequencyThresholds(low_entropy=1.0, high_entropy=1.0)

    def test_rejects_high_less_than_low(self):
        with pytest.raises(ValueError, match="must be >"):
            PathFrequencyThresholds(low_entropy=2.0, high_entropy=1.0)


# --- PathFrequencyTagger ---

class TestPathFrequencyTagger:
    """Test quantile-based action variance classification."""

    def setup_method(self):
        self.tagger = PathFrequencyTagger(
            PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0)
        )

    def test_low_variance(self):
        """Entropy below low threshold → LOW."""
        assert self.tagger.tag(0.3) == ActionVariance.LOW

    def test_high_variance(self):
        """Entropy above high threshold → HIGH."""
        assert self.tagger.tag(2.5) == ActionVariance.HIGH

    def test_normal_returns_none(self):
        """Entropy between thresholds → None."""
        assert self.tagger.tag(1.0) is None

    def test_boundary_low_is_low(self):
        """Entropy exactly at low threshold counts as LOW."""
        assert self.tagger.tag(0.5) == ActionVariance.LOW

    def test_boundary_high_is_high(self):
        """Entropy exactly at high threshold counts as HIGH."""
        assert self.tagger.tag(2.0) == ActionVariance.HIGH

    def test_tag_analytics_returns_none_when_missing(self):
        """Returns None when analytics lacks entropy."""
        assert self.tagger.tag_analytics(TrajectoryAnalytics()) is None

    def test_tag_analytics_low(self):
        analytics = TrajectoryAnalytics(vote_entropy=0.3)
        assert self.tagger.tag_analytics(analytics) == ActionVariance.LOW

    def test_tag_analytics_high(self):
        analytics = TrajectoryAnalytics(vote_entropy=2.5)
        assert self.tagger.tag_analytics(analytics) == ActionVariance.HIGH

    def test_tag_analytics_normal(self):
        analytics = TrajectoryAnalytics(vote_entropy=1.0)
        assert self.tagger.tag_analytics(analytics) is None
