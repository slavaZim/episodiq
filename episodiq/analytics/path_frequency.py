"""Action variance tagger based on vote entropy quantiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodiq.analytics.transition_types import TrajectoryAnalytics


class ActionVariance(str, Enum):
    """Action variance flag based on vote entropy quantile.

    LOW  — entropy ≤ low threshold (very predictable, few likely actions)
    HIGH — entropy ≥ high threshold (many options, unpredictable)
    """

    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class PathFrequencyThresholds:
    """Entropy quantile boundaries for variance flags."""

    low_entropy: float
    high_entropy: float

    def __post_init__(self) -> None:
        if self.high_entropy <= self.low_entropy:
            msg = f"high_entropy ({self.high_entropy}) must be > low_entropy ({self.low_entropy})"
            raise ValueError(msg)



class PathFrequencyTagger:
    """Tag trajectory points as low/high action variance by entropy quantile."""

    def __init__(self, thresholds: PathFrequencyThresholds) -> None:
        self._t = thresholds

    @property
    def thresholds(self) -> PathFrequencyThresholds:
        return self._t

    def tag(self, entropy: float) -> ActionVariance | None:
        """Return LOW / HIGH / None based on entropy quantile thresholds."""
        if entropy <= self._t.low_entropy:
            return ActionVariance.LOW
        if entropy >= self._t.high_entropy:
            return ActionVariance.HIGH
        return None

    def tag_analytics(self, analytics: TrajectoryAnalytics) -> ActionVariance | None:
        """Tag from TrajectoryAnalytics. Returns None if entropy missing."""
        if analytics.vote_entropy is None:
            return None
        return self.tag(analytics.vote_entropy)
