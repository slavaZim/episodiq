"""Trajectory analytics: deterministic loop detection + retrieval path frequency."""

from __future__ import annotations

from collections import Counter

from episodiq.analytics.transition_types import (
    LoopSignal,
    PathFrequencySignal,
    TrajectoryAnalytics,
    tail_streak,
)
from episodiq.config.config import AnalyticsConfig, get_config
from episodiq.retrieval.candidate import RetrievalCandidate
from episodiq.storage.postgres.models import TrajectoryPath
from episodiq.utils import categorical_entropy


class TransitionAnalyzer:
    """Compute trajectory analytics from a pre-retrieved candidate list.

    loop           -- deterministic; consecutive repetition of the trailing
                      action.observation duplet (tail_streak). No retrieval.
    path_frequency -- entropy over the distinct action cluster_ids of the
                      candidates' best-matching paths. Plain Counter (no Lev
                      voting): top_k retrieval already filtered by Lev sim.
    fail_frac      -- share of candidates whose trajectory ended in failure.
    """

    def __init__(self, *, config: AnalyticsConfig | None = None):
        cfg = config or get_config().analytics
        self.loop_threshold = cfg.loop_threshold

    def analyze(
        self,
        current_path: TrajectoryPath,
        candidates: list[RetrievalCandidate],
    ) -> TrajectoryAnalytics:
        loop_streak = tail_streak(current_path.trace)
        loop_signal = self._loop(current_path.trace, loop_streak)

        votes: Counter = Counter()
        for c in candidates:
            if c.best_path_action_cluster_id is not None:
                votes[c.best_path_action_cluster_id] += 1
        path_frequency_signal = self._path_frequency(votes)

        fail_frac = None
        if candidates:
            fails = sum(
                1 for c in candidates if c.trajectory_status == "failure"
            )
            fail_frac = fails / len(candidates)

        return TrajectoryAnalytics(
            path_frequency_signal=path_frequency_signal,
            loop_signal=loop_signal,
            fail_frac=fail_frac,
        )

    def _loop(self, trace: list[str], streak: int) -> LoopSignal | None:
        """Loop signal from the trailing duplet's consecutive-repetition streak."""
        if len(trace) < 2:
            return None
        return LoopSignal(
            is_detected=streak >= self.loop_threshold,
            duplet=f"{trace[-2]}.{trace[-1]}",
            streak=streak,
        )

    @staticmethod
    def _path_frequency(votes: Counter) -> PathFrequencySignal | None:
        if not votes:
            return None
        return PathFrequencySignal(
            entropy=categorical_entropy(votes),
            n_matches=sum(votes.values()),
        )
