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

    loop            -- deterministic; consecutive repetition of the trailing
                       action.observation duplet (tail_streak). No retrieval.
    path_frequency  -- entropy over the distinct action cluster_ids of the
                       candidates' best-matching paths. Plain Counter (no Lev
                       voting): top_k retrieval already filtered by Lev sim.
    fail_similarity -- per-snapshot raw share of failure-ending candidates
                       plus running cummax / cummean / cummeanmax,
                       maintained incrementally from
                       ``prev_path.data["fail_similarity"]``.
    """

    def __init__(self, *, config: AnalyticsConfig | None = None):
        cfg = config or get_config().analytics
        self.loop_threshold = cfg.loop_threshold

    def analyze(
        self,
        current_path: TrajectoryPath,
        candidates: list[RetrievalCandidate],
        prev_path: TrajectoryPath | None = None,
    ) -> TrajectoryAnalytics:
        loop_streak = tail_streak(current_path.trace)
        loop_signal = self._loop(current_path.trace, loop_streak)

        votes: Counter = Counter()
        for c in candidates:
            if c.best_path_action_cluster_id is not None:
                votes[c.best_path_action_cluster_id] += 1
        path_frequency_signal = self._path_frequency(votes)

        fail_similarity = self._roll_fail_similarity(candidates, prev_path)

        return TrajectoryAnalytics(
            path_frequency_signal=path_frequency_signal,
            loop_signal=loop_signal,
            fail_similarity=fail_similarity,
        )

    @staticmethod
    def _roll_fail_similarity(
        candidates: list[RetrievalCandidate],
        prev_path: TrajectoryPath | None,
    ) -> dict | None:
        """Build the per-snapshot similarity dict from the previous
        path's stored state plus this step's raw value. When retrieval
        returns nothing the previous dict is carried forward unchanged
        (this snapshot does not contribute). ``_count`` tracks the
        number of contributing snapshots so cummean / cummeanmax can be
        rolled without re-walking the trajectory.
        """
        prev = (
            prev_path.data.get("fail_similarity")
            if prev_path is not None and prev_path.data else None
        )
        if not candidates:
            return prev
        fails = sum(
            1 for c in candidates if c.trajectory_status == "failure"
        )
        current = fails / len(candidates)
        prev_count = (prev or {}).get("_count", 0)
        if prev_count == 0:
            return {
                "current": current,
                "cummax": current,
                "cummean": current,
                "cummeanmax": current,
                "_count": 1,
            }
        prev_cummean = prev["cummean"]
        new_cummean = (
            (prev_cummean * prev_count + current) / (prev_count + 1)
        )
        return {
            "current": current,
            "cummax": max(prev["cummax"], current),
            "cummean": new_cummean,
            "cummeanmax": max(prev["cummeanmax"], new_cummean),
            "_count": prev_count + 1,
        }

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
