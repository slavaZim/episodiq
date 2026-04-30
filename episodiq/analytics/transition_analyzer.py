"""Transition profile analytics: HNSW prefetch → sparse cosine rerank → lev voting."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median

import numpy as np
import structlog

from episodiq.analytics.transition_types import (
    ActionSignal,
    PathFrequencySignal,
    LoopSignal,
    TrajectoryAnalytics,
    tail_streak,
)
from episodiq.config.config import AnalyticsConfig, get_config
from episodiq.storage.postgres.models import TrajectoryPath
from episodiq.storage.postgres.repository import TrajectoryPathRepository
from episodiq.utils import categorical_entropy, levenshtein, sparse_cosine, trunc_suffix

logger = structlog.stdlib.get_logger(__name__)

VOTE_SUFFIX = 5


class TransitionAnalyzer:
    """Compute trajectory analytics via single-level Levenshtein voting.

    Pipeline:
        1. HNSW cosine prefetch (prefetch_n)
        2. Sparse cosine rerank on transition_profile → top_k
        3. Single lev(suffix=5) vote, drop zero-similarity candidates
        4. Signals: transition, action, path_frequency, loop,

    Requires min_voters nonzero lev matches for valid analytics.
    """

    def __init__(
        self,
        *,
        path_repo: TrajectoryPathRepository,
        config: AnalyticsConfig | None = None,
    ):
        cfg = config or get_config().analytics
        self.path_repo = path_repo
        self.prefetch_n = cfg.prefetch_n
        self.top_k = cfg.top_k
        self.min_voters = cfg.min_voters
        self.fail_risk_action_threshold = cfg.fail_risk_action_threshold
        self.success_signal_action_threshold = cfg.success_signal_action_threshold
        self.loop_threshold = cfg.loop_threshold

    async def analyze(
        self,
        current_path: TrajectoryPath,
        *,
        prefetch: list[TrajectoryPath] | None = None,
    ) -> TrajectoryAnalytics:
        if current_path.profile_embed is None:
            return TrajectoryAnalytics()

        # 1. HNSW prefetch (skip if caller provides pre-fetched pool)
        if prefetch is None:
            prefetch = await self.path_repo.prefetch_similar(
                profile_embed=current_path.profile_embed,
                exclude_trajectory_id=current_path.trajectory_id,
                limit=self.prefetch_n,
            )

        # 2. Sparse cosine rerank → top_k
        query_profile = current_path.transition_profile or {}
        prefetched_reranked = [
            (sparse_cosine(query_profile, c.transition_profile or {}), c)
            for c in prefetch
        ]
        prefetched_reranked.sort(key=lambda x: -x[0])
        candidates = [c for _, c in prefetched_reranked[:self.top_k]]

        # Per-status top_k pools from full prefetch (not from mixed candidates)
        success_candidates = [
            c for _, c in prefetched_reranked
            if c.trajectory and c.trajectory.status == "success"
        ][:self.top_k]
        failure_candidates = [
            c for _, c in prefetched_reranked
            if c.trajectory and c.trajectory.status == "failure"
        ][:self.top_k]

        # 3. Single-level lev voting (suffix=5, drop zeros)
        votes, n_voters, mean_similarity = self._lev_votes(current_path, candidates)

        # Guard: need enough voters for meaningful signal
        if n_voters < self.min_voters:
            return TrajectoryAnalytics()

        # Compute entropy once
        vote_entropy = categorical_entropy(votes) if votes else None

        # Vote distribution (normalized)
        vote_total = sum(votes.values())
        vote_distribution = (
            {a: count / vote_total for a, count in votes.items()}
            if vote_total > 0 else None
        )

        n_success = sum(
            1 for c in candidates
            if c.trajectory and c.trajectory.status == "success"
        )

        # fail_similarity = fail_lev_cosine - succ_lev_cosine (delta)
        fail_lev = self._status_similarity(current_path, failure_candidates)
        succ_lev = self._status_similarity(current_path, success_candidates)
        if fail_lev is not None and succ_lev is not None:
            fail_similarity = fail_lev - succ_lev
        else:
            fail_similarity = None

        loop_signal = self._loop(current_path, candidates, success_candidates)
        fail_risk_transition = self._fail_risk_transition(
            current_path, failure_candidates, success_candidates,
        )
        success_signal_transition = self._success_signal_transition(
            current_path, failure_candidates, success_candidates,
        )

        return TrajectoryAnalytics(
            vote_entropy=vote_entropy,
            mean_similarity=mean_similarity,
            n_voters=n_voters,
            n_success=n_success,
            vote_distribution=vote_distribution,
            fail_similarity=fail_similarity,
            path_frequency_signal=self._path_frequency(votes, vote_entropy),
            loop_signal=loop_signal,
            fail_risk_action=self._fail_risk_action(fail_lev, succ_lev),
            success_signal_action=self._success_signal_action(fail_lev, succ_lev),
            prefetched_reranked=prefetched_reranked,
            candidates=candidates,
            success_candidates=success_candidates,
            failure_candidates=failure_candidates,
            fail_risk_transition=fail_risk_transition,
            success_signal_transition=success_signal_transition,
            loop_streak=tail_streak(current_path.trace),
        )

    def _lev_votes(
        self,
        current_path: TrajectoryPath,
        candidates: list[TrajectoryPath],
    ) -> tuple[Counter[str], int, float]:
        """Single-level lev voting on suffix=5, drop zeros.

        Returns (votes, n_voters, mean_similarity).
        """
        query_trace = current_path.trace
        if not query_trace or not candidates:
            return Counter(), 0, 0.0

        query_suffix = trunc_suffix(query_trace, VOTE_SUFFIX)
        votes: Counter[str] = Counter()
        n_voters = 0
        total_sim = 0.0

        for c in candidates:
            sim = levenshtein(query_suffix, trunc_suffix(c.trace, VOTE_SUFFIX))
            if sim > 0:
                n_voters += 1
                total_sim += sim
                label = c.action_label
                if label:
                    votes[label] += sim

        mean_sim = total_sim / n_voters if n_voters > 0 else 0.0
        return votes, n_voters, mean_sim

    @staticmethod
    def _path_frequency(
        votes: Counter[str],
        vote_entropy: float | None,
    ) -> PathFrequencySignal | None:
        if not votes or vote_entropy is None:
            return None
        return PathFrequencySignal(
            entropy=vote_entropy,
            n_matches=sum(votes.values()),
        )

    @staticmethod
    def _count_triplet(trace: list[str], triplet: str) -> int:
        """Count raw occurrences of a transition triplet in the trace."""
        count = 0
        for i in range(0, len(trace) - 2, 2):
            if f"{trace[i]}.{trace[i+1]}.{trace[i+2]}" == triplet:
                count += 1
        return count

    def _loop(
        self,
        current_path: TrajectoryPath,
        candidates: list[TrajectoryPath],
        success_candidates: list[TrajectoryPath],
    ) -> LoopSignal | None:
        trace = current_path.trace
        if len(trace) < 3:
            return None

        transition = f"{trace[-3]}.{trace[-2]}.{trace[-1]}"
        repeat_count = self._count_triplet(trace, transition)

        candidate_counts = [
            self._count_triplet(c.trace, transition) for c in candidates
        ]
        success_counts = [
            self._count_triplet(c.trace, transition) for c in success_candidates
        ]

        median_count = median(candidate_counts) if candidate_counts else 0
        mean_success = mean(success_counts) if success_counts else None

        signal = LoopSignal(
            is_detected=False,
            transition=transition,
            repeat_count=repeat_count,
            median_neighbor_repeat=median_count,
            mean_success_repeat=mean_success,
        )
        signal.is_detected = signal.is_loop_at(current_path, self.loop_threshold)
        return signal

    def _fail_risk_transition(
        self,
        current_path: TrajectoryPath,
        failure_candidates: list[TrajectoryPath],
        success_candidates: list[TrajectoryPath],
    ) -> bool:
        """Last triplet present in failure candidates but absent in success."""
        trace = current_path.trace
        if len(trace) < 3:
            return False
        transition = f"{trace[-3]}.{trace[-2]}.{trace[-1]}"
        in_success = any(self._count_triplet(c.trace, transition) > 0 for c in success_candidates)
        if in_success:
            return False
        in_failure = any(self._count_triplet(c.trace, transition) > 0 for c in failure_candidates)
        return in_failure

    def _success_signal_transition(
        self,
        current_path: TrajectoryPath,
        failure_candidates: list[TrajectoryPath],
        success_candidates: list[TrajectoryPath],
    ) -> bool:
        """Last triplet present in success candidates but absent in failure."""
        trace = current_path.trace
        if len(trace) < 3:
            return False
        transition = f"{trace[-3]}.{trace[-2]}.{trace[-1]}"
        in_failure = any(self._count_triplet(c.trace, transition) > 0 for c in failure_candidates)
        if in_failure:
            return False
        in_success = any(self._count_triplet(c.trace, transition) > 0 for c in success_candidates)
        return in_success

    def _status_similarity(
        self,
        current_path: TrajectoryPath,
        status_candidates: list[TrajectoryPath],
    ) -> float | None:
        """Lev-weighted cosine between actual action and pre-filtered status candidates."""
        actual_embed = (
            current_path.action_message.embedding
            if current_path.action_message else None
        )
        if actual_embed is None:
            return None

        actual_arr = np.array(actual_embed)

        if not status_candidates:
            return None

        query_suffix = trunc_suffix(current_path.trace, VOTE_SUFFIX)
        weighted_sum = 0.0
        weight_sum = 0.0
        for c in status_candidates:
            lev_sim = levenshtein(query_suffix, trunc_suffix(c.trace, VOTE_SUFFIX))
            if lev_sim > 0 and c.action_message and c.action_message.embedding is not None:
                c_arr = np.array(c.action_message.embedding)
                cos_sim = float(np.dot(actual_arr, c_arr))
                weighted_sum += lev_sim * cos_sim
                weight_sum += lev_sim

        if weight_sum == 0:
            return None

        return weighted_sum / weight_sum

    # TODO: test mutual exclusivity — with positive thresholds, fail_risk_action
    # and success_signal_action cannot both be is_detected=True for the same path.
    def _fail_risk_action(
        self,
        fail_lev: float | None,
        succ_lev: float | None,
    ) -> ActionSignal | None:
        if fail_lev is None or succ_lev is None:
            return None

        delta = fail_lev - succ_lev
        return ActionSignal(
            is_detected=delta >= self.fail_risk_action_threshold,
            similarity=delta,
        )

    def _success_signal_action(
        self,
        fail_lev: float | None,
        succ_lev: float | None,
    ) -> ActionSignal | None:
        if fail_lev is None or succ_lev is None:
            return None

        delta = succ_lev - fail_lev
        return ActionSignal(
            is_detected=delta >= self.success_signal_action_threshold,
            similarity=delta,
        )
