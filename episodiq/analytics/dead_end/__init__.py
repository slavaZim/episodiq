"""Dead-end prediction: feature extraction from TrajectoryAnalytics."""

from __future__ import annotations

from collections import Counter

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.analytics.transition_types import TrajectoryAnalytics
from episodiq.storage.postgres.models import TrajectoryPath
from episodiq.utils import sparse_cosine

NEIGHBOR_FEATURE_NAMES = [
    "vote_entropy",
    "top1_share",
    "n_vote_labels",
    "fail_risk_action_rate",
    "success_signal_action_rate",
    "profile_sparsity",
    "cos_vs_success",
    "cos_vs_failure",
    "fail_risk_transition_rate",
    "success_signal_transition_rate",
    "last_triplet_ratio",
]

FEATURE_NAMES = NEIGHBOR_FEATURE_NAMES

from episodiq.config.dead_end_defaults import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD  # noqa: E402

__all__ = ["DEFAULT_MODEL_PATH", "DEFAULT_THRESHOLD"]


def _mean_profile(paths: list[TrajectoryPath]) -> dict[str, float]:
    agg: Counter[str] = Counter()
    n = 0
    for p in paths:
        prof = p.transition_profile or {}
        if prof:
            for k, v in prof.items():
                agg[k] += v
            n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in agg.items()}


def _last_triplet_ratio(
    current_path: TrajectoryPath,
    success_candidates: list[TrajectoryPath],
    failure_candidates: list[TrajectoryPath],
) -> float:
    """Ratio of success candidates containing the last triplet vs all candidates.

    Returns 0.5 when no candidates contain the triplet (neutral).
    """
    trace = current_path.trace
    if len(trace) < 3:
        return 0.5

    transition = f"{trace[-3]}.{trace[-2]}.{trace[-1]}"
    n_succ = sum(
        1 for c in success_candidates
        if TransitionAnalyzer._count_triplet(c.trace, transition) > 0
    )
    n_fail = sum(
        1 for c in failure_candidates
        if TransitionAnalyzer._count_triplet(c.trace, transition) > 0
    )
    total = n_succ + n_fail
    if total == 0:
        return 0.5
    return n_succ / total


def extract_neighbor_features(
    current_path: TrajectoryPath,
    analytics: TrajectoryAnalytics,
) -> list[float] | None:
    """Extract 11 neighbor-based features from TrajectoryAnalytics.

    Returns None if insufficient data.
    """
    trace = current_path.trace
    if len(trace) < 5:
        return None

    if analytics.vote_entropy is None or analytics.candidates is None:
        return None

    candidates = analytics.candidates
    top_k = len(candidates)
    if top_k == 0:
        return None

    dist = analytics.vote_distribution
    if not dist:
        return None

    path_index = current_path.index or 0

    # Signal rates: cumulative count normalized by path position
    fail_risk_action_rate = current_path.fail_risk_action_count / (path_index + 1)
    success_signal_action_rate = current_path.success_signal_action_count / (path_index + 1)
    fail_risk_transition_rate = current_path.fail_risk_transition_count / (path_index + 1)
    success_signal_transition_rate = current_path.success_signal_transition_count / (path_index + 1)

    # cosine vs mean profiles (per-status top_k from full prefetch)
    query_profile = current_path.transition_profile or {}
    mean_success = _mean_profile(analytics.success_candidates or [])
    mean_failure = _mean_profile(analytics.failure_candidates or [])

    # last_triplet_ratio: computed on-the-fly from candidates
    triplet_ratio = _last_triplet_ratio(
        current_path,
        analytics.success_candidates or [],
        analytics.failure_candidates or [],
    )

    return [
        analytics.vote_entropy,
        max(dist.values()),
        len(dist),
        fail_risk_action_rate,
        success_signal_action_rate,
        len(current_path.transition_profile or {}),
        sparse_cosine(query_profile, mean_success) if mean_success else 0.0,
        sparse_cosine(query_profile, mean_failure) if mean_failure else 0.0,
        fail_risk_transition_rate,
        success_signal_transition_rate,
        triplet_ratio,
    ]
