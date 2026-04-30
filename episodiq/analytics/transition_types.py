"""Dataclasses for transition profile analytics signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodiq.storage.postgres.models import TrajectoryPath


@dataclass
class PathFrequencySignal:
    """Action variance signal based on vote entropy.

    Low entropy  → few likely actions (low variance)
    High entropy → many options (high variance)
    """
    entropy: float
    n_matches: int


def tail_streak(trace: list[str]) -> int:
    """Count consecutive repetitions of the last duplet from the end of trace.

    Duplet = trace[-2].trace[-1]. Scans backwards in steps of 2, gap=0.
    Returns 0 if trace has fewer than 2 elements.

    streak=1 means the duplet appeared once (no repetition).
    streak>=2 means the agent is repeating the same action-observation pair.
    """
    n = len(trace)
    if n < 2:
        return 0
    duplet = f"{trace[-2]}.{trace[-1]}"
    count = 0
    i = n - 2
    while i >= 0:
        if f"{trace[i]}.{trace[i + 1]}" == duplet:
            count += 1
        else:
            break
        i -= 2
    return count


@dataclass
class LoopSignal:
    """Whether the agent is repeating the same transition excessively."""
    is_detected: bool
    transition: str
    repeat_count: int
    median_neighbor_repeat: float = 0
    mean_success_repeat: float | None = None

    @staticmethod
    def is_loop_at(path: TrajectoryPath, threshold: float) -> bool:
        return tail_streak(path.trace) >= threshold


@dataclass
class ActionSignal:
    """Action-level signal based on Levenshtein-weighted cosine similarity.

    similarity = fail_lev_cosine - succ_lev_cosine (delta).
    For fail_risk_action: is_detected when similarity >= threshold (closer to failure).
    For success_signal_action: is_detected when -similarity >= threshold (closer to success).
    """
    is_detected: bool
    similarity: float | None


@dataclass
class TrajectoryAnalytics:
    """Aggregated analytics for a trajectory at a given point."""
    vote_entropy: float | None = None
    mean_similarity: float | None = None
    n_voters: int | None = None
    n_success: int | None = None
    vote_distribution: dict[str, float] | None = None
    fail_similarity: float | None = None
    path_frequency_signal: PathFrequencySignal | None = None
    loop_signal: LoopSignal | None = None
    fail_risk_action: ActionSignal | None = None
    success_signal_action: ActionSignal | None = None
    prefetched_reranked: list[tuple[float, TrajectoryPath]] | None = None
    candidates: list[TrajectoryPath] | None = None
    success_candidates: list[TrajectoryPath] | None = None
    failure_candidates: list[TrajectoryPath] | None = None
    fail_risk_transition: bool = False
    success_signal_transition: bool = False
    loop_streak: int = 0
    model_score: float | None = None
