"""Dataclasses for transition profile analytics signals."""

from __future__ import annotations

from dataclasses import dataclass

# Re-export from the metric utility so callers see one canonical tuple.
from episodiq.analytics.metrics import SIMILARITY_METRICS  # noqa: E402,F401

DEFAULT_SIMILARITY_METRIC = "cummax"
# Earliest path index whose ``fail_similarity`` is surfaced in the
# rendered report. The running aggregate is built from earlier
# contributors regardless; only display is gated.
DEFAULT_MIN_FAIL_SIMILARITY_STEP = 50


@dataclass
class PathFrequencySignal:
    """Action variance signal based on vote entropy.

    Low entropy  -> few likely actions (low variance)
    High entropy -> many options (high variance)
    """
    entropy: float
    n_matches: float


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
    """The agent is repeating the same trailing action.observation duplet."""
    is_detected: bool
    duplet: str
    streak: int


@dataclass
class TrajectoryAnalytics:
    """Aggregated analytics for a trajectory at a given point.

    ``fail_similarity`` is the structured dict produced by
    ``TransitionAnalyzer`` for this snapshot:
    ``{"current": x, "cummax": ..., "cummean": ..., "cummeanmax": ...}``.
    All three metrics are kept so the report renderer can pick any of
    them at display time without re-running retrieval. ``None`` means
    no candidates and no previous state to carry forward.
    """
    path_frequency_signal: PathFrequencySignal | None = None
    loop_signal: LoopSignal | None = None
    fail_similarity: dict | None = None
