"""Build a full trajectory report: per-path analytics + persisted signal metadata."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from episodiq.analytics.log_builder import LogBuilder
from episodiq.analytics.path_frequency import (
    PathFrequencyTagger,
    PathFrequencyThresholds,
)
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.config.config import AnalyticsConfig
from episodiq.config.retrieval_config import RetrievalConfig
from episodiq.retrieval.retrieval import Retrieval
from episodiq.storage.postgres.models import Trajectory
from episodiq.storage.postgres.repository import TrajectoryPathRepository


@dataclass(frozen=True)
class TrajectoryReport:
    """Built trajectory report: rendered log entries + summary counts."""

    trajectory: Trajectory
    entry_pairs: list[tuple[dict, dict]]
    loop_count: int
    unclassified_step_count: int
    peak_fail_frac: float | None
    variance_high_count: int
    variance_low_count: int


class TrajectoryReportBuilder:
    """Analyze a completed trajectory's paths, persist per-path signal metadata
    to ``TrajectoryPath.data``, and build structured log entries for rendering.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        analytics_config: AnalyticsConfig,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self._session = session
        self._path_repo = TrajectoryPathRepository(session)
        self._retrieval_config = retrieval_config
        self._analyzer = TransitionAnalyzer(config=analytics_config)
        self._builder = LogBuilder(
            path_frequency_tagger=PathFrequencyTagger(
                PathFrequencyThresholds(
                    analytics_config.low_entropy,
                    analytics_config.high_entropy,
                ),
            ),
        )

    async def build(
        self, trajectory_id: UUID, *, analytics: bool = False,
    ) -> TrajectoryReport | None:
        """Build the report for ``trajectory_id``.

        Returns None if the trajectory does not exist. Raises ValueError if
        it exists but has no completed paths.

        When ``analytics`` is False (default), the report skips retrieval +
        transition analysis entirely — entries contain only the static
        labels/annotations and no loop / fail-frac / variance metadata.
        Enable with the CLI ``-a`` flag.
        """
        trajectory = await self._session.get(Trajectory, trajectory_id)
        if trajectory is None:
            return None

        paths = await self._path_repo.get_trajectory_paths(trajectory_id)
        if not paths:
            raise ValueError(f"No completed paths for trajectory {trajectory_id}")

        entry_pairs: list[tuple[dict, dict]] = []
        loop_count = 0

        if not analytics:
            for path in paths:
                obs, act = self._builder.build(path, None)
                entry_pairs.append((obs, act))
        else:
            retrieval = Retrieval(self._path_repo, self._retrieval_config)
            for path in paths:
                candidates = await retrieval.search(path)
                signals = self._analyzer.analyze(path, candidates)

                obs, act = self._builder.build(path, signals)
                entry_pairs.append((obs, act))

                if signals.loop_signal and signals.loop_signal.is_detected:
                    loop_count += 1
                path.data = {
                    "fail_frac": signals.fail_frac,
                    "loop_streak": (
                        signals.loop_signal.streak if signals.loop_signal else 0
                    ),
                    "loop_detected": bool(
                        signals.loop_signal and signals.loop_signal.is_detected
                    ),
                }
            await self._session.commit()

        unclassified = sum(
            1 for obs, act in entry_pairs
            if "annotation" not in obs or "annotation" not in act
        )
        fail_fracs = [
            obs["fail_frac"] for obs, _ in entry_pairs if "fail_frac" in obs
        ]
        variance_high = sum(
            1 for _, act in entry_pairs if act.get("action_variance") == "high"
        )
        variance_low = sum(
            1 for _, act in entry_pairs if act.get("action_variance") == "low"
        )

        return TrajectoryReport(
            trajectory=trajectory,
            entry_pairs=entry_pairs,
            loop_count=loop_count,
            unclassified_step_count=unclassified,
            peak_fail_frac=max(fail_fracs) if fail_fracs else None,
            variance_high_count=variance_high,
            variance_low_count=variance_low,
        )
