"""Build a full trajectory report: per-path analytics + persisted signal metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from episodiq.analytics.log_builder import LogBuilder
from episodiq.analytics.path_frequency import (
    PathFrequencyTagger,
    PathFrequencyThresholds,
)
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.analytics.transition_types import (
    DEFAULT_MIN_FAIL_SIMILARITY_STEP,
    DEFAULT_SIMILARITY_METRIC,
    SIMILARITY_METRICS,
)
from episodiq.config.config import AnalyticsConfig
from episodiq.config.retrieval_config import RetrievalConfig, WindowMinHashConfig
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.retrieval import Retrieval
from episodiq.storage.postgres.models import Trajectory
from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)


@dataclass(frozen=True)
class TrajectoryReport:
    """Built trajectory report: rendered log entries + summary counts.

    ``fail_similarity`` is the chosen-metric trajectory-level aggregate
    (read from the last contributing path's stored dict) — ``None`` if
    no path ever produced retrieval candidates.
    """

    trajectory: Trajectory
    entry_pairs: list[tuple[dict, dict]]
    loop_count: int
    unclassified_step_count: int
    fail_similarity: float | None
    variance_high_count: int
    variance_low_count: int
    # Wallclock from first observation to last action — derived from
    # message timestamps so it survives DB row re-updates (the trajectory
    # row's audit columns tick on every UPDATE).
    started_at: datetime
    ended_at: datetime


class TrajectoryReportBuilder:
    """Analyze a completed trajectory's paths, persist per-path signal metadata
    to ``TrajectoryPath.data``, and build structured log entries for rendering.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        analytics_config: AnalyticsConfig,
        minhash_config: WindowMinHashConfig,
        retrieval_config: RetrievalConfig,
        scoring_config: AggShiftConfig,
        metric: str = DEFAULT_SIMILARITY_METRIC,
        min_fail_similarity_step: int = DEFAULT_MIN_FAIL_SIMILARITY_STEP,
    ) -> None:
        if metric not in SIMILARITY_METRICS:
            raise ValueError(
                f"metric must be one of {SIMILARITY_METRICS}; got {metric!r}",
            )
        self._session = session
        self._path_repo = TrajectoryPathRepository(session)
        self._lsh_repo = TrajectoryWindowLSHRepository(session)
        self._minhash_config = minhash_config
        self._retrieval_config = retrieval_config
        self._scoring_config = scoring_config
        self._metric = metric
        self._analyzer = TransitionAnalyzer(config=analytics_config)
        self._builder = LogBuilder(
            path_frequency_tagger=PathFrequencyTagger(
                PathFrequencyThresholds(
                    analytics_config.low_entropy,
                    analytics_config.high_entropy,
                ),
            ),
            min_fail_similarity_step=min_fail_similarity_step,
            metric=metric,
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

        last_sim_path = None
        if not analytics:
            for path in paths:
                obs, act = self._builder.build(path, None)
                entry_pairs.append((obs, act))
        else:
            retrieval = Retrieval(
                self._path_repo, self._lsh_repo,
                minhash_config=self._minhash_config,
                retrieval_config=self._retrieval_config,
                scoring_config=self._scoring_config,
            )
            prev_path = None
            for path in paths:
                candidates = await retrieval.search(path)
                signals = self._analyzer.analyze(path, candidates, prev_path)

                obs, act = self._builder.build(path, signals)
                entry_pairs.append((obs, act))

                if signals.loop_signal and signals.loop_signal.is_detected:
                    loop_count += 1
                path.data = {
                    "fail_similarity": signals.fail_similarity,
                    "loop_streak": (
                        signals.loop_signal.streak if signals.loop_signal else 0
                    ),
                    "loop_detected": bool(
                        signals.loop_signal and signals.loop_signal.is_detected
                    ),
                }
                if signals.fail_similarity is not None:
                    last_sim_path = path
                prev_path = path
            await self._session.commit()

        unclassified = sum(
            1 for obs, act in entry_pairs
            if "annotation" not in obs or "annotation" not in act
        )
        variance_high = sum(
            1 for _, act in entry_pairs if act.get("action_variance") == "high"
        )
        variance_low = sum(
            1 for _, act in entry_pairs if act.get("action_variance") == "low"
        )
        fail_similarity = (
            last_sim_path.data["fail_similarity"].get(self._metric)
            if last_sim_path is not None else None
        )

        return TrajectoryReport(
            trajectory=trajectory,
            entry_pairs=entry_pairs,
            loop_count=loop_count,
            unclassified_step_count=unclassified,
            fail_similarity=fail_similarity,
            variance_high_count=variance_high,
            variance_low_count=variance_low,
            started_at=datetime.fromisoformat(entry_pairs[0][0]["timestamp"]),
            ended_at=datetime.fromisoformat(entry_pairs[-1][1]["timestamp"]),
        )
