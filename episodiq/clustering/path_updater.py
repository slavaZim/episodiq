"""TrajectoryPathUpdater: rebuilds trajectory paths with transition profiles.

Matches online BuildPathStep behavior: one row per observation,
transition_profile built incrementally with exponential decay (λ=0.8),
profile_embed feature-hashed.
"""

import asyncio
import logging
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.storage.postgres.repository import MessageRepository, TrajectoryPathRepository

logger = logging.getLogger(__name__)

WORKERS = 10


class TrajectoryPathUpdater:
    """Drop all trajectory paths, rebuild from message cluster labels.

    Creates one row per observation (matching online BuildPathStep):
    - transition_profile: all prior transitions with exponential decay
    - profile_embed: feature-hashed vector for cosine search
    - trace: full sequence of observation/action labels up to this point
    """

    def __init__(
        self,
        msg_repo: MessageRepository,
        path_repo: TrajectoryPathRepository,
        calc: PathStateCalculator,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        workers: int = WORKERS,
    ):
        self._msg_repo = msg_repo
        self._path_repo = path_repo
        self._calc = calc
        self._session_factory = session_factory
        self._workers = workers

    async def update(self, *, fill_signals: bool = False) -> int:
        """Rebuild all trajectory paths. Returns total rows created.

        Args:
            fill_signals: If True, run a second pass to populate signal counts
                via TransitionAnalyzer. Expensive.
        """
        await self._path_repo.delete_all()

        traj_ids = await self._msg_repo.get_distinct_trajectory_ids()
        logger.info("build_paths_start trajectories=%d fill_signals=%s", len(traj_ids), fill_signals)

        total = 0
        for i, tid in enumerate(traj_ids, 1):
            total += await self._build_trajectory(tid)
            if i % 100 == 0:
                logger.info("build_paths_progress %d/%d trajectories paths=%d", i, len(traj_ids), total)

        logger.info("build_paths_done trajectories=%d paths=%d", len(traj_ids), total)

        await self._path_repo.sync_trajectory_status()

        if fill_signals:
            if self._session_factory:
                # Flush created paths so _fill_signal_counts sees them in new sessions
                await self._path_repo.session.commit()
            await self._fill_signal_counts()

        return total

    async def _build_trajectory(self, trajectory_id) -> int:
        """Build paths for a single trajectory. Returns rows created."""
        rows = await self._msg_repo.get_trajectory_with_clusters(trajectory_id)
        msgs = [m for m in rows if m.role != "system"]
        if not msgs:
            return 0

        prev_path = None
        count = 0

        for i in range(2, len(msgs), 2):
            profile, embed, trace = self._calc.step(prev_path, msgs[i - 2].cluster_label)
            prev_path = await self._path_repo.create(
                trajectory_id=trajectory_id,
                from_observation_id=msgs[i - 2].id,
                action_message_id=msgs[i - 1].id,
                to_observation_id=msgs[i].id,
                transition_profile=profile,
                profile_embed=embed,
                trace=trace,
            )
            count += 1

        # Trailing observation — only if trajectory ends on observation (no dangling action)
        if len(msgs) % 2 == 1:
            last_obs = msgs[-1]
            profile, embed, trace = self._calc.step(prev_path, last_obs.cluster_label)
            await self._path_repo.create(
                trajectory_id=trajectory_id,
                from_observation_id=last_obs.id,
                action_message_id=None,
                transition_profile=profile,
                profile_embed=embed,
                trace=trace,
            )
            count += 1

        return count

    async def _fill_signal_counts(self) -> None:
        """Second pass: run TransitionAnalyzer per completed path, accumulate signal counts."""
        all_paths = await self._path_repo.get_completed()
        total_paths = len(all_paths)
        logger.info("fill_signals_start paths=%d", total_paths)

        by_traj: dict[str, list] = defaultdict(list)
        for p in all_paths:
            by_traj[str(p.trajectory_id)].append(p)

        sem = asyncio.Semaphore(self._workers)
        done = 0

        async def fill_counts(paths: list) -> int:
            nonlocal done
            async with sem:
                if self._session_factory:
                    async with self._session_factory() as session:
                        path_repo = TrajectoryPathRepository(session)
                        count = await self._do_fill_signal_counts(paths, path_repo)
                        await session.commit()
                else:
                    count = await self._do_fill_signal_counts(paths, self._path_repo)
                done += len(paths)
                if done % 1000 < len(paths):
                    logger.info("fill_signals_progress %d/%d paths", done, total_paths)
                return count

        results = await asyncio.gather(
            *[fill_counts(paths) for paths in by_traj.values()],
            return_exceptions=True,
        )
        updated = sum(r for r in results if isinstance(r, int))
        logger.info("fill_signals_done updated=%d", updated)

    async def _do_fill_signal_counts(self, paths: list, path_repo: TrajectoryPathRepository) -> int:
        analyzer = TransitionAnalyzer(path_repo=path_repo)
        paths.sort(key=lambda p: p.created_at)
        fail_risk_action_count = 0
        fail_risk_transition_count = 0
        success_signal_action_count = 0
        success_signal_transition_count = 0
        loop_count = 0
        updated = 0

        for path in paths:
            analytics = await analyzer.analyze(path)

            if analytics.fail_risk_action and analytics.fail_risk_action.is_detected:
                fail_risk_action_count += 1
            if analytics.fail_risk_transition:
                fail_risk_transition_count += 1
            if analytics.success_signal_action and analytics.success_signal_action.is_detected:
                success_signal_action_count += 1
            if analytics.success_signal_transition:
                success_signal_transition_count += 1
            if analytics.loop_signal and analytics.loop_signal.is_detected:
                loop_count += 1

            any_signal = (
                fail_risk_action_count > 0
                or fail_risk_transition_count > 0
                or success_signal_action_count > 0
                or success_signal_transition_count > 0
                or loop_count > 0
            )
            if any_signal:
                await path_repo.update(
                    path.id,
                    fail_risk_action_count=fail_risk_action_count,
                    fail_risk_transition_count=fail_risk_transition_count,
                    success_signal_action_count=success_signal_action_count,
                    success_signal_transition_count=success_signal_transition_count,
                    loop_count=loop_count,
                )
                updated += 1

        return updated
