"""Step that builds trajectory path with incremental transition profile."""

import structlog

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.storage.postgres.repository import ClusterRepository, TrajectoryPathRepository
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class BuildPathStep(Step):
    """Build trajectory path with incremental transition profile.

    Flow per request:
    1. Close previous path: set to_observation_id = current obs
    2. Compute profile/trace via PathStateCalculator
    3. Create new path
    4. Run signal analysis on new path, update signal counts if detected
    """

    step_id = "build_path"
    deferred = True

    async def exec(self) -> StepResult:
        if not self.ctx.trajectory_id or not self.ctx.input_messages:
            return StepResult(passable=True)

        obs_msg = self.ctx.input_messages[-1]
        action_msg = self.ctx.output_message

        async with self.ctx.session_factory() as session:
            # Skip path building when no clusters exist yet
            cluster_repo = ClusterRepository(session)
            if not await cluster_repo.has_any():
                return StepResult(passable=True)

            repo = TrajectoryPathRepository(session)

            last_row = await repo.get_last(self.ctx.trajectory_id)

            # Close previous path
            if last_row is not None:
                await repo.update(last_row.id, to_observation_id=obs_msg.id)

            calc = PathStateCalculator()
            obs_label = await repo.get_cluster_label(obs_msg.id)
            profile, embed, trace = calc.step(last_row, obs_label)

            fail_risk_action_count = last_row.fail_risk_action_count if last_row else 0
            fail_risk_transition_count = last_row.fail_risk_transition_count if last_row else 0
            success_signal_action_count = last_row.success_signal_action_count if last_row else 0
            success_signal_transition_count = last_row.success_signal_transition_count if last_row else 0
            loop_count = last_row.loop_count if last_row else 0

            await repo.create(
                trajectory_id=self.ctx.trajectory_id,
                from_observation_id=obs_msg.id,
                action_message_id=action_msg.id,
                transition_profile=profile,
                profile_embed=embed,
                trace=trace,
                fail_risk_action_count=fail_risk_action_count,
                fail_risk_transition_count=fail_risk_transition_count,
                success_signal_action_count=success_signal_action_count,
                success_signal_transition_count=success_signal_transition_count,
                loop_count=loop_count,
            )

            # Signal analysis on current path
            if embed:
                new_path = await repo.get_last(self.ctx.trajectory_id)
                analyzer = TransitionAnalyzer(path_repo=repo)
                analytics = await analyzer.analyze(new_path)
                self.ctx.analytics = analytics
                updates = {}
                if analytics.fail_risk_action and analytics.fail_risk_action.is_detected:
                    updates["fail_risk_action_count"] = fail_risk_action_count + 1
                if analytics.fail_risk_transition:
                    updates["fail_risk_transition_count"] = fail_risk_transition_count + 1
                if analytics.success_signal_action and analytics.success_signal_action.is_detected:
                    updates["success_signal_action_count"] = success_signal_action_count + 1
                if analytics.success_signal_transition:
                    updates["success_signal_transition_count"] = success_signal_transition_count + 1
                if analytics.loop_signal and analytics.loop_signal.is_detected:
                    updates["loop_count"] = loop_count + 1
                if updates:
                    await repo.update(new_path.id, **updates)

            await session.commit()

        logger.info(
            "path_built",
            trajectory_id=str(self.ctx.trajectory_id),
            profile_size=len(profile) if profile else 0,
            trace_len=len(trace),
        )

        return StepResult(passable=True, terminal=True)
