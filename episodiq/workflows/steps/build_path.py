"""Step that builds a trajectory path row with its incremental trace."""

import structlog

from episodiq.api_adapters.base import CanonicalMessage
from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.retrieval.path_state import ActObs, PathStateCalculator
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class BuildPathStep(Step):
    """Build a trajectory path row with its incremental cluster-label trace
    and act_obs token sequence.

    Per request:
    1. Close previous pending path: set ``to_observation_id`` to the first
       new observation (or, for a parallel tool-call batch, expand to N
       closed paths — one per tool response — all sharing the same
       ``action_message_id``, ``from_observation_id``, and
       ``parallel_group``). Trace tokens are computed once per batch via
       ``PathStateCalculator.token_step``, which sorts ASC for ordering
       invariance, so each path in the batch gets the same canonical
       cumulative ``trace_tokens``.
    2. Create new pending path from the last new observation pointing at
       the new assistant action.
    """

    step_id = "build_path"
    deferred = True

    async def exec(self) -> StepResult:
        if not self.ctx.trajectory_id or not self.ctx.input_messages:
            return StepResult(passable=True)

        input_messages = self.ctx.input_messages
        action_msg = self.ctx.output_message

        async with self.ctx.session_factory() as session:
            cluster_repo = ClusterRepository(session)
            if not await cluster_repo.has_any():
                return StepResult(passable=True)

            repo = TrajectoryPathRepository(session)
            last_row = await repo.get_last(self.ctx.trajectory_id)

            assigner = TokenAssigner(
                TokenMappingRepository(session),
                TokenClusterRepository(session),
                cluster_repo,
            )
            calc = PathStateCalculator(assigner)

            obs_info = [
                (m, *(await repo.get_cluster_info(m.id)))
                for m in input_messages
            ]
            last_obs_msg, _, last_obs_label = obs_info[-1]

            trace = calc.granular_step(last_row, last_obs_label)
            trace_tokens: list[int] | None = None

            if last_row is not None:
                parallel_group = None
                if len(input_messages) > 1 and last_row.action_message is not None:
                    canon = CanonicalMessage.from_db(last_row.action_message)
                    if canon.tool_calls_count > 1:
                        parallel_group = last_row.action_message.index

                a_cid = (
                    last_row.action_message.cluster_id
                    if last_row.action_message else None
                )
                cat = (
                    last_row.action_message.category
                    if last_row.action_message else None
                )
                act_obs = [
                    ActObs(
                        a_cluster_id=a_cid,
                        o_cluster_id=ocid,
                        action_category=cat,
                    )
                    for _, ocid, _ in obs_info
                ]
                trace_tokens, wins = await calc.token_step(
                    last_row, act_obs[0] if len(act_obs) == 1 else act_obs,
                )

                await repo.update(
                    last_row.id,
                    to_observation_id=input_messages[0].id,
                    trace=trace,
                    trace_tokens=trace_tokens,
                    parallel_group=parallel_group,
                )
                for obs_msg, _, _ in obs_info[1:]:
                    await repo.create(
                        trajectory_id=self.ctx.trajectory_id,
                        from_observation_id=last_row.from_observation_id,
                        action_message_id=last_row.action_message_id,
                        to_observation_id=obs_msg.id,
                        trace=trace,
                        trace_tokens=trace_tokens,
                        parallel_group=parallel_group,
                    )

                if wins:
                    lsh_repo = TrajectoryWindowLSHRepository(session)
                    await lsh_repo.bulk_insert([
                        (self.ctx.trajectory_id, w.step, bi, bh)
                        for w in wins for bi, bh in enumerate(w.bands)
                    ])

            await repo.create(
                trajectory_id=self.ctx.trajectory_id,
                from_observation_id=last_obs_msg.id,
                action_message_id=action_msg.id,
                trace=trace,
                trace_tokens=trace_tokens,
            )

            await session.commit()

        logger.info(
            "path_built",
            trajectory_id=str(self.ctx.trajectory_id),
            trace_len=len(trace),
            input_count=len(input_messages),
        )

        return StepResult(passable=True, terminal=True)
