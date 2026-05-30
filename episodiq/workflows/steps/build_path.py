"""Step that builds a trajectory path row with its incremental trace."""

import structlog

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
)
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class BuildPathStep(Step):
    """Build a trajectory path row with its incremental cluster-label trace
    and act_obs token sequence.

    Flow per request:
    1. Close previous path: set to_observation_id and extend its trace,
       trace_tokens and minhash_sig with its (action, to_observation)
       transition (mirrors TrajectoryPathTokenUpdater).
    2. Create new pending path carrying cumulative trace + trace_tokens
       forward, so the next close can extend them without re-querying.
       minhash_sig is left null and recomputed fresh from trace_tokens on
       each close.
    """

    step_id = "build_path"
    deferred = True

    async def exec(self) -> StepResult:
        if not self.ctx.trajectory_id or not self.ctx.input_messages:
            return StepResult(passable=True)

        obs_msg = self.ctx.input_messages[-1]
        action_msg = self.ctx.output_message

        async with self.ctx.session_factory() as session:
            cluster_repo = ClusterRepository(session)
            if not await cluster_repo.has_any():
                return StepResult(passable=True)

            repo = TrajectoryPathRepository(session)
            last_row = await repo.get_last(self.ctx.trajectory_id)

            calc = PathStateCalculator()
            obs_cluster_id, obs_label = await repo.get_cluster_info(obs_msg.id)

            trace = calc.granular_step(last_row, obs_label)
            trace_tokens: list[int] | None = None

            if last_row is not None:
                a_cid = (
                    last_row.action_message.cluster_id
                    if last_row.action_message else None
                )
                ordinal: int | None = None
                if a_cid is not None and obs_cluster_id is not None:
                    assigner = TokenAssigner(
                        TokenMappingRepository(session),
                        TokenClusterRepository(session),
                        cluster_repo,
                    )
                    ordinal = await assigner.assign(a_cid, obs_cluster_id)
                if ordinal is not None:
                    trace_tokens, minhash_sig = calc.token_step(last_row, ordinal)
                else:
                    # Unmapped pair (missing cluster_id, or centroid fallback
                    # below similarity threshold) — carry prev cumulative
                    # tokens forward and keep the previous minhash_sig.
                    trace_tokens = last_row.trace_tokens
                    minhash_sig = last_row.minhash_sig

                await repo.update(
                    last_row.id,
                    to_observation_id=obs_msg.id,
                    trace=trace,
                    trace_tokens=trace_tokens,
                    minhash_sig=minhash_sig,
                )

            await repo.create(
                trajectory_id=self.ctx.trajectory_id,
                from_observation_id=obs_msg.id,
                action_message_id=action_msg.id,
                trace=trace,
                trace_tokens=trace_tokens,
            )

            await session.commit()

        logger.info(
            "path_built",
            trajectory_id=str(self.ctx.trajectory_id),
            trace_len=len(trace),
        )

        return StepResult(passable=True, terminal=True)
