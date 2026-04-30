"""Deferred step: embed input messages and assign clusters."""

import structlog

from episodiq.clustering.assigner import ClusterAssigner
from episodiq.storage.postgres.repository import MessageRepository
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class ProcessInputStep(Step):
    """Embed input messages and assign clusters (deferred)."""

    step_id = "process_input"
    deferred = True

    async def exec(self) -> StepResult:
        if not self.ctx.input_messages:
            return StepResult(passable=True)

        for msg in self.ctx.input_messages:
            emb = await self.ctx.embedder.embed_text(msg.to_embedder_format())
            msg.embedding = emb

            async with self.ctx.session_factory() as session:
                repo = MessageRepository(session)
                await repo.update(msg.id, embedding=emb)
                assigner = ClusterAssigner(repo)
                await assigner.assign(msg.id, emb, msg.cluster_type, msg.category)
                await session.commit()

        logger.info("input_processed", count=len(self.ctx.input_messages))
        return StepResult(passable=True)
