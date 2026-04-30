"""Deferred step: embed output message and assign cluster."""

import structlog

from episodiq.clustering.assigner import ClusterAssigner
from episodiq.storage.postgres.repository import MessageRepository
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class ProcessOutputStep(Step):
    """Embed output message and assign cluster (deferred)."""

    step_id = "process_output"
    deferred = True

    async def exec(self) -> StepResult:
        msg = self.ctx.output_message
        if msg is None:
            return StepResult(passable=True)

        emb = await self.ctx.embedder.embed_text(msg.to_embedder_format())
        msg.embedding = emb

        async with self.ctx.session_factory() as session:
            repo = MessageRepository(session)
            await repo.update(msg.id, embedding=emb)
            assigner = ClusterAssigner(repo)
            await assigner.assign(msg.id, emb, msg.cluster_type, msg.category)
            await session.commit()

        logger.info("output_processed", message_id=str(msg.id))
        return StepResult(passable=True)
