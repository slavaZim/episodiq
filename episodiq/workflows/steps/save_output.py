"""Step that saves the assistant response to DB (no embedding)."""

import structlog

from episodiq.storage.postgres.repository import MessageRepository
from episodiq.workflows.context import OutputMessage
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class SaveOutputStep(Step):
    """Save assistant response from pending_response without embedding."""

    step_id = "save_output"

    async def exec(self) -> StepResult:
        pending = self.ctx.pending_response
        if pending is None or pending.canonical_msg is None:
            return StepResult(passable=True)

        canonical_msg = pending.canonical_msg

        async with self.ctx.session_factory() as session:
            repo = MessageRepository(session)
            db_msg = await repo.save(
                trajectory_id=self.ctx.trajectory_id,
                message=canonical_msg,
                embedding=None,
                category=canonical_msg.category,
                cluster_type=canonical_msg.cluster_type,
            )
            await session.commit()

        self.ctx.output_message = OutputMessage(
            role=canonical_msg.role,
            content=canonical_msg.content,
            id=db_msg.id,
            embedding=None,
        )

        logger.info("output_saved", message_id=str(db_msg.id))
        return StepResult(passable=True)
