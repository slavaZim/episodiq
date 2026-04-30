"""Step that extracts, deduplicates, and saves input messages (no embedding)."""

import structlog

from episodiq.api_adapters.base import Role
from episodiq.storage.postgres.repository import MessageRepository
from episodiq.workflows.context import InputMessage
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class SaveInputStep(Step):
    """Extract, deduplicate, and save input messages without embedding."""

    step_id = "save_input"

    async def exec(self) -> StepResult:
        all_messages = self.ctx.api_adapter.extract_request_messages(self.ctx.body)
        if not all_messages:
            return StepResult(passable=False, reason="No messages in request")

        async with self.ctx.session_factory() as session:
            repo = MessageRepository(session)
            last_idx = await repo.get_max_index(self.ctx.trajectory_id)

        start = (last_idx + 1) if last_idx is not None else 0
        new_messages = all_messages[start:]

        if not new_messages:
            return StepResult(passable=False, reason="No new messages")

        if any(m.role == Role.ASSISTANT for m in new_messages):
            logger.warning("multi_turn_unsupported")
            return StepResult(
                passable=False,
                reason="currently multi-turn single request input not supported",
            )

        input_messages = []
        async with self.ctx.session_factory() as session:
            msg_repo = MessageRepository(session)
            for msg in new_messages:
                db_msg = await msg_repo.save(
                    self.ctx.trajectory_id,
                    msg,
                    embedding=None,
                    category=msg.category,
                    cluster_type=msg.cluster_type,
                )
                input_messages.append(
                    InputMessage(
                        role=msg.role,
                        content=msg.content,
                        id=db_msg.id,
                        embedding=None,
                    )
                )
            await session.commit()

        self.ctx.input_messages = input_messages
        logger.info("messages_saved", total=len(all_messages), new=len(new_messages))
        return StepResult(passable=True)
