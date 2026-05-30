"""MessageUpdater: applies cluster assignments to messages."""

import logging
from dataclasses import dataclass
from uuid import UUID

from episodiq.storage.postgres.repository import MessageRepository

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    message_id: UUID
    cluster_id: UUID


class MessageUpdater:
    """Bulk update Message.cluster_id from assignments."""

    def __init__(self, msg_repo: MessageRepository):
        self._repo = msg_repo

    async def update(self, assignments: list[ClusterAssignment]) -> None:
        if not assignments:
            return

        for a in assignments:
            await self._repo.update(a.message_id, cluster_id=a.cluster_id)

        logger.debug("updated messages=%d", len(assignments))
