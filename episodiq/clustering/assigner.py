"""ClusterAssigner: online KNN-based cluster assignment via pgvector."""

import logging
from uuid import UUID

from episodiq.storage.postgres.repository import MessageRepository

logger = logging.getLogger(__name__)


class ClusterAssigner:
    """Assign a message to the nearest cluster via KNN vote using pgvector."""

    def __init__(
        self,
        repo: MessageRepository,
        k: int = 7,
        confidence_threshold: float = 0.5,
    ):
        self._repo = repo
        self._k = k
        self._confidence_threshold = confidence_threshold

    async def assign(
        self, message_id: UUID, embedding: list[float],
        cluster_type: str, category: str,
    ) -> UUID | None:
        """Find k nearest non-noise neighbors, distance-weighted vote.

        Returns cluster_id if confident, None otherwise (message stays cluster_id=NULL).
        """
        neighbors = await self._repo.find_neighbors(
            embedding, cluster_type, category, message_id, k=self._k,
        )

        if not neighbors:
            return None

        # Distance-weighted vote: similarity = 1 - cosine_distance
        votes: dict[UUID, float] = {}
        for cluster_id, distance in neighbors:
            sim = 1.0 - distance
            votes[cluster_id] = votes.get(cluster_id, 0.0) + sim

        total = sum(votes.values())
        if total <= 0:
            return None

        best_id = max(votes, key=votes.get)
        confidence = votes[best_id] / total

        if confidence < self._confidence_threshold:
            return None

        await self._repo.update(message_id, cluster_id=best_id)

        logger.debug(
            "assigned message=%s cluster=%s conf=%.2f",
            message_id, best_id, confidence,
        )
        return best_id
