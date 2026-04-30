"""ClusterSaver: persists clustering results to DB."""

import logging
from collections import defaultdict

from episodiq.clustering.manager import CategoryResult
from episodiq.clustering.updater import ClusterAssignment
from episodiq.storage.postgres.repository import ClusterRepository

logger = logging.getLogger(__name__)


class ClusterSaver:
    """Drop old clusters for affected type+category, create new ones.

    Returns list of ClusterAssignment for the caller to update messages.
    """

    def __init__(self, cluster_repo: ClusterRepository):
        self._repo = cluster_repo

    async def save(self, results: list[CategoryResult]) -> list[ClusterAssignment]:
        assignments: list[ClusterAssignment] = []

        for cat_result in results:
            await self._repo.delete_by_type_category(cat_result.type, cat_result.category)

            # Group message_ids by label
            label_to_ids: dict[str, list] = defaultdict(list)
            for msg_id, label in zip(cat_result.message_ids, cat_result.labels):
                label_to_ids[label].append(msg_id)

            # Create one Cluster row per unique label (skip noise)
            for label, msg_ids in label_to_ids.items():
                if label.endswith(":?"):
                    continue

                cluster = await self._repo.create(
                    type=cat_result.type,
                    category=cat_result.category,
                    label=label,
                )

                for msg_id in msg_ids:
                    assignments.append(ClusterAssignment(msg_id, cluster.id))

            logger.debug(
                "saved type=%s category=%s clusters=%d messages=%d",
                cat_result.type, cat_result.category,
                len(label_to_ids), len(cat_result.message_ids),
            )

        return assignments
