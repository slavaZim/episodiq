"""AnnotationSaver: applies annotation results to DB.

Takes Annotation list from ClusterAnnotator, picks representative
cluster per group, reassigns messages, deletes absorbed.
Path rebuild is a separate step via `episodiq cluster rebuild-paths`.
"""

import logging
from uuid import UUID

from episodiq.clustering.annotator.annotator import Annotation

logger = logging.getLogger(__name__)


class AnnotationSaver:
    """Applies annotations to DB: update annotations, merge absorbed clusters."""

    def __init__(self, cluster_repo, message_repo):
        self._cluster_repo = cluster_repo
        self._message_repo = message_repo

    async def save(self, annotations: list[Annotation]) -> int:
        """Apply annotations to DB. Returns count of merged (deleted) clusters."""
        absorbed_ids: list[UUID] = []

        for ann in annotations:
            if not ann.text:
                continue

            if len(ann.merged_ids) == 1:
                cid = next(iter(ann.merged_ids))
                await self._cluster_repo.update(cid, annotation=ann.text)
                continue

            # Multiple clusters — pick representative (smallest label)
            rep_id = await self._pick_representative(ann.merged_ids)
            others = ann.merged_ids - {rep_id}

            await self._cluster_repo.update(rep_id, annotation=ann.text)

            # Reassign messages from absorbed → representative
            for old_id in others:
                msgs = await self._message_repo.find_by(cluster_id=old_id)
                for msg in msgs:
                    await self._message_repo.update(msg.id, cluster_id=rep_id)
                absorbed_ids.append(old_id)

        # Delete absorbed clusters
        for cid in absorbed_ids:
            await self._cluster_repo.delete(cid)
        if absorbed_ids:
            logger.info("deleted %d absorbed clusters", len(absorbed_ids))

        return len(absorbed_ids)

    async def _pick_representative(self, cluster_ids: set[UUID]) -> UUID:
        """Pick cluster with smallest label as representative."""
        clusters = []
        for cid in cluster_ids:
            found = await self._cluster_repo.find_by(id=cid)
            clusters.extend(found)
        clusters.sort(key=lambda c: c.label)
        return clusters[0].id
