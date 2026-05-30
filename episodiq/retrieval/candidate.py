"""Retrieval output type."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RetrievalCandidate:
    """One returned candidate trajectory from Retrieval.search().

    best_path_action_cluster_id is the action_message.cluster_id of the
    candidate's path whose Lev sim to the query was highest. It's exposed so
    downstream analytics (e.g. path-frequency entropy) can group candidates by
    semantic action without re-running retrieval.
    """
    trajectory_id: UUID
    score: float
    best_path_action_cluster_id: UUID | None
    trajectory_status: str
