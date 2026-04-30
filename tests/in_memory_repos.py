"""In-memory repositories for unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np

@dataclass
class OriginResponse:
    adapter_id: str | None
    external_id: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    message_id: UUID | None = None
    internal_message_id: UUID | None = None


@dataclass
class Cluster:
    id: UUID
    type: str
    category: str
    label: str
    annotation: str | None = None


PREFIXES = {"observation": "o", "action": "a"}


@dataclass
class Message:
    id: UUID
    trajectory_id: UUID
    role: str
    content: list[dict]
    index: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = None
    origin_response: OriginResponse | None = None
    cluster_id: UUID | None = None
    category: str | None = None
    cluster_type: str | None = None
    summary: str | None = None
    cluster: Cluster | None = None

    @property
    def cluster_label(self) -> str:
        if self.cluster:
            return self.cluster.label
        prefix = PREFIXES.get("observation" if self.role in ("user", "tool") else "action", "?")
        cat = self.category or "?"
        return f"{prefix}:{cat}:?"


@dataclass
class Trajectory:
    id: UUID
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    paths: list = field(default_factory=list)


class InMemoryTrajectoryRepository:
    """Fake replacement for TrajectoryRepository for unit tests."""

    def __init__(self):
        self._trajectories: dict[UUID, Trajectory] = {}

    async def find_or_create(self, trajectory_id: UUID) -> Trajectory:
        if trajectory_id not in self._trajectories:
            self._trajectories[trajectory_id] = Trajectory(id=trajectory_id)
        return self._trajectories[trajectory_id]

    async def update_status(self, trajectory_id: UUID, status: str) -> None:
        self._trajectories[trajectory_id].status = status

    async def get_with_completed_paths(
        self,
        status: str,
        limit: int | None = None,
        require_embed: bool = False,
    ) -> list[Trajectory]:
        result = [t for t in self._trajectories.values() if t.status == status]
        if require_embed:
            result = [
                t for t in result
                if any(getattr(p, "profile_embed", None) for p in t.paths)
            ]
        if limit is not None:
            result = result[:limit]
        return result


class InMemoryMessageRepository:
    """Fake replacement for MessageRepository for unit tests."""

    def __init__(self):
        self._messages: list[Message] = []
        self._by_id: dict[UUID, Message] = {}
        self._clusters: dict[UUID, Cluster] = {}

    def add_cluster(self, cluster: Cluster) -> None:
        self._clusters[cluster.id] = cluster

    async def save(
        self,
        trajectory_id: UUID,
        message: Any,
        embedding: list[float] | None = None,
        cluster_id: UUID | None = None,
        category: str | None = None,
        cluster_type: str | None = None,
    ) -> Message:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        content = message.content

        index = sum(1 for m in self._messages if m.trajectory_id == trajectory_id)

        cluster = self._clusters.get(cluster_id) if cluster_id else None

        msg = Message(
            id=uuid4(),
            trajectory_id=trajectory_id,
            role=role,
            content=content,
            index=index,
            embedding=list(embedding) if embedding else None,
            cluster_id=cluster_id,
            category=category,
            cluster_type=cluster_type,
            cluster=cluster,
        )

        is_assistant = message.__class__.__name__ == "CanonicalAssistantMessage" or role in ("assistant", "ASSISTANT")

        if is_assistant:
            usage = getattr(message, "usage", None)
            msg.origin_response = OriginResponse(
                message_id=msg.id,
                adapter_id=getattr(message, "adapter_id", None),
                external_id=getattr(message, "external_id", None),
                model=getattr(message, "model", None),
                input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            )

        self._messages.append(msg)
        self._by_id[msg.id] = msg
        return msg

    async def get_max_index(self, trajectory_id: UUID) -> int | None:
        indices = [m.index for m in self._messages if m.trajectory_id == trajectory_id]
        return max(indices) if indices else None

    async def update_embedding(self, message_id: UUID, embedding: list[float]) -> None:
        self._by_id[message_id].embedding = list(embedding)

    async def get_message_count(self, trajectory_id: UUID) -> int:
        return sum(1 for m in self._messages if m.trajectory_id == trajectory_id)

    async def get_trajectory(self, trajectory_id: UUID) -> list[Message]:
        return [m for m in self._messages if m.trajectory_id == trajectory_id]

    async def find_neighbors(
        self,
        embedding: list[float],
        cluster_type: str,
        category: str,
        exclude_message_id: UUID,
        k: int = 10,
    ):
        from episodiq.storage.postgres.repository import ClusterNeighbor

        query = np.array(embedding)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []

        scored = []
        for m in self._messages:
            if m.id == exclude_message_id or m.embedding is None or m.cluster_id is None:
                continue
            if m.cluster_type != cluster_type or m.category != category:
                continue
            vec = np.array(m.embedding)
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            distance = 1.0 - float(np.dot(query, vec) / (q_norm * v_norm))
            scored.append(ClusterNeighbor(m.cluster_id, distance))

        scored.sort(key=lambda x: x.distance)
        return scored[:k]

    async def update(self, id: UUID, **kwargs) -> None:
        msg = self._by_id[id]
        for k, v in kwargs.items():
            setattr(msg, k, v)

    async def find_by(self, **kwargs) -> list[Message]:
        result = []
        for m in self._messages:
            if all(getattr(m, k, None) == v for k, v in kwargs.items()):
                result.append(m)
        return result

    async def get_messages_for_clustering(
        self, cluster_type: str, category: str,
    ) -> list[Message]:
        return [
            m for m in self._messages
            if m.cluster_type == cluster_type
            and m.category == category
            and m.embedding is not None
        ]

    async def get_distinct_categories(self, cluster_type: str) -> list[str]:
        return list({
            m.category for m in self._messages
            if m.cluster_type == cluster_type and m.category is not None
        })

    async def get_distinct_trajectory_ids(self) -> list[UUID]:
        return list({m.trajectory_id for m in self._messages})

    async def get_trajectory_with_clusters(self, trajectory_id: UUID) -> list[Message]:
        msgs = [m for m in self._messages if m.trajectory_id == trajectory_id]
        msgs.sort(key=lambda m: m.index)
        return msgs

    async def sample_by_cluster(self, cluster_id: UUID, n: int) -> list[Message]:
        """Return up to n messages from a cluster (random in prod, deterministic here)."""
        msgs = [m for m in self._messages if m.cluster_id == cluster_id]
        return msgs[:n]

    def add_message(self, msg: Message) -> None:
        """Add a pre-built Message directly (for clustering tests)."""
        self._messages.append(msg)
        self._by_id[msg.id] = msg


class InMemoryClusterRepository:
    """Fake replacement for ClusterRepository for unit tests."""

    def __init__(self):
        self._clusters: list[Cluster] = []

    async def has_any(self) -> bool:
        return len(self._clusters) > 0

    async def delete_by_type_category(self, type: str, category: str) -> None:
        self._clusters = [
            c for c in self._clusters
            if not (c.type == type and c.category == category)
        ]

    async def create(self, **kwargs) -> Cluster:
        cluster = Cluster(id=uuid4(), **kwargs)
        self._clusters.append(cluster)
        return cluster

    async def update(self, id: UUID, **kwargs) -> None:
        cluster = next((c for c in self._clusters if c.id == id), None)
        if cluster:
            for k, v in kwargs.items():
                setattr(cluster, k, v)

    async def find_by(self, **kwargs) -> list[Cluster]:
        return [
            c for c in self._clusters
            if all(getattr(c, k, None) == v for k, v in kwargs.items())
        ]

    async def delete(self, id: UUID) -> None:
        self._clusters = [c for c in self._clusters if c.id != id]

    def get_by_label(self, label: str) -> Cluster | None:
        return next((c for c in self._clusters if c.label == label), None)

    async def get_centroids(self, cluster_ids: set[UUID]) -> list[tuple[UUID, str, Any]]:
        """Compute average embedding per cluster from linked messages in message_repo."""
        # This requires access to messages — store a reference if provided
        results = []
        for cluster in self._clusters:
            if cluster.id not in cluster_ids:
                continue
            msgs = self._messages_by_cluster.get(cluster.id, [])
            embeddings = [m.embedding for m in msgs if m.embedding is not None]
            if embeddings:
                avg = np.mean(embeddings, axis=0).tolist()
                results.append((cluster.id, cluster.label, avg))
        return results

    def link_messages(self, message_repo: "InMemoryMessageRepository") -> None:
        """Link to message repo for centroid computation."""
        self._messages_by_cluster: dict[UUID, list[Message]] = {}
        for m in message_repo._messages:
            if m.cluster_id is not None:
                self._messages_by_cluster.setdefault(m.cluster_id, []).append(m)


@dataclass
class InMemoryPath:
    id: UUID
    trajectory_id: UUID
    from_observation_id: UUID
    action_message_id: UUID | None = None
    to_observation_id: UUID | None = None
    transition_profile: dict | None = None
    profile_embed: list[float] | None = None
    trace: list[str] = field(default_factory=list)
    trajectory_status: str = "pending"
    fail_risk_action_count: int = 0
    fail_risk_transition_count: int = 0
    success_signal_action_count: int = 0
    success_signal_transition_count: int = 0
    loop_count: int = 0
    index: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    from_obs_label: str | None = None
    action_label: str | None = None
    to_obs_label: str | None = None
    trajectory: Trajectory | None = None
    action_message: Message | None = None


class InMemoryTrajectoryPathRepository:
    """Fake replacement for TrajectoryPathRepository for unit tests."""

    def __init__(self, msg_repo: InMemoryMessageRepository | None = None):
        self._paths: list[InMemoryPath] = []
        self._msg_repo = msg_repo

    async def delete_all(self) -> None:
        self._paths.clear()

    async def sync_trajectory_status(self) -> None:
        for p in self._paths:
            if p.trajectory:
                p.trajectory_status = p.trajectory.status

    async def create(
        self,
        trajectory_id: UUID,
        from_observation_id: UUID,
        transition_profile: dict | None = None,
        profile_embed: list[float] | None = None,
        trace: list[str] | None = None,
        action_message_id: UUID | None = None,
        to_observation_id: UUID | None = None,
        fail_risk_action_count: int = 0,
        fail_risk_transition_count: int = 0,
        success_signal_action_count: int = 0,
        success_signal_transition_count: int = 0,
        loop_count: int = 0,
        trajectory_status: str = "pending",
    ) -> InMemoryPath:
        # Auto-increment index per trajectory
        existing = [p for p in self._paths if p.trajectory_id == trajectory_id]
        index = max((p.index for p in existing), default=-1) + 1

        # Resolve labels from linked message repo
        from_obs_label = None
        action_label = None
        to_obs_label = None
        if self._msg_repo:
            msg = self._msg_repo._by_id.get(from_observation_id)
            from_obs_label = msg.cluster_label if msg else None
            if action_message_id:
                msg = self._msg_repo._by_id.get(action_message_id)
                action_label = msg.cluster_label if msg else None
            if to_observation_id:
                msg = self._msg_repo._by_id.get(to_observation_id)
                to_obs_label = msg.cluster_label if msg else None

        path = InMemoryPath(
            id=uuid4(),
            trajectory_id=trajectory_id,
            from_observation_id=from_observation_id,
            action_message_id=action_message_id,
            to_observation_id=to_observation_id,
            transition_profile=transition_profile,
            profile_embed=profile_embed,
            trace=trace or [],
            trajectory_status=trajectory_status,
            fail_risk_action_count=fail_risk_action_count,
            fail_risk_transition_count=fail_risk_transition_count,
            success_signal_action_count=success_signal_action_count,
            success_signal_transition_count=success_signal_transition_count,
            loop_count=loop_count,
            index=index,
            from_obs_label=from_obs_label,
            action_label=action_label,
            to_obs_label=to_obs_label,
        )
        self._paths.append(path)
        return path

    async def update(self, id: UUID, **kwargs) -> None:
        for p in self._paths:
            if p.id == id:
                for k, v in kwargs.items():
                    setattr(p, k, v)
                return

    async def get_last(self, trajectory_id: UUID) -> InMemoryPath | None:
        traj_paths = [p for p in self._paths if p.trajectory_id == trajectory_id]
        return traj_paths[-1] if traj_paths else None

    async def get_cluster_label(self, message_id: UUID) -> str:
        return self._msg_repo._by_id[message_id].cluster_label

    async def get_completed(
        self, limit: int | None = None, require_embed: bool = False,
    ) -> list[InMemoryPath]:
        result = [p for p in self._paths if p.to_observation_id is not None]
        if require_embed:
            result = [p for p in result if p.profile_embed is not None]
        if limit is not None:
            result = result[:limit]
        return result

    async def prefetch_similar(
        self,
        profile_embed: list[float],
        exclude_trajectory_id: UUID,
        limit: int = 1000,
        defer_embed: bool = True,
    ) -> list["InMemoryPath"]:
        """Brute-force cosine search for unit tests."""
        query = np.array(profile_embed)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []

        scored = []
        for p in self._paths:
            if p.trajectory_id == exclude_trajectory_id:
                continue
            if p.profile_embed is None:
                continue
            if p.trajectory_status not in ("success", "failure"):
                continue
            vec = np.array(p.profile_embed)
            v_norm = np.linalg.norm(vec)
            if v_norm == 0:
                continue
            distance = 1.0 - float(np.dot(query, vec) / (q_norm * v_norm))
            scored.append((distance, p))

        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:limit]]


