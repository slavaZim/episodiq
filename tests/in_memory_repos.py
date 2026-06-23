"""In-memory repositories for unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from episodiq.storage.postgres.repository import Centroid

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
        status: str | list[str],
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Trajectory]:
        if isinstance(status, str):
            statuses = {status}
        else:
            statuses = set(status)
        result = [
            t for t in sorted(self._trajectories.values(), key=lambda t: t.id)
            if t.status in statuses
        ]
        end = offset + limit if limit is not None else len(result)
        return result[offset:end]


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

    async def get_categories(self, cluster_type: str) -> list[str]:
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

    async def get_categories(self, type: str) -> list[str]:
        """Sorted distinct ``category`` values for clusters of ``type``.
        TokenAssigner reads this list to compute per-category noise
        ordinals; stable order matters.
        """
        return sorted({
            c.category for c in self._clusters
            if c.type == type and c.category is not None
        })

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

    async def get_centroids(self, cluster_ids: set[UUID]) -> "list[Centroid]":
        """Compute average embedding per cluster from linked messages."""
        from episodiq.storage.postgres.repository import Centroid
        results: list[Centroid] = []
        for cluster in self._clusters:
            if cluster.id not in cluster_ids:
                continue
            msgs = self._messages_by_cluster.get(cluster.id, [])
            embeddings = [m.embedding for m in msgs if m.embedding is not None]
            if embeddings:
                avg = np.mean(embeddings, axis=0).tolist()
                results.append(Centroid(
                    cluster_type=cluster.type,
                    category=cluster.category,
                    embedding=avg,
                    cluster_id=cluster.id,
                    label=cluster.label,
                ))
        return results

    async def get_category_centroids(self) -> "list[Centroid]":
        """Per-(cluster_type, category) centroids from noise-side
        messages (cluster_id IS NULL). Mirrors the production query
        — TokenAssigner falls back to these when a direct mapping is
        absent.
        """
        from episodiq.storage.postgres.repository import Centroid
        groups: dict[tuple[str, str], list[list[float]]] = {}
        if not hasattr(self, "_messages_by_cluster"):
            return []
        msg_repo_messages = []
        for cluster_msgs in self._messages_by_cluster.values():
            msg_repo_messages.extend(cluster_msgs)
        # Walk every message — noise rows live where cluster_id is None
        # but cluster_type/category are still set on the message itself.
        for m in self._all_messages():
            if m.cluster_id is not None:
                continue
            if m.cluster_type is None or m.category is None:
                continue
            if m.embedding is None:
                continue
            groups.setdefault(
                (m.cluster_type, m.category), [],
            ).append(m.embedding)
        return [
            Centroid(
                cluster_type=ct, category=cat,
                embedding=np.mean(embs, axis=0).tolist(),
            )
            for (ct, cat), embs in groups.items()
        ]

    def _all_messages(self) -> list[Message]:
        """All messages from the linked MessageRepository, if any."""
        return list(getattr(self, "_all_msgs_ref", []))

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
    data: dict | None = None
    trace: list[str] = field(default_factory=list)
    trace_tokens: list[int] | None = None
    minhash_sig: list[int] | None = None
    parallel_group: int | None = None
    trajectory_status: str = "pending"
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
        *,
        trajectory_id: UUID,
        from_observation_id: UUID,
        action_message_id: UUID | None = None,
        to_observation_id: UUID | None = None,
        data: dict | None = None,
        trace: list[str] | None = None,
        trace_tokens: list[int] | None = None,
        minhash_sig: list[int] | None = None,
        trajectory_status: str = "pending",
        parallel_group: int | None = None,
    ) -> InMemoryPath:
        # Auto-increment index per trajectory
        existing = [p for p in self._paths if p.trajectory_id == trajectory_id]
        index = max((p.index for p in existing), default=-1) + 1

        # Resolve labels + action_message from linked message repo
        from_obs_label = None
        action_label = None
        to_obs_label = None
        action_message = None
        if self._msg_repo:
            msg = self._msg_repo._by_id.get(from_observation_id)
            from_obs_label = msg.cluster_label if msg else None
            if action_message_id:
                action_message = self._msg_repo._by_id.get(action_message_id)
                action_label = action_message.cluster_label if action_message else None
            if to_observation_id:
                msg = self._msg_repo._by_id.get(to_observation_id)
                to_obs_label = msg.cluster_label if msg else None

        path = InMemoryPath(
            id=uuid4(),
            trajectory_id=trajectory_id,
            from_observation_id=from_observation_id,
            action_message_id=action_message_id,
            to_observation_id=to_observation_id,
            data=data,
            trace=trace or [],
            trace_tokens=trace_tokens,
            minhash_sig=minhash_sig,
            parallel_group=parallel_group,
            trajectory_status=trajectory_status,
            index=index,
            from_obs_label=from_obs_label,
            action_label=action_label,
            to_obs_label=to_obs_label,
            action_message=action_message,
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

    async def get_cluster_info(self, message_id: UUID) -> tuple[UUID | None, str]:
        msg = self._msg_repo._by_id[message_id]
        return msg.cluster_id, msg.cluster_label

    async def get_completed(
        self, limit: int | None = None, require_tokens: bool = False,
    ) -> list[InMemoryPath]:
        result = [p for p in self._paths if p.to_observation_id is not None]
        if require_tokens:
            result = [p for p in result if p.trace_tokens is not None]
        if limit is not None:
            result = result[:limit]
        return result

    async def get_trajectory_paths(self, trajectory_id: UUID) -> list[InMemoryPath]:
        return [
            p for p in self._paths
            if p.trajectory_id == trajectory_id
            and p.to_observation_id is not None
        ]

    async def get_minhash_corpus(
        self,
    ) -> list[tuple[UUID, UUID | None, str, list[int]]]:
        """Mirror of TrajectoryPathRepository.get_minhash_corpus."""
        out: list[tuple[UUID, UUID | None, str, list[int]]] = []
        for p in self._paths:
            if not p.minhash_sig:
                continue
            if p.trajectory_status not in ("success", "failure"):
                continue
            action_cluster_id = None
            if self._msg_repo and p.action_message_id:
                msg = self._msg_repo._by_id.get(p.action_message_id)
                action_cluster_id = msg.cluster_id if msg else None
            out.append((p.trajectory_id, action_cluster_id,
                        p.trajectory_status, p.minhash_sig))
        return out

    async def minhash_prefilter(
        self,
        query_signature: list[int],
        exclude_trajectory_id: UUID,
        limit: int,
        min_similarity: float = 0.0,
        corpus: list[tuple[UUID, UUID | None, str, list[int]]] | None = None,
    ) -> list[tuple[UUID, float, UUID | None, str]]:
        """In-memory mirror of the prod prefilter: MAX-pool sim per traj,
        filter by min_similarity, sort desc, slice to limit. Excludes the
        query trajectory.
        """
        from episodiq.retrieval.minhash import max_pool_jaccard_per_key

        if not query_signature:
            return []
        if corpus is None:
            corpus = await self.get_minhash_corpus()
        meta: dict[UUID, tuple[UUID | None, str]] = {}
        keyed: list[tuple[UUID, list[int]]] = []
        for tid, action_cid, status, sig in corpus:
            if tid == exclude_trajectory_id:
                continue
            if tid not in meta:
                meta[tid] = (action_cid, status)
            keyed.append((tid, sig))
        per_traj_sim = max_pool_jaccard_per_key(query_signature, keyed)
        ranked = [
            (tid, sim, meta[tid][0], meta[tid][1])
            for tid, sim in per_traj_sim.items()
            if sim >= min_similarity
        ]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:limit]

    async def get_latest_trace_tokens_for_trajectories(
        self, trajectory_ids: list[UUID],
    ) -> dict[UUID, tuple[UUID | None, str, list[int]]]:
        """trajectory_id → (action_cluster_id, status, trace_tokens) for the
        highest-index path with non-NULL trace_tokens per trajectory.
        """
        if not trajectory_ids:
            return {}
        wanted = set(trajectory_ids)
        best: dict[UUID, tuple[int, UUID | None, str, list[int]]] = {}
        for p in self._paths:
            if p.trajectory_id not in wanted or p.trace_tokens is None:
                continue
            idx = p.index if p.index is not None else -1
            existing = best.get(p.trajectory_id)
            if existing is not None and existing[0] >= idx:
                continue
            action_cluster_id = None
            if self._msg_repo and p.action_message_id:
                msg = self._msg_repo._by_id.get(p.action_message_id)
                action_cluster_id = msg.cluster_id if msg else None
            best[p.trajectory_id] = (
                idx, action_cluster_id, p.trajectory_status, p.trace_tokens,
            )
        return {
            tid: (action_cid, status, list(tokens))
            for tid, (_idx, action_cid, status, tokens) in best.items()
        }

    async def collect_act_obs(self) -> list[tuple[UUID, UUID]]:
        """Distinct (action_cluster_id, to_obs_cluster_id) act_obs across paths."""
        act_obs: set[tuple[UUID, UUID]] = set()
        for p in self._paths:
            if not (self._msg_repo and p.action_message_id and p.to_observation_id):
                continue
            a_msg = self._msg_repo._by_id.get(p.action_message_id)
            o_msg = self._msg_repo._by_id.get(p.to_observation_id)
            if not a_msg or not o_msg:
                continue
            if a_msg.cluster_id is None or o_msg.cluster_id is None:
                continue
            act_obs.add((a_msg.cluster_id, o_msg.cluster_id))
        return list(act_obs)


@dataclass
class InMemoryTokenCluster:
    id: UUID
    cluster_id: int
    centroid: list[float]


class InMemoryTokenClusterRepository:
    """Fake replacement for TokenClusterRepository for unit tests."""

    def __init__(self) -> None:
        self._rows: list[InMemoryTokenCluster] = []

    async def delete_all(self) -> None:
        self._rows.clear()

    async def get_centroids(self) -> list[tuple[UUID, int, list[float]]]:
        return [(r.id, r.cluster_id, r.centroid) for r in self._rows]

    def add(self, cluster_id: int, centroid: list[float]) -> InMemoryTokenCluster:
        row = InMemoryTokenCluster(
            id=uuid4(), cluster_id=cluster_id, centroid=list(centroid),
        )
        self._rows.append(row)
        return row


class InMemoryTrajectoryWindowLSHRepository:
    """Fake replacement for TrajectoryWindowLSHRepository for unit
    tests. Mirrors the production ``lookup`` signature: temporal
    filter via ``[step_min, step_max]`` over ``window_center``, top
    ``top_uniq`` candidates per anchor by ``aggregation`` of band-hit
    counts (``"min_distance"`` → max similarity, ``"mean"`` → average).
    """

    def __init__(self) -> None:
        # (tid, window_center, band_index, band_hash)
        self._rows: list[tuple[UUID, int, int, int]] = []

    async def delete_for_trajectories(self, trajectory_ids: list[UUID]) -> None:
        wanted = set(trajectory_ids)
        self._rows = [r for r in self._rows if r[0] not in wanted]

    async def bulk_insert(
        self, rows: list[tuple[UUID, int, int, int]],
    ) -> None:
        seen = {(tid, wc, bi) for tid, wc, bi, _ in self._rows}
        for tid, wc, bi, bh in rows:
            if (tid, wc, bi) in seen:
                continue
            seen.add((tid, wc, bi))
            self._rows.append((tid, wc, bi, bh))

    async def lookup(
        self,
        band_pairs: list[tuple[int, int]],
        *,
        step_min: int,
        step_max: int,
        top_uniq: int,
        exclude_trajectory_id: UUID | None = None,
        aggregation: str = "mean",
    ) -> list[tuple[UUID, float]]:
        pairs = set(band_pairs)
        # (tid, wc) → band_count
        per_window: dict[tuple[UUID, int], int] = {}
        for tid, wc, bi, bh in self._rows:
            if (bi, bh) not in pairs:
                continue
            if not (step_min <= wc <= step_max):
                continue
            if exclude_trajectory_id is not None and tid == exclude_trajectory_id:
                continue
            per_window[(tid, wc)] = per_window.get((tid, wc), 0) + 1

        # Per-tid agg over windows in [step_min, step_max].
        per_tid: dict[UUID, list[int]] = {}
        for (tid, _wc), cnt in per_window.items():
            per_tid.setdefault(tid, []).append(cnt)

        ranked: list[tuple[UUID, float]] = []
        for tid, counts in per_tid.items():
            if aggregation == "min_distance":
                score = float(max(counts))
            else:  # mean
                score = sum(counts) / len(counts)
            ranked.append((tid, score))
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_uniq]


@dataclass
class InMemoryTokenMapping:
    id: UUID
    action_label: str
    observation_label: str
    action_cluster_id: UUID | None
    observation_cluster_id: UUID | None
    token_cluster_id: UUID | None


class InMemoryTokenMappingRepository:
    """Fake replacement for TokenMappingRepository for unit tests."""

    def __init__(self) -> None:
        self._rows: list[InMemoryTokenMapping] = []

    async def delete_all(self) -> None:
        self._rows.clear()

    async def find_by(self, **kwargs: Any) -> list[InMemoryTokenMapping]:
        out = []
        for r in self._rows:
            if all(getattr(r, k, None) == v for k, v in kwargs.items()):
                out.append(r)
        return out

    async def find_by_cluster_ids(
        self, action_cluster_id: UUID, observation_cluster_id: UUID,
    ) -> UUID | None:
        for r in self._rows:
            if (
                r.action_cluster_id == action_cluster_id
                and r.observation_cluster_id == observation_cluster_id
            ):
                return r.token_cluster_id
        return None

    def add(
        self,
        action_label: str,
        observation_label: str,
        action_cluster_id: UUID | None = None,
        observation_cluster_id: UUID | None = None,
        token_cluster_id: UUID | None = None,
    ) -> InMemoryTokenMapping:
        row = InMemoryTokenMapping(
            id=uuid4(),
            action_label=action_label,
            observation_label=observation_label,
            action_cluster_id=action_cluster_id,
            observation_cluster_id=observation_cluster_id,
            token_cluster_id=token_cluster_id,
        )
        self._rows.append(row)
        return row
