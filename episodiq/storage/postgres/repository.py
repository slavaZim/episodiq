from typing import Generic, NamedTuple, TypeVar
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, contains_eager, defer, joinedload

from episodiq.api_adapters.base import (
    CanonicalMessage,
    CanonicalAssistantMessage,
)
from episodiq.config import get_config
from episodiq.retrieval.minhash import max_pool_jaccard_per_key
from episodiq.storage.postgres.models import (
    Base,
    Cluster,
    Message,
    OriginResponse,
    TokenCluster,
    TokenMapping,
    Trajectory,
    TrajectoryPath,
)

ModelT = TypeVar("ModelT", bound=Base)

# fetch_similar dedup-by-trajectory paging: page size = limit * DEDUP_PAGE_FACTOR;
# pages through the HNSW nearest paths until `limit` distinct trajectories are
# collected, the index is exhausted, or DEDUP_MAX_PAGES is reached.
DEDUP_PAGE_FACTOR = 8
DEDUP_MAX_PAGES = 4


class ClusterNeighbor(NamedTuple):
    cluster_id: UUID
    distance: float


class BaseRepository(Generic[ModelT]):
    """Generic CRUD base for SQLAlchemy models with UUID primary key."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by(self, **kwargs) -> list[ModelT]:
        stmt = select(self.model).filter_by(**kwargs)
        return list((await self.session.execute(stmt)).scalars().all())

    async def create(self, **kwargs) -> ModelT:
        obj = self.model(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def update(self, id: UUID, **kwargs) -> None:
        await self.session.execute(
            update(self.model).where(self.model.id == id).values(**kwargs)
        )

    async def delete(self, id: UUID) -> None:
        await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )


class TrajectoryRepository(BaseRepository[Trajectory]):
    model = Trajectory

    async def find_or_create(self, trajectory_id: UUID) -> Trajectory:
        """Return existing trajectory or create a new one with status='pending'."""
        traj = await self.session.get(Trajectory, trajectory_id)
        if traj is None:
            traj = Trajectory(id=trajectory_id)
            self.session.add(traj)
            await self.session.flush()
        return traj

    async def get_with_completed_paths(
        self,
        status: str | list[str],
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Trajectory]:
        """Load trajectories by status with their completed paths eager-loaded.
        Results ordered by Trajectory.id for deterministic offset slicing
        (so tune/eval splits are reproducible).
        """
        path_filters = [TrajectoryPath.to_observation_id.isnot(None)]

        status_filter = (
            Trajectory.status.in_(status)
            if isinstance(status, (list, tuple, set))
            else Trajectory.status == status
        )

        # LIMIT on trajectory IDs, ordered for deterministic offset slicing.
        traj_ids_q = (
            select(Trajectory.id)
            .join(TrajectoryPath)
            .where(status_filter, *path_filters)
            .group_by(Trajectory.id)
            .order_by(Trajectory.id)
            .offset(offset)
        )
        if limit is not None:
            traj_ids_q = traj_ids_q.limit(limit)

        stmt = (
            select(Trajectory)
            .join(TrajectoryPath)
            .where(
                Trajectory.id.in_(traj_ids_q),
                *path_filters,
            )
            .options(
                contains_eager(Trajectory.paths)
                .joinedload(TrajectoryPath.action_message)
                .joinedload(Message.cluster),
            )
            .order_by(Trajectory.id)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().unique().all())


class ClusterRepository(BaseRepository[Cluster]):
    model = Cluster

    async def has_any(self) -> bool:
        """Check if any clusters exist."""
        result = await self.session.execute(
            select(func.count()).select_from(Cluster).limit(1)
        )
        return result.scalar_one() > 0

    async def delete_by_type_category(self, type: str, category: str) -> None:
        """Delete all clusters matching type + category."""
        await self.session.execute(
            delete(Cluster).where(Cluster.type == type, Cluster.category == category)
        )

    async def get_centroids(
        self, cluster_ids: set[UUID],
    ) -> list[tuple[UUID, str, list[float]]]:
        """Compute AVG(embedding) per cluster, cast back to vector(dims) so
        pgvector returns a parsed list of floats (AVG strips the Vector type)."""
        dims = get_config().message_dims
        stmt = (
            select(
                Cluster.id,
                Cluster.label,
                cast(func.avg(Message.embedding), Vector(dims)).label("centroid"),
            )
            .join(Message, Message.cluster_id == Cluster.id)
            .where(Cluster.id.in_(cluster_ids), Message.embedding.is_not(None))
            .group_by(Cluster.id, Cluster.label)
        )
        return list((await self.session.execute(stmt)).all())


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def get_max_index(self, trajectory_id: UUID) -> int | None:
        """Return the highest message index for a trajectory, or None if empty."""
        result = await self.session.execute(
            select(func.max(Message.index)).where(
                Message.trajectory_id == trajectory_id
            )
        )
        return result.scalar_one_or_none()

    async def save(
        self,
        trajectory_id: UUID,
        message: CanonicalMessage,
        embedding: list[float] | None = None,
        category: str | None = None,
        cluster_type: str | None = None,
    ) -> Message:
        msg = Message(
            trajectory_id=trajectory_id,
            role=message.role.value,
            content=message.content,
            embedding=embedding,
            category=category,
            cluster_type=cluster_type,
        )
        self.session.add(msg)
        await self.session.flush()

        if isinstance(message, CanonicalAssistantMessage):
            self.session.add(
                OriginResponse(
                    message_id=msg.id,
                    adapter_id=message.adapter_id,
                    external_id=message.external_id,
                    model=message.model,
                    input_tokens=message.usage.input_tokens if message.usage else None,
                    output_tokens=message.usage.output_tokens if message.usage else None,
                )
            )
        return msg

    async def get_messages_for_clustering(
        self, cluster_type: str, category: str,
    ) -> list[Message]:
        """Load messages with embeddings for a cluster_type + category pair.

        When ``skip_initial_observation`` is enabled and ``cluster_type`` is
        ``"observation"``, each trajectory's initial observation (the task
        prompt — the observation message with the lowest index in that
        trajectory) is excluded from the result.
        """
        stmt = (
            select(Message)
            .where(
                Message.cluster_type == cluster_type,
                Message.category == category,
                Message.embedding.is_not(None),
            )
        )

        if cluster_type == "observation" and get_config().skip_initial_observation:
            # Exclude each trajectory's initial observation (the task prompt):
            # keep an observation only if an EARLIER observation exists in the
            # same trajectory -- the lowest-index one has no such predecessor.
            inner = aliased(Message)
            earlier_observation = (
                select(inner.id)
                .where(
                    inner.trajectory_id == Message.trajectory_id,
                    inner.cluster_type == "observation",
                    inner.index < Message.index,
                )
            )
            stmt = stmt.where(earlier_observation.exists())

        return list((await self.session.execute(stmt)).scalars().all())

    async def get_distinct_categories(self, cluster_type: str) -> list[str]:
        """Return distinct non-null category values for a cluster_type."""
        stmt = (
            select(Message.category)
            .where(
                Message.cluster_type == cluster_type,
                Message.category.is_not(None),
            )
            .distinct()
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def find_neighbors(
        self,
        embedding: list[float],
        cluster_type: str,
        category: str,
        exclude_message_id: UUID,
        k: int = 10,
    ) -> list[ClusterNeighbor]:
        """Find k nearest clustered neighbors via pgvector cosine distance.

        Uses iterative_scan=strict_order so the HNSW index yields rows
        one-by-one until k pass the filter — no JOIN required.
        """
        dist = Message.embedding.cosine_distance(embedding)
        await self.session.execute(
            text("SET LOCAL hnsw.iterative_scan = strict_order")
        )
        stmt = (
            select(Message.cluster_id, dist.label("distance"))
            .where(
                Message.cluster_type == cluster_type,
                Message.category == category,
                Message.cluster_id.isnot(None),
                Message.id != exclude_message_id,
            )
            .order_by(dist)
            .limit(k)
        )
        rows = (await self.session.execute(stmt)).all()
        return [ClusterNeighbor(row.cluster_id, float(row.distance)) for row in rows]

    async def sample_by_cluster(self, cluster_id: UUID, n: int) -> list[Message]:
        """Sample n random messages from a cluster."""
        stmt = (
            select(Message)
            .where(Message.cluster_id == cluster_id)
            .order_by(func.random())
            .limit(n)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_distinct_trajectory_ids(self) -> list:
        """Return all distinct trajectory IDs that have messages."""
        result = await self.session.execute(
            select(Message.trajectory_id).distinct()
        )
        return list(result.scalars().all())

    async def get_trajectory_with_clusters(self, trajectory_id) -> list[Message]:
        """Load messages for a trajectory with cluster relationship, ordered by index."""
        result = await self.session.execute(
            select(Message)
            .options(joinedload(Message.cluster))
            .where(Message.trajectory_id == trajectory_id)
            .order_by(Message.index)
        )
        return list(result.scalars().unique().all())


class ActObsCluster(NamedTuple):
    a_cluster_id: UUID
    o_cluster_id: UUID


class TrajectoryPathRepository(BaseRepository[TrajectoryPath]):
    model = TrajectoryPath

    async def get_minhash_corpus(
        self,
    ) -> list[tuple[UUID, UUID | None, str, list[int]]]:
        """All retrieval-eligible paths: ``(trajectory_id,
        action_cluster_id, trajectory_status, minhash_sig)``. Used by the
        live prefilter and by offline sweeps that pre-load the corpus once.
        """
        stmt = (
            select(
                TrajectoryPath.trajectory_id,
                Message.cluster_id.label("action_cluster_id"),
                TrajectoryPath.trajectory_status,
                TrajectoryPath.minhash_sig,
            )
            .join(Message, Message.id == TrajectoryPath.action_message_id)
            .where(TrajectoryPath.minhash_sig.is_not(None))
            .where(TrajectoryPath.trajectory_status.in_(("success", "failure")))
        )
        result = await self.session.execute(stmt)
        return [
            (row.trajectory_id, row.action_cluster_id, row.trajectory_status,
             row.minhash_sig)
            for row in result
            if row.minhash_sig
        ]

    async def minhash_prefilter(
        self,
        query_signature: list[int],
        exclude_trajectory_id: UUID,
        limit: int,
        min_similarity: float = 0.0,
        corpus: list[tuple[UUID, UUID | None, str, list[int]]] | None = None,
    ) -> list[tuple[UUID, float, UUID | None, str]]:
        """MinHash Jaccard-estimate prefilter. Computes per-trajectory
        MAX-pool similarity between the query signature and the retrieval
        corpus, and returns the top ``limit`` rows ``(trajectory_id, sim,
        best_path_action_cluster_id, trajectory_status)`` with sim >=
        ``min_similarity``, sorted by sim descending. Excludes the query
        trajectory.

        If ``corpus`` is provided, it is used as-is (no DB round-trip);
        otherwise the full corpus is fetched. Callers that issue many
        searches against the same corpus should cache it and pass it in.
        """
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

    async def get_paths_with_tokens_for_trajectories(
        self, trajectory_ids: list[UUID],
    ) -> list[tuple[UUID, UUID | None, str, list[int]]]:
        """Fetch (trajectory_id, action_cluster_id, trajectory_status,
        trace_tokens) per path. Skips paths with NULL trace_tokens. action's
        cluster_id is taken from the joined action_message; trajectory_status
        is denormalized on TrajectoryPath.
        """
        if not trajectory_ids:
            return []
        stmt = (
            select(
                TrajectoryPath.trajectory_id,
                Message.cluster_id.label("action_cluster_id"),
                TrajectoryPath.trajectory_status,
                TrajectoryPath.trace_tokens,
            )
            .join(Message, Message.id == TrajectoryPath.action_message_id)
            .where(TrajectoryPath.trajectory_id.in_(trajectory_ids))
            .where(TrajectoryPath.trace_tokens.is_not(None))
        )
        result = await self.session.execute(stmt)
        return [
            (row.trajectory_id, row.action_cluster_id, row.trajectory_status,
             row.trace_tokens)
            for row in result
        ]

    async def collect_act_obs(self) -> list[ActObsCluster]:
        """Distinct (action_cluster_id, to_observation_cluster_id) act_obs over
        all completed trajectory_paths. Noise filtered server-side: both
        cluster_ids on the joined messages must be set.
        """
        Action = aliased(Message)
        ToObs = aliased(Message)
        stmt = (
            select(
                Action.cluster_id.label("a_cluster_id"),
                ToObs.cluster_id.label("o_cluster_id"),
            )
            .select_from(TrajectoryPath)
            .join(Action, Action.id == TrajectoryPath.action_message_id)
            .join(ToObs, ToObs.id == TrajectoryPath.to_observation_id)
            .where(Action.cluster_id.is_not(None))
            .where(ToObs.cluster_id.is_not(None))
            .distinct()
        )
        result = await self.session.execute(stmt)
        return [ActObsCluster(a, o) for a, o in result.all()]

    async def get_last(self, trajectory_id: UUID) -> TrajectoryPath | None:
        """Last path row for trajectory (most recent by created_at).

        Eager-loads from_observation, action_message, to_observation with
        their cluster relationships for label properties.
        """
        result = await self.session.execute(
            select(TrajectoryPath)
            .where(TrajectoryPath.trajectory_id == trajectory_id)
            .options(
                joinedload(TrajectoryPath.from_observation).joinedload(Message.cluster),
                joinedload(TrajectoryPath.action_message).joinedload(Message.cluster),
                joinedload(TrajectoryPath.to_observation).joinedload(Message.cluster),
            )
            .order_by(TrajectoryPath.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_trajectory_paths(self, trajectory_id: UUID) -> list[TrajectoryPath]:
        """All completed paths for a trajectory ordered by index."""
        result = await self.session.execute(
            select(TrajectoryPath)
            .where(
                TrajectoryPath.trajectory_id == trajectory_id,
                TrajectoryPath.to_observation_id.isnot(None),
            )
            .options(
                joinedload(TrajectoryPath.from_observation).joinedload(Message.cluster),
                joinedload(TrajectoryPath.action_message).joinedload(Message.cluster),
                joinedload(TrajectoryPath.to_observation).joinedload(Message.cluster),
                joinedload(TrajectoryPath.trajectory),
            )
            .order_by(TrajectoryPath.index)
        )
        return list(result.scalars().unique().all())

    async def delete_all(self) -> None:
        """Delete all trajectory path rows."""
        await self.session.execute(delete(TrajectoryPath))

    async def sync_trajectory_status(self) -> None:
        """Bulk-sync trajectory_status from trajectories table."""
        await self.session.execute(
            update(TrajectoryPath)
            .values(trajectory_status=Trajectory.status)
            .where(TrajectoryPath.trajectory_id == Trajectory.id)
        )

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
    ) -> TrajectoryPath:
        """Create path row. Pending if action/to_obs are None."""
        row = TrajectoryPath(
            trajectory_id=trajectory_id,
            from_observation_id=from_observation_id,
            action_message_id=action_message_id,
            to_observation_id=to_observation_id,
            data=data,
            trace=trace or [],
            trace_tokens=trace_tokens,
            minhash_sig=minhash_sig,
            trajectory_status=trajectory_status,
        )
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row, ["index"])
        return row

    async def get_cluster_info(self, message_id: UUID) -> tuple[UUID | None, str]:
        """Get (cluster_id, cluster_label) for a message (label falls back to
        'o:cat:?' / 'a:cat:?' when unclustered)."""
        result = await self.session.execute(
            select(Message)
            .options(joinedload(Message.cluster))
            .where(Message.id == message_id)
        )
        msg = result.scalar_one()
        return msg.cluster_id, msg.cluster_label

    async def fetch_similar(
        self,
        profile: list[float],
        exclude_trajectory_id: UUID,
        limit: int,
        defer_embed: bool = True,
    ) -> list[TrajectoryPath]:
        """HNSW L2 (euclidean) fetch, deduplicated to one path per trajectory.

        Pages through the HNSW nearest paths in batches of limit*DEDUP_PAGE_FACTOR,
        keeping the nearest path of each trajectory (first hit wins), until
        `limit` distinct trajectories are collected, the index is exhausted, or
        DEDUP_MAX_PAGES is reached. Returns the `limit` nearest distinct
        trajectories, eager-loading trajectory + action_message.cluster.

        Uses strict_order iterative scan: with no rerank stage the retrieval
        order is final, and the per-trajectory dedup ("first hit = nearest")
        plus OFFSET paging both require exact distance order.
        """
        dist = TrajectoryPath.profile.l2_distance(profile)
        where = (
            TrajectoryPath.trajectory_id != exclude_trajectory_id,
            TrajectoryPath.to_observation_id.isnot(None),
            TrajectoryPath.trajectory_status.in_(["success", "failure"]),
        )
        await self.session.execute(
            text("SET LOCAL hnsw.iterative_scan = strict_order")
        )
        page_size = limit * DEDUP_PAGE_FACTOR

        nearest: dict[UUID, UUID] = {}  # trajectory_id -> nearest path id
        for page in range(DEDUP_MAX_PAGES):
            offset = page * page_size
            # pgvector caps hnsw.ef_search at 1000.
            ef_search = min(1000, max(40, offset + page_size))
            await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
            rows = (
                await self.session.execute(
                    select(TrajectoryPath.id, TrajectoryPath.trajectory_id)
                    .where(*where)
                    .order_by(dist)
                    .offset(offset)
                    .limit(page_size)
                )
            ).all()
            if not rows:
                break
            for path_id, traj_id in rows:
                if traj_id not in nearest:
                    nearest[traj_id] = path_id
            if len(nearest) >= limit:
                break

        # `nearest` is insertion-ordered = distance-ordered -> first `limit` win
        path_ids = list(nearest.values())[:limit]
        if not path_ids:
            return []

        opts = [
            joinedload(TrajectoryPath.trajectory),
            joinedload(TrajectoryPath.action_message).joinedload(Message.cluster),
            joinedload(TrajectoryPath.from_observation),
        ]
        if defer_embed:
            opts.append(defer(TrajectoryPath.profile))
        entities = (
            await self.session.execute(
                select(TrajectoryPath)
                .where(TrajectoryPath.id.in_(path_ids))
                .options(*opts)
            )
        ).scalars().unique().all()
        by_id = {e.id: e for e in entities}
        return [by_id[pid] for pid in path_ids if pid in by_id]

    async def get_completed(
        self, limit: int | None = None, require_tokens: bool = False,
    ) -> list[TrajectoryPath]:
        """Load completed paths with relationships.

        Args:
            limit: Max rows to return. None = all.
            require_tokens: Filter to paths with trace_tokens IS NOT NULL.
        """
        filters = [TrajectoryPath.to_observation_id.isnot(None)]
        if require_tokens:
            filters.append(TrajectoryPath.trace_tokens.isnot(None))
        stmt = (
            select(TrajectoryPath)
            .options(
                joinedload(TrajectoryPath.trajectory),
                joinedload(TrajectoryPath.action_message).joinedload(Message.cluster),
            )
            .where(*filters)
            .order_by(TrajectoryPath.created_at)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list((await self.session.execute(stmt)).scalars().unique().all())


class TokenClusterRepository(BaseRepository[TokenCluster]):
    model = TokenCluster

    async def delete_all(self) -> None:
        await self.session.execute(delete(TokenCluster))

    async def get_centroids(self) -> list[tuple[UUID, int, list[float]]]:
        """All token clusters with their centroids. Used by TokenAssigner for
        online nearest-centroid lookup.
        """
        stmt = select(TokenCluster.id, TokenCluster.cluster_id, TokenCluster.centroid)
        return list((await self.session.execute(stmt)).all())


class TokenMappingRepository(BaseRepository[TokenMapping]):
    model = TokenMapping

    async def delete_all(self) -> None:
        await self.session.execute(delete(TokenMapping))

    async def find_by_cluster_ids(
        self, action_cluster_id: UUID, observation_cluster_id: UUID,
    ) -> UUID | None:
        """Direct lookup: (action_cluster_id, observation_cluster_id) →
        token_cluster_id (or None if not indexed).
        """
        stmt = select(TokenMapping.token_cluster_id).where(
            TokenMapping.action_cluster_id == action_cluster_id,
            TokenMapping.observation_cluster_id == observation_cluster_id,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
