from typing import Generic, NamedTuple, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager, defer, joinedload

from episodiq.api_adapters.base import (
    CanonicalMessage,
    CanonicalAssistantMessage,
)
from episodiq.storage.postgres.models import Base, Cluster, Message, OriginResponse, Trajectory, TrajectoryPath

ModelT = TypeVar("ModelT", bound=Base)


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
        status: str,
        limit: int | None = None,
        require_embed: bool = False,
    ) -> list[Trajectory]:
        """Load trajectories by status with their completed paths eager-loaded.

        Args:
            status: Trajectory status filter ("success", "failure").
            limit: Max number of trajectories. None = all.
            require_embed: Only include paths with profile_embed.
        """
        path_filters = [TrajectoryPath.to_observation_id.isnot(None)]
        if require_embed:
            path_filters.append(TrajectoryPath.profile_embed.isnot(None))

        # LIMIT on trajectory IDs, not joined rows
        traj_ids_q = (
            select(Trajectory.id)
            .join(TrajectoryPath)
            .where(Trajectory.status == status, *path_filters)
            .group_by(Trajectory.id)
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

    async def get_centroids(self, cluster_ids: set[UUID]) -> list[tuple[UUID, str, any]]:
        """Compute AVG(embedding) per cluster. Returns (cluster_id, label, centroid_raw)."""
        stmt = (
            select(
                Cluster.id,
                Cluster.label,
                func.avg(Message.embedding).label("centroid"),
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
        """Load messages with embeddings for a cluster_type + category pair."""
        stmt = (
            select(Message)
            .where(
                Message.cluster_type == cluster_type,
                Message.category == category,
                Message.embedding.is_not(None),
            )
        )
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


class TrajectoryPathRepository(BaseRepository[TrajectoryPath]):
    model = TrajectoryPath

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
    ) -> TrajectoryPath:
        """Create path row. Pending if action/to_obs are None."""
        row = TrajectoryPath(
            trajectory_id=trajectory_id,
            from_observation_id=from_observation_id,
            transition_profile=transition_profile,
            profile_embed=profile_embed,
            trace=trace or [],
            action_message_id=action_message_id,
            to_observation_id=to_observation_id,
            fail_risk_action_count=fail_risk_action_count,
            fail_risk_transition_count=fail_risk_transition_count,
            success_signal_action_count=success_signal_action_count,
            success_signal_transition_count=success_signal_transition_count,
            loop_count=loop_count,
            trajectory_status=trajectory_status,
        )
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row, ["index"])
        return row

    async def get_cluster_label(self, message_id: UUID) -> str:
        """Get cluster label for a message (with fallback for unclustered)."""
        result = await self.session.execute(
            select(Message)
            .options(joinedload(Message.cluster))
            .where(Message.id == message_id)
        )
        return result.scalar_one().cluster_label

    async def prefetch_similar(
        self,
        profile_embed: list[float],
        exclude_trajectory_id: UUID,
        limit: int,
        defer_embed: bool = True,
    ) -> list[TrajectoryPath]:
        """HNSW cosine prefetch. Eager-loads trajectory + action_message.cluster."""
        dist = TrajectoryPath.profile_embed.cosine_distance(profile_embed)
        opts = [
            joinedload(TrajectoryPath.trajectory),
            joinedload(TrajectoryPath.action_message).joinedload(Message.cluster),
            joinedload(TrajectoryPath.from_observation),
        ]
        if defer_embed:
            opts.append(defer(TrajectoryPath.profile_embed))
        ef_search = min(1000, max(40, limit * 2))
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        await self.session.execute(
            text("SET LOCAL hnsw.iterative_scan = relaxed_order")
        )
        stmt = (
            select(TrajectoryPath)
            .options(*opts)
            .where(
                TrajectoryPath.trajectory_id != exclude_trajectory_id,
                TrajectoryPath.to_observation_id.isnot(None),
                TrajectoryPath.trajectory_status.in_(["success", "failure"]),
            )
            .order_by(dist)
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().unique().all())

    async def get_completed(
        self, limit: int | None = None, require_embed: bool = False,
    ) -> list[TrajectoryPath]:
        """Load completed paths with relationships.

        Args:
            limit: Max rows to return. None = all.
            require_embed: Filter to paths with profile_embed IS NOT NULL.
        """
        filters = [TrajectoryPath.to_observation_id.isnot(None)]
        if require_embed:
            filters.append(TrajectoryPath.profile_embed.isnot(None))
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
