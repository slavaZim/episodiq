from typing import Generic, NamedTuple, TypeVar
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY, BigInteger, Column, Integer, MetaData, SmallInteger, Table,
    cast, delete, func, or_, select, text, update,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, contains_eager, joinedload

from episodiq.api_adapters.base import (
    CanonicalMessage,
    CanonicalAssistantMessage,
)
from episodiq.config import get_config
from episodiq.storage.postgres.models import (
    Base,
    Cluster,
    Message,
    OriginResponse,
    TokenCluster,
    TokenMapping,
    Trajectory,
    TrajectoryPath,
    TrajectoryWindowLSH,
)

ModelT = TypeVar("ModelT", bound=Base)

class ClusterNeighbor(NamedTuple):
    cluster_id: UUID
    distance: float


class Centroid(NamedTuple):
    """Average embedding for a clustered group OR a noise group.

    For HDBSCAN-clustered groups: ``cluster_id`` and ``label`` are set.
    For noise groups (cluster_id IS NULL on the constituent messages):
    ``cluster_id is None``, ``label is None``; identified by ``(cluster_type,
    category)``.
    """
    cluster_type: str
    category: str
    embedding: list[float]
    cluster_id: UUID | None = None
    label: str | None = None


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

    async def get_categories(self, type: str) -> list[str]:
        """Sorted distinct ``category`` values for clusters of ``type``.

        Stable order is required by the per-category noise encoding: a
        noise ordinal of ``-1 - i`` carries the i-th category from this
        list. Used by TokenAssigner after clusters are built; pre-cluster
        flows go through MessageRepository.get_categories.
        """
        stmt = (
            select(Cluster.category)
            .where(Cluster.type == type)
            .distinct()
            .order_by(Cluster.category)
        )
        return [row[0] for row in (await self.session.execute(stmt)).all()]

    async def get_centroids(self, cluster_ids: set[UUID]) -> list[Centroid]:
        """Compute AVG(embedding) per cluster. Casts back to vector(dims) so
        pgvector returns parsed floats (AVG strips the Vector type)."""
        dims = get_config().message_dims
        stmt = (
            select(
                Cluster.id,
                Cluster.type,
                Cluster.category,
                Cluster.label,
                cast(func.avg(Message.embedding), Vector(dims)).label("centroid"),
            )
            .join(Message, Message.cluster_id == Cluster.id)
            .where(Cluster.id.in_(cluster_ids), Message.embedding.is_not(None))
            .group_by(Cluster.id, Cluster.type, Cluster.category, Cluster.label)
        )
        return [
            Centroid(
                cluster_type=r.type,
                category=r.category,
                embedding=list(r.centroid),
                cluster_id=r.id,
                label=r.label,
            )
            for r in (await self.session.execute(stmt)).all()
        ]


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

        # Stable order so downstream UMAP+HDBSCAN are reproducible across
        # grid search and final clustering.
        stmt = stmt.order_by(Message.id)
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_categories(self, cluster_type: str) -> list[str]:
        """Sorted distinct non-null category values for a ``cluster_type``.

        Used pre-clustering (no Cluster rows yet); ClusterRepository.get_categories
        is the post-clustering equivalent.
        """
        stmt = (
            select(Message.category)
            .where(
                Message.cluster_type == cluster_type,
                Message.category.is_not(None),
            )
            .distinct()
            .order_by(Message.category)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get_category_centroids(self) -> list[Centroid]:
        """AVG(embedding) over message-level NOISE messages, grouped by
        (cluster_type, category). Returns one ``Centroid`` per category
        with ``cluster_id=None`` (noise) and ``label=None``.

        Used by ActObsBuilder to substitute a category-specific anchor
        for one-side-noise pairs in the clustering pool.
        """
        dims = get_config().message_dims
        stmt = (
            select(
                Message.cluster_type,
                Message.category,
                cast(func.avg(Message.embedding), Vector(dims)).label("centroid"),
            )
            .where(
                Message.cluster_id.is_(None),
                Message.embedding.is_not(None),
                Message.cluster_type.is_not(None),
                Message.category.is_not(None),
            )
            .group_by(Message.cluster_type, Message.category)
        )
        return [
            Centroid(
                cluster_type=ct,
                category=cat,
                embedding=list(centroid),
            )
            for ct, cat, centroid in (await self.session.execute(stmt)).all()
            if centroid is not None
        ]

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
    a_cluster_id: UUID | None
    o_cluster_id: UUID | None
    a_category: str | None
    o_category: str | None


class TrajectoryPathRepository(BaseRepository[TrajectoryPath]):
    model = TrajectoryPath

    async def get_latest_trace_tokens_for_trajectories(
        self, trajectory_ids: list[UUID],
    ) -> dict[UUID, tuple[UUID | None, str, list[int]]]:
        """``trajectory_id → (action_cluster_id, trajectory_status, tokens)``
        for the highest-index path with non-NULL ``trace_tokens`` per traj.

        Empty result for trajectories with no indexed path. Used by the
        cascade Stage-3 exact min-shift rerank.
        """
        if not trajectory_ids:
            return {}
        stmt = (
            select(
                TrajectoryPath.trajectory_id,
                Message.cluster_id.label("action_cluster_id"),
                TrajectoryPath.trajectory_status,
                TrajectoryPath.trace_tokens,
            )
            .distinct(TrajectoryPath.trajectory_id)
            .join(Message, Message.id == TrajectoryPath.action_message_id)
            .where(TrajectoryPath.trajectory_id.in_(trajectory_ids))
            .where(TrajectoryPath.trace_tokens.is_not(None))
            .order_by(TrajectoryPath.trajectory_id, TrajectoryPath.index.desc())
        )
        result = await self.session.execute(stmt)
        return {
            row.trajectory_id: (
                row.action_cluster_id, row.trajectory_status, list(row.trace_tokens),
            )
            for row in result
        }

    async def collect_act_obs(self) -> list[ActObsCluster]:
        """Distinct (action_cluster_id, to_observation_cluster_id) act_obs
        pairs over all completed trajectory_paths.

        Includes pairs where one side fell into message-level noise
        (cluster_id IS NULL), as long as the other side is clustered.
        The builder substitutes the per-category noise centroid for the
        NULL side so HDBSCAN sees those pairs as their own dense regions.

        Both-noise pairs are excluded — they perturbed HDBSCAN topology
        (~-0.03 rmean) without contributing signal; path_updater
        carry-forwards on unmapped pairs.

        Sorted by COUNT(*) DESC — density prior. NULL-side pairs sink to
        the end via NULLS LAST tie-break.
        """
        Action = aliased(Message)
        ToObs = aliased(Message)
        pair_count = func.count().label("pair_count")

        stmt = (
            select(
                Action.cluster_id.label("a_cluster_id"),
                ToObs.cluster_id.label("o_cluster_id"),
                Action.category.label("a_category"),
                ToObs.category.label("o_category"),
                pair_count,
            )
            .select_from(TrajectoryPath)
            .join(Action, Action.id == TrajectoryPath.action_message_id)
            .join(ToObs, ToObs.id == TrajectoryPath.to_observation_id)
            .where(
                or_(
                    Action.cluster_id.is_not(None),
                    ToObs.cluster_id.is_not(None),
                ),
                Action.category.is_not(None),
                ToObs.category.is_not(None),
            )
            .group_by(
                Action.cluster_id, ToObs.cluster_id,
                Action.category, ToObs.category,
            )
            .order_by(
                pair_count.desc(),
                Action.cluster_id.nulls_last(),
                ToObs.cluster_id.nulls_last(),
            )
        )
        result = await self.session.execute(stmt)
        return [
            ActObsCluster(
                a_cluster_id=r.a_cluster_id,
                o_cluster_id=r.o_cluster_id,
                a_category=r.a_category,
                o_category=r.o_category,
            )
            for r in result.all()
        ]

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
        trajectory_status: str = "pending",
        parallel_group: int | None = None,
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
            trajectory_status=trajectory_status,
            parallel_group=parallel_group,
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


def make_window_lsh_table(name: str) -> Table:
    """Build a SQLAlchemy Table mirroring the TrajectoryWindowLSH schema under
    a custom physical name. Used by sweep to spin up alt tables that the
    production repo can read/write without copying any query logic.
    """
    return Table(
        name,
        MetaData(),
        Column("trajectory_id", PGUUID(as_uuid=True), primary_key=True),
        Column("window_center", Integer, primary_key=True),
        Column("band_index", SmallInteger, primary_key=True),
        Column("band_hash", BigInteger, nullable=False),
    )


class TrajectoryWindowLSHRepository:
    """LSH band index repo. Defaults to the production
    ``trajectory_window_lsh`` table; pass ``table=make_window_lsh_table(...)``
    to point the same query logic at an alternative physical table
    (sweep use-case, isolated experiments).
    """

    def __init__(
        self, session: AsyncSession, *, table: Table | None = None,
    ) -> None:
        self.session = session
        self._table = table if table is not None else TrajectoryWindowLSH.__table__

    async def delete_for_trajectories(self, trajectory_ids: list[UUID]) -> None:
        if not trajectory_ids:
            return
        t = self._table
        await self.session.execute(
            delete(t).where(t.c.trajectory_id.in_(trajectory_ids)),
        )

    async def bulk_insert(
        self, rows: list[tuple[UUID, int, int, int]],
    ) -> None:
        """Insert ``(trajectory_id, window_center, band_index, band_hash)`` rows.

        Caller must dedupe (table primary key is the first three columns).
        """
        if not rows:
            return
        await self.session.execute(
            self._table.insert(),
            [
                dict(
                    trajectory_id=tid, window_center=wc,
                    band_index=bi, band_hash=bh,
                )
                for tid, wc, bi, bh in rows
            ],
        )

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
        """Top ``top_uniq`` candidate trajectories for one query anchor,
        ranked by ``aggregation`` (``"min_distance"`` or ``"mean"``) of band-hit
        counts over the anchor's neighborhood ``[step_min, step_max]``.
        Validation is the caller's responsibility (RetrievalConfig).
        """
        if not band_pairs:
            return []
        idx_arr = [int(bi) for bi, _ in band_pairs]
        hash_arr = [int(bh) for _, bh in band_pairs]
        t = self._table

        per_window = (
            select(
                t.c.trajectory_id.label("tid"),
                t.c.window_center.label("wc"),
                func.count().label("band_count"),
            )
            .where(
                func.row(t.c.band_index, t.c.band_hash).in_(
                    select(
                        func.unnest(cast(idx_arr, ARRAY(SmallInteger))).label("bi"),
                        func.unnest(cast(hash_arr, ARRAY(BigInteger))).label("bh"),
                    ),
                ),
                t.c.window_center >= step_min,
                t.c.window_center <= step_max,
            )
            .group_by(t.c.trajectory_id, t.c.window_center)
        )
        if exclude_trajectory_id is not None:
            per_window = per_window.where(
                t.c.trajectory_id != exclude_trajectory_id,
            )
        per_window = per_window.subquery()

        if aggregation == "min_distance":
            # Optimistic — pick the anchor whose match-count is highest
            # (= minimum distance) per candidate, mirroring the
            # agg-shift kernel's best-shift-wins semantics downstream.
            score = func.max(per_window.c.band_count)
        else:  # mean
            score = func.avg(cast(per_window.c.band_count, Integer))
        score = score.label("score")
        stmt = (
            select(per_window.c.tid, score)
            .group_by(per_window.c.tid)
            .order_by(score.desc())
            .limit(top_uniq)
        )
        result = await self.session.execute(stmt)
        return [(tid, float(s)) for tid, s in result.all()]


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
