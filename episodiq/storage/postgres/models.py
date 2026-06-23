import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Integer, SmallInteger, String, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from episodiq.config import get_config


_PREFIXES = {"observation": "o", "action": "a"}

MESSAGE_DIMS = get_config().message_dims


class Base(DeclarativeBase):
    pass


class Trajectory(Base):
    __tablename__ = "trajectories"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(20), server_default="pending")
    meta: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    paths: Mapped[list["TrajectoryPath"]] = relationship(back_populates="trajectory")


class Cluster(Base):
    __tablename__ = "clusters"
    __table_args__ = (
        CheckConstraint(
            "type IN ('action', 'observation')",
            name="ck_clusters_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    annotation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    trajectory_id: Mapped[uuid.UUID] = mapped_column(UUID)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[dict | list | str] = mapped_column(JSONB)
    index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(MESSAGE_DIMS), nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    cluster_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    cluster: Mapped["Cluster | None"] = relationship()
    origin_response: Mapped["OriginResponse | None"] = relationship(
        back_populates="message"
    )

    @property
    def cluster_label(self) -> str:
        """Cluster label or fallback like 'o:text:?' / 'a:bash:?'."""
        if self.cluster:
            return self.cluster.label
        prefix = _PREFIXES.get("observation" if self.role in ("user", "tool") else "action", "?")
        cat = self.category or "?"
        return f"{prefix}:{cat}:?"


class TrajectoryPath(Base):
    __tablename__ = "trajectory_paths"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("trajectories.id"), nullable=False
    )
    from_observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("messages.id"), nullable=False
    )
    action_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("messages.id"), nullable=True
    )
    to_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("messages.id"), nullable=True
    )
    # per-path signal metadata: running cummax, current failure score, loop counts
    data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Set on paths emitted from a parallel tool-call batch — all N paths in
    # the same assistant message share this group id (= the assistant
    # message's own ``index``). NULL for ordinary sequential paths. The
    # tokenizer sorts tokens within a group to make ordering invariant.
    parallel_group: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace: Mapped[list] = mapped_column(JSONB, server_default="[]", nullable=False)
    # per-position act_obs-level token cluster ids (one int per act_obs position)
    trace_tokens: Mapped[list[int] | None] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    trajectory_status: Mapped[str] = mapped_column(String(20), server_default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
    )

    trajectory: Mapped["Trajectory"] = relationship(back_populates="paths")
    from_observation: Mapped["Message"] = relationship(foreign_keys=[from_observation_id])
    action_message: Mapped["Message | None"] = relationship(foreign_keys=[action_message_id])
    to_observation: Mapped["Message | None"] = relationship(foreign_keys=[to_observation_id])

    @property
    def from_obs_label(self) -> str:
        return self.from_observation.cluster_label

    @property
    def action_label(self) -> str | None:
        return self.action_message.cluster_label if self.action_message else None

    @property
    def to_obs_label(self) -> str | None:
        return self.to_observation.cluster_label if self.to_observation else None


class TokenCluster(Base):
    """Token cluster — re-clustering of act_obs embeddings (concat of action
    and observation label centroids). The integer token used in
    trajectory_paths.trace_tokens.

    Distinct from message-level Cluster (which holds action/observation).
    """

    __tablename__ = "token_clusters"
    __table_args__ = (
        UniqueConstraint("cluster_id", name="uq_token_clusters_cluster_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    # ordinal cluster id (0..n_clusters-1), used as the integer token in trace_tokens
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    centroid: Mapped[list[float]] = mapped_column(
        Vector(2 * MESSAGE_DIMS), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
    )


class TokenMapping(Base):
    """Maps a concrete (action_label, observation_label) pair to its
    sep A/O message-level cluster ids and token cluster id.
    Replaces in-memory hdbscan_lookup.
    """

    __tablename__ = "token_mapping"
    __table_args__ = (
        UniqueConstraint(
            "action_label", "observation_label", name="uq_token_mapping_labels"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    action_label: Mapped[str] = mapped_column(String, nullable=False)
    observation_label: Mapped[str] = mapped_column(String, nullable=False)
    action_cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True
    )
    observation_cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("clusters.id", ondelete="SET NULL"), nullable=True
    )
    token_cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("token_clusters.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
    )


class OriginResponse(Base):
    __tablename__ = "origin_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID, ForeignKey("messages.id"), nullable=True
    )
    adapter_id: Mapped[str] = mapped_column(String)
    external_id: Mapped[str | None] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    message: Mapped["Message | None"] = relationship(back_populates="origin_response")


class TrajectoryWindowLSH(Base):
    """LSH band index over per-window MinHash signatures.

    One row per (trajectory, window_center, band_index). ``band_hash``
    is the rolled hash of the corresponding signature slice; the lookup
    index on ``(band_index, band_hash, window_center)`` powers fast
    band+temporal candidate retrieval.
    """
    __tablename__ = "trajectory_window_lsh"

    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("trajectories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    window_center: Mapped[int] = mapped_column(Integer, primary_key=True)
    band_index: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    band_hash: Mapped[int] = mapped_column(BigInteger, nullable=False)
