import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
    transition_profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    profile_embed: Mapped[list[float] | None] = mapped_column(Vector(2000), nullable=True)
    index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace: Mapped[list] = mapped_column(JSONB, server_default="[]", nullable=False)
    trajectory_status: Mapped[str] = mapped_column(String(20), server_default="pending", nullable=False)
    fail_risk_action_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    fail_risk_transition_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    success_signal_action_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    success_signal_transition_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    loop_count: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
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
