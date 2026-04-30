from dataclasses import dataclass
from typing import Any
from uuid import UUID

from episodiq.analytics.transition_types import TrajectoryAnalytics

from fastapi import Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.api_adapters.base import (
    ApiAdapter,
    CanonicalAssistantMessage,
    CanonicalMessage,
)
from episodiq.inference.embedder import Embedder


@dataclass(frozen=True)
class Input:
    """Immutable input data for workflow."""
    request: Request
    body: dict[str, Any]


@dataclass(frozen=True)
class Dependencies:
    """Immutable dependencies for workflow."""
    api_adapter: ApiAdapter
    session_factory: async_sessionmaker[AsyncSession]
    embedder: Embedder
    failsafe: bool


@dataclass
class InputMessage(CanonicalMessage):
    """Canonical message enriched with DB id and embedding."""
    id: UUID = None
    embedding: list[float] | None = None


@dataclass
class OutputMessage(CanonicalMessage):
    """Assistant message enriched with DB id and embedding."""
    id: UUID = None
    embedding: list[float] | None = None


@dataclass
class PendingResponse:
    """Response awaiting persistence by ForwardStep."""
    response: Response
    canonical_msg: CanonicalAssistantMessage | None = None


@dataclass
class WorkflowContext:
    """Mutable context passed through pipeline steps."""

    input: Input
    dependencies: Dependencies

    # Populated by steps
    trajectory_id: UUID | None = None
    input_messages: list[InputMessage] | None = None

    # Request metadata from X-Meta header
    request_meta: dict | None = None

    # Pending response (set by steps for lazy ForwardStep)
    pending_response: PendingResponse | None = None

    # Output message (set by ProcessOutputStep)
    output_message: OutputMessage | None = None

    # Analytics result (set by BuildPathStep)
    analytics: TrajectoryAnalytics | None = None

    # Shortcuts
    @property
    def request(self) -> Request:
        return self.input.request

    @property
    def body(self) -> dict[str, Any]:
        return self.input.body

    @property
    def api_adapter(self) -> ApiAdapter:
        return self.dependencies.api_adapter

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self.dependencies.session_factory

    @property
    def embedder(self) -> Embedder:
        return self.dependencies.embedder

    @property
    def failsafe(self) -> bool:
        return self.dependencies.failsafe
