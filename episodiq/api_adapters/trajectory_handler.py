"""Trajectory ID handling."""

import uuid
from typing import Any, Protocol

from fastapi import HTTPException, Request


class InvalidTrajectoryIDError(HTTPException):
    def __init__(self, value: str):
        super().__init__(
            status_code=400,
            detail=f"Invalid X-Trajectory-ID: '{value}' is not a valid UUID",
        )


def parse_trajectory_id(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise InvalidTrajectoryIDError(value)


class TrajectoryHandler(Protocol):
    """Protocol for trajectory ID extraction and injection."""

    def get_trajectory_id(self, request: Request, body: dict[str, Any]) -> uuid.UUID:
        """Extract or generate trajectory ID from request."""
        ...

    def apply_trajectory_id(self, headers: dict[str, str], trajectory_id: uuid.UUID) -> None:
        """Add trajectory ID to response headers (modifies in-place)."""
        ...


class DefaultTrajectoryHandler:
    """Default handler using X-Trajectory-ID header."""

    def get_trajectory_id(self, request: Request, body: dict[str, Any]) -> uuid.UUID:
        header = request.headers.get("X-Trajectory-ID")
        if header:
            return parse_trajectory_id(header)
        return uuid.uuid4()

    def apply_trajectory_id(self, headers: dict[str, str], trajectory_id: uuid.UUID) -> None:
        headers["X-Trajectory-ID"] = str(trajectory_id)
