"""Episodiq management API router."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from episodiq.storage.postgres.models import Trajectory
from episodiq.storage.postgres.repository import TrajectoryRepository

router = APIRouter(prefix="/episodiq", tags=["episodiq"])


@router.get("/health")
async def health():
    return {"status": "ok"}


class UpdateTrajectoryRequest(BaseModel):
    status: Literal["success", "failure"]


@router.patch("/trajectories/{trajectory_id}")
async def update_trajectory(
    trajectory_id: UUID,
    body: UpdateTrajectoryRequest,
    request: Request,
):
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        traj = await session.get(Trajectory, trajectory_id)
        if not traj:
            raise HTTPException(404, "Trajectory not found")
        if traj.status in ("success", "failure"):
            raise HTTPException(409, f"Trajectory already marked as '{traj.status}'")
        repo = TrajectoryRepository(session)
        await repo.update(trajectory_id, status=body.status)
        await session.commit()
    return {"id": str(trajectory_id), "status": body.status}
