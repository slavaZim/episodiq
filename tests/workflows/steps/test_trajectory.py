"""Tests for TrajectoryStep."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from episodiq.workflows.context import Dependencies, Input, WorkflowContext
from episodiq.workflows.steps.trajectory import TrajectoryStep
from tests.conftest import mock_session_factory as create_mock_session_factory
from tests.helpers import MockEmbedder
from tests.in_memory_repos import InMemoryTrajectoryRepository


def _make_context() -> WorkflowContext:
    adapter = MagicMock()
    adapter.trajectory_handler.get_trajectory_id = MagicMock(return_value=uuid4())

    request = MagicMock()
    request.headers = MagicMock()
    request.headers.get = MagicMock(return_value=None)

    return WorkflowContext(
        input=Input(request=request, body={}),
        dependencies=Dependencies(
            api_adapter=adapter,
            embedder=MockEmbedder(),
            session_factory=create_mock_session_factory(),
            failsafe=True,
        ),
    )


class TestTrajectoryStep:

    @pytest.fixture(autouse=True)
    def setup_repo(self, monkeypatch):
        self.traj_repo = InMemoryTrajectoryRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.trajectory.TrajectoryRepository",
            lambda s: self.traj_repo,
        )

    async def test_creates_trajectory_and_passes(self):
        """New trajectory is created with status='pending', step passes."""
        ctx = _make_context()
        step = TrajectoryStep(ctx)
        result = await step.exec()

        assert result.passable is True
        assert ctx.trajectory_id is not None
        assert ctx.trajectory_id in self.traj_repo._trajectories
        assert self.traj_repo._trajectories[ctx.trajectory_id].status == "pending"

    async def test_active_trajectory_passes(self):
        """Existing active trajectory passes."""
        ctx = _make_context()
        traj_id = ctx.dependencies.api_adapter.trajectory_handler.get_trajectory_id.return_value
        await self.traj_repo.find_or_create(traj_id)

        step = TrajectoryStep(ctx)
        result = await step.exec()

        assert result.passable is True

    async def test_completed_trajectory_returns_not_passable(self):
        """Trajectory with status='success' returns passable=False."""
        ctx = _make_context()
        traj_id = ctx.dependencies.api_adapter.trajectory_handler.get_trajectory_id.return_value
        await self.traj_repo.find_or_create(traj_id)
        await self.traj_repo.update_status(traj_id, "success")

        step = TrajectoryStep(ctx)
        result = await step.exec()

        assert result.passable is False
        assert "success" in result.reason

    async def test_failed_trajectory_returns_not_passable(self):
        """Trajectory with status='failure' returns passable=False."""
        ctx = _make_context()
        traj_id = ctx.dependencies.api_adapter.trajectory_handler.get_trajectory_id.return_value
        await self.traj_repo.find_or_create(traj_id)
        await self.traj_repo.update_status(traj_id, "failure")

        step = TrajectoryStep(ctx)
        result = await step.exec()

        assert result.passable is False
        assert "failure" in result.reason

    async def test_completed_trajectory_logs_warning(self):
        """Completed trajectory logs a warning with trajectory ID and status."""
        ctx = _make_context()
        traj_id = ctx.dependencies.api_adapter.trajectory_handler.get_trajectory_id.return_value
        await self.traj_repo.find_or_create(traj_id)
        await self.traj_repo.update_status(traj_id, "failure")

        step = TrajectoryStep(ctx)
        with patch("episodiq.workflows.steps.trajectory.logger") as mock_logger:
            await step.exec()
            mock_logger.warning.assert_called_once_with("trajectory_inactive", status="failure")
