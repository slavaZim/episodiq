import asyncio
from unittest.mock import MagicMock
from uuid import uuid4


from episodiq.workflows.context import (
    Dependencies,
    Input,
    WorkflowContext,
)
from episodiq.workflows.steps.base import Step, StepResult
from episodiq.workflows.trajectory_manager import TrajectoryManager
from tests.conftest import mock_session_factory


class RecorderStep(Step):
    """Records execution order."""

    step_id = "recorder"
    deferred = True
    log: list[str] = []

    async def exec(self) -> StepResult:
        RecorderStep.log.append(self.step_id)
        return StepResult(passable=True)


class ErrorDeferredStep(Step):
    """Raises exception in deferred execution."""

    step_id = "error_deferred"
    deferred = True

    async def exec(self) -> StepResult:
        raise RuntimeError("Deferred step failed")


class SlowDeferredStep(Step):
    """Takes too long — for timeout testing."""

    step_id = "slow_deferred"
    deferred = True

    async def exec(self) -> StepResult:
        await asyncio.sleep(5.0)
        return StepResult(passable=True)


class FailDeferredStep(Step):
    """Returns passable=False."""

    step_id = "fail_deferred"
    deferred = True

    async def exec(self) -> StepResult:
        return StepResult(passable=False, reason="Intentional deferred failure")


def _make_ctx(session_factory=None) -> WorkflowContext:
    sf = session_factory or mock_session_factory()
    ctx = WorkflowContext(
        input=Input(request=MagicMock(), body={}),
        dependencies=Dependencies(
            api_adapter=MagicMock(),
            session_factory=sf,
            embedder=MagicMock(),
            failsafe=True,
        ),
    )
    ctx.trajectory_id = uuid4()
    return ctx


class TestTrajectoryManager:
    async def test_executes_deferred_steps(self):
        """Deferred steps execute after push."""
        RecorderStep.log = []

        manager = TrajectoryManager()
        ctx = _make_ctx()

        manager.push(ctx.trajectory_id, ctx, [RecorderStep])
        await manager.shutdown()

        assert RecorderStep.log == ["recorder"]

    async def test_sequential_within_trajectory(self):
        """Multiple pushes to same trajectory execute in FIFO order."""
        log: list[int] = []

        class Step1(Step):
            step_id = "s1"
            deferred = True
            async def exec(self) -> StepResult:
                log.append(1)
                await asyncio.sleep(0.01)
                return StepResult(passable=True)

        class Step2(Step):
            step_id = "s2"
            deferred = True
            async def exec(self) -> StepResult:
                log.append(2)
                return StepResult(passable=True)

        manager = TrajectoryManager()
        ctx = _make_ctx()

        manager.push(ctx.trajectory_id, ctx, [Step1])
        manager.push(ctx.trajectory_id, ctx, [Step2])
        await manager.shutdown()

        assert log == [1, 2]

    async def test_parallel_across_trajectories(self):
        """Different trajectories execute in parallel."""
        started = []

        class TrackStep(Step):
            step_id = "track"
            deferred = True
            async def exec(self) -> StepResult:
                started.append(self.ctx.trajectory_id)
                await asyncio.sleep(0.05)
                return StepResult(passable=True)

        manager = TrajectoryManager()
        ctx1 = _make_ctx()
        ctx2 = _make_ctx()

        manager.push(ctx1.trajectory_id, ctx1, [TrackStep])
        manager.push(ctx2.trajectory_id, ctx2, [TrackStep])

        # Both should start before either finishes (parallel)
        await asyncio.sleep(0.02)
        assert len(started) == 2

        await manager.shutdown()

    async def test_error_marks_internal_error(self, session_factory):
        """Exception in deferred step marks trajectory as internal_error."""
        from episodiq.storage.postgres.repository import TrajectoryRepository

        ctx = _make_ctx(session_factory=session_factory)

        # Create trajectory in DB
        async with session_factory() as session:
            repo = TrajectoryRepository(session)
            await repo.find_or_create(ctx.trajectory_id)
            await session.commit()

        manager = TrajectoryManager()
        manager.push(ctx.trajectory_id, ctx, [ErrorDeferredStep])
        await manager.shutdown()

        # Verify trajectory is marked internal_error
        async with session_factory() as session:
            repo = TrajectoryRepository(session)
            trajs = await repo.find_by(id=ctx.trajectory_id)
            assert len(trajs) == 1
            assert trajs[0].status == "internal_error"

    async def test_timeout_marks_internal_error(self, session_factory):
        """Timeout in deferred step marks trajectory as internal_error."""
        from episodiq.storage.postgres.repository import TrajectoryRepository

        ctx = _make_ctx(session_factory=session_factory)

        async with session_factory() as session:
            repo = TrajectoryRepository(session)
            await repo.find_or_create(ctx.trajectory_id)
            await session.commit()

        manager = TrajectoryManager(postprocess_timeout=0.05)
        manager.push(ctx.trajectory_id, ctx, [SlowDeferredStep])
        await manager.shutdown()

        async with session_factory() as session:
            repo = TrajectoryRepository(session)
            trajs = await repo.find_by(id=ctx.trajectory_id)
            assert len(trajs) == 1
            assert trajs[0].status == "internal_error"

    async def test_passable_false_stops_chain(self):
        """Step returning passable=False stops remaining deferred steps."""
        log: list[str] = []

        class AfterFailStep(Step):
            step_id = "after_fail"
            deferred = True
            async def exec(self) -> StepResult:
                log.append("after_fail")
                return StepResult(passable=True)

        manager = TrajectoryManager()
        ctx = _make_ctx()

        manager.push(ctx.trajectory_id, ctx, [FailDeferredStep, AfterFailStep])
        await manager.shutdown()

        assert "after_fail" not in log

    async def test_shutdown_idempotent(self):
        """Shutdown on empty manager is a no-op."""
        manager = TrajectoryManager()
        await manager.shutdown()  # Should not raise
