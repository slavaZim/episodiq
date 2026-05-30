"""Per-trajectory FIFO queue for deferred step execution."""

import asyncio
from dataclasses import dataclass, field
from typing import Type
from uuid import UUID

import structlog

from episodiq.storage.postgres.repository import TrajectoryRepository
from episodiq.workflows.context import WorkflowContext
from episodiq.workflows.steps.base import Step

logger = structlog.stdlib.get_logger(__name__)


@dataclass
class DeferredJob:
    ctx: WorkflowContext
    steps: list[Type[Step]]


@dataclass
class TrajectorySlot:
    trajectory_id: UUID
    queue: asyncio.Queue[DeferredJob | None] = field(default_factory=asyncio.Queue)
    worker: asyncio.Task | None = field(default=None, repr=False)


class TrajectoryManager:
    """Manages per-trajectory async workers for deferred step execution.

    Each trajectory gets its own FIFO queue — turns within a trajectory
    execute sequentially, different trajectories run in parallel.
    """

    def __init__(self, postprocess_timeout: float = 30.0):
        self._slots: dict[UUID, TrajectorySlot] = {}
        self._timeout = postprocess_timeout

    def push(
        self,
        trajectory_id: UUID,
        ctx: WorkflowContext,
        steps: list[Type[Step]],
    ) -> None:
        if trajectory_id not in self._slots:
            slot = TrajectorySlot(trajectory_id=trajectory_id)
            slot.worker = asyncio.create_task(
                self._consume(slot),
                name=f"traj-{str(trajectory_id)[:8]}",
            )
            self._slots[trajectory_id] = slot

        self._slots[trajectory_id].queue.put_nowait(
            DeferredJob(ctx=ctx, steps=steps)
        )

    async def _consume(self, slot: TrajectorySlot) -> None:
        while True:
            job = await slot.queue.get()
            if job is None:
                break
            try:
                await asyncio.wait_for(
                    self._run_deferred(job, slot.trajectory_id),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "deferred_timeout",
                    trajectory_id=str(slot.trajectory_id),
                )
                await self._mark_internal_error(job, slot.trajectory_id)
            except Exception:
                logger.exception(
                    "deferred_failed",
                    trajectory_id=str(slot.trajectory_id),
                )
                await self._mark_internal_error(job, slot.trajectory_id)
            finally:
                slot.queue.task_done()

    async def _run_deferred(
        self,
        job: DeferredJob,
        trajectory_id: UUID,
    ) -> None:
        import time

        for step_cls in job.steps:
            step = step_cls(job.ctx)
            structlog.contextvars.bind_contextvars(step=step.step_id)
            t0 = time.monotonic()

            result = await step.exec()

            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "deferred_step_completed",
                step=step.step_id,
                duration_ms=duration_ms,
                passable=result.passable,
            )

            if not result.passable:
                logger.warning(
                    "deferred_step_aborted",
                    step=step.step_id,
                    reason=result.reason,
                )
                break

            if result.terminal:
                break

        structlog.contextvars.unbind_contextvars("step")

    async def _mark_internal_error(
        self,
        job: DeferredJob,
        trajectory_id: UUID,
    ) -> None:
        try:
            async with job.ctx.session_factory() as session:
                repo = TrajectoryRepository(session)
                await repo.update(trajectory_id, status="internal_error")
                await session.commit()
        except Exception:
            logger.exception(
                "failed_to_mark_internal_error",
                trajectory_id=str(trajectory_id),
            )

    async def shutdown(self) -> None:
        for slot in self._slots.values():
            await slot.queue.join()
            slot.queue.put_nowait(None)
        await asyncio.gather(
            *(s.worker for s in self._slots.values() if s.worker),
            return_exceptions=True,
        )
        self._slots.clear()
