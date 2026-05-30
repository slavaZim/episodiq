import asyncio
import time
from typing import Any, Type

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.api_adapters.base import ApiAdapter
from episodiq.config import get_config
from episodiq.inference import Embedder
from episodiq.storage.postgres.repository import TrajectoryRepository
from episodiq.workflows.context import Dependencies, Input, WorkflowContext
from episodiq.workflows.steps.base import Step
from episodiq.workflows.trajectory_manager import TrajectoryManager

logger = structlog.stdlib.get_logger(__name__)


class Workflow:
    """Pipeline-based workflow executor."""

    def __init__(
        self,
        api_adapter: ApiAdapter,
        steps: list[Type[Step]],
        fallback_step: Type[Step],
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        trajectory_manager: TrajectoryManager | None = None,
        failsafe: bool = True,
    ):
        # Validate: deferred steps must come after sync steps
        seen_deferred = False
        for s in steps:
            if seen_deferred and not s.deferred:
                raise ValueError(
                    f"Sync step {s.step_id} after deferred step — "
                    f"deferred steps must be last"
                )
            seen_deferred = seen_deferred or s.deferred

        self.api_adapter = api_adapter
        self.steps = steps
        self.fallback_step = fallback_step
        self.failsafe = failsafe
        self._trajectory_manager = trajectory_manager
        self._dependencies = Dependencies(
            api_adapter=api_adapter,
            session_factory=session_factory,
            embedder=embedder,
            failsafe=failsafe,
        )

    async def run(self, request: Request, body: dict[str, Any]) -> Response:
        structlog.contextvars.clear_contextvars()
        start_time = time.monotonic()

        model = body.get("model", "unknown")
        structlog.contextvars.bind_contextvars(model=model)
        logger.info(
            "request_received",
            method=request.method,
            path=str(request.url.path),
        )

        ctx = WorkflowContext(
            input=Input(request=request, body=body),
            dependencies=self._dependencies,
        )

        try:
            result = await self._run(ctx)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("request_completed", status=result.status_code, duration_ms=duration_ms)
            return result
        except Exception as exc:
            if not self.failsafe:
                raise

            if isinstance(exc, asyncio.TimeoutError):
                logger.warning("workflow_timeout", exc_info=exc)
            else:
                logger.exception("workflow_error", exc_info=exc)

            await self._mark_internal_error(ctx)
            result = await self._fallback(ctx)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("request_completed", status=result.status_code, duration_ms=duration_ms, fallback=True)
            return result

    async def _run(self, ctx: WorkflowContext) -> Response:
        cfg = get_config()
        deadline = time.monotonic() + cfg.process_input_timeout
        deferred_queue: list[Type[Step]] = []

        for step_cls in self.steps:
            if step_cls.deferred:
                deferred_queue.append(step_cls)
                continue

            step = step_cls(ctx)
            structlog.contextvars.bind_contextvars(step=step.step_id)
            t0 = time.monotonic()

            if step.has_timeout:
                remaining = deadline - t0
                result = await asyncio.wait_for(step.exec(), timeout=remaining)
            else:
                result = await step.exec()

            duration_ms = int((time.monotonic() - t0) * 1000)

            if not result.passable:
                logger.info("step_completed", passable=False, duration_ms=duration_ms)
                if result.reason:
                    logger.warning("fallback_triggered", reason=result.reason)
                structlog.contextvars.unbind_contextvars("step")
                return await self._fallback(ctx)

            logger.info("step_completed", passable=True, duration_ms=duration_ms)

            if result.terminal:
                break

        structlog.contextvars.unbind_contextvars("step")

        if ctx.pending_response is None:
            raise RuntimeError("Pipeline completed without pending_response")

        if deferred_queue and self._trajectory_manager and ctx.trajectory_id:
            self._trajectory_manager.push(
                ctx.trajectory_id, ctx, deferred_queue,
            )

        return ctx.pending_response.response

    async def _fallback(self, ctx: WorkflowContext) -> Response:
        step = self.fallback_step(ctx)
        await asyncio.shield(step.exec())
        return ctx.pending_response.response

    async def _mark_internal_error(self, ctx: WorkflowContext) -> None:
        if ctx.trajectory_id is None:
            return
        try:
            async with ctx.session_factory() as session:
                repo = TrajectoryRepository(session)
                await repo.update(ctx.trajectory_id, status="internal_error")
                await session.commit()
        except Exception:
            logger.exception("failed_to_mark_internal_error")

    async def endpoint(self, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return Response(content="Invalid JSON", status_code=400)
        return await self.run(request, body)

    def build_router(self) -> APIRouter:
        router = APIRouter(prefix=self.api_adapter.mount_path, tags=[self.api_adapter.id])
        for route in self.api_adapter.routes:
            operation_id = f"{self.api_adapter.id}_{route.operation_id}"
            router.api_route(route.path, methods=route.methods, operation_id=operation_id)(self.endpoint)
        return router
