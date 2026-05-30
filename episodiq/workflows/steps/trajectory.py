import json

import structlog

from episodiq.storage.postgres.repository import TrajectoryRepository
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class TrajectoryStep(Step):
    """Extract trajectory ID, ensure trajectory row exists, check status."""

    step_id = "trajectory"

    async def exec(self) -> StepResult:
        self.ctx.trajectory_id = self.ctx.api_adapter.trajectory_handler.get_trajectory_id(
            self.ctx.request,
            self.ctx.body,
        )

        meta_header = self.ctx.request.headers.get("X-Meta")
        if meta_header:
            try:
                self.ctx.request_meta = json.loads(meta_header)
            except (json.JSONDecodeError, TypeError):
                logger.warning("invalid_x_meta_header", value=meta_header)

        structlog.contextvars.bind_contextvars(trajectory_id=str(self.ctx.trajectory_id))

        async with self.ctx.session_factory() as session:
            repo = TrajectoryRepository(session)
            trajectory = await repo.find_or_create(self.ctx.trajectory_id)
            await session.commit()

        if trajectory.status != "pending":
            logger.warning(
                "trajectory_inactive",
                status=trajectory.status,
            )
            return StepResult(passable=False, reason=f"trajectory status is '{trajectory.status}'")

        return StepResult(passable=True)
