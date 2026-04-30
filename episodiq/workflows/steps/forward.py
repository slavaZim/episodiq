import json

import structlog

from episodiq.workflows.context import PendingResponse
from episodiq.workflows.steps.base import Step, StepResult

logger = structlog.stdlib.get_logger(__name__)


class ForwardStep(Step):
    """Forward to upstream, set pending_response. No embed/save."""

    step_id = "forward"
    has_timeout = False

    async def exec(self) -> StepResult:
        if self.ctx.pending_response is not None:
            logger.info("forward_upstream", provider=self.ctx.api_adapter.id, pending=True)
            return StepResult(passable=True)

        response = await self.ctx.api_adapter.forward(self.ctx.request, self.ctx.body)
        logger.info("forward_upstream", provider=self.ctx.api_adapter.id, status=response.status_code)

        if response.status_code == 200:
            body = json.loads(response.body)
            canonical_msg = self.ctx.api_adapter.extract_response_message(body)
            self.ctx.pending_response = PendingResponse(response, canonical_msg)
        else:
            self.ctx.pending_response = PendingResponse(response, None)

        if self.ctx.trajectory_id:
            self.ctx.api_adapter.trajectory_handler.apply_trajectory_id(
                response.headers, self.ctx.trajectory_id
            )

        return StepResult(passable=True)
