import asyncio

from episodiq.workflows.context import PendingResponse
from episodiq.workflows.steps.base import Step, StepResult


class FailStep(Step):
    """Always fails - for testing fallback behavior."""

    step_id = "fail"

    async def exec(self) -> StepResult:
        return StepResult(passable=False, reason="Intentional failure for testing")


class ErrorStep(Step):
    """Raises exception - for testing error handling."""

    step_id = "error"

    async def exec(self) -> StepResult:
        raise ValueError("Intentional error for testing")


class SlowStep(Step):
    """Sleeps before returning - for testing timeouts."""

    step_id = "slow"
    sleep_time: float = 1.0

    async def exec(self) -> StepResult:
        await asyncio.sleep(self.sleep_time)
        return StepResult(passable=True)


class ProtectedFallbackStep(Step):
    """Slow fallback - for testing shield behavior."""

    step_id = "protected_fallback"
    sleep_time: float = 0.3
    call_count: int = 0
    cancelled: bool = False

    async def exec(self) -> StepResult:
        from fastapi.responses import JSONResponse

        ProtectedFallbackStep.call_count += 1
        try:
            await asyncio.sleep(self.sleep_time)
            response = JSONResponse({"role": "assistant", "content": "Protected"})
            self.ctx.pending_response = PendingResponse(response)
            return StepResult(passable=True)
        except asyncio.CancelledError:
            ProtectedFallbackStep.cancelled = True
            raise
