import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.responses import JSONResponse

from episodiq.workflows.base import Workflow
from episodiq.workflows.context import PendingResponse
from episodiq.workflows.steps.base import Step, StepResult
from tests.workflows.steps.fixtures import (
    FailStep,
    ErrorStep,
    SlowStep,
    ProtectedFallbackStep,
)


class PassStep(Step):
    """Always passes."""

    step_id = "pass"

    async def exec(self) -> StepResult:
        return StepResult(passable=True)


class SetResponseStep(Step):
    """Sets pending_response — simulates ForwardStep."""

    step_id = "set_response"
    timeout_group = None

    async def exec(self) -> StepResult:
        response = JSONResponse({"role": "assistant", "content": "Success"})
        self.ctx.pending_response = PendingResponse(response)
        return StepResult(passable=True)


class FallbackStep(Step):
    """Simple fallback that sets pending_response."""

    step_id = "test_fallback"

    async def exec(self) -> StepResult:
        response = JSONResponse({"role": "assistant", "content": "Fallback"})
        self.ctx.pending_response = PendingResponse(response)
        return StepResult(passable=True)


class DeferredRecorderStep(Step):
    """Deferred step that records execution."""

    step_id = "deferred_recorder"
    deferred = True
    executed: bool = False

    async def exec(self) -> StepResult:
        DeferredRecorderStep.executed = True
        return StepResult(passable=True)


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.forward = AsyncMock(return_value=JSONResponse({"role": "assistant", "content": "Forwarded"}))
    return adapter


@pytest.fixture
def mock_request():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    return MagicMock()


class TestWorkflow:
    async def test_passable_false_triggers_fallback(self, mock_adapter, mock_embedder, mock_request):
        """When step returns passable=False, fallback is executed."""
        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, FailStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Fallback"}'

    async def test_exception_triggers_fallback(self, mock_adapter, mock_embedder, mock_request):
        """When step raises exception, fallback is executed."""
        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, ErrorStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Fallback"}'

    async def test_timeout_triggers_fallback(self, mock_adapter, mock_embedder, mock_request, monkeypatch):
        """When pipeline times out, fallback is executed."""
        mock_cfg = MagicMock()
        mock_cfg.process_input_timeout = 0.1
        monkeypatch.setattr("episodiq.workflows.base.get_config", lambda: mock_cfg)

        SlowStep.sleep_time = 1.0

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, SlowStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Fallback"}'

    async def test_successful_pipeline(self, mock_adapter, mock_embedder, mock_request):
        """When all steps pass and pending_response is set, response is returned."""
        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, SetResponseStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Success"}'

    async def test_failsafe_false_reraises_exception(self, mock_adapter, mock_embedder, mock_request):
        """With failsafe=False, exceptions are re-raised."""
        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, ErrorStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
            failsafe=False,
        )

        with pytest.raises(ValueError, match="Intentional error"):
            await workflow.run(mock_request, {})

    async def test_failsafe_false_reraises_timeout(self, mock_adapter, mock_embedder, mock_request, monkeypatch):
        """With failsafe=False, timeout is re-raised."""
        mock_cfg = MagicMock()
        mock_cfg.process_input_timeout = 0.1
        monkeypatch.setattr("episodiq.workflows.base.get_config", lambda: mock_cfg)

        SlowStep.sleep_time = 1.0

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, SlowStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
            failsafe=False,
        )

        with pytest.raises(asyncio.TimeoutError):
            await workflow.run(mock_request, {})

    async def test_fallback_error_propagates(self, mock_adapter, mock_embedder, mock_request):
        """When fallback itself fails, error propagates."""
        class BrokenFallbackStep(Step):
            step_id = "broken_fallback"

            async def exec(self) -> StepResult:
                raise RuntimeError("Fallback failed")

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[ErrorStep],
            fallback_step=BrokenFallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
            failsafe=True,
        )

        with pytest.raises(RuntimeError, match="Fallback failed"):
            await workflow.run(mock_request, {})

    async def test_fallback_not_called_twice(self, mock_adapter, mock_embedder, mock_request):
        """When fallback fails, it is not retried."""
        call_count = 0

        class CountingFallbackStep(Step):
            step_id = "counting_fallback"

            async def exec(self) -> StepResult:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Fallback failed")

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[ErrorStep],
            fallback_step=CountingFallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
            failsafe=True,
        )

        with pytest.raises(RuntimeError):
            await workflow.run(mock_request, {})

        assert call_count == 1

    async def test_protected_fallback_not_cancelled(self, mock_adapter, mock_embedder, mock_request):
        """Protected fallback step completes without cancellation."""
        ProtectedFallbackStep.call_count = 0
        ProtectedFallbackStep.cancelled = False

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[FailStep],
            fallback_step=ProtectedFallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Protected"}'
        assert ProtectedFallbackStep.call_count == 1
        assert ProtectedFallbackStep.cancelled is False

    async def test_no_pending_response_raises(self, mock_adapter, mock_embedder, mock_request):
        """Pipeline without pending_response raises RuntimeError."""
        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
            failsafe=False,
        )

        with pytest.raises(RuntimeError, match="pending_response"):
            await workflow.run(mock_request, {})

    async def test_terminal_step_stops_sync_pipeline(self, mock_adapter, mock_embedder, mock_request):
        """Step returning terminal=True stops sync processing."""
        class TerminalResponseStep(Step):
            step_id = "terminal_response"
            timeout_group = None

            async def exec(self) -> StepResult:
                response = JSONResponse({"role": "assistant", "content": "Terminal"})
                self.ctx.pending_response = PendingResponse(response)
                return StepResult(passable=True, terminal=True)

        after_called = False

        class AfterTerminalStep(Step):
            step_id = "after_terminal"

            async def exec(self) -> StepResult:
                nonlocal after_called
                after_called = True
                return StepResult(passable=True)

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[PassStep, TerminalResponseStep, AfterTerminalStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Terminal"}'
        assert after_called is False

    async def test_deferred_steps_skipped_in_sync(self, mock_adapter, mock_embedder, mock_request):
        """Deferred steps are not executed synchronously without TrajectoryManager."""
        DeferredRecorderStep.executed = False

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[SetResponseStep, DeferredRecorderStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert DeferredRecorderStep.executed is False

    async def test_deferred_validation_rejects_sync_after_deferred(self, mock_adapter, mock_embedder):
        """Sync step after deferred step raises ValueError at construction."""
        with pytest.raises(ValueError, match="deferred steps must be last"):
            Workflow(
                api_adapter=mock_adapter,
                steps=[DeferredRecorderStep, PassStep],
                fallback_step=FallbackStep,
                session_factory=MagicMock(),
                embedder=mock_embedder,
            )

    async def test_deadline_shared_within_preprocess(
        self, mock_adapter, mock_embedder, mock_request, monkeypatch,
    ):
        """Two PREPROCESS steps share one deadline."""
        mock_cfg = MagicMock()
        mock_cfg.process_input_timeout = 0.15
        monkeypatch.setattr("episodiq.workflows.base.get_config", lambda: mock_cfg)

        class SlowPreprocess1(Step):
            step_id = "slow_pre_1"

            async def exec(self) -> StepResult:
                await asyncio.sleep(0.1)
                return StepResult(passable=True)

        class SlowPreprocess2(Step):
            step_id = "slow_pre_2"

            async def exec(self) -> StepResult:
                await asyncio.sleep(0.1)
                return StepResult(passable=True)

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[SlowPreprocess1, SlowPreprocess2],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Fallback"}'

    async def test_no_timeout_step_skips_deadline(
        self, mock_adapter, mock_embedder, mock_request, monkeypatch,
    ):
        """Step with has_timeout=False is not subject to deadline."""
        mock_cfg = MagicMock()
        mock_cfg.process_input_timeout = 0.05
        monkeypatch.setattr("episodiq.workflows.base.get_config", lambda: mock_cfg)

        class SlowNoTimeoutStep(Step):
            step_id = "slow_no_timeout"
            has_timeout = False

            async def exec(self) -> StepResult:
                await asyncio.sleep(0.1)
                response = JSONResponse({"role": "assistant", "content": "Done"})
                self.ctx.pending_response = PendingResponse(response)
                return StepResult(passable=True)

        workflow = Workflow(
            api_adapter=mock_adapter,
            steps=[SlowNoTimeoutStep],
            fallback_step=FallbackStep,
            session_factory=MagicMock(),
            embedder=mock_embedder,
        )

        response = await workflow.run(mock_request, {})

        assert response.status_code == 200
        assert response.body == b'{"role":"assistant","content":"Done"}'
