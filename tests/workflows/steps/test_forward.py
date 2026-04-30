"""Tests for ForwardStep."""

import json
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

from fastapi.responses import Response

from episodiq.api_adapters.base import CanonicalAssistantMessage
from episodiq.workflows.context import Dependencies, Input, PendingResponse, WorkflowContext
from episodiq.workflows.steps.forward import ForwardStep
from tests.helpers import MockEmbedder


class TestForwardStep:

    async def test_forwards_and_sets_pending_response(self):
        """ForwardStep forwards to upstream and sets ctx.pending_response."""
        from episodiq.api_adapters.trajectory_handler import DefaultTrajectoryHandler

        api_adapter = MagicMock()
        api_adapter.forward = AsyncMock(return_value=Response(
            content=json.dumps({"choices": [{"message": {"role": "assistant", "content": "answer"}}]}),
            status_code=200,
        ))
        api_adapter.extract_response_message = MagicMock(return_value=CanonicalAssistantMessage.build(
            text="answer", adapter_id="test",
        ))
        api_adapter.trajectory_handler = DefaultTrajectoryHandler()

        ctx = WorkflowContext(
            input=Input(request=MagicMock(), body={}),
            dependencies=Dependencies(
                api_adapter=api_adapter,
                embedder=MockEmbedder(),
                session_factory=MagicMock(),
                failsafe=True,
            ),
        )
        ctx.trajectory_id = uuid4()

        step = ForwardStep(ctx)
        result = await step.exec()

        assert result.passable is True
        assert ctx.pending_response is not None
        assert ctx.pending_response.response.status_code == 200
        assert ctx.pending_response.canonical_msg is not None

    async def test_uses_existing_pending_response(self):
        """When ctx.pending_response is already set, ForwardStep returns without calling forward."""
        api_adapter = MagicMock()
        api_adapter.forward = AsyncMock()

        ctx = WorkflowContext(
            input=Input(request=MagicMock(), body={}),
            dependencies=Dependencies(
                api_adapter=api_adapter,
                embedder=MockEmbedder(),
                session_factory=MagicMock(),
                failsafe=True,
            ),
        )

        pending_response = Response(content=b'{"ok": true}', status_code=200)
        ctx.pending_response = PendingResponse(pending_response, CanonicalAssistantMessage.build(
            adapter_id="test", text="cached",
        ))

        step = ForwardStep(ctx)
        result = await step.exec()

        assert result.passable is True
        api_adapter.forward.assert_not_awaited()

    async def test_non_200_sets_pending_with_no_canonical(self):
        """Non-200 response sets pending_response with canonical_msg=None."""
        api_adapter = MagicMock()
        api_adapter.forward = AsyncMock(return_value=Response(
            content=b'{"error": "bad"}', status_code=500,
        ))
        from episodiq.api_adapters.trajectory_handler import DefaultTrajectoryHandler
        api_adapter.trajectory_handler = DefaultTrajectoryHandler()

        ctx = WorkflowContext(
            input=Input(request=MagicMock(), body={}),
            dependencies=Dependencies(
                api_adapter=api_adapter,
                embedder=MockEmbedder(),
                session_factory=MagicMock(),
                failsafe=True,
            ),
        )
        ctx.trajectory_id = uuid4()

        step = ForwardStep(ctx)
        result = await step.exec()

        assert result.passable is True
        assert ctx.pending_response.response.status_code == 500
        assert ctx.pending_response.canonical_msg is None
