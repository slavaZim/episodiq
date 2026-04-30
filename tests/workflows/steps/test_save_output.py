"""Tests for SaveOutputStep."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from episodiq.api_adapters.base import CanonicalAssistantMessage, CanonicalToolCall, Role
from episodiq.workflows.context import Dependencies, Input, PendingResponse, WorkflowContext
from episodiq.workflows.steps.save_output import SaveOutputStep
from tests.conftest import mock_session_factory as create_mock_session_factory
from tests.helpers import MockEmbedder
from tests.in_memory_repos import InMemoryMessageRepository


def _make_context(**overrides) -> WorkflowContext:
    return WorkflowContext(
        input=Input(request=MagicMock(), body={}),
        dependencies=Dependencies(
            api_adapter=MagicMock(),
            embedder=MockEmbedder(),
            session_factory=create_mock_session_factory(),
            failsafe=True,
        ),
        **overrides,
    )


class TestSaveOutputStep:

    @pytest.fixture(autouse=True)
    def setup_repo(self, monkeypatch):
        self.msg_repo = InMemoryMessageRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.save_output.MessageRepository",
            lambda s: self.msg_repo,
        )

    async def test_saves_assistant_message(self):
        """Saves canonical assistant message without embedding."""
        tid = uuid4()
        canonical = CanonicalAssistantMessage.build(adapter_id="test", text="hello")
        ctx = _make_context(
            trajectory_id=tid,
            pending_response=PendingResponse(response=MagicMock(), canonical_msg=canonical),
        )

        result = await SaveOutputStep(ctx).exec()

        assert result.passable is True
        assert ctx.output_message is not None
        assert ctx.output_message.role == Role.ASSISTANT
        assert ctx.output_message.embedding is None
        assert ctx.output_message.id is not None

        assert len(self.msg_repo._messages) == 1
        saved = self.msg_repo._messages[0]
        assert saved.embedding is None
        assert saved.cluster_type == "action"
        assert saved.category == "text"

    async def test_saves_tool_call_message(self):
        """Assistant with tool_call gets category from tool name."""
        tid = uuid4()
        canonical = CanonicalAssistantMessage.build(
            adapter_id="test",
            tool_calls=[CanonicalToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
        )
        ctx = _make_context(
            trajectory_id=tid,
            pending_response=PendingResponse(response=MagicMock(), canonical_msg=canonical),
        )

        result = await SaveOutputStep(ctx).exec()

        assert result.passable is True
        saved = self.msg_repo._messages[0]
        assert saved.cluster_type == "action"
        assert saved.category == "bash"

    async def test_skip_no_pending_response(self):
        """No pending_response → passable=True, nothing saved."""
        ctx = _make_context(trajectory_id=uuid4(), pending_response=None)

        result = await SaveOutputStep(ctx).exec()

        assert result.passable is True
        assert ctx.output_message is None
        assert len(self.msg_repo._messages) == 0

    async def test_skip_no_canonical_msg(self):
        """pending_response without canonical_msg → passable=True, nothing saved."""
        ctx = _make_context(
            trajectory_id=uuid4(),
            pending_response=PendingResponse(response=MagicMock(), canonical_msg=None),
        )

        result = await SaveOutputStep(ctx).exec()

        assert result.passable is True
        assert ctx.output_message is None
        assert len(self.msg_repo._messages) == 0
