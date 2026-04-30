"""Tests for SaveInputStep."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from episodiq.api_adapters.base import (
    CanonicalAssistantMessage,
    CanonicalUserMessage,
    CanonicalToolMessage,
)
from episodiq.workflows.context import Dependencies, Input, WorkflowContext
from episodiq.workflows.steps.save_input import SaveInputStep
from tests.conftest import mock_session_factory as create_mock_session_factory
from tests.helpers import MockEmbedder
from tests.in_memory_repos import InMemoryMessageRepository


def _make_context(*, messages, **overrides) -> WorkflowContext:
    adapter = MagicMock()
    adapter.extract_request_messages.return_value = messages
    return WorkflowContext(
        input=Input(request=MagicMock(), body={}),
        dependencies=Dependencies(
            api_adapter=adapter,
            embedder=MockEmbedder(),
            session_factory=create_mock_session_factory(),
            failsafe=True,
        ),
        **overrides,
    )


class TestSaveInputStep:

    @pytest.fixture(autouse=True)
    def setup_repo(self, monkeypatch):
        self.msg_repo = InMemoryMessageRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.save_input.MessageRepository",
            lambda s: self.msg_repo,
        )

    async def test_saves_user_message_with_cluster_type(self):
        """User text message → cluster_type='observation', category='text'."""
        tid = uuid4()
        ctx = _make_context(
            messages=[CanonicalUserMessage.build("hello")],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is True
        assert len(self.msg_repo._messages) == 1
        saved = self.msg_repo._messages[0]
        assert saved.cluster_type == "observation"
        assert saved.category == "text"
        assert saved.embedding is None

    async def test_saves_tool_message_with_cluster_type(self):
        """Tool response → cluster_type='observation', category from tool name."""
        tid = uuid4()
        ctx = _make_context(
            messages=[CanonicalToolMessage.build("c1", "bash", "ok")],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is True
        saved = self.msg_repo._messages[0]
        assert saved.cluster_type == "observation"
        assert saved.category == "bash"

    async def test_saves_multiple_messages(self):
        """Multiple new messages are all saved with correct cluster_type."""
        tid = uuid4()
        ctx = _make_context(
            messages=[
                CanonicalUserMessage.build("question"),
                CanonicalToolMessage.build("c1", "editor", "done"),
            ],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is True
        assert len(self.msg_repo._messages) == 2
        assert self.msg_repo._messages[0].cluster_type == "observation"
        assert self.msg_repo._messages[0].category == "text"
        assert self.msg_repo._messages[1].cluster_type == "observation"
        assert self.msg_repo._messages[1].category == "editor"

        assert len(ctx.input_messages) == 2

    async def test_deduplicates_already_saved(self):
        """Messages already in DB (by index) are skipped."""
        tid = uuid4()
        # Pre-populate: index 0 already saved
        existing = CanonicalUserMessage.build("old")
        await self.msg_repo.save(tid, existing, category="text", cluster_type="observation")

        ctx = _make_context(
            messages=[
                CanonicalUserMessage.build("old"),     # index 0 — skip
                CanonicalUserMessage.build("new"),     # index 1 — save
            ],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is True
        assert len(ctx.input_messages) == 1
        assert len(self.msg_repo._messages) == 2  # 1 existing + 1 new

    async def test_no_new_messages_returns_not_passable(self):
        """All messages already saved → passable=False."""
        tid = uuid4()
        existing = CanonicalUserMessage.build("old")
        await self.msg_repo.save(tid, existing, category="text", cluster_type="observation")

        ctx = _make_context(
            messages=[CanonicalUserMessage.build("old")],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is False
        assert "No new messages" in result.reason

    async def test_empty_messages_returns_not_passable(self):
        """No messages extracted → passable=False."""
        ctx = _make_context(messages=[], trajectory_id=uuid4())

        result = await SaveInputStep(ctx).exec()

        assert result.passable is False

    async def test_multi_turn_with_assistant_rejected(self):
        """New messages containing assistant role → passable=False."""
        tid = uuid4()
        ctx = _make_context(
            messages=[
                CanonicalUserMessage.build("hi"),
                CanonicalAssistantMessage.build(adapter_id="test", text="hello"),
                CanonicalUserMessage.build("follow up"),
            ],
            trajectory_id=tid,
        )

        result = await SaveInputStep(ctx).exec()

        assert result.passable is False
        assert "multi-turn" in result.reason
