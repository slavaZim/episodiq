"""Tests for ProcessInputStep (deferred: embed + assign cluster)."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from episodiq.api_adapters.base import CanonicalUserMessage
from episodiq.workflows.context import Dependencies, Input, InputMessage, WorkflowContext
from episodiq.workflows.steps.process_input import ProcessInputStep
from tests.conftest import mock_session_factory as create_mock_session_factory
from tests.helpers import MockEmbedder, text_to_embedding
from tests.in_memory_repos import Cluster, InMemoryMessageRepository, Message


def _make_context(**overrides) -> WorkflowContext:
    return WorkflowContext(
        input=Input(request=MagicMock(), body={}),
        dependencies=Dependencies(
            api_adapter=MagicMock(),
            embedder=MockEmbedder(embedding_fn=text_to_embedding, dims=50),
            session_factory=create_mock_session_factory(),
            failsafe=True,
        ),
        **overrides,
    )


def _input_msg(text: str = "hello") -> InputMessage:
    msg = CanonicalUserMessage.build(text)
    return InputMessage(role=msg.role, content=msg.content, id=uuid4())


def _seed_neighbors(msg_repo: InMemoryMessageRepository, cluster: Cluster, n: int = 5):
    """Seed msg_repo with clustered messages so KNN can vote."""
    emb = text_to_embedding("hello", 50)
    for i in range(n):
        m = Message(
            id=uuid4(), trajectory_id=uuid4(), role="user",
            content=[{"type": "text", "text": "hello"}], index=0,
            embedding=emb, cluster_id=cluster.id, category="text",
            cluster_type="observation", cluster=cluster,
        )
        msg_repo.add_message(m)
        msg_repo._clusters[cluster.id] = cluster


class TestProcessInputStep:

    @pytest.fixture(autouse=True)
    def setup_repo(self, monkeypatch):
        self.msg_repo = InMemoryMessageRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.process_input.MessageRepository",
            lambda s: self.msg_repo,
        )

    async def test_embeds_and_updates(self):
        """Embeds input message and updates DB record."""
        msg = _input_msg("hello world")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None)

        ctx = _make_context(trajectory_id=uuid4(), input_messages=[msg])
        result = await ProcessInputStep(ctx).exec()

        assert result.passable is True
        assert msg.embedding is not None
        assert len(msg.embedding) == 50

    async def test_assigns_cluster(self):
        """Assigns cluster via KNN when neighbors exist."""
        cluster = Cluster(id=uuid4(), type="observation", category="text", label="o:text:0")
        _seed_neighbors(self.msg_repo, cluster)

        msg = _input_msg("hello")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None, cluster_id=None)

        ctx = _make_context(trajectory_id=uuid4(), input_messages=[msg])
        result = await ProcessInputStep(ctx).exec()

        assert result.passable is True
        # Message should be assigned to the cluster
        updated = self.msg_repo._by_id[msg.id]
        assert updated.cluster_id == cluster.id

    async def test_no_cluster_without_neighbors(self):
        """No neighbors → cluster_id stays None."""
        msg = _input_msg("hello")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None, cluster_id=None)

        ctx = _make_context(trajectory_id=uuid4(), input_messages=[msg])
        result = await ProcessInputStep(ctx).exec()

        assert result.passable is True
        updated = self.msg_repo._by_id[msg.id]
        assert updated.cluster_id is None

    async def test_skip_no_input_messages(self):
        """No input messages → passable=True."""
        ctx = _make_context(trajectory_id=uuid4(), input_messages=[])
        result = await ProcessInputStep(ctx).exec()
        assert result.passable is True

    async def test_skip_none_input_messages(self):
        """None input messages → passable=True."""
        ctx = _make_context(trajectory_id=uuid4(), input_messages=None)
        result = await ProcessInputStep(ctx).exec()
        assert result.passable is True

    async def test_multiple_messages_all_embedded(self):
        """Multiple input messages: all get embeddings."""
        msgs = [_input_msg("first"), _input_msg("second")]
        for m in msgs:
            self.msg_repo._by_id[m.id] = MagicMock(id=m.id, embedding=None)

        ctx = _make_context(trajectory_id=uuid4(), input_messages=msgs)
        result = await ProcessInputStep(ctx).exec()

        assert result.passable is True
        assert all(m.embedding is not None for m in msgs)
