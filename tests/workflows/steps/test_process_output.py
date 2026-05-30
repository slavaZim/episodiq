"""Tests for ProcessOutputStep (deferred: embed + assign cluster)."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from episodiq.api_adapters.base import CanonicalAssistantMessage
from episodiq.workflows.context import Dependencies, Input, OutputMessage, WorkflowContext
from episodiq.workflows.steps.process_output import ProcessOutputStep
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


def _output_msg(text: str = "hello") -> OutputMessage:
    msg = CanonicalAssistantMessage.build(adapter_id="test", text=text)
    return OutputMessage(role=msg.role, content=msg.content, id=uuid4())


def _seed_neighbors(msg_repo: InMemoryMessageRepository, cluster: Cluster, n: int = 5):
    """Seed msg_repo with clustered assistant messages so KNN can vote."""
    emb = text_to_embedding("hello", 50)
    for i in range(n):
        m = Message(
            id=uuid4(), trajectory_id=uuid4(), role="assistant",
            content=[{"type": "text", "text": "hello"}], index=0,
            embedding=emb, cluster_id=cluster.id, category="text",
            cluster_type="action", cluster=cluster,
        )
        msg_repo.add_message(m)
        msg_repo._clusters[cluster.id] = cluster


class TestProcessOutputStep:

    @pytest.fixture(autouse=True)
    def setup_repo(self, monkeypatch):
        self.msg_repo = InMemoryMessageRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.process_output.MessageRepository",
            lambda s: self.msg_repo,
        )

    async def test_embeds_and_updates(self):
        """Embeds output message and updates DB record."""
        msg = _output_msg("hi there")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None)

        ctx = _make_context(trajectory_id=uuid4(), output_message=msg)
        result = await ProcessOutputStep(ctx).exec()

        assert result.passable is True
        assert msg.embedding is not None
        assert len(msg.embedding) == 50

    async def test_assigns_cluster(self):
        """Assigns cluster via KNN when neighbors exist."""
        cluster = Cluster(id=uuid4(), type="action", category="text", label="a:text:0")
        _seed_neighbors(self.msg_repo, cluster)

        msg = _output_msg("hello")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None, cluster_id=None)

        ctx = _make_context(trajectory_id=uuid4(), output_message=msg)
        result = await ProcessOutputStep(ctx).exec()

        assert result.passable is True
        updated = self.msg_repo._by_id[msg.id]
        assert updated.cluster_id == cluster.id

    async def test_no_cluster_without_neighbors(self):
        """No neighbors → cluster_id stays None."""
        msg = _output_msg("hello")
        self.msg_repo._by_id[msg.id] = MagicMock(id=msg.id, embedding=None, cluster_id=None)

        ctx = _make_context(trajectory_id=uuid4(), output_message=msg)
        result = await ProcessOutputStep(ctx).exec()

        assert result.passable is True
        updated = self.msg_repo._by_id[msg.id]
        assert updated.cluster_id is None

    async def test_skip_no_output_message(self):
        """No output message → passable=True."""
        ctx = _make_context(trajectory_id=uuid4(), output_message=None)
        result = await ProcessOutputStep(ctx).exec()
        assert result.passable is True
