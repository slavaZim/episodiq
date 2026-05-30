"""Tests for BuildPathStep (close previous path, create new one with
incremental cluster-label trace)."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from episodiq.api_adapters.base import Role
from episodiq.workflows.context import (
    Dependencies,
    Input,
    InputMessage,
    OutputMessage,
    WorkflowContext,
)
from episodiq.workflows.steps.build_path import BuildPathStep
from tests.conftest import mock_session_factory as create_mock_session_factory
from tests.helpers import MockEmbedder
from tests.in_memory_repos import (
    Cluster,
    InMemoryClusterRepository,
    InMemoryMessageRepository,
    InMemoryTrajectoryPathRepository,
    Message,
)


def _cluster(type_: str, category: str, label: str) -> Cluster:
    return Cluster(id=uuid4(), type=type_, category=category, label=label)


def _obs(tid, index, cluster=None, category="text") -> Message:
    return Message(
        id=uuid4(), trajectory_id=tid, role="user", content=[], index=index,
        cluster=cluster, cluster_id=cluster.id if cluster else None,
        category=category,
    )


def _act(tid, index, cluster=None, category="text") -> Message:
    return Message(
        id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=index,
        cluster=cluster, cluster_id=cluster.id if cluster else None,
        category=category,
    )


def _input_msg(msg: Message) -> InputMessage:
    im = InputMessage(role=Role(msg.role), content=[{"type": "text", "text": "hi"}])
    im.id = msg.id
    return im


def _output_msg(msg: Message) -> OutputMessage:
    om = OutputMessage(role=Role(msg.role), content=[{"type": "text", "text": "ok"}])
    om.id = msg.id
    return om


class TestBuildPathStep:

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self.msg_repo = InMemoryMessageRepository()
        self.path_repo = InMemoryTrajectoryPathRepository(self.msg_repo)
        self.cluster_repo = InMemoryClusterRepository()
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.TrajectoryPathRepository",
            lambda s: self.path_repo,
        )
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.ClusterRepository",
            lambda s: self.cluster_repo,
        )
        # Stub TokenAssigner so close path doesn't hit the DB through
        # TokenMappingRepository / TokenClusterRepository on the mock session.
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.TokenMappingRepository",
            lambda s: MagicMock(),
        )
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.TokenClusterRepository",
            lambda s: MagicMock(),
        )
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.TokenAssigner.assign",
            AsyncMock(return_value=None),
        )

    def _make_context(self, **overrides) -> WorkflowContext:
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

    def _register_cluster(self, cluster: Cluster) -> None:
        """Force has_any() to return True."""
        self.cluster_repo._clusters.append(cluster)

    @pytest.mark.asyncio
    async def test_skip_no_trajectory(self):
        ctx = self._make_context(trajectory_id=None, input_messages=[])
        result = await BuildPathStep(ctx).exec()
        assert result.passable is True
        assert len(self.path_repo._paths) == 0

    @pytest.mark.asyncio
    async def test_skip_no_input_messages(self):
        ctx = self._make_context(trajectory_id=uuid4(), input_messages=None)
        result = await BuildPathStep(ctx).exec()
        assert result.passable is True
        assert len(self.path_repo._paths) == 0

    @pytest.mark.asyncio
    async def test_skip_no_clusters(self):
        """No clusters in DB → skip path building entirely."""
        tid = uuid4()
        obs = _obs(tid, 0)
        act = _act(tid, 1)
        self.msg_repo.add_message(obs)
        self.msg_repo.add_message(act)
        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs)],
            output_message=_output_msg(act),
        )
        result = await BuildPathStep(ctx).exec()
        assert result.passable is True
        assert len(self.path_repo._paths) == 0

    @pytest.mark.asyncio
    async def test_first_observation_creates_pending_path(self):
        tid = uuid4()
        c_obs = _cluster("observation", "text", "o:text:0")
        c_act = _cluster("action", "text", "a:text:0")
        self._register_cluster(c_obs)

        obs = _obs(tid, 0, c_obs)
        act = _act(tid, 1, c_act)
        self.msg_repo.add_message(obs)
        self.msg_repo.add_message(act)

        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs)],
            output_message=_output_msg(act),
        )
        result = await BuildPathStep(ctx).exec()
        assert result.passable is True
        assert result.terminal is True
        assert len(self.path_repo._paths) == 1
        path = self.path_repo._paths[0]
        assert path.trajectory_id == tid
        assert path.from_observation_id == obs.id
        assert path.action_message_id == act.id
        assert path.trace == ["o:text:0"]

    @pytest.mark.asyncio
    async def test_second_observation_closes_previous_path(self):
        tid = uuid4()
        c_obs0 = _cluster("observation", "text", "o:text:0")
        c_act0 = _cluster("action", "text", "a:text:0")
        c_obs1 = _cluster("observation", "text", "o:text:1")
        c_act1 = _cluster("action", "text", "a:text:1")
        for c in (c_obs0, c_act0, c_obs1, c_act1):
            self._register_cluster(c)

        first_obs = _obs(tid, 0, c_obs0)
        first_act = _act(tid, 1, c_act0)
        second_obs = _obs(tid, 2, c_obs1)
        second_act = _act(tid, 3, c_act1)
        for m in (first_obs, first_act, second_obs, second_act):
            self.msg_repo.add_message(m)

        # First request — pending path.
        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(first_obs)],
            output_message=_output_msg(first_act),
        )
        await BuildPathStep(ctx).exec()

        # Second request — should close previous and create another.
        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(second_obs)],
            output_message=_output_msg(second_act),
        )
        await BuildPathStep(ctx).exec()

        paths = sorted(self.path_repo._paths, key=lambda p: p.index)
        assert len(paths) == 2

        # Previous path closed (to_observation_id set) and its trace extended
        # to include its (action, to_observation) transition.
        assert paths[0].to_observation_id == second_obs.id
        assert paths[0].trace == ["o:text:0", "a:text:0", "o:text:1"]

        # New pending path carries the same cumulative trace forward.
        assert paths[1].from_observation_id == second_obs.id
        assert paths[1].trace == ["o:text:0", "a:text:0", "o:text:1"]

    @pytest.mark.asyncio
    async def test_close_writes_trace_tokens_and_minhash(self, monkeypatch):
        """When TokenAssigner returns an ordinal, the closed path gets cumulative
        trace_tokens + a fresh minhash_sig (once enough tokens accumulate to
        form n-grams). The new pending path inherits trace_tokens but never
        minhash_sig.
        """
        monkeypatch.setattr(
            "episodiq.workflows.steps.build_path.TokenAssigner.assign",
            AsyncMock(side_effect=[7, 8, 9]),
        )
        tid = uuid4()
        clusters = [
            _cluster("observation", "text", f"o:text:{i}") for i in range(4)
        ] + [
            _cluster("action", "text", f"a:text:{i}") for i in range(4)
        ]
        for c in clusters:
            self._register_cluster(c)

        msgs = []
        for i in range(4):
            o = _obs(tid, 2 * i, clusters[i])
            a = _act(tid, 2 * i + 1, clusters[4 + i])
            self.msg_repo.add_message(o)
            self.msg_repo.add_message(a)
            msgs.append((o, a))

        for o, a in msgs:
            ctx = self._make_context(
                trajectory_id=tid,
                input_messages=[_input_msg(o)],
                output_message=_output_msg(a),
            )
            await BuildPathStep(ctx).exec()

        paths = sorted(self.path_repo._paths, key=lambda p: p.index)
        # Three closes → cumulative tokens on each closed path.
        assert paths[0].trace_tokens == [7]
        assert paths[1].trace_tokens == [7, 8]
        assert paths[2].trace_tokens == [7, 8, 9]
        # n=3 ngrams need >= 3 tokens, so signature is computed only here.
        assert paths[0].minhash_sig is None
        assert paths[1].minhash_sig is None
        assert paths[2].minhash_sig is not None
        # Latest pending path inherits tokens but never minhash_sig.
        assert paths[3].trace_tokens == [7, 8, 9]
        assert paths[3].minhash_sig is None
