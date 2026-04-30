"""Tests for BuildPathStep."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from episodiq.analytics.transition_types import ActionSignal, LoopSignal, TrajectoryAnalytics
from episodiq.api_adapters.base import Role
from episodiq.workflows.context import Dependencies, Input, InputMessage, OutputMessage, WorkflowContext
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


def _cluster(type: str, category: str, label: str) -> Cluster:
    return Cluster(id=uuid4(), type=type, category=category, label=label)


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
    """Create InputMessage matching a DB Message."""
    im = InputMessage(role=Role(msg.role), content=[{"type": "text", "text": "hi"}])
    im.id = msg.id
    return im


def _output_msg(msg: Message) -> OutputMessage:
    """Create OutputMessage matching a DB Message."""
    om = OutputMessage(role=Role(msg.role), content=[{"type": "text", "text": "hello"}])
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

    def _add_cluster(self, cluster: Cluster) -> None:
        """Register cluster so has_any() returns True."""
        self.cluster_repo._clusters.append(cluster)

    async def test_skip_no_trajectory(self):
        """No trajectory_id → passable=True, no paths created."""
        ctx = self._make_context(trajectory_id=None, input_messages=[])
        result = await BuildPathStep(ctx).exec()

        assert result.passable is True
        assert len(self.path_repo._paths) == 0

    async def test_skip_no_input_messages(self):
        ctx = self._make_context(trajectory_id=uuid4(), input_messages=None)
        result = await BuildPathStep(ctx).exec()

        assert result.passable is True
        assert len(self.path_repo._paths) == 0

    async def test_skip_no_clusters(self):
        """No clusters in DB → skip path building entirely."""
        tid = uuid4()
        obs = _obs(tid, 0, category="text")
        act = _act(tid, 1, category="text")
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

    async def test_first_observation(self):
        """First obs → trace=[obs_label], no profile, no prev path to close."""
        tid = uuid4()
        c_obs = _cluster("observation", "text", "o:text:0")
        obs = _obs(tid, 0, c_obs)
        c_act = _cluster("action", "text", "a:text:0")
        act = _act(tid, 1, c_act)
        self._add_cluster(c_obs)
        self._add_cluster(c_act)
        self.msg_repo.add_message(obs)
        self.msg_repo.add_message(act)

        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs)],
            output_message=_output_msg(act),
        )

        result = await BuildPathStep(ctx).exec()

        assert result.passable is True
        assert len(self.path_repo._paths) == 1

        p = self.path_repo._paths[0]
        assert p.from_observation_id == obs.id
        assert p.action_message_id == act.id
        assert p.trace == ["o:text:0"]
        assert p.transition_profile is None
        assert p.profile_embed is None
        assert p.fail_risk_action_count == 0

    async def test_second_observation_closes_prev(self):
        """Second obs → closes prev path, creates new with profile."""
        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")
        for c in [c_o0, c_a0, c_o1, c_a1]:
            self._add_cluster(c)

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _obs(tid, 2, c_o1)
        act1 = _act(tid, 3, c_a1)
        for m in [obs0, act0, obs1, act1]:
            self.msg_repo.add_message(m)

        # Simulate first request
        ctx1 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs0)],
            output_message=_output_msg(act0),
        )
        await BuildPathStep(ctx1).exec()

        # Second request — should close prev path and create new with profile
        ctx2 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs1)],
            output_message=_output_msg(act1),
        )
        with patch(
            "episodiq.workflows.steps.build_path.TransitionAnalyzer.analyze",
            new_callable=AsyncMock,
            return_value=TrajectoryAnalytics(),
        ):
            await BuildPathStep(ctx2).exec()

        assert len(self.path_repo._paths) == 2

        # Previous path closed
        p0 = self.path_repo._paths[0]
        assert p0.to_observation_id == obs1.id

        # New path has profile from prev triplet
        p1 = self.path_repo._paths[1]
        assert p1.trace == ["o:text:0", "a:text:0", "o:text:1"]
        assert p1.transition_profile == {"o:text:0.a:text:0.o:text:1": 1.0}
        assert p1.profile_embed is not None
        assert len(p1.profile_embed) == 2000

    async def test_fail_risk_action_increments_count(self):
        """fail_risk_action detected → fail_risk_action_count incremented on new path."""
        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")
        for c in [c_o0, c_a0, c_o1, c_a1]:
            self._add_cluster(c)

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _obs(tid, 2, c_o1)
        act1 = _act(tid, 3, c_a1)
        for m in [obs0, act0, obs1, act1]:
            self.msg_repo.add_message(m)

        # First request
        ctx1 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs0)],
            output_message=_output_msg(act0),
        )
        await BuildPathStep(ctx1).exec()

        # Second request — analyzer detects fail_risk_action
        signal_analytics = TrajectoryAnalytics(
            fail_risk_action=ActionSignal(
                is_detected=True,
                similarity=0.5,
            ),
        )
        ctx2 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs1)],
            output_message=_output_msg(act1),
        )
        with patch(
            "episodiq.workflows.steps.build_path.TransitionAnalyzer.analyze",
            new_callable=AsyncMock,
            return_value=signal_analytics,
        ):
            await BuildPathStep(ctx2).exec()

        p1 = self.path_repo._paths[1]
        assert p1.fail_risk_action_count == 1
        assert ctx2.analytics is signal_analytics

    async def test_no_fail_risk_action_keeps_count(self):
        """No fail_risk_action → fail_risk_action_count stays same."""
        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")
        for c in [c_o0, c_a0, c_o1, c_a1]:
            self._add_cluster(c)

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _obs(tid, 2, c_o1)
        act1 = _act(tid, 3, c_a1)
        for m in [obs0, act0, obs1, act1]:
            self.msg_repo.add_message(m)

        ctx1 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs0)],
            output_message=_output_msg(act0),
        )
        await BuildPathStep(ctx1).exec()

        ctx2 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs1)],
            output_message=_output_msg(act1),
        )
        with patch(
            "episodiq.workflows.steps.build_path.TransitionAnalyzer.analyze",
            new_callable=AsyncMock,
            return_value=TrajectoryAnalytics(),
        ):
            await BuildPathStep(ctx2).exec()

        p1 = self.path_repo._paths[1]
        assert p1.fail_risk_action_count == 0

    async def test_loop_increments_count(self):
        """Loop detected → loop_count incremented on new path."""
        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")
        for c in [c_o0, c_a0, c_o1, c_a1]:
            self._add_cluster(c)

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _obs(tid, 2, c_o1)
        act1 = _act(tid, 3, c_a1)
        for m in [obs0, act0, obs1, act1]:
            self.msg_repo.add_message(m)

        ctx1 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs0)],
            output_message=_output_msg(act0),
        )
        await BuildPathStep(ctx1).exec()

        loop_analytics = TrajectoryAnalytics(
            loop_signal=LoopSignal(
                is_detected=True,
                transition="o:text:0.a:text:0.o:text:1",
                repeat_count=5,
                mean_success_repeat=1.0,
            ),
        )
        ctx2 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs1)],
            output_message=_output_msg(act1),
        )
        with patch(
            "episodiq.workflows.steps.build_path.TransitionAnalyzer.analyze",
            new_callable=AsyncMock,
            return_value=loop_analytics,
        ):
            await BuildPathStep(ctx2).exec()

        p1 = self.path_repo._paths[1]
        assert p1.loop_count == 1

    async def test_all_signal_counts_incremented(self):
        """All 5 signals detected → all counts incremented on new path."""
        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")
        for c in [c_o0, c_a0, c_o1, c_a1]:
            self._add_cluster(c)

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _obs(tid, 2, c_o1)
        act1 = _act(tid, 3, c_a1)
        for m in [obs0, act0, obs1, act1]:
            self.msg_repo.add_message(m)

        ctx1 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs0)],
            output_message=_output_msg(act0),
        )
        await BuildPathStep(ctx1).exec()

        all_signals = TrajectoryAnalytics(
            fail_risk_action=ActionSignal(is_detected=True, similarity=0.5),
            success_signal_action=ActionSignal(is_detected=True, similarity=-0.5),
            fail_risk_transition=True,
            success_signal_transition=True,
            loop_signal=LoopSignal(
                is_detected=True,
                transition="o:text:0.a:text:0.o:text:1",
                repeat_count=5,
                mean_success_repeat=1.0,
            ),
        )
        ctx2 = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs1)],
            output_message=_output_msg(act1),
        )
        with patch(
            "episodiq.workflows.steps.build_path.TransitionAnalyzer.analyze",
            new_callable=AsyncMock,
            return_value=all_signals,
        ):
            await BuildPathStep(ctx2).exec()

        p1 = self.path_repo._paths[1]
        assert p1.fail_risk_action_count == 1
        assert p1.fail_risk_transition_count == 1
        assert p1.success_signal_action_count == 1
        assert p1.success_signal_transition_count == 1
        assert p1.loop_count == 1

    async def test_unclustered_observation(self):
        """Unclustered obs → fallback label like o:text:?."""
        tid = uuid4()
        obs = Message(
            id=uuid4(), trajectory_id=tid, role="user", content=[], index=0,
            category="text",
        )
        c_act = _cluster("action", "text", "a:text:0")
        act = _act(tid, 1, c_act)
        self._add_cluster(c_act)
        self.msg_repo.add_message(obs)
        self.msg_repo.add_message(act)

        ctx = self._make_context(
            trajectory_id=tid,
            input_messages=[_input_msg(obs)],
            output_message=_output_msg(act),
        )

        result = await BuildPathStep(ctx).exec()

        assert result.passable is True
        assert self.path_repo._paths[0].trace == ["o:text:?"]

