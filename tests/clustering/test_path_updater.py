"""Tests for TrajectoryPathUpdater."""

from unittest.mock import patch
from uuid import uuid4

import pytest

from episodiq.analytics.path_state import PathStateCalculator
from episodiq.analytics.transition_types import ActionSignal, LoopSignal, TrajectoryAnalytics
from episodiq.clustering.path_updater import TrajectoryPathUpdater
from tests.in_memory_repos import (
    Cluster,
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


def _tool_obs(tid, index, cluster=None, category="bash") -> Message:
    return Message(
        id=uuid4(), trajectory_id=tid, role="tool", content=[], index=index,
        cluster=cluster, cluster_id=cluster.id if cluster else None,
        category=category,
    )


def _act(tid, index, cluster=None, category="text") -> Message:
    return Message(
        id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=index,
        cluster=cluster, cluster_id=cluster.id if cluster else None,
        category=category,
    )


class TestTrajectoryPathUpdater:

    async def test_single_observation(self):
        """One observation → one path, no profile, trace = [obs_label]."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_obs = _cluster("observation", "text", "o:text:0")
        obs = _obs(tid, 0, c_obs)
        msg_repo.add_message(obs)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        assert total == 1
        p = path_repo._paths[0]
        assert p.from_observation_id == obs.id
        assert p.action_message_id is None
        assert p.to_observation_id is None
        assert p.trace == ["o:text:0"]
        assert p.transition_profile is None

    async def test_one_full_step_text(self):
        """obs0 → act → obs1 (text): two paths, second has transition profile."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")

        obs0 = _obs(tid, 0, c_o0)
        act = _act(tid, 1, c_a)
        obs1 = _obs(tid, 2, c_o1)
        for m in [obs0, act, obs1]:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        assert total == 2

        p0 = path_repo._paths[0]
        assert p0.from_observation_id == obs0.id
        assert p0.action_message_id == act.id
        assert p0.to_observation_id == obs1.id
        assert p0.trace == ["o:text:0"]
        assert p0.transition_profile is None

        p1 = path_repo._paths[1]
        assert p1.from_observation_id == obs1.id
        assert p1.trace == ["o:text:0", "a:text:0", "o:text:1"]
        assert p1.transition_profile == {"o:text:0.a:text:0.o:text:1": 1.0}

    async def test_one_full_step_tool(self):
        """obs0 → tool_call → tool_response: works with tool categories."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "bash", "a:bash:0")
        c_o1 = _cluster("observation", "bash", "o:bash:0")

        obs0 = _obs(tid, 0, c_o0)
        act = _act(tid, 1, c_a, category="bash")
        obs1 = _tool_obs(tid, 2, c_o1, category="bash")
        for m in [obs0, act, obs1]:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        assert total == 2

        p1 = path_repo._paths[1]
        assert p1.trace == ["o:text:0", "a:bash:0", "o:bash:0"]
        assert p1.transition_profile == {"o:text:0.a:bash:0.o:bash:0": 1.0}

    async def test_incremental_trace_and_arriving_transition(self):
        """Each new obs extends trace; profile records the transition arriving at it, not leaving."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "bash", "o:bash:0")
        c_a1 = _cluster("action", "bash", "a:bash:0")
        c_o2 = _cluster("observation", "text", "o:text:1")

        obs0 = _obs(tid, 0, c_o0)
        act0 = _act(tid, 1, c_a0)
        obs1 = _tool_obs(tid, 2, c_o1, category="bash")
        act1 = _act(tid, 3, c_a1, category="bash")
        obs2 = _obs(tid, 4, c_o2)
        for m in [obs0, act0, obs1, act1, obs2]:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        # p0: obs0 — no prior history
        p0 = path_repo._paths[0]
        assert p0.trace == ["o:text:0"]
        assert p0.transition_profile is None

        # p1: obs1 — has the transition that arrived here (o0→a0→o1)
        p1 = path_repo._paths[1]
        assert p1.trace == ["o:text:0", "a:text:0", "o:bash:0"]
        assert p1.transition_profile == {"o:text:0.a:text:0.o:bash:0": 1.0}

        # p2: obs2 — profile has both transitions, first one decayed
        p2 = path_repo._paths[2]
        assert p2.trace == ["o:text:0", "a:text:0", "o:bash:0", "a:bash:0", "o:text:1"]
        t_first = "o:text:0.a:text:0.o:bash:0"
        t_second = "o:bash:0.a:bash:0.o:text:1"
        assert p2.transition_profile[t_first] == pytest.approx(0.8)
        assert p2.transition_profile[t_second] == pytest.approx(1.0)

    async def test_decay_on_repeated_transitions(self):
        """Same transition repeated accumulates with decay."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        msgs = [
            _obs(tid, 0, c_o),
            _act(tid, 1, c_a),
            _obs(tid, 2, c_o),
            _act(tid, 3, c_a),
            _obs(tid, 4, c_o),
        ]
        for m in msgs:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        p2 = path_repo._paths[2]
        key = "o:text:0.a:text:0.o:text:0"
        assert p2.transition_profile[key] == pytest.approx(0.8 + 1.0)

    async def test_skips_system_messages(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")

        sys_msg = Message(
            id=uuid4(), trajectory_id=tid, role="system", content=[], index=0,
        )
        obs = _obs(tid, 1, c_o)
        msg_repo.add_message(sys_msg)
        msg_repo.add_message(obs)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        assert total == 1
        assert path_repo._paths[0].from_observation_id == obs.id

    async def test_deletes_old_paths(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        await path_repo.create(uuid4(), uuid4())

        tid = uuid4()
        msg_repo.add_message(_obs(tid, 0, _cluster("observation", "text", "o:text:0")))

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        assert len(path_repo._paths) == 1

    async def test_unclustered_messages_use_fallback_label(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        obs = Message(
            id=uuid4(), trajectory_id=tid, role="user", content=[], index=0,
            category="text",
        )
        msg_repo.add_message(obs)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        assert path_repo._paths[0].trace == ["o:text:?"]

    async def test_multiple_trajectories(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        c_o = _cluster("observation", "text", "o:text:0")
        tid1, tid2 = uuid4(), uuid4()
        msg_repo.add_message(_obs(tid1, 0, c_o))
        msg_repo.add_message(_obs(tid2, 0, c_o))

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        assert total == 2
        tids = {p.trajectory_id for p in path_repo._paths}
        assert tids == {tid1, tid2}

    async def test_trailing_action_skipped(self):
        """Trajectory ending with action (no next observation) doesn't create trailing path."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        # obs → act (even number, ends on action)
        obs = _obs(tid, 0, c_o)
        act = _act(tid, 1, c_a)
        msg_repo.add_message(obs)
        msg_repo.add_message(act)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        # No trailing path created — only complete triplets produce paths
        assert total == 0
        assert len(path_repo._paths) == 0

    async def test_trailing_action_after_full_step_skipped(self):
        """obs0 → act0 → obs1 → act1 (trailing action): 2 paths, not 3."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o0 = _cluster("observation", "text", "o:text:0")
        c_a0 = _cluster("action", "text", "a:text:0")
        c_o1 = _cluster("observation", "text", "o:text:1")
        c_a1 = _cluster("action", "text", "a:text:1")

        msgs = [
            _obs(tid, 0, c_o0),
            _act(tid, 1, c_a0),
            _obs(tid, 2, c_o1),
            _act(tid, 3, c_a1),  # trailing action
        ]
        for m in msgs:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        total = await updater.update()

        # 1 complete step + 1 trailing obs (obs1 has no trailing action after it... wait)
        # Actually: 4 msgs (even) → no trailing obs path
        # Loop creates 1 path: obs0→act0→obs1
        assert total == 1
        p = path_repo._paths[0]
        assert p.from_observation_id == msgs[0].id
        assert p.action_message_id == msgs[1].id
        assert p.to_observation_id == msgs[2].id

    async def test_profile_embed_computed(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")
        for m in [_obs(tid, 0, c_o), _act(tid, 1, c_a), _obs(tid, 2, c_o)]:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        assert path_repo._paths[0].profile_embed is None
        assert path_repo._paths[1].profile_embed is not None
        assert len(path_repo._paths[1].profile_embed) == 2000


def _signal_analytics(
    fail_risk_action: bool = False,
    fail_risk_transition: bool = False,
    success_signal_action: bool = False,
    success_signal_transition: bool = False,
    loop: bool = False,
) -> TrajectoryAnalytics:
    return TrajectoryAnalytics(
        fail_risk_action=ActionSignal(
            is_detected=fail_risk_action,
            similarity=0.5 if fail_risk_action else -0.1,
        ),
        success_signal_action=ActionSignal(
            is_detected=success_signal_action,
            similarity=0.5 if success_signal_action else -0.1,
        ),
        fail_risk_transition=fail_risk_transition,
        success_signal_transition=success_signal_transition,
        loop_signal=LoopSignal(
            is_detected=loop,
            transition="o:text:0.a:text:0.o:text:0",
            repeat_count=5 if loop else 0,
            mean_success_repeat=1.0,
        ),
    )


class TestFillSignalCounts:

    async def test_fail_risk_action_counts_accumulated(self):
        """Detected fail_risk_action signals increment count within a trajectory."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        msgs = [
            _obs(tid, 0, c_o), _act(tid, 1, c_a),
            _obs(tid, 2, c_o), _act(tid, 3, c_a),
            _obs(tid, 4, c_o), _act(tid, 5, c_a),
            _obs(tid, 6, c_o),
        ]
        for m in msgs:
            msg_repo.add_message(m)

        call_count = 0
        async def mock_analyze(current_path):
            nonlocal call_count
            detected = call_count in (0, 2)
            call_count += 1
            return _signal_analytics(fail_risk_action=detected)

        with patch(
            "episodiq.clustering.path_updater.TransitionAnalyzer.analyze",
            side_effect=mock_analyze,
        ):
            updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
            await updater.update(fill_signals=True)

        completed = [p for p in path_repo._paths if p.to_observation_id is not None]
        assert len(completed) == 3
        assert completed[0].fail_risk_action_count == 1
        assert completed[1].fail_risk_action_count == 1
        assert completed[2].fail_risk_action_count == 2

    async def test_loop_counts_accumulated(self):
        """Detected loop signals increment loop_count within a trajectory."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        msgs = [
            _obs(tid, 0, c_o), _act(tid, 1, c_a),
            _obs(tid, 2, c_o), _act(tid, 3, c_a),
            _obs(tid, 4, c_o), _act(tid, 5, c_a),
            _obs(tid, 6, c_o),
        ]
        for m in msgs:
            msg_repo.add_message(m)

        # loop on path 1 and 2, not on 0
        call_count = 0
        async def mock_analyze(current_path):
            nonlocal call_count
            detected = call_count in (1, 2)
            call_count += 1
            return _signal_analytics(loop=detected)

        with patch(
            "episodiq.clustering.path_updater.TransitionAnalyzer.analyze",
            side_effect=mock_analyze,
        ):
            updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
            await updater.update(fill_signals=True)

        completed = [p for p in path_repo._paths if p.to_observation_id is not None]
        assert len(completed) == 3
        assert completed[0].loop_count == 0
        assert completed[1].loop_count == 1
        assert completed[2].loop_count == 2

    async def test_fill_signals_false_skips(self):
        """Default fill_signals=False leaves counts at 0."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        tid = uuid4()
        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        for m in [_obs(tid, 0, c_o), _act(tid, 1, c_a), _obs(tid, 2, c_o)]:
            msg_repo.add_message(m)

        updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
        await updater.update()

        for p in path_repo._paths:
            assert p.fail_risk_action_count == 0
            assert p.fail_risk_transition_count == 0
            assert p.success_signal_action_count == 0
            assert p.success_signal_transition_count == 0
            assert p.loop_count == 0

    async def test_signal_counts_per_trajectory(self):
        """Each trajectory accumulates counts independently."""
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)

        c_o = _cluster("observation", "text", "o:text:0")
        c_a = _cluster("action", "text", "a:text:0")

        tid1, tid2 = uuid4(), uuid4()
        for tid in [tid1, tid2]:
            for m in [_obs(tid, 0, c_o), _act(tid, 1, c_a), _obs(tid, 2, c_o)]:
                msg_repo.add_message(m)

        async def mock_analyze(current_path):
            return _signal_analytics(
                fail_risk_action=True, fail_risk_transition=True,
                success_signal_action=True, success_signal_transition=True,
                loop=True,
            )

        with patch(
            "episodiq.clustering.path_updater.TransitionAnalyzer.analyze",
            side_effect=mock_analyze,
        ):
            updater = TrajectoryPathUpdater(msg_repo, path_repo, PathStateCalculator(), workers=1)
            await updater.update(fill_signals=True)

        completed = [p for p in path_repo._paths if p.to_observation_id is not None]
        assert len(completed) == 2
        for p in completed:
            assert p.fail_risk_action_count == 1
            assert p.fail_risk_transition_count == 1
            assert p.success_signal_action_count == 1
            assert p.success_signal_transition_count == 1
            assert p.loop_count == 1
