"""Unit tests for ``TrajectoryPathTokenUpdater`` orchestration —
mocked ``PathStateCalculator`` to pin the call chain (``prev_path``
threading, single-vs-list ``ActObs`` per call), the LSH-clear-before-
write invariant, and how window bands fan out into LSH rows.
``PathStateCalculator.token_step`` arithmetic is covered by
``test_path_state.py`` and is not re-validated here. End-to-end
against real Postgres lives in
``tests/storage/test_tokenizer_path_updater_db.py``.
"""

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from episodiq.clustering.tokenizer.path_updater import (
    TrajectoryPathTokenUpdater,
)
from episodiq.retrieval.path_state import ActObs, WindowSig

from tests.in_memory_repos import (
    Cluster,
    InMemoryMessageRepository,
    InMemoryTrajectoryPathRepository,
    InMemoryTrajectoryWindowLSHRepository,
    Message,
)


# ----------------------------------------------------------------------
# Shared seeding helpers
# ----------------------------------------------------------------------


def _build_repos(num_trajectories: int = 1):
    """Wire repos with ``num_trajectories`` distinct trajectory ids
    registered via at least one Message each so
    ``get_distinct_trajectory_ids`` returns them all.
    """
    msg_repo = InMemoryMessageRepository()
    path_repo = InMemoryTrajectoryPathRepository(msg_repo)
    lsh_repo = InMemoryTrajectoryWindowLSHRepository()
    tids = [uuid4() for _ in range(num_trajectories)]
    for tid in tids:
        msg_repo.add_message(Message(
            id=uuid4(), trajectory_id=tid, role="user",
            content=[], index=0,
        ))
    return msg_repo, path_repo, lsh_repo, tids


async def _add_path(
    path_repo: InMemoryTrajectoryPathRepository,
    msg_repo: InMemoryMessageRepository,
    *,
    trajectory_id: UUID,
    parallel_group: int | None = None,
    action_cluster_id: UUID | None = None,
    obs_cluster_id: UUID | None = None,
    action_category: str | None = None,
) -> None:
    """Create one path with action/observation messages linked so
    ``ActObs.from_path`` resolves the cluster ids the calculator
    sees in production."""
    obs0 = Message(
        id=uuid4(), trajectory_id=trajectory_id, role="user",
        content=[], index=0,
    )
    act = Message(
        id=uuid4(), trajectory_id=trajectory_id, role="assistant",
        content=[], index=1,
        cluster_id=action_cluster_id,
        cluster=Cluster(
            id=action_cluster_id or uuid4(), type="action",
            category=action_category or "exec", label="a",
        ) if action_cluster_id else None,
    )
    obs1 = Message(
        id=uuid4(), trajectory_id=trajectory_id, role="user",
        content=[], index=2,
        cluster_id=obs_cluster_id,
        cluster=Cluster(
            id=obs_cluster_id or uuid4(), type="observation",
            category="text", label="o",
        ) if obs_cluster_id else None,
    )
    for m in (obs0, act, obs1):
        msg_repo.add_message(m)
    path = await path_repo.create(
        trajectory_id=trajectory_id,
        from_observation_id=obs0.id,
        action_message_id=act.id,
        to_observation_id=obs1.id,
        parallel_group=parallel_group,
    )
    # ``ActObs.from_path`` reads ``path.action_message`` /
    # ``path.to_observation`` as eager-loaded relationships — the in-
    # memory repo doesn't auto-populate them, so wire them up by hand.
    path.action_message = act
    path.to_observation = obs1


# ----------------------------------------------------------------------
# Orchestration tests (mocked calculator)
# ----------------------------------------------------------------------


class TestOrchestration:

    @pytest.mark.asyncio
    async def test_empty_repos_return_zero(self):
        msg_repo = InMemoryMessageRepository()
        path_repo = InMemoryTrajectoryPathRepository(msg_repo)
        lsh_repo = InMemoryTrajectoryWindowLSHRepository()
        calc = AsyncMock()

        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        assert await updater.update() == 0
        calc.token_step.assert_not_called()

    @pytest.mark.asyncio
    async def test_clears_lsh_before_writing(self):
        """Pre-write delete invariant: ``delete_for_trajectories``
        must wipe any stale LSH bands before fresh rows are written.

        Set-up: seed a stale LSH row (band_hash=999) under the
        trajectory we'll update. Stub ``calc.token_step`` to return no
        windows so the updater writes nothing back. Without the
        ``delete_for_trajectories`` call, the stale row would survive
        — so its absence afterwards proves the delete fired.
        """
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        await _add_path(path_repo, msg_repo, trajectory_id=tid)
        await lsh_repo.bulk_insert([(tid, 0, 0, 999)])  # stale band

        calc = AsyncMock()
        calc.token_step.return_value = ([], [])  # no new bands

        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()
        assert lsh_repo._rows == []

    @pytest.mark.asyncio
    async def test_lsh_delete_called_with_all_trajectory_ids(self):
        """One ``delete_for_trajectories([tidA, tidB, ...])`` call up
        front covering EVERY trajectory the updater walks — not
        per-trajectory calls (would be a perf regression)."""
        msg_repo, path_repo, _real_lsh, tids = _build_repos(num_trajectories=2)
        await _add_path(path_repo, msg_repo, trajectory_id=tids[0])
        await _add_path(path_repo, msg_repo, trajectory_id=tids[1])

        lsh_repo = AsyncMock()
        lsh_repo.delete_for_trajectories = AsyncMock()
        lsh_repo.bulk_insert = AsyncMock()
        calc = AsyncMock()
        calc.token_step.return_value = ([], [])

        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()

        lsh_repo.delete_for_trajectories.assert_awaited_once()
        passed_tids = lsh_repo.delete_for_trajectories.call_args.args[0]
        assert set(passed_tids) == set(tids)


# ----------------------------------------------------------------------
# Call chain into PathStateCalculator
# ----------------------------------------------------------------------


class TestCalculatorCallChain:
    """``PathStateCalculator.token_step`` is the only place the updater
    delegates to. These tests pin the shape of those delegations:
    ``prev_path`` is threaded forward correctly between paths, and the
    ``ActObs`` shape (single vs list) reflects whether the run is
    sequential or a parallel-batch group.
    """

    @pytest.mark.asyncio
    async def test_first_call_has_prev_path_none(self):
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        await _add_path(path_repo, msg_repo, trajectory_id=tid)

        calc = AsyncMock()
        calc.token_step.return_value = ([1], [])
        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()

        # Exactly one call; prev_path arg is None (no predecessor).
        assert calc.token_step.await_count == 1
        first_call = calc.token_step.call_args_list[0]
        assert first_call.args[0] is None

    @pytest.mark.asyncio
    async def test_subsequent_calls_thread_prev_path_through(self):
        """Path 2's ``token_step`` must receive path 1 as ``prev_path``,
        and path 3's must receive path 2 — chain order matters for
        cumulative trace_tokens."""
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        await _add_path(path_repo, msg_repo, trajectory_id=tid)
        await _add_path(path_repo, msg_repo, trajectory_id=tid)
        await _add_path(path_repo, msg_repo, trajectory_id=tid)

        calc = AsyncMock()
        calc.token_step.return_value = ([1], [])
        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()

        paths_in_order = await path_repo.get_trajectory_paths(tid)
        assert calc.token_step.await_count == 3
        # Call N receives path[N-1] as prev_path (call 0 receives None).
        assert calc.token_step.call_args_list[0].args[0] is None
        assert calc.token_step.call_args_list[1].args[0] is paths_in_order[0]
        assert calc.token_step.call_args_list[2].args[0] is paths_in_order[1]

    @pytest.mark.asyncio
    async def test_sequential_paths_pass_single_actobs(self):
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        await _add_path(path_repo, msg_repo, trajectory_id=tid)
        await _add_path(path_repo, msg_repo, trajectory_id=tid)

        calc = AsyncMock()
        calc.token_step.return_value = ([1], [])
        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()

        for call in calc.token_step.call_args_list:
            assert isinstance(call.args[1], ActObs)

    @pytest.mark.asyncio
    async def test_parallel_group_passes_list_of_actobs(self):
        """A run of paths sharing ``parallel_group`` becomes ONE
        ``token_step(prev, list[ActObs])`` call — the calculator
        sorts the batch internally so tokens are tool-call-order
        invariant. Both batch members get the same trace_tokens
        written back."""
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        # Two parallel paths + one trailing sequential path.
        await _add_path(path_repo, msg_repo, trajectory_id=tid, parallel_group=42)
        await _add_path(path_repo, msg_repo, trajectory_id=tid, parallel_group=42)
        await _add_path(path_repo, msg_repo, trajectory_id=tid, parallel_group=None)

        calc = AsyncMock()
        calc.token_step.side_effect = [
            ([1, 2], [WindowSig(step=10, bands=[7, 9])]),    # batch
            ([1, 2, 3], [WindowSig(step=11, bands=[5])]),    # sequential
        ]
        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        count = await updater.update()

        assert count == 3
        # Batch call: ``list[ActObs]`` of length 2.
        batch_arg = calc.token_step.call_args_list[0].args[1]
        assert isinstance(batch_arg, list)
        assert len(batch_arg) == 2
        assert all(isinstance(ao, ActObs) for ao in batch_arg)
        # Trailing call: single ActObs.
        last_arg = calc.token_step.call_args_list[1].args[1]
        assert isinstance(last_arg, ActObs)

        # Both parallel paths carry the same tokens (the batch result).
        paths = await path_repo.get_trajectory_paths(tid)
        assert paths[0].trace_tokens == [1, 2]
        assert paths[1].trace_tokens == [1, 2]
        assert paths[2].trace_tokens == [1, 2, 3]


# ----------------------------------------------------------------------
# LSH row materialisation
# ----------------------------------------------------------------------


class TestLshRows:
    """Bands returned by ``token_step`` fan out into one LSH row per
    ``(band_index, band_hash)`` per ``WindowSig.step``."""

    @pytest.mark.asyncio
    async def test_window_bands_become_lsh_rows(self):
        msg_repo, path_repo, lsh_repo, [tid] = _build_repos()
        await _add_path(path_repo, msg_repo, trajectory_id=tid)
        await _add_path(path_repo, msg_repo, trajectory_id=tid)

        calc = AsyncMock()
        calc.token_step.side_effect = [
            ([7], [WindowSig(step=5, bands=[111, 222])]),
            ([7, 8], [WindowSig(step=6, bands=[333, 444])]),
        ]
        updater = TrajectoryPathTokenUpdater(
            msg_repo, path_repo, lsh_repo, calc,
        )
        await updater.update()

        # 2 windows × 2 bands each = 4 rows.
        assert sorted(lsh_repo._rows) == sorted([
            (tid, 5, 0, 111), (tid, 5, 1, 222),
            (tid, 6, 0, 333), (tid, 6, 1, 444),
        ])


