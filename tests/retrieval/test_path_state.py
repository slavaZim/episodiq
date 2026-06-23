"""Tests for PathStateCalculator (granular trace + token_step)."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import pytest

from episodiq.config.retrieval_config import WindowMinHashConfig
from episodiq.retrieval.path_state import ActObs, PathStateCalculator
from episodiq.retrieval.window_minhash import WindowMinHasher


@dataclass
class FakePath:
    """Stand-in for TrajectoryPath (only the fields token_step reads)."""
    trace: list[str] = field(default_factory=list)
    trace_tokens: list[int] | None = None
    action_label: str | None = None


class FakeAssigner:
    """Deterministic mock assigner: returns the ordinal stored in the
    mapping. ``assign`` returns ``None`` for unknown ``(a_cid, o_cid)``.
    """
    def __init__(self, mapping: dict[tuple[UUID, UUID], int]) -> None:
        self._mapping = mapping

    async def assign(self, a_cid, o_cid, _category) -> int | None:
        return self._mapping.get((a_cid, o_cid))


class FakeHasher:
    """Deterministic stand-in for ``WindowMinHasher``: emits bands whose
    values mirror the contents of the window being hashed. Lets us
    assert that the calculator handed the right slice to the hasher.
    """
    def __init__(self, cfg: WindowMinHashConfig) -> None:
        self.config = cfg
        self.calls: list[tuple[list[int], int]] = []

    def bands_for_window(self, tokens, start: int) -> np.ndarray:
        slice_ = list(tokens[start:start + self.config.window])
        self.calls.append((slice_, start))
        # One band per token in the window — easy to assert against.
        return np.array(slice_, dtype=np.int64)


def _calc(
    window: int = 4, assigner=None, hasher: WindowMinHasher | None = None,
) -> PathStateCalculator:
    if hasher is None:
        hasher = WindowMinHasher(WindowMinHashConfig(window=window))
    return PathStateCalculator(assigner=assigner, hasher=hasher)


class TestGranularStep:
    def test_first_observation(self):
        trace = _calc().granular_step(None, "o:text:greeting")
        assert trace == ["o:text:greeting"]

    def test_completed_triplet_extends_trace(self):
        prev = FakePath(
            trace=["o:text:greeting"], action_label="a:text:response",
        )
        trace = _calc().granular_step(prev, "o:text:followup")
        assert trace == ["o:text:greeting", "a:text:response", "o:text:followup"]

    def test_no_action_label_yields_single_obs(self):
        prev = FakePath(trace=["o:text:foo"], action_label=None)
        trace = _calc().granular_step(prev, "o:text:bar")
        assert trace == ["o:text:bar"]


@pytest.mark.asyncio
class TestTokenStepSequential:
    async def test_appends_one_token(self):
        a, o = uuid4(), uuid4()
        calc = _calc(assigner=FakeAssigner({(a, o): 7}))
        tokens, wins = await calc.token_step(
            None, ActObs(a, o, None),
        )
        assert tokens == [7]
        assert wins == []  # Below W=4

    async def test_window_emitted_at_W(self):
        a, o = uuid4(), uuid4()
        calc = _calc(window=4, assigner=FakeAssigner({(a, o): 9}))
        prev = FakePath(trace_tokens=[1, 2, 3])
        tokens, wins = await calc.token_step(prev, ActObs(a, o, None))
        assert tokens == [1, 2, 3, 9]
        assert len(wins) == 1
        assert wins[0].step == 0 + 2  # first_token + half_window

    async def test_unresolved_carries_prev_unchanged(self):
        a, o = uuid4(), uuid4()
        # mapping is empty → assigner returns None
        calc = _calc(assigner=FakeAssigner({}))
        prev = FakePath(trace_tokens=[5, 6])
        tokens, wins = await calc.token_step(prev, ActObs(a, o, None))
        assert tokens == [5, 6]
        assert wins == []


@pytest.mark.asyncio
class TestTokenStepParallel:
    async def test_sorts_ordinals_ascending(self):
        """Two parallel calls in reversed order — final tokens come out
        sorted ASC regardless of input order."""
        a1, o1 = uuid4(), uuid4()
        a2, o2 = uuid4(), uuid4()
        calc = _calc(assigner=FakeAssigner({(a1, o1): 9, (a2, o2): 3}))
        prev = FakePath(trace_tokens=[1])
        tokens, _ = await calc.token_step(
            prev, [ActObs(a1, o1, None), ActObs(a2, o2, None)],
        )
        assert tokens == [1, 3, 9]

        # Reversed call order → identical output.
        tokens_rev, _ = await calc.token_step(
            prev, [ActObs(a2, o2, None), ActObs(a1, o1, None)],
        )
        assert tokens_rev == tokens

    async def test_emits_one_window_per_appended_token_once_full(self):
        """W=4, prev_len=3, append 3 → tokens at positions 3,4,5; all three
        positions have right_edge >= 4, so 3 windows emitted. The bands
        each window emits reflect the sliding W-token slice the hasher
        was actually called on."""
        keys = [(uuid4(), uuid4()) for _ in range(3)]
        mapping = {keys[i]: ord_ for i, ord_ in enumerate([5, 6, 7])}
        cfg = WindowMinHashConfig(window=4)
        fake = FakeHasher(cfg)
        calc = _calc(window=4, assigner=FakeAssigner(mapping), hasher=fake)
        prev = FakePath(trace_tokens=[1, 2, 3])
        tokens, wins = await calc.token_step(
            prev, [ActObs(a, o, None) for a, o in keys],
        )
        # Sorted ASC inside the parallel batch → final tokens [1,2,3,5,6,7].
        assert tokens == [1, 2, 3, 5, 6, 7]
        assert len(wins) == 3
        # Each window's slice is the trailing 4 tokens AT THE TIME it
        # formed — sliding-window over the cumulative trace.
        assert fake.calls == [
            ([1, 2, 3, 5], 0),
            ([2, 3, 5, 6], 1),
            ([3, 5, 6, 7], 2),
        ]
        # Steps advance by 1 (= first_token + half_window).
        assert [w.step for w in wins] == [2, 3, 4]
        # Bands == the slice the hasher saw.
        assert wins[0].bands == [1, 2, 3, 5]
        assert wins[2].bands == [3, 5, 6, 7]

    async def test_unresolved_pairs_skipped_in_parallel(self):
        """Unresolved ordinals are dropped, remaining sorted."""
        a1, o1 = uuid4(), uuid4()
        a_bad, o_bad = uuid4(), uuid4()   # no mapping
        a3, o3 = uuid4(), uuid4()
        calc = _calc(assigner=FakeAssigner({(a1, o1): 5, (a3, o3): 2}))
        prev = FakePath(trace_tokens=[])
        tokens, _ = await calc.token_step(
            prev,
            [ActObs(a1, o1, None), ActObs(a_bad, o_bad, None),
             ActObs(a3, o3, None)],
        )
        assert tokens == [2, 5]
