"""Tests for PrefetchTopkTuner._eval_grid_point and _suggest."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.tune.prefetch_topk import GridPoint, PrefetchTopkTuner
from episodiq.utils import l2_normalize
from tests.in_memory_repos import InMemoryPath, InMemoryTrajectoryPathRepository, Trajectory


BASE_TRACE = ["o:1", "a:1", "o:2", "a:2", "o:3"]
BASE_PROFILE = {"o:1.a:1.o:2": 1.0, "a:1.o:2.a:2": 1.0}


def _embed(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.random(64).tolist())


def _path(
    *,
    action_label: str,
    trace: list[str] = BASE_TRACE,
    profile: dict[str, float] = BASE_PROFILE,
    seed: int = 0,
    status: str = "success",
) -> InMemoryPath:
    tid = uuid4()
    return InMemoryPath(
        id=uuid4(),
        trajectory_id=tid,
        from_observation_id=uuid4(),
        action_label=action_label,
        trace=trace,
        transition_profile=profile,
        profile_embed=_embed(seed),
        trajectory=Trajectory(id=tid, status=status),
    )


class TestEvalGridPoint:
    @pytest.mark.asyncio
    async def test_perfect_hit(self):
        """All neighbors vote for the query's action → hit_at_5 = 1.0."""
        tuner = PrefetchTopkTuner(InMemoryTrajectoryPathRepository())
        query = _path(action_label="a:target", seed=0)
        neighbors = [_path(action_label="a:target", seed=i + 1) for i in range(5)]

        hit, n = await tuner._eval_grid_point([(query, neighbors)], prefetch_n=10, top_k=5)
        assert n == 1
        assert hit == 1.0

    @pytest.mark.asyncio
    async def test_zero_hit(self):
        """No neighbor votes for the query's action → hit_at_5 = 0.0."""
        tuner = PrefetchTopkTuner(InMemoryTrajectoryPathRepository())
        query = _path(action_label="a:target", seed=0)
        neighbors = [_path(action_label=f"a:other_{i}", seed=i + 1) for i in range(6)]

        hit, n = await tuner._eval_grid_point([(query, neighbors)], prefetch_n=10, top_k=6)
        assert n == 1
        assert hit == 0.0

    @pytest.mark.asyncio
    async def test_partial_hit(self):
        """One of two queries hits → hit_at_5 = 0.5."""
        tuner = PrefetchTopkTuner(InMemoryTrajectoryPathRepository())

        q_hit = _path(action_label="a:target", seed=0)
        n_hit = [_path(action_label="a:target", seed=i + 10) for i in range(3)]

        q_miss = _path(action_label="a:rare", seed=20)
        n_miss = [_path(action_label="a:other", seed=i + 30) for i in range(3)]

        hit, n = await tuner._eval_grid_point(
            [(q_hit, n_hit), (q_miss, n_miss)], prefetch_n=10, top_k=5,
        )
        assert n == 2
        assert hit == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_voters_excluded(self):
        """Completely different traces → lev=0 → no voters → not evaluated."""
        tuner = PrefetchTopkTuner(InMemoryTrajectoryPathRepository())

        query = _path(
            action_label="a:target", seed=0,
            trace=["x:1", "x:2", "x:3", "x:4", "x:5"],
            profile={"x:1.x:2.x:3": 1.0},
        )
        neighbors = [
            _path(
                action_label="a:target", seed=i + 1,
                trace=["y:1", "y:2", "y:3", "y:4", "y:5"],
                profile={"y:1.y:2.y:3": 1.0},
            )
            for i in range(5)
        ]

        hit, n = await tuner._eval_grid_point([(query, neighbors)], prefetch_n=10, top_k=5)
        assert n == 0
        assert hit == 0.0

    @pytest.mark.asyncio
    async def test_topk_controls_rerank_cutoff(self):
        """top_k=1 keeps only highest-cosine neighbor; top_k=5 includes lower-cosine ones."""
        tuner = PrefetchTopkTuner(InMemoryTrajectoryPathRepository())

        query = _path(action_label="a:target", seed=0)

        # Highest cosine (identical profile) but wrong action
        wrong = _path(action_label="a:wrong", seed=1, profile=BASE_PROFILE)
        # Low cosine (disjoint profile keys) but correct action
        correct = [
            _path(action_label="a:target", seed=i + 10, profile={"z:1.z:2.z:3": 0.5})
            for i in range(3)
        ]

        pool = [wrong] + correct

        # top_k=1: only `wrong` survives rerank → miss
        hit_1, _ = await tuner._eval_grid_point([(query, pool)], prefetch_n=10, top_k=1)
        assert hit_1 == 0.0

        # top_k=5: all pass rerank, 3 vote "a:target" vs 1 "a:wrong" → hit
        hit_5, _ = await tuner._eval_grid_point([(query, pool)], prefetch_n=10, top_k=5)
        assert hit_5 == 1.0


class TestSuggest:
    def test_picks_minimal_above_threshold(self):
        """Smallest (prefetch_n, top_k) within tolerance of best."""
        grid = [
            GridPoint(prefetch_n=200, top_k=10, hit_at_5=0.80, n_evaluated=100),
            GridPoint(prefetch_n=200, top_k=25, hit_at_5=0.85, n_evaluated=100),
            GridPoint(prefetch_n=500, top_k=10, hit_at_5=0.89, n_evaluated=100),
            GridPoint(prefetch_n=500, top_k=25, hit_at_5=0.90, n_evaluated=100),
        ]
        # best=0.90, tolerance=0.05 > binomial(0.90, 100)≈0.059 → margin=0.059
        # threshold ≈ 0.841 → (200,25) at 0.85 > 0.841 → picked
        pn, tk, _ = PrefetchTopkTuner._suggest(grid, n=100)
        assert pn == 200
        assert tk == 25

    def test_empty_grid(self):
        assert PrefetchTopkTuner._suggest([], n=100) == (0, 0, 0.0)

    def test_tolerance_overrides_tight_binomial(self):
        """Default tolerance=5% is wider than binomial margin for large n."""
        grid = [
            GridPoint(prefetch_n=200, top_k=10, hit_at_5=0.88, n_evaluated=2000),
            GridPoint(prefetch_n=500, top_k=10, hit_at_5=0.89, n_evaluated=2000),
            GridPoint(prefetch_n=1000, top_k=10, hit_at_5=0.90, n_evaluated=2000),
        ]
        # best=0.90, binomial≈0.013, tolerance=0.05 wins → threshold=0.85
        # (200,10) at 0.88 ≥ 0.85 → picked (cheapest)
        pn, tk, margin = PrefetchTopkTuner._suggest(grid, n=2000)
        assert pn == 200
        assert tk == 10
        assert margin == 0.05

    def test_small_tolerance_uses_binomial(self):
        """Tiny tolerance falls back to binomial margin."""
        grid = [
            GridPoint(prefetch_n=200, top_k=10, hit_at_5=0.88, n_evaluated=2000),
            GridPoint(prefetch_n=500, top_k=10, hit_at_5=0.89, n_evaluated=2000),
            GridPoint(prefetch_n=1000, top_k=10, hit_at_5=0.90, n_evaluated=2000),
        ]
        # tolerance=0.01 < binomial≈0.013 → margin=binomial
        # threshold ≈ 0.887 → (200,10) at 0.88 < 0.887 → skip
        # (500,10) at 0.89 ≥ 0.887 → pick
        pn, tk, margin = PrefetchTopkTuner._suggest(grid, n=2000, tolerance=0.01)
        assert pn == 500
        assert tk == 10
        assert margin > 0.01

    def test_single_point(self):
        """Single grid point → always selected."""
        grid = [GridPoint(prefetch_n=200, top_k=10, hit_at_5=0.75, n_evaluated=50)]
        pn, tk, _ = PrefetchTopkTuner._suggest(grid, n=50)
        assert pn == 200
        assert tk == 10
