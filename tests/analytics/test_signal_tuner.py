"""Tests for SignalTuner."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.transition_types import TrajectoryAnalytics
from episodiq.analytics.tune.signal_tuner import SignalTuner
from episodiq.utils import bootstrap_auc_ci
from tests.conftest import mock_session_factory
from tests.in_memory_repos import InMemoryPath, InMemoryTrajectoryRepository, Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ids(n: int) -> list:
    return [uuid4() for _ in range(n)]


def _statuses(success_ids, failure_ids):
    d = {tid: "success" for tid in success_ids}
    d.update({tid: "failure" for tid in failure_ids})
    return d


def _n_paths(ids, n: int = 10):
    return {tid: n for tid in ids}


# ---------------------------------------------------------------------------
# _eval_threshold
# ---------------------------------------------------------------------------

class TestEvalThreshold:
    def test_perfect_separation(self):
        """Failures have high similarity, successes have low → AUC near 1."""
        s_ids = _make_ids(5)
        f_ids = _make_ids(5)
        statuses = _statuses(s_ids, f_ids)
        n_paths = _n_paths(s_ids + f_ids, n=10)

        signals = {}
        for tid in s_ids:
            signals[tid] = [0.1] * 10
        for tid in f_ids:
            signals[tid] = [0.9] * 10

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is not None
        assert result.auc == pytest.approx(1.0)
        assert result.threshold == 0.5

    def test_no_separation(self):
        """Both classes have varied but uncorrelated rates → AUC near 0.5."""
        s_ids = _make_ids(10)
        f_ids = _make_ids(10)
        statuses = _statuses(s_ids, f_ids)
        n_paths = _n_paths(s_ids + f_ids, n=10)

        rng = np.random.RandomState(42)
        signals = {tid: rng.uniform(0.3, 0.8, 10).tolist() for tid in s_ids + f_ids}

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is not None
        assert 0.3 < result.auc < 0.7

    def test_returns_none_single_class(self):
        """Only one class present → can't compute AUC."""
        ids = _make_ids(5)
        statuses = {tid: "success" for tid in ids}
        n_paths = _n_paths(ids, n=10)
        signals = {tid: [0.8] * 10 for tid in ids}

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is None

    def test_returns_none_uniform_scores(self):
        """All rates identical → can't compute AUC."""
        s_ids = _make_ids(3)
        f_ids = _make_ids(3)
        statuses = _statuses(s_ids, f_ids)
        n_paths = _n_paths(s_ids + f_ids, n=10)

        signals = {tid: [0.1] * 10 for tid in s_ids + f_ids}

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is None

    def test_threshold_affects_signal_count(self):
        """Higher threshold → fewer signals counted."""
        s_ids = _make_ids(3)
        f_ids = _make_ids(3)
        statuses = _statuses(s_ids, f_ids)
        n_paths = _n_paths(s_ids + f_ids, n=10)

        signals = {}
        for tid in s_ids:
            signals[tid] = [0.1, 0.2, 0.3, 0.4, 0.45, 0.15, 0.25, 0.35, 0.42, 0.38]
        for tid in f_ids:
            signals[tid] = [0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.62, 0.78]

        low = SignalTuner._eval_threshold(0.2, statuses, n_paths, signals)
        high = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)

        assert low is not None and high is not None
        assert low.signal_rate > high.signal_rate

    def test_empty_signals(self):
        """Trajectories with no signals → rate=0 for all → None."""
        s_ids = _make_ids(3)
        f_ids = _make_ids(3)
        statuses = _statuses(s_ids, f_ids)
        n_paths = _n_paths(s_ids + f_ids, n=10)
        signals: dict = {}

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is None

    def test_signal_rate_computation(self):
        """Verify mean signal_rate reflects per-trajectory rates."""
        tid_s = uuid4()
        tid_f = uuid4()
        statuses = {tid_s: "success", tid_f: "failure"}
        n_paths = {tid_s: 4, tid_f: 4}
        signals = {
            tid_s: [0.1, 0.6, 0.2, 0.3],
            tid_f: [0.6, 0.7, 0.8, 0.3],
        }

        result = SignalTuner._eval_threshold(0.5, statuses, n_paths, signals)
        assert result is not None
        assert result.signal_rate == pytest.approx(0.5)  # mean of 0.25 and 0.75
        assert result.auc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap_auc_ci
# ---------------------------------------------------------------------------

class TestBootstrapAucCi:
    def test_perfect_auc_tight_ci(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo >= 0.95
        assert hi == pytest.approx(1.0)

    def test_random_auc_wide_ci(self):
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = rng.random(100)

        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo < 0.5 < hi

    def test_single_class_returns_zeros(self):
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo == 0.0
        assert hi == 0.0


# ---------------------------------------------------------------------------
# SignalTuner.run
# ---------------------------------------------------------------------------

def _make_trajectory(status: str, fail_similarities: list[float]):
    tid = uuid4()
    paths = [
        InMemoryPath(
            id=uuid4(),
            trajectory_id=tid,
            from_observation_id=uuid4(),
            profile_embed=[1.0] * 64,
        )
        for _ in fail_similarities
    ]
    traj = Trajectory(id=tid, status=status, paths=paths)
    for p in paths:
        p.trajectory = traj
    return traj, fail_similarities


def _build_tuner(trajectories, similarity_map):
    repo = InMemoryTrajectoryRepository()
    for traj in trajectories:
        repo._trajectories[traj.id] = traj

    return SignalTuner(repo, mock_session_factory()), similarity_map


def _patch_analyzer(sim_map):
    patcher = patch("episodiq.analytics.tune.signal_tuner.TransitionAnalyzer")

    MockAnalyzer = patcher.start()

    async def mock_analyze(path):
        return TrajectoryAnalytics(fail_similarity=sim_map[path.id])

    MockAnalyzer.return_value.analyze = mock_analyze
    return patcher


def _sim_map_from(*traj_sims_pairs):
    m = {}
    for traj, sims in traj_sims_pairs:
        for path, sim in zip(traj.paths, sims):
            m[path.id] = sim
    return m


class TestRun:
    @pytest.mark.asyncio
    async def test_both_directions_from_same_data(self):
        """Failures positive, successes negative fail_similarity → both directions tune."""
        # Successes: succ_lev > fail_lev → fail_similarity negative
        s1, s1_sims = _make_trajectory("success", [-0.7, -0.8, -0.75])
        s2, s2_sims = _make_trajectory("success", [-0.6, -0.65, -0.72])
        # Failures: fail_lev > succ_lev → fail_similarity positive
        f1, f1_sims = _make_trajectory("failure", [0.8, 0.9, 0.85])
        f2, f2_sims = _make_trajectory("failure", [0.7, 0.75, 0.82])

        sim_map = _sim_map_from((s1, s1_sims), (s2, s2_sims), (f1, f1_sims), (f2, f2_sims))
        tuner, _ = _build_tuner([s1, s2, f1, f2], sim_map)
        patcher = _patch_analyzer(sim_map)

        try:
            result = await tuner.run(
                sample_size=4, percentiles=[50, 80, 95],
                min_rate=0.0, max_rate=1.0,
            )
        finally:
            patcher.stop()

        assert result.n_trajectories == 4
        assert result.n_success == 2
        assert result.n_failure == 2
        # fail_risk: high fail_similarity → failure → max AUC
        assert result.fail_risk_suggested is not None
        assert result.fail_risk_suggested.auc >= 0.8
        # success_signal: high -fail_similarity → success → min AUC (against failure=1)
        assert result.success_signal_suggested is not None
        assert result.success_signal_suggested.auc <= 0.3

    @pytest.mark.asyncio
    async def test_empty_trajectories(self):
        """No trajectories → empty result."""
        tuner, _ = _build_tuner([], {})
        patcher = _patch_analyzer({})

        try:
            result = await tuner.run(sample_size=10, percentiles=[50])
        finally:
            patcher.stop()

        assert result.n_trajectories == 0
        assert result.fail_risk_thresholds == []
        assert result.fail_risk_suggested is None
        assert result.success_signal_thresholds == []
        assert result.success_signal_suggested is None

    @pytest.mark.asyncio
    async def test_none_fail_similarity_skipped(self):
        """Paths returning None fail_similarity are excluded."""
        s1, _ = _make_trajectory("success", [None, None])
        f1, _ = _make_trajectory("failure", [None, None])

        sim_map = _sim_map_from((s1, [None, None]), (f1, [None, None]))
        tuner, _ = _build_tuner([s1, f1], sim_map)
        patcher = _patch_analyzer(sim_map)

        try:
            result = await tuner.run(
                sample_size=4, percentiles=[50], min_rate=0.0, max_rate=1.0,
            )
        finally:
            patcher.stop()

        assert result.fail_risk_thresholds == []
        assert result.fail_risk_suggested is None

    @pytest.mark.asyncio
    async def test_min_max_rate_filters_thresholds(self):
        """Thresholds producing rate outside [min_rate, max_rate] are dropped."""
        s1, s1_sims = _make_trajectory("success", [0.1, 0.15, 0.12, 0.08])
        s2, s2_sims = _make_trajectory("success", [0.11, 0.13, 0.09, 0.14])
        f1, f1_sims = _make_trajectory("failure", [0.5, 0.6, 0.55, 0.58])
        f2, f2_sims = _make_trajectory("failure", [0.52, 0.61, 0.57, 0.53])

        sim_map = _sim_map_from((s1, s1_sims), (s2, s2_sims), (f1, f1_sims), (f2, f2_sims))
        tuner, _ = _build_tuner([s1, s2, f1, f2], sim_map)
        patcher = _patch_analyzer(sim_map)

        try:
            wide = await tuner.run(sample_size=4, min_rate=0.0, max_rate=1.0)
            tight = await tuner.run(sample_size=4, min_rate=0.15, max_rate=0.2)
        finally:
            patcher.stop()

        assert len(tight.fail_risk_thresholds) < len(wide.fail_risk_thresholds)
        for t in tight.fail_risk_thresholds:
            assert 0.15 <= t.signal_rate <= 0.2
