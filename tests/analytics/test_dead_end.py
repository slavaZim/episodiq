"""Tests for dead-end feature extraction, training, and inference."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.dead_end import (
    NEIGHBOR_FEATURE_NAMES,
    _mean_profile,
    extract_neighbor_features,
)
from episodiq.analytics.dead_end.inference import DeadEndPrediction, DeadEndPredictor
from episodiq.analytics.dead_end.train import (
    DeadEndTrainer,
)
from episodiq.analytics.transition_types import TrajectoryAnalytics
from episodiq.config.config import AnalyticsConfig
from episodiq.utils import l2_normalize
from tests.in_memory_repos import InMemoryPath, InMemoryTrajectoryPathRepository, Trajectory


def _make_path(
    trace: list[str] | None = None,
    index: int = 5,
    fail_risk_action_count: int = 0,
    fail_risk_transition_count: int = 0,
    success_signal_action_count: int = 0,
    success_signal_transition_count: int = 0,
    loop_count: int = 0,
    transition_profile: dict | None = None,
    status: str | None = None,
) -> InMemoryPath:
    traj = Trajectory(id=uuid4(), status=status) if status else None
    return InMemoryPath(
        id=uuid4(),
        trajectory_id=uuid4(),
        from_observation_id=uuid4(),
        to_observation_id=uuid4(),
        trace=trace or ["o:a", "a:b", "o:c", "a:d", "o:e"],
        index=index,
        fail_risk_action_count=fail_risk_action_count,
        fail_risk_transition_count=fail_risk_transition_count,
        success_signal_action_count=success_signal_action_count,
        success_signal_transition_count=success_signal_transition_count,
        loop_count=loop_count,
        transition_profile={"o:a.a:b.o:c": 1.0} if transition_profile is None else transition_profile,
        trajectory=traj,
    )


def _make_analytics(
    n_candidates: int = 5,
    statuses: list[str] | None = None,
) -> TrajectoryAnalytics:
    statuses = statuses or ["success"] * n_candidates
    candidates = [
        _make_path(status=s, index=10, transition_profile={"o:a.a:b.o:c": 1.0})
        for s in statuses
    ]
    success_candidates = [c for c in candidates if c.trajectory and c.trajectory.status == "success"]
    failure_candidates = [c for c in candidates if c.trajectory and c.trajectory.status == "failure"]

    return TrajectoryAnalytics(
        vote_entropy=1.5,
        mean_similarity=0.6,
        n_voters=15,
        n_success=len(success_candidates),
        vote_distribution={"a:text:reply": 0.6, "a:text:ask": 0.4},
        candidates=candidates,
        success_candidates=success_candidates,
        failure_candidates=failure_candidates,
    )


# --- _mean_profile ---

class TestMeanProfile:
    def test_empty_list(self):
        assert _mean_profile([]) == {}

    def test_single_path(self):
        p = _make_path(transition_profile={"a.b.c": 2.0, "d.e.f": 4.0})
        result = _mean_profile([p])
        assert result == {"a.b.c": 2.0, "d.e.f": 4.0}

    def test_averages_across_paths(self):
        p1 = _make_path(transition_profile={"a.b.c": 2.0})
        p2 = _make_path(transition_profile={"a.b.c": 4.0})
        result = _mean_profile([p1, p2])
        assert result["a.b.c"] == pytest.approx(3.0)

    def test_skips_empty_profiles(self):
        p1 = _make_path(transition_profile={"a.b.c": 6.0})
        p2 = _make_path(transition_profile={})
        result = _mean_profile([p1, p2])
        assert result["a.b.c"] == pytest.approx(6.0)


# --- extract_neighbor_features ---

class TestExtractNeighborFeatures:
    def test_returns_11_features(self):
        path = _make_path()
        analytics = _make_analytics()
        features = extract_neighbor_features(path, analytics)
        assert features is not None
        assert len(features) == 11
        assert len(features) == len(NEIGHBOR_FEATURE_NAMES)

    def test_returns_none_short_trace(self):
        path = _make_path(trace=["o:a", "a:b", "o:c"])
        analytics = _make_analytics()
        assert extract_neighbor_features(path, analytics) is None

    def test_returns_none_missing_entropy(self):
        analytics = TrajectoryAnalytics()
        path = _make_path()
        assert extract_neighbor_features(path, analytics) is None

    def test_returns_none_no_candidates(self):
        analytics = TrajectoryAnalytics(vote_entropy=1.0, candidates=[])
        path = _make_path()
        assert extract_neighbor_features(path, analytics) is None

    def test_returns_none_no_vote_distribution(self):
        analytics = TrajectoryAnalytics(
            vote_entropy=1.0,
            candidates=[_make_path(status="success")],
        )
        path = _make_path()
        assert extract_neighbor_features(path, analytics) is None

    def test_top1_share(self):
        analytics = _make_analytics()
        # vote_distribution = {"a:text:reply": 0.6, "a:text:ask": 0.4}
        features = extract_neighbor_features(_make_path(), analytics)
        assert features is not None
        # top1_share = max(dist.values()) = 0.6
        assert features[1] == pytest.approx(0.6)

    def test_fail_risk_action_rate(self):
        path = _make_path(fail_risk_action_count=3, index=5)
        analytics = _make_analytics()
        features = extract_neighbor_features(path, analytics)
        assert features is not None
        # fail_risk_action_rate = count / (index + 1) = 3 / 6
        assert features[3] == pytest.approx(0.5)

    def test_fail_risk_transition_rate(self):
        path = _make_path(fail_risk_transition_count=2, index=5)
        analytics = _make_analytics()
        features = extract_neighbor_features(path, analytics)
        assert features is not None
        # fail_risk_transition_rate = 2 / 6
        assert features[8] == pytest.approx(2.0 / 6.0)

    def test_success_signal_transition_rate(self):
        path = _make_path(success_signal_transition_count=4, index=5)
        analytics = _make_analytics()
        features = extract_neighbor_features(path, analytics)
        assert features is not None
        # success_signal_transition_rate = 4 / 6
        assert features[9] == pytest.approx(4.0 / 6.0)

    def test_last_triplet_ratio_default(self):
        """Default path → last_triplet_ratio computed from candidates."""
        analytics = _make_analytics()
        features = extract_neighbor_features(_make_path(), analytics)
        assert features is not None
        assert 0.0 <= features[10] <= 1.0


# --- DeadEndPredictor ---

class TestDeadEndPredictor:
    def test_not_available_before_load(self):
        predictor = DeadEndPredictor(Path("/nonexistent/model.joblib"), threshold=0.5)
        assert predictor.is_available is False

    def test_load_returns_false_missing_file(self):
        predictor = DeadEndPredictor(Path("/nonexistent/model.joblib"), threshold=0.5)
        assert predictor.load() is False
        assert predictor.is_available is False


class TestDeadEndPrediction:
    def test_fields(self):
        pred = DeadEndPrediction(probability=0.7, confidence=0.7, is_dead_end=True)
        assert pred.probability == 0.7
        assert pred.is_dead_end is True


# --- DeadEndTrainer ---

# Shared trace suffix so lev voting produces matches (min_voters=1 in tests)
_BASE_TRACE = ["o:start", "a:go", "o:mid", "a:step", "o:end"]


def _populate_repo(
    repo: InMemoryTrajectoryPathRepository,
    n_traj: int = 15,
    steps_per_traj: int = 3,
) -> None:
    """Populate InMemory repo with paths for training tests.

    Creates multiple trajectories (mix of success/failure) with similar
    embeddings and traces so TransitionAnalyzer can produce valid analytics.
    """
    rng = np.random.default_rng(42)
    base_embed = l2_normalize(rng.random(64).tolist())

    for t in range(n_traj):
        tid = uuid4()
        status = "failure" if t % 3 == 0 else "success"
        traj = Trajectory(id=tid, status=status)

        for s in range(steps_per_traj):
            # Small noise so prefetch finds all paths as similar
            noise = rng.normal(0, 0.01, 64)
            embed = l2_normalize((np.array(base_embed) + noise).tolist())
            # Trace: shared suffix for lev voting, unique middle for DPM
            trace = ["o:start", "a:go", "o:mid"] + [f"a:{t}"] * (s + 1) + ["o:end"]
            profile = {"o:start.a:go.o:mid": float(s + 1)}

            path = InMemoryPath(
                id=uuid4(),
                trajectory_id=tid,
                from_observation_id=uuid4(),
                to_observation_id=uuid4(),
                profile_embed=embed,
                transition_profile=profile,
                trace=trace,
                index=s,
                action_label=f"a:{t}",
                trajectory=traj,
                trajectory_status=status,
            )
            repo._paths.append(path)


_TEST_ANALYTICS_CONFIG = AnalyticsConfig(
    min_voters=1, prefetch_n=50, top_k=10,
    fail_risk_action_threshold=0.0, success_signal_action_threshold=0.0, loop_threshold=0,
    low_entropy=0.5, high_entropy=2.5,
    dead_end_model="dead_end.joblib", dead_end_threshold=0.85,
)


class TestDeadEndTrainer:
    @pytest.mark.asyncio
    async def test_run_and_save(self, tmp_path):
        repo = InMemoryTrajectoryPathRepository()
        _populate_repo(repo)

        trainer = DeadEndTrainer(
            path_repo=repo,
            analytics_config=_TEST_ANALYTICS_CONFIG,
            test_size=0.2,
            threshold=0.85,
            min_trace=3,
            concurrency=5,
        )
        result = await trainer.run()
        trainer.save(tmp_path / "model.joblib")

        assert result.n_train_samples > 0
        assert result.feature_shape[1] == 11  # 11 neighbor features
        assert result.classification is None
        assert result.walk is None
        assert (tmp_path / "model.joblib").exists()

    @pytest.mark.asyncio
    async def test_eval_returns_both_strategies(self):
        repo = InMemoryTrajectoryPathRepository()
        _populate_repo(repo, n_traj=15)

        trainer = DeadEndTrainer(
            path_repo=repo,
            analytics_config=_TEST_ANALYTICS_CONFIG,
            test_size=0.3,
            threshold=0.85,
            min_trace=3,
            concurrency=5,
        )
        result = await trainer.run(eval=True)

        assert result.classification is not None
        assert len(result.classification.y_test) > 0
        assert len(result.classification.y_pred) == len(result.classification.y_test)
        assert len(result.classification.y_proba) == len(result.classification.y_test)

        assert result.walk is not None
        assert result.walk.n_detected + result.walk.n_missed >= 0
        assert len(result.walk.trajectories) > 0

    @pytest.mark.asyncio
    async def test_walk_turns_remaining(self):
        """turns_remaining = total_steps - flagged_at."""
        repo = InMemoryTrajectoryPathRepository()
        _populate_repo(repo, n_traj=15)

        trainer = DeadEndTrainer(
            path_repo=repo,
            analytics_config=_TEST_ANALYTICS_CONFIG,
            test_size=0.3,
            threshold=0.5,
            min_trace=3,
            concurrency=5,
        )
        result = await trainer.run(eval=True)

        for r in result.walk.trajectories:
            if r.flagged_at is not None:
                assert r.turns_remaining == r.total_steps - r.flagged_at

