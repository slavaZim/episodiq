"""Real-DB integration tests for TransitionAnalyzer.

Builds Trajectory + Message + TrajectoryPath rows directly in PostgreSQL with
distant (or identical) embeddings so KNN ordering and signal computation are
fully deterministic. Uses small prefetch_n / top_k / min_voters values so the
candidate pool is exercised end-to-end without statistical noise.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from episodiq.analytics.transition_analyzer import TransitionAnalyzer
from episodiq.config.config import AnalyticsConfig
from episodiq.storage.postgres.models import Cluster, Message, Trajectory, TrajectoryPath
from episodiq.storage.postgres.repository import TrajectoryPathRepository
from episodiq.utils import l2_normalize


OBS_LABEL = "o:text:0"
ACT_LABEL = "a:bash:1"
DEFAULT_TRACE = [OBS_LABEL, ACT_LABEL, OBS_LABEL]
DEFAULT_PROFILE = {f"{OBS_LABEL}.{ACT_LABEL}.{OBS_LABEL}": 1.0}


def _embed_2000(seed: int) -> list[float]:
    """L2-normalized random 2000-d vector for profile_embed (HNSW search)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2000)
    return l2_normalize(v).tolist()


def _embed_1024(seed: int) -> list[float]:
    """L2-normalized random 1024-d vector for action message embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024)
    return l2_normalize(v).tolist()


def _config(**overrides) -> AnalyticsConfig:
    """AnalyticsConfig with small prefetch/top_k/min_voters for deterministic tests."""
    defaults = dict(
        prefetch_n=5,
        top_k=3,
        min_voters=1,
        fail_risk_action_threshold=0.06,
        success_signal_action_threshold=0.06,
        loop_threshold=2,
        low_entropy=0.5,
        high_entropy=2.5,
        dead_end_model="",
        dead_end_threshold=0.5,
    )
    defaults.update(overrides)
    return AnalyticsConfig(**defaults)


async def _make_path(
    session,
    *,
    trace: list[str],
    action_label: str,
    status: str,
    profile_embed: list[float] | None,
    transition_profile: dict[str, float] | None = None,
    action_embed: list[float] | None = None,
) -> TrajectoryPath:
    """Build a full Trajectory+Cluster+Message+TrajectoryPath row set.

    Sets relationships in-memory after flush so the analyzer can read
    `path.action_message.embedding` and `path.action_message.cluster.label`
    without async lazy loading.
    """
    tid = uuid4()
    traj = Trajectory(id=tid, status=status)
    session.add(traj)

    obs_cluster = Cluster(type="observation", category="text", label=OBS_LABEL)
    act_cluster = Cluster(type="action", category="text", label=action_label)
    session.add_all([obs_cluster, act_cluster])
    await session.flush()

    from_obs = Message(
        trajectory_id=tid, role="user", content=[],
        category="text", cluster_type="observation", cluster_id=obs_cluster.id,
    )
    action_msg = Message(
        trajectory_id=tid, role="assistant", content=[],
        category="text", cluster_type="action", cluster_id=act_cluster.id,
        embedding=action_embed,
    )
    to_obs = Message(
        trajectory_id=tid, role="user", content=[],
        category="text", cluster_type="observation", cluster_id=obs_cluster.id,
    )
    session.add_all([from_obs, action_msg, to_obs])
    await session.flush()

    from_obs.cluster = obs_cluster
    action_msg.cluster = act_cluster
    to_obs.cluster = obs_cluster

    path = TrajectoryPath(
        trajectory_id=tid,
        from_observation_id=from_obs.id,
        action_message_id=action_msg.id,
        to_observation_id=to_obs.id,
        transition_profile=transition_profile or {},
        profile_embed=profile_embed,
        trace=trace,
        trajectory_status=status,
    )
    session.add(path)
    await session.flush()

    path.trajectory = traj
    path.from_observation = from_obs
    path.action_message = action_msg
    path.to_observation = to_obs
    return path


@pytest.mark.asyncio(loop_scope="session")
class TestAnalyze:
    async def test_returns_empty_when_no_profile_embed(self, db_session):
        """profile_embed=None short-circuits before any DB work."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        path = await _make_path(
            db_session, trace=[OBS_LABEL], action_label=ACT_LABEL,
            status="pending", profile_embed=None,
        )

        result = await analyzer.analyze(path)
        assert result.vote_entropy is None
        assert result.candidates is None
        assert result.path_frequency_signal is None

    async def test_returns_empty_when_all_voters_zero_lev(self, db_session):
        """Candidate trace disjoint from query → lev=0 → 0 voters → empty result."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config(min_voters=1))

        await _make_path(
            db_session,
            trace=["x:1", "x:2", "x:3"], action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile={"x:1.x:2.x:3": 1.0},
            action_embed=_embed_1024(1),
        )

        current = await _make_path(
            db_session,
            trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(1),
            transition_profile=DEFAULT_PROFILE,
            action_embed=_embed_1024(2),
        )

        result = await analyzer.analyze(current)
        assert result.vote_entropy is None
        assert result.path_frequency_signal is None
        assert result.fail_risk_action is None

    async def test_min_voters_threshold(self, db_session):
        """min_voters=2: 1 voter → empty; 2 voters → analytics returned."""
        repo = TrajectoryPathRepository(db_session)

        # One overlapping-trace neighbor (lev>0), one disjoint (lev=0)
        await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(10),
        )
        await _make_path(
            db_session, trace=["x:1", "x:2", "x:3"], action_label="a:other",
            status="success", profile_embed=_embed_2000(1),
            transition_profile={"x:1.x:2.x:3": 1.0},
            action_embed=_embed_1024(11),
        )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(99),
        )

        # min_voters=2: only 1 candidate has lev>0 → empty
        analyzer_strict = TransitionAnalyzer(path_repo=repo, config=_config(min_voters=2))
        strict_result = await analyzer_strict.analyze(current)
        assert strict_result.vote_entropy is None
        assert strict_result.candidates is None

        # min_voters=1: 1 voter is enough → analytics returned
        analyzer_loose = TransitionAnalyzer(path_repo=repo, config=_config(min_voters=1))
        loose_result = await analyzer_loose.analyze(current)
        assert loose_result.n_voters == 1
        assert loose_result.vote_entropy is not None

    async def test_sparse_cosine_rerank_keeps_top_k(self, db_session):
        """top_k cuts the prefetch pool by sparse_cosine on transition_profile.

        Five candidates with strictly decreasing cosine to the query profile.
        With top_k=3, only the three highest-cosine candidates appear in
        `candidates`, even though all five remain in `prefetched_reranked`.
        """
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        # Cosine to query={K0:1.0} decreases as more keys are added.
        # C1 cos=1.0, C2 cos≈0.707, C3 cos≈0.577, C4 cos=0.5, C5 cos≈0.447
        profiles = [
            {"K0": 1.0},
            {"K0": 1.0, "K1": 1.0},
            {"K0": 1.0, "K1": 1.0, "K2": 1.0},
            {"K0": 1.0, "K1": 1.0, "K2": 1.0, "K3": 1.0},
            {"K0": 1.0, "K1": 1.0, "K2": 1.0, "K3": 1.0, "K4": 1.0},
        ]
        created_ids = []
        for i, prof in enumerate(profiles):
            p = await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=prof, action_embed=_embed_1024(50 + i),
            )
            created_ids.append(p.id)

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(99),
            transition_profile={"K0": 1.0}, action_embed=_embed_1024(99),
        )

        result = await analyzer.analyze(current)

        assert result.prefetched_reranked is not None
        assert len(result.prefetched_reranked) == 5
        # rerank order is strictly decreasing
        cosines = [c for c, _ in result.prefetched_reranked]
        assert cosines == sorted(cosines, reverse=True)

        assert result.candidates is not None
        assert len(result.candidates) == 3
        kept_ids = {c.id for c in result.candidates}
        assert kept_ids == set(created_ids[:3])

    async def test_lev_weighted_votes(self, db_session):
        """Vote weight per candidate equals levenshtein(query_suffix, c_suffix).

        Two candidates with different action labels: one with identical trace
        (lev=1.0), one with a single substitution (lev=0.8). Verify
        mean_similarity, vote_distribution, and vote weights.
        """
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config(min_voters=1))

        # 5-element trace so trunc_suffix(.., 5) returns the whole trace
        identical_trace = [OBS_LABEL, ACT_LABEL, OBS_LABEL, ACT_LABEL, OBS_LABEL]
        # one substitution at the last position
        edited_trace = [OBS_LABEL, ACT_LABEL, OBS_LABEL, ACT_LABEL, "o:text:9"]

        await _make_path(
            db_session, trace=identical_trace, action_label="a:identical",
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(1),
        )
        await _make_path(
            db_session, trace=edited_trace, action_label="a:edited",
            status="success", profile_embed=_embed_2000(1),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(2),
        )

        current = await _make_path(
            db_session, trace=identical_trace, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(99),
        )

        result = await analyzer.analyze(current)

        # both vote → 2 voters, mean_sim = (1.0 + 0.8) / 2 = 0.9
        assert result.n_voters == 2
        assert result.mean_similarity == pytest.approx(0.9)

        # vote_distribution normalized by total weight (1.0 + 0.8 = 1.8)
        assert result.vote_distribution is not None
        assert result.vote_distribution["a:identical"] == pytest.approx(1.0 / 1.8)
        assert result.vote_distribution["a:edited"] == pytest.approx(0.8 / 1.8)

    async def test_path_frequency_consensus(self, db_session):
        """Three matching neighbors all vote same action → entropy=0, n_success=3."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        for i in range(3):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE,
                action_embed=_embed_1024(100 + i),
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(200),
        )

        result = await analyzer.analyze(current)

        assert result.n_voters == 3
        assert result.n_success == 3
        assert result.vote_entropy == pytest.approx(0.0)
        assert result.vote_distribution == {ACT_LABEL: pytest.approx(1.0)}
        assert result.path_frequency_signal is not None
        assert result.path_frequency_signal.entropy == pytest.approx(0.0)
        assert result.mean_similarity == pytest.approx(1.0)

    async def test_vote_distribution_split(self, db_session):
        """Two distinct vote labels → entropy>0, distribution splits proportionally."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(10 + i),
            )
        await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label="a:bash:other",
            status="failure", profile_embed=_embed_2000(2),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(20),
        )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(99),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(30),
        )

        result = await analyzer.analyze(current)

        assert result.n_voters == 3
        assert result.vote_distribution is not None
        assert set(result.vote_distribution.keys()) == {ACT_LABEL, "a:bash:other"}
        assert result.vote_distribution[ACT_LABEL] == pytest.approx(2 / 3)
        assert result.vote_distribution["a:bash:other"] == pytest.approx(1 / 3)
        assert result.vote_entropy is not None
        assert result.vote_entropy > 0.0

    async def test_loop_signal_detected_via_tail_streak(self, db_session):
        """Trace ending in a repeated duplet hits loop_threshold → is_detected."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config(loop_threshold=2))

        await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(1),
        )

        # tail duplet "a:bash:1.o:text:0" repeats twice → tail_streak == 2
        looping_trace = [OBS_LABEL, ACT_LABEL, OBS_LABEL, ACT_LABEL, OBS_LABEL]
        current = await _make_path(
            db_session, trace=looping_trace, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(1),
            transition_profile={
                f"{OBS_LABEL}.{ACT_LABEL}.{OBS_LABEL}": 1.0,
                f"{ACT_LABEL}.{OBS_LABEL}.{ACT_LABEL}": 1.0,
            },
            action_embed=_embed_1024(2),
        )

        result = await analyzer.analyze(current)

        assert result.loop_signal is not None
        assert result.loop_signal.is_detected is True
        assert result.loop_streak == 2
        assert result.loop_signal.repeat_count >= 1

    async def test_loop_signal_not_detected_below_threshold(self, db_session):
        """Trace without repeated tail duplet → is_detected=False."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config(loop_threshold=2))

        await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(1),
        )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(1),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(2),
        )

        result = await analyzer.analyze(current)

        assert result.loop_signal is not None
        assert result.loop_signal.is_detected is False
        assert result.loop_streak == 1

    async def test_fail_risk_action_when_failures_match_query(self, db_session):
        """Failure action embeds == query → fail_lev≫succ_lev → fail_risk fires.

        Also verifies action signals are mutually exclusive in this scenario.
        """
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        # Identical action embedding for query and failure candidates;
        # distant random for success candidates.
        query_action = _embed_1024(0)
        far_action = _embed_1024(999)

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="failure", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE, action_embed=query_action,
            )
        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(10 + i),
                transition_profile=DEFAULT_PROFILE, action_embed=far_action,
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=query_action,
        )

        result = await analyzer.analyze(current)

        assert result.fail_risk_action is not None
        assert result.fail_risk_action.is_detected is True
        assert result.fail_risk_action.similarity is not None
        assert result.fail_risk_action.similarity >= 0.06

        assert result.success_signal_action is not None
        assert result.success_signal_action.is_detected is False

        # mutual exclusivity invariant
        assert not (
            result.fail_risk_action.is_detected
            and result.success_signal_action.is_detected
        )

        # fail_similarity = fail_lev_cosine - succ_lev_cosine, should be positive
        assert result.fail_similarity is not None
        assert result.fail_similarity > 0.0

    async def test_success_signal_action_when_successes_match_query(self, db_session):
        """Mirror: success action embeds == query → success_signal fires."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        query_action = _embed_1024(0)
        far_action = _embed_1024(999)

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE, action_embed=query_action,
            )
        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="failure", profile_embed=_embed_2000(10 + i),
                transition_profile=DEFAULT_PROFILE, action_embed=far_action,
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=query_action,
        )

        result = await analyzer.analyze(current)

        assert result.success_signal_action is not None
        assert result.success_signal_action.is_detected is True
        assert result.fail_risk_action is not None
        assert result.fail_risk_action.is_detected is False

        # mutual exclusivity invariant
        assert not (
            result.fail_risk_action.is_detected
            and result.success_signal_action.is_detected
        )

        assert result.fail_similarity is not None
        assert result.fail_similarity < 0.0

    async def test_action_signals_neutral_when_fail_eq_success(self, db_session):
        """When fail and success candidates have the same action embeds as the
        query, fail_lev ≈ succ_lev → delta ≈ 0 < threshold → both signals
        return is_detected=False."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        shared_action = _embed_1024(0)

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="failure", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE, action_embed=shared_action,
            )
        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(10 + i),
                transition_profile=DEFAULT_PROFILE, action_embed=shared_action,
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=shared_action,
        )

        result = await analyzer.analyze(current)

        assert result.fail_risk_action is not None
        assert result.success_signal_action is not None
        assert result.fail_risk_action.is_detected is False
        assert result.success_signal_action.is_detected is False
        assert result.fail_similarity == pytest.approx(0.0, abs=1e-6)

    async def test_fail_risk_transition_when_triplet_only_in_failures(self, db_session):
        """Last triplet present in failure candidates, absent in success → True."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="failure", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE,
                action_embed=_embed_1024(10 + i),
            )

        # Success candidates with disjoint trace → triplet absent
        other_trace = ["o:text:9", "a:bash:9", "o:text:9"]
        other_profile = {"o:text:9.a:bash:9.o:text:9": 1.0}
        for i in range(2):
            await _make_path(
                db_session, trace=other_trace, action_label="a:bash:9",
                status="success", profile_embed=_embed_2000(20 + i),
                transition_profile=other_profile,
                action_embed=_embed_1024(30 + i),
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(99),
        )

        result = await analyzer.analyze(current)

        assert result.fail_risk_transition is True
        assert result.success_signal_transition is False

    async def test_success_signal_transition_when_triplet_only_in_successes(self, db_session):
        """Mirror: triplet present in success candidates, absent in failures → True."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        for i in range(2):
            await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE,
                action_embed=_embed_1024(10 + i),
            )

        other_trace = ["o:text:9", "a:bash:9", "o:text:9"]
        other_profile = {"o:text:9.a:bash:9.o:text:9": 1.0}
        for i in range(2):
            await _make_path(
                db_session, trace=other_trace, action_label="a:bash:9",
                status="failure", profile_embed=_embed_2000(20 + i),
                transition_profile=other_profile,
                action_embed=_embed_1024(30 + i),
            )

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(99),
        )

        result = await analyzer.analyze(current)

        assert result.success_signal_transition is True
        assert result.fail_risk_transition is False

    async def test_candidate_lists_populated_per_status(self, db_session):
        """`candidates`, `success_candidates`, `failure_candidates` are filled
        from the prefetch pool with their own top_k cuts and status filters."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        success_ids = []
        failure_ids = []
        for i in range(2):
            p = await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="success", profile_embed=_embed_2000(i),
                transition_profile=DEFAULT_PROFILE,
                action_embed=_embed_1024(10 + i),
            )
            success_ids.append(p.id)
        for i in range(2):
            p = await _make_path(
                db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
                status="failure", profile_embed=_embed_2000(20 + i),
                transition_profile=DEFAULT_PROFILE,
                action_embed=_embed_1024(30 + i),
            )
            failure_ids.append(p.id)

        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="pending", profile_embed=_embed_2000(50),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(99),
        )

        result = await analyzer.analyze(current)

        # 4 viable rows in the prefetch pool
        assert result.prefetched_reranked is not None
        assert len(result.prefetched_reranked) == 4

        # candidates = top_k=3 of the mixed pool
        assert result.candidates is not None
        assert len(result.candidates) == 3

        # status pools cap at top_k=3 but there are only 2 of each
        assert result.success_candidates is not None
        assert result.failure_candidates is not None
        assert {c.id for c in result.success_candidates} == set(success_ids)
        assert {c.id for c in result.failure_candidates} == set(failure_ids)
        for c in result.success_candidates:
            assert c.trajectory.status == "success"
        for c in result.failure_candidates:
            assert c.trajectory.status == "failure"

    async def test_excludes_self_trajectory(self, db_session):
        """current_path's own trajectory is excluded from prefetch."""
        repo = TrajectoryPathRepository(db_session)
        analyzer = TransitionAnalyzer(path_repo=repo, config=_config())

        await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(1),
        )

        # current trajectory marked success too — must still be skipped via
        # exclude_trajectory_id, otherwise candidates would include itself.
        current = await _make_path(
            db_session, trace=DEFAULT_TRACE, action_label=ACT_LABEL,
            status="success", profile_embed=_embed_2000(0),
            transition_profile=DEFAULT_PROFILE, action_embed=_embed_1024(2),
        )

        result = await analyzer.analyze(current)

        assert result.candidates is not None
        assert len(result.candidates) == 1
        assert result.candidates[0].trajectory_id != current.trajectory_id
