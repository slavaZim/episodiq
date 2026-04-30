"""Tests for LogBuilder."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4


from episodiq.analytics.log_builder import LogBuilder
from episodiq.analytics.path_frequency import PathFrequencyTagger, PathFrequencyThresholds
from episodiq.analytics.transition_types import ActionSignal, LoopSignal, TrajectoryAnalytics


def _make_path(
    *,
    from_obs_label="o:text:0",
    action_label="a:text:0",
    annotation=None,
    action_annotation=None,
    trajectory_id=None,
    obs_category="text",
    act_category="text",
):
    tid = trajectory_id or uuid4()
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    from_obs = MagicMock()
    from_obs.created_at = ts
    from_obs.category = obs_category
    from_obs.cluster = MagicMock(annotation=annotation) if annotation else None

    action_msg = MagicMock()
    action_msg.created_at = datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    action_msg.category = act_category
    action_msg.cluster = MagicMock(annotation=action_annotation) if action_annotation else None

    path = MagicMock()
    path.trajectory_id = tid
    path.from_observation = from_obs
    path.action_message = action_msg
    path.from_obs_label = from_obs_label
    path.action_label = action_label
    return path


class TestLogBuilder:

    def test_basic_entries(self):
        """Produces observation + action entries with timestamps and labels."""
        builder = LogBuilder()
        path = _make_path()

        entries, flagged = builder.build(path)

        assert len(entries) == 2
        obs, act = entries

        assert obs["type"] == "observation"
        assert obs["label"] == "o:text:0"
        assert obs["category"] == "text"
        assert obs["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert obs["dead_end_flagged"] is False

        assert act["type"] == "action"
        assert act["label"] == "a:text:0"
        assert act["category"] == "text"
        assert act["timestamp"] == "2025-01-01T12:00:01+00:00"
        assert flagged is False

    def test_category_from_message(self):
        """Category propagated from Message model to entries."""
        builder = LogBuilder()
        path = _make_path(obs_category="text", act_category="bash")

        entries, _ = builder.build(path)
        assert entries[0]["category"] == "text"
        assert entries[1]["category"] == "bash"

    def test_unclassified_observation(self):
        """Observation ending with :? marked unclassified."""
        builder = LogBuilder()
        path = _make_path(from_obs_label="o:text:?")

        entries, _ = builder.build(path)
        assert entries[0]["unclassified"] is True

    def test_unclassified_action(self):
        """Action ending with :? marked unclassified."""
        builder = LogBuilder()
        path = _make_path(action_label="a:text:?")

        entries, _ = builder.build(path)
        assert entries[1]["unclassified"] is True

    def test_annotation_from_cluster(self):
        """Cluster annotations propagated to entries."""
        builder = LogBuilder()
        path = _make_path(annotation="greeting", action_annotation="response")

        entries, _ = builder.build(path)
        assert entries[0]["annotation"] == "greeting"
        assert entries[1]["annotation"] == "response"

    def test_loop_streak(self):
        """loop_signal.is_detected=True adds loop fields."""
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(
            loop_signal=LoopSignal(is_detected=True, transition="a:text:0", repeat_count=5),
            loop_streak=5,
        )

        entries, _ = builder.build(path, analytics=analytics)
        obs = entries[0]
        assert obs["loop"] is True
        assert obs["loop_streak"] == 5

    def test_loop_streak_below_threshold(self):
        """loop_signal.is_detected=False doesn't add loop fields."""
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(
            loop_signal=LoopSignal(is_detected=False, transition="a:text:0", repeat_count=1),
            loop_streak=1,
        )

        entries, _ = builder.build(path, analytics=analytics)
        assert "loop" not in entries[0]

    def test_fail_risk_action_detected(self):
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(
            fail_risk_action=ActionSignal(is_detected=True, similarity=0.12),
        )

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[1]["fail_risk_action"] is True

    def test_fail_risk_action_not_detected(self):
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(
            fail_risk_action=ActionSignal(is_detected=False, similarity=-0.03),
        )

        entries, _ = builder.build(path, analytics=analytics)
        assert "fail_risk_action" not in entries[1]

    def test_success_signal_action_detected(self):
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(
            success_signal_action=ActionSignal(is_detected=True, similarity=0.15),
        )

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[1]["success_signal_action"] is True

    def test_fail_risk_transition(self):
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(fail_risk_transition=True)

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[0]["fail_risk_transition"] is True

    def test_success_signal_transition(self):
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(success_signal_transition=True)

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[0]["success_signal_transition"] is True

    def test_action_variance_low(self):
        """Low entropy → action_variance=low."""
        tagger = PathFrequencyTagger(PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0))
        builder = LogBuilder(path_frequency_tagger=tagger)
        path = _make_path()
        analytics = TrajectoryAnalytics(vote_entropy=0.3)

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[1]["action_variance"] == "low"

    def test_action_variance_high(self):
        """High entropy → action_variance=high."""
        tagger = PathFrequencyTagger(PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0))
        builder = LogBuilder(path_frequency_tagger=tagger)
        path = _make_path()
        analytics = TrajectoryAnalytics(vote_entropy=2.5)

        entries, _ = builder.build(path, analytics=analytics)
        assert entries[1]["action_variance"] == "high"

    def test_no_action_variance_normal(self):
        """Normal entropy → no action_variance field."""
        tagger = PathFrequencyTagger(PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0))
        builder = LogBuilder(path_frequency_tagger=tagger)
        path = _make_path()
        analytics = TrajectoryAnalytics(vote_entropy=1.0)

        entries, _ = builder.build(path, analytics=analytics)
        assert "action_variance" not in entries[1]

    def test_no_action_variance_without_tagger(self):
        """No tagger → no action_variance field."""
        builder = LogBuilder()
        path = _make_path()
        analytics = TrajectoryAnalytics(vote_entropy=0.3)

        entries, _ = builder.build(path, analytics=analytics)
        assert "action_variance" not in entries[1]

    def test_no_action_variance_without_analytics(self):
        """No analytics → no action_variance field."""
        tagger = PathFrequencyTagger(PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0))
        builder = LogBuilder(path_frequency_tagger=tagger)
        path = _make_path()

        entries, _ = builder.build(path)
        assert "action_variance" not in entries[1]

    def test_dead_end_predictor(self):
        """Dead-end predictor runs and flags."""
        predictor = MagicMock()
        predictor.is_available = True
        predictor.predict.return_value = MagicMock(probability=0.85, is_dead_end=True)

        builder = LogBuilder(dead_end_predictor=predictor)
        path = _make_path()
        analytics = TrajectoryAnalytics()

        entries, flagged = builder.build(path, analytics=analytics)
        assert entries[0]["dead_end_prob"] == 0.85
        assert flagged is True
        assert entries[0]["dead_end_flagged"] is True

    def test_dead_end_already_flagged_skips(self):
        """Already flagged → predictor not called."""
        predictor = MagicMock()
        predictor.is_available = True

        builder = LogBuilder(dead_end_predictor=predictor)
        path = _make_path()
        analytics = TrajectoryAnalytics()

        entries, flagged = builder.build(path, analytics=analytics, dead_end_flagged=True)
        predictor.predict.assert_not_called()
        assert flagged is True
        assert entries[0]["dead_end_flagged"] is True
