"""Tests for LogBuilder (structured log entries from path + analytics)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from episodiq.analytics.log_builder import LogBuilder
from episodiq.analytics.path_frequency import (
    PathFrequencyTagger,
    PathFrequencyThresholds,
)
from episodiq.analytics.transition_types import (
    LoopSignal,
    PathFrequencySignal,
    TrajectoryAnalytics,
)


@dataclass
class FakeCluster:
    annotation: str | None = None


@dataclass
class FakeMessage:
    category: str | None = None
    cluster: FakeCluster | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )


@dataclass
class FakePath:
    id: UUID
    trajectory_id: UUID
    index: int
    from_obs_label: str
    action_label: str | None
    from_observation: FakeMessage = field(default_factory=FakeMessage)
    action_message: FakeMessage = field(default_factory=FakeMessage)


def _path(
    from_label: str = "o:text:greeting",
    action_label: str | None = "a:text:respond",
    from_category: str | None = "text",
    action_category: str | None = "text",
    from_annotation: str | None = None,
    action_annotation: str | None = None,
    index: int = 60,
) -> FakePath:
    return FakePath(
        id=uuid4(),
        trajectory_id=uuid4(),
        index=index,
        from_obs_label=from_label,
        action_label=action_label,
        from_observation=FakeMessage(
            category=from_category,
            cluster=FakeCluster(annotation=from_annotation) if from_annotation else None,
        ),
        action_message=FakeMessage(
            category=action_category,
            cluster=FakeCluster(annotation=action_annotation) if action_annotation else None,
        ),
    )


class TestLogBuilderBuild:
    def test_returns_observation_and_action_pair(self):
        entries = LogBuilder().build(_path(), analytics=None)
        assert [e["type"] for e in entries] == ["observation", "action"]

    def test_observation_carries_label_and_category(self):
        obs, _ = LogBuilder().build(
            _path(from_label="o:text:hello", from_category="text"), analytics=None,
        )
        assert obs["label"] == "o:text:hello"
        assert obs["category"] == "text"

    def test_unclassified_label_flagged(self):
        obs, _ = LogBuilder().build(
            _path(from_label="o:text:?", action_label="a:text:?"), analytics=None,
        )
        assert obs["unclassified"] is True

    def test_fail_similarity_attached_to_observation(self):
        analytics = TrajectoryAnalytics(
            fail_similarity={
                "current": 0.42, "cummax": 0.65, "_count": 7,
            },
        )
        obs, _ = LogBuilder(metric="cummax").build(
            _path(index=60), analytics=analytics,
        )
        assert obs["fail_similarity"] == {"current": 0.42, "cummax": 0.65}

    def test_fail_similarity_no_metric_omits_agg(self):
        """When LogBuilder isn't told which metric to surface (e.g. analytics
        run without a CLI flag), only the raw current value is exposed.
        """
        analytics = TrajectoryAnalytics(
            fail_similarity={
                "current": 0.42, "cummax": 0.65, "_count": 7,
            },
        )
        obs, _ = LogBuilder().build(_path(index=60), analytics=analytics)
        assert obs["fail_similarity"] == {"current": 0.42}

    def test_fail_similarity_suppressed_before_min_step(self):
        """Below the display gate the field is omitted even when the
        running aggregate already had contributors at earlier paths.
        """
        analytics = TrajectoryAnalytics(
            fail_similarity={
                "current": 0.42, "cummax": 0.42, "_count": 1,
            },
        )
        obs, _ = LogBuilder(metric="cummax").build(
            _path(index=10), analytics=analytics,
        )
        assert "fail_similarity" not in obs

    def test_loop_signal_attached_to_observation_only_when_detected(self):
        active_loop = TrajectoryAnalytics(
            loop_signal=LoopSignal(is_detected=True, duplet="a.o", streak=3),
        )
        obs, _ = LogBuilder().build(_path(), analytics=active_loop)
        assert obs.get("loop") is True
        assert obs.get("loop_streak") == 3

        idle_loop = TrajectoryAnalytics(
            loop_signal=LoopSignal(is_detected=False, duplet="a.o", streak=1),
        )
        obs, _ = LogBuilder().build(_path(), analytics=idle_loop)
        assert "loop" not in obs

    def test_action_variance_attached_when_tagger_present(self):
        tagger = PathFrequencyTagger(
            PathFrequencyThresholds(low_entropy=0.5, high_entropy=2.0),
        )
        # entropy=0.1 → below low → "consistent" variance bucket
        analytics = TrajectoryAnalytics(
            path_frequency_signal=PathFrequencySignal(entropy=0.1, n_matches=10),
        )
        _, act = LogBuilder(path_frequency_tagger=tagger).build(
            _path(), analytics=analytics,
        )
        assert "action_variance" in act

    def test_cluster_annotation_propagates(self):
        obs, act = LogBuilder().build(
            _path(from_annotation="from-note", action_annotation="act-note"),
            analytics=None,
        )
        assert obs["annotation"] == "from-note"
        assert act["annotation"] == "act-note"
