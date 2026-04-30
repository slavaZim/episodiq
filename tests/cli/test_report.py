"""Tests for the report CLI command."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from io import StringIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

from rich.console import Console

from episodiq.analytics.transition_types import (
    ActionSignal,
    LoopSignal,
    TrajectoryAnalytics,
)
from episodiq.cli.rendering import (
    LogRenderer,
    OutputFormat,
    RenderContext,
    RenderMode,
    TrajectoryStats,
)
from episodiq.cli.report import _detect_format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(label: str, annotation: str | None = None):
    c = MagicMock()
    c.label = label
    c.annotation = annotation
    return c


def _make_message(
    *,
    role: str = "user",
    category: str = "text",
    cluster_label: str = "o:text:0",
    annotation: str | None = None,
    created_at: datetime | None = None,
):
    msg = MagicMock()
    msg.role = role
    msg.category = category
    msg.cluster_label = cluster_label
    msg.created_at = created_at or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    msg.cluster = _make_cluster(cluster_label, annotation)
    return msg


def _make_path(
    *,
    trajectory_id=None,
    from_obs_label="o:text:0",
    action_label="a:bash:0",
    from_annotation=None,
    action_annotation=None,
    from_category="text",
    action_category="bash",
    from_created_at=None,
    action_created_at=None,
    fail_risk_action_count=0,
):
    tid = trajectory_id or uuid4()
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    path = MagicMock()
    path.trajectory_id = tid
    path.from_obs_label = from_obs_label
    path.action_label = action_label
    path.fail_risk_action_count = fail_risk_action_count

    path.from_observation = _make_message(
        role="user",
        category=from_category,
        cluster_label=from_obs_label,
        annotation=from_annotation,
        created_at=from_created_at or base_time,
    )
    path.action_message = _make_message(
        role="assistant",
        category=action_category,
        cluster_label=action_label,
        annotation=action_annotation,
        created_at=action_created_at or base_time + timedelta(seconds=1),
    )
    return path


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_explicit_json(self):
        assert _detect_format("json") == OutputFormat.JSON

    def test_explicit_pretty(self):
        assert _detect_format("pretty") == OutputFormat.PRETTY

    def test_auto_tty(self):
        with patch("episodiq.cli.report.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = True
            assert _detect_format("auto") == OutputFormat.PRETTY

    def test_auto_pipe(self):
        with patch("episodiq.cli.report.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = False
            assert _detect_format("auto") == OutputFormat.JSON


# ---------------------------------------------------------------------------
# Rendering integration (LogBuilder → Renderer)
# ---------------------------------------------------------------------------

class TestReportRendering:

    def test_pretty_header_and_entries(self):
        """Header shows stats, entries show annotations."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        renderer = LogRenderer(console)

        tid = uuid4()
        stats = TrajectoryStats(
            trajectory_id=str(tid),
            started_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=3,
            status="success",
            fail_risk_action_count=1,
        )
        ctx = RenderContext(mode=RenderMode.REPORT, format=OutputFormat.PRETTY)
        renderer.render_trajectory_header(stats, ctx)

        obs = {
            "type": "observation",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "trajectory_id": str(tid),
            "label": "o:text:0",
            "category": "text",
            "annotation": "User asks about deployment",
            "dead_end_flagged": False,
        }
        act = {
            "type": "action",
            "timestamp": "2025-01-01T12:00:01+00:00",
            "trajectory_id": str(tid),
            "label": "a:bash:0",
            "category": "bash",
            "annotation": "Ran deploy script",
        }
        renderer.render_entry_pair(obs, act, ctx)

        output = buf.getvalue()
        assert str(tid)[:8] in output
        assert "success" in output
        assert "3 steps" in output
        assert "fail_risk_action=1" in output
        assert "User asks about deployment" in output
        assert "Ran deploy script" in output

    def test_json_output_valid_jsonl(self, capsys):
        """JSON format produces valid JSONL lines."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        renderer = LogRenderer(console)
        ctx = RenderContext(mode=RenderMode.REPORT, format=OutputFormat.JSON)

        tid = str(uuid4())
        obs = {
            "type": "observation", "timestamp": "2025-01-01T12:00:00+00:00",
            "trajectory_id": tid, "label": "o:text:0", "category": "text",
            "dead_end_flagged": False,
        }
        act = {
            "type": "action", "timestamp": "2025-01-01T12:00:01+00:00",
            "trajectory_id": tid, "label": "a:bash:0", "category": "bash",
        }
        renderer.render_entry_pair(obs, act, ctx)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "observation"
        assert json.loads(lines[1])["type"] == "action"

    def test_delta_timestamps(self):
        """Report mode shows delta from first entry."""
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        renderer = LogRenderer(console)
        ctx = RenderContext(mode=RenderMode.REPORT, format=OutputFormat.PRETTY)

        tid = str(uuid4())
        base = {"trajectory_id": tid, "category": "text", "dead_end_flagged": False}
        renderer.render_entry_pair(
            {"type": "observation", "timestamp": "2025-01-01T12:00:00+00:00", "label": "o:0", **base},
            {"type": "action", "timestamp": "2025-01-01T12:00:01+00:00", "label": "a:0", **base},
            ctx,
        )
        renderer.render_entry_pair(
            {"type": "observation", "timestamp": "2025-01-01T12:00:10+00:00", "label": "o:0", **base},
            {"type": "action", "timestamp": "2025-01-01T12:00:11+00:00", "label": "a:0", **base},
            ctx,
        )

        output = buf.getvalue()
        assert "+0.0s" in output
        assert "+10.0s" in output


# ---------------------------------------------------------------------------
# LogBuilder integration
# ---------------------------------------------------------------------------

class TestLogBuilderReport:

    def test_build_entries_with_annotations(self):
        from episodiq.analytics.log_builder import LogBuilder

        builder = LogBuilder()
        path = _make_path(from_annotation="Hello", action_annotation="Ran cmd")
        entries, flagged = builder.build(path, None, False)

        assert len(entries) == 2
        assert entries[0]["annotation"] == "Hello"
        assert entries[1]["annotation"] == "Ran cmd"
        assert flagged is False

    def test_build_entries_unannotated(self):
        from episodiq.analytics.log_builder import LogBuilder

        builder = LogBuilder()
        path = _make_path()
        entries, _ = builder.build(path, None, False)

        assert "annotation" not in entries[0]
        assert "annotation" not in entries[1]

    def test_loop_signal_in_entry(self):
        from episodiq.analytics.log_builder import LogBuilder

        builder = LogBuilder()
        path = _make_path(from_annotation="X")
        analytics = TrajectoryAnalytics(
            loop_signal=LoopSignal(is_detected=True, transition="a:text:0", repeat_count=3),
            loop_streak=3,
        )
        entries, _ = builder.build(path, analytics, False)
        assert entries[0].get("loop") is True
        assert entries[0]["loop_streak"] == 3

    def test_fail_risk_action_in_entry(self):
        from episodiq.analytics.log_builder import LogBuilder

        builder = LogBuilder()
        path = _make_path(action_annotation="Y")
        analytics = TrajectoryAnalytics(
            fail_risk_action=ActionSignal(is_detected=True, similarity=0.15),
        )
        entries, _ = builder.build(path, analytics, False)
        assert entries[1].get("fail_risk_action") is True

    def test_dead_end_flagged_sticky(self):
        from episodiq.analytics.log_builder import LogBuilder

        builder = LogBuilder()
        path1 = _make_path(from_annotation="A")
        path2 = _make_path(from_annotation="B")

        entries1, flagged = builder.build(path1, None, False)
        assert entries1[0]["dead_end_flagged"] is False

        entries2, flagged = builder.build(path2, None, True)
        assert entries2[0]["dead_end_flagged"] is True
        assert flagged is True


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

class TestStatsComputation:

    def test_unannotated_count(self):
        entry_pairs = [
            ({"annotation": "X"}, {"annotation": "Y"}),
            ({"annotation": "X"}, {}),
            ({}, {"annotation": "Y"}),
            ({}, {}),
        ]
        unannotated = sum(
            1 for obs, act in entry_pairs
            if "annotation" not in obs or "annotation" not in act
        )
        assert unannotated == 3

    def test_dead_end_first_step(self):
        entry_pairs = [
            ({"dead_end_flagged": False}, {}),
            ({"dead_end_flagged": False}, {}),
            ({"dead_end_flagged": True}, {}),
            ({"dead_end_flagged": True}, {}),
        ]
        first = next(
            (i for i, (obs, _) in enumerate(entry_pairs) if obs.get("dead_end_flagged")),
            None,
        )
        assert first == 2

    def test_dead_end_none_when_not_flagged(self):
        entry_pairs = [
            ({"dead_end_flagged": False}, {}),
            ({"dead_end_flagged": False}, {}),
        ]
        first = next(
            (i for i, (obs, _) in enumerate(entry_pairs) if obs.get("dead_end_flagged")),
            None,
        )
        assert first is None


# ---------------------------------------------------------------------------
# Header rendering
# ---------------------------------------------------------------------------

class TestTrajectoryHeader:

    def test_failure_status(self):
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        renderer = LogRenderer(console)

        stats = TrajectoryStats(
            trajectory_id=str(uuid4()),
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=5,
            status="failure",
        )
        ctx = RenderContext(mode=RenderMode.REPORT, format=OutputFormat.PRETTY)
        renderer.render_trajectory_header(stats, ctx)

        output = buf.getvalue()
        assert "failure" in output
        assert "5 steps" in output

    def test_unannotated_in_header(self):
        buf = StringIO()
        console = Console(file=buf, no_color=True, width=120)
        renderer = LogRenderer(console)

        stats = TrajectoryStats(
            trajectory_id=str(uuid4()),
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=10,
            status="success",
            unannotated_step_count=4,
        )
        ctx = RenderContext(mode=RenderMode.REPORT, format=OutputFormat.PRETTY)
        renderer.render_trajectory_header(stats, ctx)

        output = buf.getvalue()
        assert "unannotated=4" in output
