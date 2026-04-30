"""Tests for rendering package."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from uuid import uuid4

import pytest
from rich.console import Console

from episodiq.cli.rendering import (
    LogRenderer,
    OutputFormat,
    RenderContext,
    RenderMode,
    TrajectoryStats,
)


@pytest.fixture
def console_buf():
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return console, buf


def _obs(
    *,
    annotation=None,
    category="text",
    label="o:text:0",
    loop=False,
    loop_streak=None,
    dead_end_prob=None,
    trajectory_id=None,
    timestamp="2025-01-01T12:00:00+00:00",
):
    entry = {
        "type": "observation",
        "timestamp": timestamp,
        "trajectory_id": trajectory_id or str(uuid4()),
        "label": label,
        "category": category,
        "dead_end_flagged": False,
    }
    if annotation is not None:
        entry["annotation"] = annotation
    if loop:
        entry["loop"] = True
        if loop_streak is not None:
            entry["loop_streak"] = loop_streak
    if dead_end_prob is not None:
        entry["dead_end_prob"] = dead_end_prob
    return entry


def _act(
    *,
    annotation=None,
    category="text",
    label="a:text:0",
    fail_risk_action=False,
    action_variance=None,
    trajectory_id=None,
    timestamp="2025-01-01T12:00:01+00:00",
):
    entry = {
        "type": "action",
        "timestamp": timestamp,
        "trajectory_id": trajectory_id or str(uuid4()),
        "label": label,
        "category": category,
    }
    if annotation is not None:
        entry["annotation"] = annotation
    if fail_risk_action:
        entry["fail_risk_action"] = True
    if action_variance is not None:
        entry["action_variance"] = action_variance
    return entry


def _tail_ctx():
    return RenderContext(mode=RenderMode.TAIL, format=OutputFormat.PRETTY)


def _report_ctx():
    return RenderContext(mode=RenderMode.REPORT, format=OutputFormat.PRETTY)


def _json_ctx(mode=RenderMode.TAIL):
    return RenderContext(mode=mode, format=OutputFormat.JSON)


# -- Timestamp formatting --

class TestTimestampFormatting:

    def test_tail_shows_hms(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(), _tail_ctx())
        output = buf.getvalue()
        assert "12:00:00" in output

    def test_report_first_entry_zero_delta(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(), _report_ctx())
        output = buf.getvalue()
        assert "+0.0s" in output

    def test_report_delta_from_base(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        ctx = _report_ctx()
        r.render_entry_pair(
            _obs(timestamp="2025-01-01T12:00:00+00:00"),
            _act(timestamp="2025-01-01T12:00:01+00:00"),
            ctx,
        )
        r.render_entry_pair(
            _obs(timestamp="2025-01-01T12:00:03+00:00"),
            _act(timestamp="2025-01-01T12:00:04+00:00"),
            ctx,
        )
        output = buf.getvalue()
        assert "+3.0s" in output


# -- Pretty rendering --

class TestRenderEntryPairPretty:

    def test_annotated_observation(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(annotation="User asks about deployment"),
            _act(annotation="Ran bash command"),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "User asks about deployment" in output

    def test_annotated_action(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(annotation="Question"),
            _act(annotation="Ran bash command"),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "Ran bash command" in output

    def test_unannotated_observation_fallback(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(category="text"), _act(), _tail_ctx())
        output = buf.getvalue()
        assert "unclassified (observation) (text)" in output

    def test_unannotated_action_fallback(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(category="bash"), _tail_ctx())
        output = buf.getvalue()
        assert "unclassified (action) (bash)" in output

    def test_loop_annotation(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(annotation="X", loop=True, loop_streak=5),
            _act(),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "loop x5" in output

    def test_fail_risk_action_annotation(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(),
            _act(annotation="Y", fail_risk_action=True),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "fail-risk action" in output

    def test_variance_annotation(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(),
            _act(annotation="Y", action_variance="high"),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "variance=high" in output

    def test_dead_end_on_action_line(self, console_buf):
        """dead_end_prob from obs dict rendered on action line."""
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(dead_end_prob=0.85),
            _act(annotation="Y"),
            _tail_ctx(),
        )
        output = buf.getvalue()
        lines = output.strip().split("\n")
        # dead-end should be on second line (action)
        assert "dead-end p=0.85" in lines[1]

    def test_tail_includes_trajectory_id(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        tid = "abcd1234-0000-0000-0000-000000000000"
        r.render_entry_pair(
            _obs(annotation="X", trajectory_id=tid),
            _act(),
            _tail_ctx(),
        )
        output = buf.getvalue()
        assert "abcd1234" in output

    def test_report_omits_trajectory_id(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        tid = "abcd1234-0000-0000-0000-000000000000"
        r.render_entry_pair(
            _obs(annotation="X", trajectory_id=tid),
            _act(),
            _report_ctx(),
        )
        output = buf.getvalue()
        assert "abcd1234" not in output

    def test_action_has_arrow_prefix(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(annotation="Y"), _tail_ctx())
        output = buf.getvalue()
        assert "-> " in output

    def test_multiple_right_annotations(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(
            _obs(dead_end_prob=0.9),
            _act(annotation="Y", fail_risk_action=True, action_variance="low"),
            _tail_ctx(),
        )
        output = buf.getvalue()
        lines = output.strip().split("\n")
        action_line = lines[1]
        assert "fail-risk action" in action_line
        assert "variance=low" in action_line
        assert "dead-end p=0.90" in action_line


# -- JSON rendering --

class TestRenderEntryPairJson:

    def test_json_two_lines(self, console_buf, capsys):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(), _json_ctx())
        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "observation"
        assert json.loads(lines[1])["type"] == "action"

    def test_json_preserves_fields(self, console_buf, capsys):
        console, buf = console_buf
        r = LogRenderer(console)
        obs = _obs(annotation="hello", loop=True, loop_streak=3)
        r.render_entry_pair(obs, _act(), _json_ctx())
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip().split("\n")[0])
        assert parsed["annotation"] == "hello"
        assert parsed["loop"] is True
        assert parsed["loop_streak"] == 3

    def test_json_bypasses_console(self, console_buf, capsys):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_entry_pair(_obs(), _act(), _json_ctx())
        assert buf.getvalue() == ""


# -- Trajectory header --

class TestRenderTrajectoryHeader:

    def test_header_shows_id_and_status(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        stats = TrajectoryStats(
            trajectory_id="abcd1234-full-uuid",
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=10,
            status="success",
        )
        r.render_trajectory_header(stats, _report_ctx())
        output = buf.getvalue()
        assert "abcd1234" in output
        assert "success" in output
        assert "10 steps" in output

    def test_header_shows_fail_risk_action_count(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        stats = TrajectoryStats(
            trajectory_id="x" * 36,
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=10,
            status="failure",
            fail_risk_action_count=3,
        )
        r.render_trajectory_header(stats, _report_ctx())
        output = buf.getvalue()
        assert "fail_risk_action=3" in output

    def test_header_json_noop(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        stats = TrajectoryStats(
            trajectory_id="x" * 36,
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc),
            duration_s=300.0,
            step_count=10,
            status="success",
        )
        r.render_trajectory_header(stats, _json_ctx())
        assert buf.getvalue() == ""


# -- Stream banner --

class TestRenderStreamBanner:

    def test_banner_shows_title(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_stream_banner(_tail_ctx())
        output = buf.getvalue()
        assert "Streaming trajectory logs" in output

    def test_banner_shows_filters(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_stream_banner(_tail_ctx(), filters={"status": "active"})
        output = buf.getvalue()
        assert "status=active" in output

    def test_banner_json_noop(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_stream_banner(_json_ctx())
        assert buf.getvalue() == ""


# -- Meta events --

class TestRenderMetaEvent:

    def test_meta_pretty(self, console_buf):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_meta_event("reconnecting", _tail_ctx(), reason="timeout")
        output = buf.getvalue()
        assert "reconnecting" in output
        assert "reason=timeout" in output

    def test_meta_json(self, console_buf, capsys):
        console, buf = console_buf
        r = LogRenderer(console)
        r.render_meta_event("dropped_events", _json_ctx(), count="5")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["type"] == "meta"
        assert parsed["event"] == "dropped_events"
        assert parsed["count"] == "5"
