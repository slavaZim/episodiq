"""Log entry rendering for trajectory analytics output."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.text import Text


class RenderMode(Enum):
    TAIL = "tail"
    REPORT = "report"


class OutputFormat(Enum):
    PRETTY = "pretty"
    JSON = "json"


@dataclass
class RenderContext:
    """Mutable rendering state passed through render calls."""

    mode: RenderMode
    format: OutputFormat
    verbose: bool = False
    delta_base_ts: datetime | None = None
    prev_ts: datetime | None = None


@dataclass(frozen=True)
class TrajectoryStats:
    """Summary statistics for a completed trajectory."""

    trajectory_id: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    step_count: int
    status: str
    loop_count: int = 0
    unclassified_step_count: int = 0
    fail_similarity: float | None = None
    variance_high_count: int = 0
    variance_low_count: int = 0


# -- Private helpers --

_TS_COL = 10
_TID_COL = 10
_STEP_COL = 5  # "9999 " — right-padded step index
_KIND_COL = 4  # "o:  " / "a:  " — kind marker only; category trails on the right


def _format_timestamp(iso_str: str, ctx: RenderContext) -> str:
    ts = datetime.fromisoformat(iso_str)

    if ctx.mode == RenderMode.TAIL:
        return ts.strftime("%H:%M:%S")

    if ctx.delta_base_ts is None:
        ctx.delta_base_ts = ts
        ctx.prev_ts = ts
        return "+0.0s"

    delta = (ts - ctx.delta_base_ts).total_seconds()
    ctx.prev_ts = ts
    return f"+{delta:.1f}s"


def _fallback_text(entry: dict) -> str:
    entry_type = entry["type"]
    category = entry.get("category") or "unknown"
    return f"unclassified ({entry_type}) ({category})"


def _split_label(label: str) -> tuple[str, str]:
    """``a:execute_bash:33`` → ``("a:", "execute_bash")``. The category
    portion trails the annotation in the rendered line, the kind portion
    leads it as a short marker.
    """
    parts = label.split(":", 2)
    kind = parts[0] + ":" if parts else ""
    category = parts[1] if len(parts) > 1 else ""
    return kind, category


def _obs_right_parts(entry: dict) -> list[tuple[str, str]]:
    """Returns list of (text, style) tuples for observation."""
    parts: list[tuple[str, str]] = []
    sim = entry.get("fail_similarity")
    if sim:
        agg_key = next((k for k in sim if k != "current"), None)
        if agg_key is not None:
            parts.append((
                f"fail_similarity={sim[agg_key]:.2f}",
                "magenta bold",
            ))
    if entry.get("loop"):
        streak = entry.get("loop_streak", 0)
        parts.append((f"loop x{streak}", "yellow bold"))
    return parts


def _act_right_parts(act: dict) -> list[tuple[str, str]]:
    """Returns list of (text, style) tuples for action."""
    parts: list[tuple[str, str]] = []
    if "action_variance" in act:
        parts.append((f"variance={act['action_variance']}", "dim"))
    return parts


class LogRenderer:
    """Renders LogBuilder entry pairs as Rich pretty-print or JSONL."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def render_entry_pair(
        self, obs: dict, act: dict, ctx: RenderContext,
        skip_observation: bool = False,
    ) -> None:
        if ctx.format == OutputFormat.JSON:
            if not skip_observation:
                self._json_line(obs)
            self._json_line(act)
            return

        if not skip_observation:
            self._render_observation(obs, ctx)
        self._render_action(act, ctx)

    def render_trajectory_header(
        self, stats: TrajectoryStats, ctx: RenderContext,
    ) -> None:
        if ctx.format == OutputFormat.JSON:
            return

        status_style = "green" if stats.status == "success" else "red"
        header = Text()
        header.append(f"Trajectory {stats.trajectory_id[:8]} ", style="bold cyan")
        header.append(f"[{stats.status}]", style=f"bold {status_style}")
        header.append(f"  {stats.step_count} steps, {stats.duration_s:.1f}s", style="dim")

        # Signal summary
        if stats.fail_similarity is not None:
            header.append(
                f"  fail_similarity={stats.fail_similarity:.2f}",
                style="magenta bold",
            )
        if stats.variance_high_count > 0 or stats.variance_low_count > 0:
            header.append(
                f"  variance high={stats.variance_high_count} "
                f"low={stats.variance_low_count}",
                style="dim",
            )
        if stats.loop_count > 0:
            header.append(f"  loops={stats.loop_count}", style="yellow")
        if stats.unclassified_step_count > 0:
            header.append(
                f"  unclassified={stats.unclassified_step_count}", style="dim",
            )

        self._console.print(header)
        self._console.rule(style="dim")

    def render_stream_banner(
        self, ctx: RenderContext, filters: dict | None = None,
    ) -> None:
        if ctx.format == OutputFormat.JSON:
            return

        self._console.print("Streaming trajectory logs", style="bold cyan")
        if filters:
            parts = [f"{k}={v}" for k, v in filters.items() if v]
            if parts:
                self._console.print(f"  Filters: {', '.join(parts)}", style="dim")
        self._console.rule(style="dim")

    def render_meta_event(
        self, event: str, ctx: RenderContext, **fields: str,
    ) -> None:
        if ctx.format == OutputFormat.JSON:
            self._json_line({"type": "meta", "event": event, **fields})
            return

        line = Text()
        line.append(f"  [{event}]", style="dim italic")
        for k, v in fields.items():
            line.append(f" {k}={v}", style="dim")
        self._console.print(line)

    # -- Private --

    def _render_observation(self, entry: dict, ctx: RenderContext) -> None:
        line = Text()

        step = entry.get("index")
        if step is not None:
            line.append(f"{step:>{_STEP_COL - 1}d} ", style="dim cyan")

        ts_str = _format_timestamp(entry["timestamp"], ctx)
        line.append(f"{ts_str:<{_TS_COL}}", style="dim")

        if ctx.mode == RenderMode.TAIL:
            tid = entry.get("trajectory_id", "")[:8]
            line.append(f"{tid:<{_TID_COL}}", style="cyan")

        if "label" in entry:
            kind, _ = _split_label(entry["label"])
            line.append(f"{kind:<{_KIND_COL}}", style="dim cyan")

        if "annotation" in entry:
            line.append(entry["annotation"])
        else:
            line.append(_fallback_text(entry), style="dim")

        for text, style in _obs_right_parts(entry):
            line.append("  | ", style="dim")
            line.append(text, style=style)

        self._console.print(line)

    def _render_action(self, act: dict, ctx: RenderContext) -> None:
        line = Text()

        indent = _TS_COL
        if act.get("index") is not None:
            indent += _STEP_COL
        if ctx.mode == RenderMode.TAIL:
            indent += _TID_COL

        line.append(" " * indent)
        line.append("-> ", style="dim")

        if "label" in act:
            kind, _ = _split_label(act["label"])
            line.append(f"{kind:<{_KIND_COL}}", style="dim cyan")

        if "annotation" in act:
            line.append(act["annotation"])
        else:
            line.append(_fallback_text(act), style="dim")

        for text, style in _act_right_parts(act):
            line.append("  | ", style="dim")
            line.append(text, style=style)

        self._console.print(line)

    def _json_line(self, entry: dict) -> None:
        sys.stdout.write(json.dumps(entry, default=str) + "\n")
        sys.stdout.flush()
