"""Structured log entry builder for trajectory path analytics."""

from episodiq.analytics.path_frequency import PathFrequencyTagger
from episodiq.analytics.transition_types import TrajectoryAnalytics

# Earliest path index at which the retrieval `fail_frac` signal is considered
# reliable enough to surface in the rendered report. Before this point the
# tune corpus has produced too few comparable snapshots for the score to be
# meaningful — leave the entry without `fail_frac` to avoid noise.
MIN_FAIL_FRAC_STEP = 50


class LogBuilder:
    """Build structured log entries from path + analytics signals."""

    def __init__(
        self,
        *,
        path_frequency_tagger: PathFrequencyTagger | None = None,
    ):
        self._tagger = path_frequency_tagger

    def build(
        self,
        path,
        analytics: TrajectoryAnalytics | None = None,
    ) -> list[dict]:
        """Build observation + action log entries for a path."""
        entries = []

        # --- Observation ---
        obs = {
            "type": "observation",
            "timestamp": path.from_observation.created_at.isoformat(),
            "trajectory_id": str(path.trajectory_id),
            "path_id": str(path.id),
            "index": path.index,
            "label": path.from_obs_label,
            "category": getattr(path.from_observation, "category", None),
        }

        from_cluster = getattr(getattr(path, "from_observation", None), "cluster", None)
        if from_cluster and from_cluster.annotation:
            obs["annotation"] = from_cluster.annotation

        if path.from_obs_label.endswith(":?"):
            obs["unclassified"] = True

        if analytics and analytics.loop_signal and analytics.loop_signal.is_detected:
            obs["loop"] = True
            obs["loop_streak"] = analytics.loop_signal.streak

        if (
            analytics
            and analytics.fail_frac is not None
            and path.index >= MIN_FAIL_FRAC_STEP
        ):
            obs["fail_frac"] = analytics.fail_frac

        entries.append(obs)

        # --- Action ---
        act = {
            "type": "action",
            "timestamp": path.action_message.created_at.isoformat(),
            "trajectory_id": str(path.trajectory_id),
            "path_id": str(path.id),
            "index": path.index,
            "label": path.action_label,
            "category": getattr(path.action_message, "category", None),
        }

        action_cluster = getattr(getattr(path, "action_message", None), "cluster", None)
        if action_cluster and action_cluster.annotation:
            act["annotation"] = action_cluster.annotation

        if path.action_label and path.action_label.endswith(":?"):
            act["unclassified"] = True

        if analytics and self._tagger:
            variance = self._tagger.tag_analytics(analytics)
            if variance:
                act["action_variance"] = variance.value

        entries.append(act)

        return entries
