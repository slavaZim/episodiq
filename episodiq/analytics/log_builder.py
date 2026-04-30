"""Structured log entry builder for trajectory path analytics."""

from episodiq.analytics.dead_end.inference import DeadEndPredictor
from episodiq.analytics.path_frequency import PathFrequencyTagger
from episodiq.analytics.transition_types import TrajectoryAnalytics


class LogBuilder:
    """Build structured log entries from path + analytics signals.

    Runs dead-end prediction if predictor is available and not already flagged.
    """

    def __init__(
        self,
        *,
        dead_end_predictor: DeadEndPredictor | None = None,
        path_frequency_tagger: PathFrequencyTagger | None = None,
    ):
        self._predictor = dead_end_predictor
        self._tagger = path_frequency_tagger

    def build(
        self,
        path,
        analytics: TrajectoryAnalytics | None = None,
        dead_end_flagged: bool = False,
    ) -> tuple[list[dict], bool]:
        """Build log entries.

        Returns:
            (entries, updated dead_end_flagged).
        """
        entries = []

        # --- Observation ---
        obs = {
            "type": "observation",
            "timestamp": path.from_observation.created_at.isoformat(),
            "trajectory_id": str(path.trajectory_id),
            "path_id": str(path.id),
            "label": path.from_obs_label,
            "category": getattr(path.from_observation, "category", None),
        }

        from_cluster = getattr(getattr(path, "from_observation", None), "cluster", None)
        if from_cluster and from_cluster.annotation:
            obs["annotation"] = from_cluster.annotation

        if path.from_obs_label.endswith(":?"):
            obs["unclassified"] = True

        if analytics and analytics.fail_risk_transition:
            obs["fail_risk_transition"] = True
        if analytics and analytics.success_signal_transition:
            obs["success_signal_transition"] = True

        if analytics and analytics.loop_signal and analytics.loop_signal.is_detected:
            obs["loop"] = True
            obs["loop_streak"] = analytics.loop_streak

        if analytics and not dead_end_flagged and self._predictor and self._predictor.is_available:
            prediction = self._predictor.predict(path, analytics)
            if prediction:
                obs["dead_end_prob"] = prediction.probability
                if prediction.is_dead_end:
                    dead_end_flagged = True

        obs["dead_end_flagged"] = dead_end_flagged
        entries.append(obs)

        # --- Action ---
        act = {
            "type": "action",
            "timestamp": path.action_message.created_at.isoformat(),
            "trajectory_id": str(path.trajectory_id),
            "path_id": str(path.id),
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

        if analytics and analytics.fail_risk_action and analytics.fail_risk_action.is_detected:
            act["fail_risk_action"] = True
        if analytics and analytics.success_signal_action and analytics.success_signal_action.is_detected:
            act["success_signal_action"] = True

        entries.append(act)

        return entries, dead_end_flagged
