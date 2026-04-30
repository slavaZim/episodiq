"""Incremental transition profile / trace calculator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from episodiq.utils import transition_profile_to_vector

if TYPE_CHECKING:
    from episodiq.storage.postgres.models import TrajectoryPath


class PathStateCalculator:
    """Builds transition profile and trace from previous path's completed triplet.

    Profile and trace lag one step behind: they incorporate the previous path's
    (from_obs, action, to_obs) triplet, not the current one.
    """

    def __init__(self, *, decay_lambda: float = 0.8):
        self._decay_lambda = decay_lambda

    def step(
        self,
        prev_path: TrajectoryPath | None,
        obs_label: str,
    ) -> tuple[dict[str, float] | None, list[float] | None, list[str]]:
        """Compute profile, embed, trace for a new path row.

        Args:
            prev_path: Previous TrajectoryPath (None for first observation).
            obs_label: Cluster label of the current observation.

        Returns:
            (profile, embed, trace) — profile/embed are None for first observation.
        """
        if prev_path and prev_path.action_label:
            triplet = f"{prev_path.from_obs_label}.{prev_path.action_label}.{obs_label}"
            profile = {k: v * self._decay_lambda for k, v in (prev_path.transition_profile or {}).items()}
            profile[triplet] = profile.get(triplet, 0.0) + 1.0
            trace = list(prev_path.trace) + [prev_path.action_label, obs_label]
            embed = transition_profile_to_vector(profile)
            return profile, embed, trace

        return None, None, [obs_label]
