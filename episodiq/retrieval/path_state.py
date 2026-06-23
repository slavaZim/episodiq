"""Incremental trace / token state calculator + per-window LSH bands.

Walks a trajectory one path at a time: each ``token_step`` resolves the
act_obs pair to an ordinal via ``TokenAssigner``, appends it to the
previous path's ``trace_tokens``, and emits LSH band hashes for the new
window that formed at this step (once trace reaches ``W = 2w`` tokens).

A window of width ``W`` is identified by its *center* position
``step = first_token_index + w``. The last window of a trace of length
``L`` covers tokens ``[L-W, L)`` and has ``step = L - w``.

Callers persist:
  * ``trace_tokens`` on ``trajectory_paths`` (cumulative);
  * one row per band in ``trajectory_window_lsh`` for the ``WindowSig``
    returned by ``token_step``, keyed by ``(trajectory_id, step,
    band_index)``.

Stateless — prev_path carries the cumulative trace; each call is a pure
function of ``(prev_path, a_cluster_id, o_cluster_id, action_category)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from episodiq.clustering.tokenizer.assigner import TokenAssigner
from episodiq.config.retrieval_config import WindowMinHashConfig
from episodiq.retrieval.window_minhash import WindowMinHasher

if TYPE_CHECKING:
    from episodiq.storage.postgres.models import TrajectoryPath


@dataclass(frozen=True)
class WindowSig:
    """LSH bands for a single window of a path's ``trace_tokens``.

    ``step`` is the center position of the window (= first_token + w).
    Window content is ``tokens[step - w : step + w]`` (length W).
    Each band in ``bands`` becomes one row in ``trajectory_window_lsh``
    keyed by ``(trajectory_id, step, band_index)``.
    """
    step: int
    bands: list[int]


@dataclass(frozen=True)
class ActObs:
    """One ``(action, observation)`` pair from a trajectory path.

    The triplet ``(a_cluster_id, o_cluster_id, action_category)`` is what
    the ``TokenAssigner`` needs to resolve a token ordinal.
    """
    a_cluster_id: UUID | None
    o_cluster_id: UUID | None
    action_category: str | None

    @classmethod
    def from_path(cls, path: "TrajectoryPath") -> "ActObs":
        """Pull the act-obs triplet from a path's joined messages."""
        action = path.action_message
        obs = path.to_observation
        return cls(
            a_cluster_id=action.cluster_id if action else None,
            o_cluster_id=obs.cluster_id if obs else None,
            action_category=action.category if action else None,
        )


class PathStateCalculator:
    """Per-path incremental state: trace, trace_tokens, and per-window LSH
    bands. Stateless — holds only immutable hash tables and the assigner.
    """

    def __init__(
        self,
        assigner: TokenAssigner | None = None,
        hasher: WindowMinHasher | None = None,
    ) -> None:
        self._assigner = assigner
        # Default to env-driven config so prod picks up overrides; tests
        # inject a custom hasher to exercise specific window sizes.
        self._wmh = hasher or WindowMinHasher(WindowMinHashConfig.from_env())
        self._cfg = self._wmh.config

    def granular_step(
        self,
        prev_path: TrajectoryPath | None,
        obs_label: str,
    ) -> list[str]:
        """Trace of alternating obs/action cluster labels; lags one step
        behind the current observation.
        """
        if prev_path and prev_path.action_label:
            return list(prev_path.trace) + [prev_path.action_label, obs_label]
        return [obs_label]

    async def token_step(
        self,
        prev_path: TrajectoryPath | None,
        act_obs: ActObs | list[ActObs],
    ) -> tuple[list[int], list[WindowSig]]:
        """Append token(s) and emit any LSH windows that form.

        ``act_obs`` is either a single ``ActObs`` (sequential step —
        appends at most one token) or a ``list[ActObs]`` (parallel
        tool-call batch — resolves N tokens, sorts ASC for ordering
        invariance, and appends them all). Unresolved tokens (assigner
        returns ``None``) are skipped — previous tokens carried forward.
        Returns the cumulative ``trace_tokens`` and the list of windows
        whose right edge landed inside the appended span.
        """
        if self._assigner is None:
            raise RuntimeError(
                "token_step requires a TokenAssigner; construct "
                "PathStateCalculator(assigner=...)",
            )
        if isinstance(act_obs, ActObs):
            act_obs = [act_obs]
        prev_tokens = (prev_path.trace_tokens if prev_path else None) or []
        new_tokens: list[int] = []
        for ao in act_obs:
            if ao.a_cluster_id is None or ao.o_cluster_id is None:
                continue
            token = await self._assigner.assign(
                ao.a_cluster_id, ao.o_cluster_id, ao.action_category,
            )
            if token is not None:
                new_tokens.append(token)
        if not new_tokens:
            return list(prev_tokens), []
        # Canonical order within a parallel batch — sorted ASC makes the
        # token sequence invariant to the model's tool_call ordering.
        if len(new_tokens) > 1:
            new_tokens.sort()
        tokens = list(prev_tokens) + new_tokens
        W = self._cfg.window
        wins: list[WindowSig] = []
        for k in range(len(prev_tokens), len(tokens)):
            right_edge = k + 1
            if right_edge < W:
                continue
            first = right_edge - W
            step = first + self._cfg.half_window
            bands = self._wmh.bands_for_window(tokens, first)
            wins.append(WindowSig(step=step, bands=[int(b) for b in bands]))
        return tokens, wins
