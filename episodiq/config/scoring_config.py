"""Scoring config for the agg-shift / cascade retrieval pipelines."""

import os
from dataclasses import dataclass

from .retrieval_config import (
    retrieval_window_from_env,
    validate_retrieval_window,
)


PENALTY_SHAPES: tuple[str, ...] = ("lin", "const", "quad", "gauss")


@dataclass(frozen=True)
class AggShiftConfig:
    """Hyperparameters for agg-shift scoring over a Lev cost grid.

    ``window`` shares the W = 2w token window with
    ``WindowMinHashConfig.window`` (single source: ``EPISODIQ_RETRIEVAL_WINDOW``).
    The agg-shift search radius is ``half_window = W/2 = w`` — same as the
    cascade's per-anchor neighborhood, so the rerank's notion of "local"
    matches Stage-1's.

    ``penalty_shape`` selects the shift-penalty kernel applied per query row
    when aggregating costs across candidate shifts ``d ∈ [-w, w]``:

      - ``"lin"``:  ``M[i, j] + lam · |d|``                  — linear
      - ``"const"``: ``M[i, j] + lam · (d != 0)``             — flat
      - ``"quad"``:  ``M[i, j] + lam · d²``                  — quadratic
      - ``"gauss"``: ``M[i, j] + lam · (1 - exp(-d²/(2σ²)))`` — Gaussian

    ``sigma`` is only used by the Gaussian shape; the other three
    ignore it.

    When ``gap_open == gap_extend == 1.0`` the scorer takes the
    uniform-Lev fast path; otherwise the Gotoh affine kernel is used.
    """
    window: int = 10                  # = W = 2w; same as WindowMinHashConfig.window.
    lam: float = 1.0                  # shift-penalty coefficient.
    penalty_shape: str = "lin"        # one of PENALTY_SHAPES.
    gap_open: float = 1.0             # affine gap opening penalty.
    gap_extend: float = 1.0           # affine gap extension penalty.
    sigma: float = 1.0                # Gaussian-only: shift std-dev.

    def __post_init__(self) -> None:
        validate_retrieval_window(self.window)
        if self.penalty_shape not in PENALTY_SHAPES:
            raise ValueError(
                f"penalty_shape must be one of {PENALTY_SHAPES}; "
                f"got {self.penalty_shape!r}",
            )

    @property
    def half_window(self) -> int:
        """``w = W / 2`` — agg-shift max_shift radius."""
        return self.window // 2

    @property
    def is_uniform(self) -> bool:
        return self.gap_open == 1.0 and self.gap_extend == 1.0

    @classmethod
    def from_env(cls) -> "AggShiftConfig":
        return cls(
            window=retrieval_window_from_env(),
            lam=float(os.getenv("EPISODIQ_AS_LAM", "1.0")),
            penalty_shape=os.getenv("EPISODIQ_AS_PENALTY_SHAPE", "lin"),
            gap_open=float(os.getenv("EPISODIQ_AS_GAP_OPEN", "1.0")),
            gap_extend=float(os.getenv("EPISODIQ_AS_GAP_EXTEND", "1.0")),
            sigma=float(os.getenv("EPISODIQ_AS_SIGMA", "1.0")),
        )
