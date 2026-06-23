"""trajectory_window_lsh: composite (band_index, band_hash, window_center) index

Revision ID: 009_lsh_band_step_index
Revises: 008_trajectory_window_lsh
Create Date: 2026-06-16

Replaces ``ix_twl_band_lookup`` ((band_index, band_hash)) with a wider
composite index that also covers ``window_center``. The cascade Stage-1
lookup is ``band IN (...) AND window_center BETWEEN s-w AND s+w`` (step
range narrows candidates to anchors temporally aligned with the query)
— adding window_center to the index lets postgres do an index-only
scan instead of a heap fetch per matching band row.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "009_lsh_band_step_index"
down_revision: Union[str, None] = "008_trajectory_window_lsh"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_twl_band_step",
        "trajectory_window_lsh",
        ["band_index", "band_hash", "window_center"],
    )
    op.drop_index("ix_twl_band_lookup", table_name="trajectory_window_lsh")


def downgrade() -> None:
    op.create_index(
        "ix_twl_band_lookup",
        "trajectory_window_lsh",
        ["band_index", "band_hash"],
    )
    op.drop_index("ix_twl_band_step", table_name="trajectory_window_lsh")
