"""trajectory_paths: parallel_group for tool-call batches

Revision ID: 010_trajectory_paths_parallel_group
Revises: 009_lsh_band_step_index
Create Date: 2026-06-18

Adds ``parallel_group`` to ``trajectory_paths``. When an assistant message
emits multiple tool_calls in a single batch, every resulting path carries
the same ``parallel_group`` (= the assistant message's own ``index``).
Sequential paths leave it NULL. The tokenizer uses it to sort tokens
within a group so their order becomes invariant to the original tool-call
ordering chosen by the model.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010_paths_parallel_group"
down_revision: Union[str, None] = "009_lsh_band_step_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trajectory_paths",
        sa.Column("parallel_group", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trajectory_paths", "parallel_group")
