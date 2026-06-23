"""trajectory_window_lsh: rename window_start → window_center

Revision ID: 011_window_center_rename
Revises: 010_paths_parallel_group
Create Date: 2026-06-21

The column actually stores the window's CENTER step (``first_token +
half_window``), not its starting offset. Migration 008 used the legacy
name ``window_start`` for what was always the center. This migration
renames it on existing databases — on fresh databases migration 008
already creates the column with the correct ``window_center`` name, so
the rename is wrapped in a name-existence guard and is a no-op there.

Postgres' index on the renamed column updates automatically — no need
to drop and recreate ``ix_twl_band_step``.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "011_window_center_rename"
down_revision: Union[str, None] = "010_paths_parallel_group"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trajectory_window_lsh'
                  AND column_name = 'window_start'
            ) THEN
                ALTER TABLE trajectory_window_lsh
                    RENAME COLUMN window_start TO window_center;
            END IF;
        END $$;
        """,
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'trajectory_window_lsh'
                  AND column_name = 'window_center'
            ) THEN
                ALTER TABLE trajectory_window_lsh
                    RENAME COLUMN window_center TO window_start;
            END IF;
        END $$;
        """,
    )
