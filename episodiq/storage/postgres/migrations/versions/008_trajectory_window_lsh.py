"""trajectory_window_lsh table + drop legacy minhash_sig column

Revision ID: 008_trajectory_window_lsh
Revises: 007_token_mapping
Create Date: 2026-06-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '008_trajectory_window_lsh'
down_revision: Union[str, None] = '007_token_mapping'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'trajectory_window_lsh',
        sa.Column('trajectory_id', sa.UUID(), nullable=False),
        sa.Column('window_center', sa.Integer(), nullable=False),
        sa.Column('band_index', sa.SmallInteger(), nullable=False),
        sa.Column('band_hash', sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ['trajectory_id'], ['trajectories.id'], ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint(
            'trajectory_id', 'window_center', 'band_index',
        ),
    )
    op.create_index(
        'ix_twl_band_lookup', 'trajectory_window_lsh',
        ['band_index', 'band_hash'],
    )
    op.create_index(
        'ix_twl_trajectory_id', 'trajectory_window_lsh', ['trajectory_id'],
    )
    # Legacy per-path MinHash is replaced by per-window LSH bands.
    op.drop_column('trajectory_paths', 'minhash_sig')


def downgrade() -> None:
    op.add_column(
        'trajectory_paths',
        sa.Column('minhash_sig', postgresql.ARRAY(sa.BigInteger()), nullable=True),
    )
    op.drop_index('ix_twl_trajectory_id', table_name='trajectory_window_lsh')
    op.drop_index('ix_twl_band_lookup', table_name='trajectory_window_lsh')
    op.drop_table('trajectory_window_lsh')
