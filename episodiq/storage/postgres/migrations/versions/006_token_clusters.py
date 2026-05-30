"""token_clusters table

Revision ID: 006_token_clusters
Revises: 005_origin_responses
Create Date: 2026-05-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from episodiq.config import get_config


revision: str = '006_token_clusters'
down_revision: Union[str, None] = '005_origin_responses'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'token_clusters',
        sa.Column('id', sa.UUID(), nullable=False),
        # ordinal cluster id (0..n_clusters-1), the integer token in trace_tokens
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('centroid', Vector(2 * get_config().message_dims), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cluster_id', name='uq_token_clusters_cluster_id'),
    )


def downgrade() -> None:
    op.drop_table('token_clusters')
