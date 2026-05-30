"""clusters table + cluster_id on messages

Revision ID: 003_clusters
Revises: 002_trajectories
Create Date: 2026-03-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '003_clusters'
down_revision: Union[str, None] = '002_trajectories'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'clusters',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('type', sa.String(20), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('annotation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("type IN ('action', 'observation')", name='ck_clusters_type'),
    )

    op.create_index('ix_clusters_type_category', 'clusters', ['type', 'category'])

    op.add_column('messages', sa.Column('cluster_id', sa.UUID(), sa.ForeignKey('clusters.id', ondelete='SET NULL'), nullable=True))
    op.add_column('messages', sa.Column('category', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('cluster_type', sa.String(20), nullable=True))
    op.create_index('ix_messages_cluster_id', 'messages', ['cluster_id'])
    op.create_index('ix_messages_category', 'messages', ['category'])
    op.create_index('ix_messages_cluster_type_category', 'messages', ['cluster_type', 'category'])


def downgrade() -> None:
    op.drop_index('ix_messages_cluster_type_category', table_name='messages')
    op.drop_index('ix_messages_category', table_name='messages')
    op.drop_index('ix_messages_cluster_id', table_name='messages')
    op.drop_column('messages', 'cluster_type')
    op.drop_column('messages', 'category')
    op.drop_column('messages', 'cluster_id')
    op.drop_index('ix_clusters_type_category', table_name='clusters')
    op.drop_table('clusters')
