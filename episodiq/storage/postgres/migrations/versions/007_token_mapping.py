"""token_mapping table

Revision ID: 007_token_mapping
Revises: 006_token_clusters
Create Date: 2026-05-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '007_token_mapping'
down_revision: Union[str, None] = '006_token_clusters'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'token_mapping',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('action_label', sa.String(), nullable=False),
        sa.Column('observation_label', sa.String(), nullable=False),
        # FKs to existing clusters table (message-level action/observation clusters)
        sa.Column('action_cluster_id', sa.UUID(),
                  sa.ForeignKey('clusters.id', ondelete='SET NULL'), nullable=True),
        sa.Column('observation_cluster_id', sa.UUID(),
                  sa.ForeignKey('clusters.id', ondelete='SET NULL'), nullable=True),
        # FK to token_clusters (pair_ao token cluster)
        sa.Column('token_cluster_id', sa.UUID(),
                  sa.ForeignKey('token_clusters.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('clock_timestamp()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('action_label', 'observation_label',
                            name='uq_token_mapping_labels'),
    )


def downgrade() -> None:
    op.drop_table('token_mapping')
