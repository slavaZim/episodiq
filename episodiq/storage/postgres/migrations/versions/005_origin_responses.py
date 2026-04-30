"""origin_responses table

Revision ID: 005_origin_responses
Revises: 004_trajectory_paths
Create Date: 2026-03-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '005_origin_responses'
down_revision: Union[str, None] = '004_trajectory_paths'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'origin_responses',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('message_id', sa.UUID(), sa.ForeignKey('messages.id'), nullable=True),
        sa.Column('adapter_id', sa.String(), nullable=False),
        sa.Column('external_id', sa.String(), nullable=True),
        sa.Column('model', sa.String(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_index('ix_origin_responses_message_id', 'origin_responses', ['message_id'])
    op.create_index('ix_origin_responses_external', 'origin_responses', ['adapter_id', 'external_id'])


def downgrade() -> None:
    op.drop_index('ix_origin_responses_external', table_name='origin_responses')
    op.drop_index('ix_origin_responses_message_id', table_name='origin_responses')
    op.drop_table('origin_responses')
