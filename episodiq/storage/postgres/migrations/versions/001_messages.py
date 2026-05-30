"""messages table

Revision ID: 001_messages
Revises:
Create Date: 2026-03-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from episodiq.config import get_config

revision: str = '001_messages'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table(
        'messages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('trajectory_id', sa.UUID(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column('index', sa.Integer(), nullable=True),
        sa.Column('embedding', Vector(get_config().message_dims), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_index('ix_messages_trajectory_id', 'messages', ['trajectory_id'])
    op.create_index('ix_messages_trajectory_created', 'messages', ['trajectory_id', 'created_at'])
    op.create_unique_constraint('uq_messages_trajectory_index', 'messages', ['trajectory_id', 'index'])

    op.execute("""
        CREATE INDEX ix_messages_embedding_hnsw ON messages
        USING hnsw (embedding vector_cosine_ops)
    """)

    # Auto-increment index per trajectory
    op.execute("""
        CREATE OR REPLACE FUNCTION set_message_index()
        RETURNS TRIGGER AS $$
        BEGIN
            SELECT COALESCE(MAX(index), -1) + 1
            INTO NEW.index
            FROM messages
            WHERE trajectory_id = NEW.trajectory_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trigger_set_message_index
        BEFORE INSERT ON messages
        FOR EACH ROW
        EXECUTE FUNCTION set_message_index();
    """)


def downgrade() -> None:
    op.execute('DROP TRIGGER IF EXISTS trigger_set_message_index ON messages')
    op.execute('DROP FUNCTION IF EXISTS set_message_index()')
    op.execute('DROP INDEX IF EXISTS ix_messages_embedding_hnsw')
    op.drop_constraint('uq_messages_trajectory_index', 'messages')
    op.drop_index('ix_messages_trajectory_created', table_name='messages')
    op.drop_index('ix_messages_trajectory_id', table_name='messages')
    op.drop_table('messages')
