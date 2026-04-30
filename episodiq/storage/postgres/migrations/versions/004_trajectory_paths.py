"""trajectory_paths table

Revision ID: 004_trajectory_paths
Revises: 003_clusters
Create Date: 2026-03-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '004_trajectory_paths'
down_revision: Union[str, None] = '003_clusters'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'trajectory_paths',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('trajectory_id', sa.UUID(), sa.ForeignKey('trajectories.id'), nullable=False),
        sa.Column('from_observation_id', sa.UUID(), sa.ForeignKey('messages.id'), nullable=False),
        sa.Column('action_message_id', sa.UUID(), sa.ForeignKey('messages.id'), nullable=True),
        sa.Column('to_observation_id', sa.UUID(), sa.ForeignKey('messages.id'), nullable=True),
        sa.Column('transition_profile', sa.JSON(), nullable=True),
        sa.Column('trace', sa.JSON(), server_default='[]', nullable=False),
        sa.Column('trajectory_status', sa.String(20), server_default='pending', nullable=False),
        sa.Column('fail_risk_action_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('fail_risk_transition_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('success_signal_action_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('success_signal_transition_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('loop_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('clock_timestamp()'), nullable=False),
        sa.Column('index', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "(action_message_id IS NULL AND to_observation_id IS NULL) OR "
            "(action_message_id IS NOT NULL AND to_observation_id IS NULL) OR "
            "(action_message_id IS NOT NULL AND to_observation_id IS NOT NULL)",
            name="ck_tp_resolved_or_pending",
        ),
    )

    op.execute("ALTER TABLE trajectory_paths ADD COLUMN profile_embed vector(2000)")

    op.create_index('ix_trajectory_paths_trajectory_id', 'trajectory_paths', ['trajectory_id'])
    op.create_unique_constraint('uq_tp_trajectory_index', 'trajectory_paths', ['trajectory_id', 'index'])

    # Auto-increment index per trajectory
    op.execute("""
        CREATE OR REPLACE FUNCTION set_trajectory_path_index()
        RETURNS TRIGGER AS $$
        BEGIN
            SELECT COALESCE(MAX(index), -1) + 1
            INTO NEW.index
            FROM trajectory_paths
            WHERE trajectory_id = NEW.trajectory_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trigger_set_tp_index
        BEFORE INSERT ON trajectory_paths
        FOR EACH ROW
        EXECUTE FUNCTION set_trajectory_path_index();
    """)
    # Cascade trajectory status to trajectory_paths
    op.execute("""
        CREATE OR REPLACE FUNCTION cascade_trajectory_status()
        RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.status IS DISTINCT FROM NEW.status THEN
                UPDATE trajectory_paths
                SET trajectory_status = NEW.status
                WHERE trajectory_id = NEW.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trigger_cascade_trajectory_status
        AFTER UPDATE OF status ON trajectories
        FOR EACH ROW
        EXECUTE FUNCTION cascade_trajectory_status();
    """)

    # Partial HNSW index: only completed trajectories
    op.execute("""
        CREATE INDEX ix_tp_profile_hnsw ON trajectory_paths
        USING hnsw (profile_embed vector_cosine_ops)
        WHERE trajectory_status IN ('success', 'failure')
    """)


def downgrade() -> None:
    op.execute('DROP TRIGGER IF EXISTS trigger_cascade_trajectory_status ON trajectories')
    op.execute('DROP FUNCTION IF EXISTS cascade_trajectory_status()')
    op.execute('DROP TRIGGER IF EXISTS trigger_set_tp_index ON trajectory_paths')
    op.execute('DROP FUNCTION IF EXISTS set_trajectory_path_index()')
    op.drop_constraint('uq_tp_trajectory_index', 'trajectory_paths')
    op.drop_index('ix_tp_profile_hnsw', table_name='trajectory_paths')
    op.drop_index('ix_trajectory_paths_trajectory_id', table_name='trajectory_paths')
    op.drop_table('trajectory_paths')
