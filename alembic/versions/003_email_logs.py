"""email_logs

Revision ID: 003_email_logs
Revises: 002_chat_sessions
Create Date: 20260321_180713

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_email_logs'
down_revision: Union[str, None] = '002_chat_sessions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'email_logs',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=True),
        sa.Column('to_email', sa.String(length=255), nullable=False),
        sa.Column('subject', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_email_logs_created_at'), 'email_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_email_logs_user_id'), 'email_logs', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_email_logs_user_id'), table_name='email_logs')
    op.drop_index(op.f('ix_email_logs_created_at'), table_name='email_logs')
    op.drop_table('email_logs')
