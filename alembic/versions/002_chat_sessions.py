"""add chat_sessions table

Revision ID: 002
Revises: 001
Create Date: 2026-03-03 00:00:00.000000

WHY a separate table instead of embedding in classifications:
    Chat sessions are independent of coin analysis — a user can ask general
    numismatic questions without classifying any coin.  A separate table
    keeps the schema clean and lets us query/delete chat history independently.

WHY JSONB messages:
    Each message is {role, content, sources?, provider?}.  The structure is
    already serialisable JSON (used by the frontend directly), so JSONB is the
    natural choice.  We never need to query inside individual messages.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column(
            "id",
            UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title",      sa.String(200),              nullable=False),
        sa.Column("messages",   JSONB,                        nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_sessions_user_id",     "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_created_at",  "chat_sessions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_created_at", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_user_id",    table_name="chat_sessions")
    op.drop_table("chat_sessions")
