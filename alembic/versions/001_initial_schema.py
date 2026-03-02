"""initial schema — users, classifications, feedback, audit_log, email_verifications, refresh_tokens

Revision ID: 001
Revises:
Create Date: 2026-03-02 00:00:00.000000

WHY explicit DDL instead of --autogenerate:
    Autogenerate requires a live database connection at revision-generation time.
    Writing the migration explicitly means any developer can run `alembic upgrade head`
    from a cold clone without having a running PostgreSQL first — they just need Docker.

TABLE CREATION ORDER (respects FK constraints):
    1. users                  — no FKs
    2. classifications         — FK → users
    3. feedback               — FK → classifications, users
    4. audit_log              — FK → users
    5. email_verifications    — FK → users
    6. refresh_tokens         — FK → users
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enum types ── must be created before the columns that use them
    op.execute("CREATE TYPE user_role AS ENUM ('admin', 'curator', 'analyst')")
    op.execute("CREATE TYPE user_status AS ENUM ('pending', 'active', 'suspended')")

    # ── 1. users ──────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",                 UUID(as_uuid=False), primary_key=True,   server_default=sa.text("gen_random_uuid()")),
        sa.Column("email",              sa.String(255),      nullable=False),
        sa.Column("hashed_password",    sa.String(255),      nullable=False),
        sa.Column("display_name",       sa.String(100),      nullable=True),
        sa.Column("role",               sa.Enum("admin", "curator", "analyst", name="user_role"),   nullable=False, server_default="analyst"),
        sa.Column("status",             sa.Enum("pending", "active", "suspended", name="user_status"), nullable=False, server_default="pending"),
        sa.Column("created_at",         sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",         sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_login_at",      sa.DateTime(timezone=True), nullable=True),
        sa.Column("email_verified_at",  sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── 2. classifications ────────────────────────────────────────────────────
    op.create_table(
        "classifications",
        sa.Column("id",             UUID(as_uuid=False), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",        UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("timestamp",      sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("label",          sa.String(50),  nullable=False),
        sa.Column("confidence",     sa.Float,       nullable=False),
        sa.Column("route_taken",    sa.String(50),  nullable=False),
        sa.Column("image_filename", sa.String(255), nullable=True),
        sa.Column("pdf_path",       sa.String(512), nullable=True),
        sa.Column("payload",        JSONB,          nullable=False),
    )
    op.create_index("ix_classifications_user_id",   "classifications", ["user_id"])
    op.create_index("ix_classifications_timestamp",  "classifications", ["timestamp"])
    op.create_index("ix_classifications_label",      "classifications", ["label"])
    op.create_index("ix_classifications_user_ts",    "classifications", ["user_id", "timestamp"])

    # ── 3. feedback ───────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id",                 UUID(as_uuid=False), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("classification_id",  UUID(as_uuid=False), sa.ForeignKey("classifications.id", ondelete="CASCADE"),  nullable=False),
        sa.Column("user_id",            UUID(as_uuid=False), sa.ForeignKey("users.id",           ondelete="SET NULL"), nullable=True),
        sa.Column("correct_type_id",    sa.String(50),  nullable=False),
        sa.Column("note",               sa.Text,        nullable=True),
        sa.Column("created_at",         sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_feedback_classification_id", "feedback", ["classification_id"])
    op.create_index("ix_feedback_user_id",           "feedback", ["user_id"])

    # ── 4. audit_log ──────────────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id",            UUID(as_uuid=False), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",       UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action",        sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50),  nullable=True),
        sa.Column("resource_id",   sa.String(255), nullable=True),
        sa.Column("payload",       JSONB,          nullable=True),
        sa.Column("ip_address",    sa.String(45),  nullable=True),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_audit_log_user_id",   "audit_log", ["user_id"])
    op.create_index("ix_audit_log_action",    "audit_log", ["action"])
    op.create_index("ix_audit_log_created_at","audit_log", ["created_at"])

    # ── 5. email_verifications ────────────────────────────────────────────────
    op.create_table(
        "email_verifications",
        sa.Column("id",         UUID(as_uuid=False), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",    UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token",      sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at",    sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_email_verifications_user_id", "email_verifications", ["user_id"])
    op.create_index("ix_email_verifications_token",   "email_verifications", ["token"], unique=True)

    # ── 6. refresh_tokens ─────────────────────────────────────────────────────
    op.create_table(
        "refresh_tokens",
        sa.Column("id",          UUID(as_uuid=False), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",     UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash",  sa.String(64),  nullable=False),
        sa.Column("expires_at",  sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at",  sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at",  sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("ip_address",  sa.String(45),  nullable=True),
    )
    op.create_index("ix_refresh_tokens_user_id",    "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True)


def downgrade() -> None:
    # Drop in reverse FK-dependency order
    op.drop_table("refresh_tokens")
    op.drop_table("email_verifications")
    op.drop_table("audit_log")
    op.drop_table("feedback")
    op.drop_table("classifications")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS user_status")
    op.execute("DROP TYPE IF EXISTS user_role")
