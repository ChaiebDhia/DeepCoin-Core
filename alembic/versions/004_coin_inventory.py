"""add coin_inventory table

Revision ID: 004_coin_inventory
Revises: 003_email_logs
Create Date: 2026-04-24 12:20:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision: str = "004_coin_inventory"
down_revision: Union[str, None] = "003_email_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "coin_inventory",
        sa.Column("id", UUID(as_uuid=False), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("type_id", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("denomination", sa.String(length=120), nullable=False),
        sa.Column("authority", sa.String(length=255), nullable=True),
        sa.Column("region", sa.String(length=120), nullable=True),
        sa.Column("mint", sa.String(length=120), nullable=True),
        sa.Column("date_range", sa.String(length=120), nullable=True),
        sa.Column("material", sa.String(length=80), nullable=True),
        sa.Column("obverse", sa.Text(), nullable=True),
        sa.Column("reverse", sa.Text(), nullable=True),
        sa.Column("provenance", sa.Text(), nullable=True),
        sa.Column("source_name", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("source_type", sa.String(length=50), server_default="manual", nullable=False),
        sa.Column("cartography", sa.Text(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("in_training_set", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("ai_prefilled", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("ai_confidence", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("gallery_images", JSONB, server_default="[]", nullable=False),
        sa.Column("created_by_user_id", UUID(as_uuid=False), nullable=True),
        sa.Column("updated_by_user_id", UUID(as_uuid=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["updated_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("type_id", name="uq_coin_inventory_type_id"),
    )

    op.create_index("ix_coin_inventory_type_id", "coin_inventory", ["type_id"], unique=True)
    op.create_index("ix_coin_inventory_denomination", "coin_inventory", ["denomination"], unique=False)
    op.create_index("ix_coin_inventory_authority", "coin_inventory", ["authority"], unique=False)
    op.create_index("ix_coin_inventory_region", "coin_inventory", ["region"], unique=False)
    op.create_index("ix_coin_inventory_mint", "coin_inventory", ["mint"], unique=False)
    op.create_index("ix_coin_inventory_material", "coin_inventory", ["material"], unique=False)
    op.create_index("ix_coin_inventory_created_by_user_id", "coin_inventory", ["created_by_user_id"], unique=False)
    op.create_index("ix_coin_inventory_updated_by_user_id", "coin_inventory", ["updated_by_user_id"], unique=False)
    op.create_index("ix_coin_inventory_created_at", "coin_inventory", ["created_at"], unique=False)
    op.create_index("ix_coin_inventory_updated_at", "coin_inventory", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_coin_inventory_updated_at", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_created_at", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_updated_by_user_id", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_created_by_user_id", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_material", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_mint", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_region", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_authority", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_denomination", table_name="coin_inventory")
    op.drop_index("ix_coin_inventory_type_id", table_name="coin_inventory")
    op.drop_table("coin_inventory")
