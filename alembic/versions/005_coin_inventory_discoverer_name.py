"""add discoverer_name to coin_inventory

Revision ID: 005_coin_inv_disc_name
Revises: 004_coin_inventory
Create Date: 2026-04-24 18:10:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "005_coin_inv_disc_name"
down_revision: Union[str, None] = "004_coin_inventory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("coin_inventory", sa.Column("discoverer_name", sa.String(length=255), nullable=True))
    op.create_index("ix_coin_inventory_discoverer_name", "coin_inventory", ["discoverer_name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_coin_inventory_discoverer_name", table_name="coin_inventory")
    op.drop_column("coin_inventory", "discoverer_name")
