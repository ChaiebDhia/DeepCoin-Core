"""
alembic/env.py
==============
Alembic migration environment — async edition.

WHY async:
    Our SQLAlchemy engine uses asyncpg (PostgreSQL async driver). Alembic's
    default synchronous `run_migrations_online()` pattern cannot connect through
    an async engine. We use the asyncio pattern:
        1. Build an async engine from the config URL.
        2. `async with engine.connect()` inside an async function.
        3. Wrap that function in `asyncio.run()` which Alembic calls synchronously.

WHY NullPool:
    Alembic is a CLI tool — it runs once, applies migrations, and exits.
    A connection pool with recycling logic adds no value here and can leave
    lingering connections that block a DROP DATABASE command during testing.
    NullPool creates one connection, uses it, and closes it immediately.

WHY `include_object`:
    The JSONB + UUID types from PostgreSQL dialects must be included.
    We also exclude the `spatial_ref_sys` table that PostGIS adds (not present
    in this project, but the guard prevents errors if PostGIS is ever enabled).
"""
from __future__ import annotations

import asyncio
import os

from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Import the metadata from our ORM models ──────────────────────────────────
# All models must be imported before Base.metadata is used, otherwise Alembic
# will not detect the tables and will generate empty migrations.
from src.api.db.base import Base
import src.api.db.models  # noqa: F401 — import triggers table registration

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url with DATABASE_URL from the environment (if set).
# This is the production-safe approach: the .ini file has a dev default, but
# the real URL is injected via the environment in CI and Docker.
_db_url = os.getenv("DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

# ── Logging setup (from alembic.ini [loggers] section) ───────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Metadata for autogenerate ─────────────────────────────────────────────────
# `target_metadata` tells Alembic to compare the live DB schema against our
# ORM models and generate the diff.
target_metadata = Base.metadata


def include_object(obj, name: str, type_: str, reflected: bool, compare_to) -> bool:
    """
    Filter for which database objects Alembic should manage.

    WHY:
        Without this, autogenerate would try to DROP/CREATE PostgreSQL internal
        tables such as `spatial_ref_sys` (from PostGIS) or `pg_stat_statements`
        (from the stats extension) which are not part of our schema.
    """
    if type_ == "table" and name.startswith("pg_"):
        return False
    return True


# ── Offline migrations (generate SQL without a live DB connection) ─────────────
def run_migrations_offline() -> None:
    """
    Run migrations without connecting to a live database.

    Output: SQL DDL printed to stdout (useful for review or applying manually).
    Usage:  `alembic upgrade head --sql`
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (connect and apply to a live DB) ─────────────────────────
def do_run_migrations(connection: Connection) -> None:
    """Apply migrations using an existing synchronous-compatible connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Build an async engine, obtain a connection, then run migrations.

    WHY NullPool:
        Alembic is a short-lived CLI process. A pool wastes resources and can
        prevent the process from exiting cleanly.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode (the default)."""
    asyncio.run(run_async_migrations())


# ── Dispatch ──────────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
