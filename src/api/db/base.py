"""
src/api/db/base.py
==================
SQLAlchemy 2 declarative base with constraint naming conventions.

WHY naming conventions:
    Alembic autogenerate compares the current DB schema against the ORM models
    to produce migration scripts.  Without consistent constraint names, Alembic
    cannot match a DB constraint to its ORM definition and generates spurious
    DROP/CREATE pairs on every run.

    The naming_convention dict below follows the Alembic recommended pattern:
        ix_  — index
        uq_  — unique constraint
        ck_  — check constraint
        fk_  — foreign key
        pk_  — primary key

    SQLAlchemy fills in the table/column names automatically.

WHY a shared Base:
    All ORM models inherit from this one Base.  This means:
        Base.metadata.create_all(engine)  creates every table at once.
        Alembic's env.py uses Base.metadata as target_metadata for autogenerate.
    If each model had its own Base, Alembic would only see one model at a time.
"""
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# ── Constraint naming convention (required for Alembic autogenerate) ─────────

_naming_convention: dict[str, str] = {
    "ix":  "ix_%(column_0_label)s",
    "uq":  "uq_%(table_name)s_%(column_0_name)s",
    "ck":  "ck_%(table_name)s_%(constraint_name)s",
    "fk":  "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk":  "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Shared declarative base for all DeepCoin ORM models.

    WHAT: Inheriting from this class registers the model with SQLAlchemy's
          mapping system and attaches it to the shared metadata object.

    WHY one base: Alembic's env.py reads Base.metadata to discover all tables.
                  All models must share the same metadata for autogenerate to work.
    """
    metadata = MetaData(naming_convention=_naming_convention)
