"""
src/api/db/__init__.py
======================
Database layer package.

Re-exports the three public symbols every module needs:
    Base        — DeclarativeBase shared across all ORM models
    get_db      — FastAPI dependency yielding an AsyncSession
    AsyncSessionLocal — for use outside request context (background tasks)
"""
from src.api.db.base    import Base                          # noqa: F401
from src.api.db.session import get_db, AsyncSessionLocal     # noqa: F401
