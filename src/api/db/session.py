"""
src/api/db/session.py
=====================
Async SQLAlchemy engine + session factory + FastAPI dependency.

WHY async:
    FastAPI is an async framework.  Using a synchronous SQLAlchemy engine
    would block the event loop on every DB query — killing concurrency.
    asyncpg is the pure-Python async PostgreSQL driver;
    SQLAlchemy's AsyncEngine wraps it with the full ORM query interface.

WHY pool_pre_ping:
    On cold starts or after a long idle period, the PostgreSQL server may
    have closed idle connections.  pool_pre_ping issues a lightweight
    "SELECT 1" before each checkout and reconnects if the connection is dead.
    Without it, the first request after idle gets a "connection closed" error.

WHY expire_on_commit=False:
    SQLAlchemy's default behavior expires all attributes after commit(), so
    reading `user.email` after `await session.commit()` would trigger a lazy
    load — which is illegal in async context.  expire_on_commit=False keeps
    the in-memory state valid after commit.

WHY get_db as an async generator:
    FastAPI's Depends() mechanism runs the code up to `yield` before the route
    handler, and the code after `yield` after the response is sent.  This
    guarantees that:
      1. Every request gets a fresh session
      2. The session is always closed (even on exception) via the finally block
      3. Rollback happens automatically on unhandled exceptions
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Engine ────────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://deepcoin:deepcoin@localhost:5432/deepcoin",
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # reconnect if server closed idle connections
    pool_size=5,          # 5 persistent connections (single-worker API)
    max_overflow=10,      # allow 10 extra connections under spike load
    echo=False,           # set True in dev to log every SQL statement
)

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields one AsyncSession per request.

    Usage in a route:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    The session is committed, rolled back, and closed automatically.
    Never call session.close() manually inside a route — the dependency handles it.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
