"""
tests/integration/conftest.py
==============================
Shared fixtures for all integration tests.

WHAT
----
This module sets up a lightweight test environment that mirrors production
without requiring any external services:
  - No PostgreSQL  → get_db overridden with AsyncMock
  - No GPU / model files → Gatekeeper replaced with _MockGatekeeper
  - No LLM API key → chat falls through to structured fallback
  - No real uploads directory → classify cleans up after itself

HOW
---
1. Environment variables set BEFORE any src.* import (module-level code).
   WHY: src/api/db/session.py creates the SQLAlchemy engine at module-import
   time by calling os.getenv("DATABASE_URL"). Once that module is imported,
   the engine URL is fixed. Setting env vars here ensures test config is
   picked up on first import.

2. Gatekeeper patched at session scope (once for all tests).
   WHY session scope: loading the mock once avoids the overhead of patching
   and un-patching for every test, and the mock is stateless.

3. get_db dependency overridden per-test with AsyncMock.
   WHY AsyncMock: The real session executes async SQL against PostgreSQL.
   AsyncMock intercepts every await call (execute, commit, rollback) and
   returns a MagicMock — no real DB connection ever established.

4. Auth dependencies overridden for tests that need an authenticated user.

5. httpx.AsyncClient with ASGITransport — no real network, no external port.
   The ASGI transport calls the app's lifespan, so startup/shutdown hooks
   run exactly as in production, with the Gatekeeper patched.
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

# ── 1. Set test env vars BEFORE importing any src.* module ───────────────────
#
# WHY here and not in a fixture:
#   session.py creates the engine at module level. If this conftest runs
#   its env-setting code in a fixture, the engine is already created with
#   the real DATABASE_URL before the fixture even fires.
#   Module-level code in conftest.py is executed by pytest BEFORE collecting
#   any test, so it is guaranteed to run first.
#
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./tests/test_ci.db")
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-exactly-32-chars!!!")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
# Disable LLM providers so chat falls back to structured context (no API calls)
os.environ.pop("GITHUB_TOKEN", None)
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("OLLAMA_HOST", None)


# ── 2. Mock Gatekeeper (defined before importing app) ────────────────────────

# Minimal valid JPEG (SOI marker + JFIF header + EOI)
# Used by classify tests — magic bytes pass the MIME validation in classify.py
MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0"   # SOI + APP0 marker
    b"\x00\x10"           # APP0 length = 16 bytes
    b"JFIF\x00"           # identifier
    b"\x01\x01"           # version 1.1
    b"\x00"               # pixel aspect ratio unit
    b"\x00\x01\x00\x01"  # 1×1 pixels
    b"\x00\x00"           # no embedded thumbnail
    b"\xff\xd9"           # EOI
)

# Minimal valid PNG (8-byte signature + IHDR chunk)
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature (8 bytes)
    b"\x00\x00\x00\rIHDR"  # IHDR chunk (13 bytes)
    b"\x00\x00\x00\x01"   # width = 1
    b"\x00\x00\x00\x01"   # height = 1
    b"\x08\x02\x00\x00\x00"  # bit depth=8, colour type=RGB, compression/filter/interlace
    b"\x90wS\xde"         # CRC
    b"\x00\x00\x00\nIDATx" # IDAT chunk
    b"\x9cc\xf8\x0f\x00\x00\x11\x00\x01\x00"  # minimal compressed RGB
    b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
)


class _MockGatekeeper:
    """
    Lightweight Gatekeeper stub used in all integration tests.

    WHAT: Replaces the real Gatekeeper (EfficientNet-B3 + LangGraph agents)
          with a deterministic, instant-return stub.

    WHY needed:
        The real Gatekeeper loads a 79 MB model into CUDA VRAM and calls
        Ollama / GitHub Models. Neither is available in CI. The mock returns
        a canned result that covers every field the classify route reads.

    Contract: the return value mirrors gatekeeper.analyze() exactly.
        {
            "state": CoinState dict with all agent result fields populated,
            "pdf_path": None  (no PDF generated in tests)
        }
    """

    def analyze(self, image_path: str, tta: bool = True) -> dict:
        return {
            "state": {
                "cnn_prediction": {
                    "class_id":          0,
                    "label":             "1015",
                    "confidence":        0.912,
                    "top5": [
                        {"rank": 1, "class_id": 0, "label": "1015", "confidence": 0.912},
                        {"rank": 2, "class_id": 1, "label": "1017", "confidence": 0.043},
                        {"rank": 3, "class_id": 2, "label": "10708", "confidence": 0.018},
                        {"rank": 4, "class_id": 3, "label": "10810", "confidence": 0.012},
                        {"rank": 5, "class_id": 4, "label": "1087",  "confidence": 0.007},
                    ],
                    "inference_time_ms": 543,
                    "tta_used":          tta,
                    "vote_fraction":     0.875 if tta else None,
                    "tta_passes":        8 if tta else 1,
                    "temperature":       1.0,
                },
                "route_taken":    "historian",
                "historian_result": {
                    "narrative":    "Test narrative about Maroneia drachm.",
                    "mint":         "Maroneia",
                    "region":       "Thrace",
                    "date":         "c.365-330 BC",
                    "material":     "silver",
                    "denomination": "drachm",
                },
                "validator_result":    {},
                "investigator_result": {},
                "node_timings":        {"cnn": "0.5s", "historian": "1.0s"},
            },
            "pdf_path": None,
        }


# ── 3. Mock DB session factory ────────────────────────────────────────────────

def _make_mock_db_session() -> AsyncMock:
    """
    Create a fresh AsyncMock that behaves like a SQLAlchemy AsyncSession.

    WHY AsyncMock (not MagicMock):
        Route handlers call `await db.execute(...)`, `await db.commit()`, etc.
        Regular MagicMock cannot be awaited. AsyncMock returns an awaitable
        for every async call, preventing ``TypeError: object MagicMock can't
        be used in 'await' expression``.

    Configured return values:
        execute → result with .scalars().all() returning [] by default
                  and .scalar_one_or_none() returning None by default.
        scalar() → None (used in COUNT queries)
    """
    session        = AsyncMock()
    result_mock    = MagicMock()
    scalars_mock   = MagicMock()
    scalars_mock.all.return_value           = []
    scalars_mock.first.return_value         = None
    scalars_mock.one_or_none.return_value   = None
    result_mock.scalars.return_value        = scalars_mock
    result_mock.scalar.return_value         = 0
    result_mock.scalar_one_or_none.return_value = None
    session.execute.return_value            = result_mock
    session.scalar.return_value             = 0
    session.commit                          = AsyncMock()
    session.rollback                        = AsyncMock()
    session.add                             = MagicMock()     # synchronous
    session.delete                          = AsyncMock()     # async in SQLAlchemy 2.x
    session.flush                           = AsyncMock()
    session.refresh                         = AsyncMock()
    return session


# ── 4. Session-scoped Gatekeeper patch ───────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """
    Reset the in-memory SlowAPI rate-limit counter before every test.

    WHY needed:
        slowapi's MemoryStorage accumulates request counts across the test
        session.  All classify tests share a single event-loop / process, and
        they all originate from the same IP (127.0.0.1).  Without reset, the
        10/minute limit is exhausted after 10 tests, causing the 11th onward
        to receive HTTP 429 instead of the expected 200/415/422.

    WHY reset() not a patch/mock of the limiter:
        reset() clears all buckets in the in-memory backend.  It is the exact
        inverse of "1 minute has passed" — perfect for per-test isolation.

    Note: if the limiter backend changes to Redis in production, this fixture
    stays correct because the in-process memory backend is ALWAYS used in the
    test environment (no REDIS_URL set in conftest env vars).
    """
    from src.api.limiter import limiter
    try:
        # limits >= 3.x: MemoryStorage has reset()
        limiter._storage.reset()
    except AttributeError:
        # Older limits versions: clear the internal dict directly
        try:
            limiter._storage._storage.clear()
        except AttributeError:
            pass  # If neither exists, the test environment uses a different backend
    yield


@pytest.fixture(scope="session", autouse=True)
def _patch_gatekeeper_globally():
    """
    Replace src.agents.gatekeeper.Gatekeeper with _MockGatekeeper for the
    entire test session.

    WHY session scope:
        The Gatekeeper is instantiated inside the FastAPI lifespan function
        via ``from src.agents.gatekeeper import Gatekeeper; app.state.gk = Gatekeeper()``.
        Patching at session scope means the mock is active before the first
        test's AsyncClient triggers the lifespan, and stays active for all tests.

    WHY autouse=True:
        Every integration test needs this — there's no point running any
        coverage test against the real 79 MB model in CI.
    """
    patcher = patch("src.agents.gatekeeper.Gatekeeper", return_value=_MockGatekeeper())
    patcher.start()
    yield
    patcher.stop()


# ── 5. Per-test fixtures ──────────────────────────────────────────────────────

@pytest.fixture
async def override_db():
    """
    Override the get_db FastAPI dependency with an AsyncMock session.

    LIFETIME: applied before the test, torn down after.
    WHY use app.dependency_overrides:
        This is FastAPI's official override mechanism. It replaces the
        dependency function for the duration of the test without patching
        the module, so it does not affect other tests running in parallel.
    """
    from src.api.db.session import get_db  # imported here (after env setup)
    from src.api.main import app

    mock_session = _make_mock_db_session()

    async def _override_get_db() -> AsyncGenerator:
        yield mock_session

    app.dependency_overrides[get_db] = _override_get_db
    yield mock_session
    del app.dependency_overrides[get_db]


@pytest.fixture
def mock_current_user():
    """
    Build a deterministic mock User object for tests that need authentication.

    WHY a MagicMock and not a real User ORM object:
        Creating a real User row requires a DB INSERT. In unit-level integration
        tests, we want to test the HTTP/routing layer only, not the ORM layer.
        MagicMock with the right attributes passes all attribute accesses
        that the route handlers perform (user.id, user.role, user.email, etc.).
    """
    from src.api.db.models import UserRole, UserStatus
    user             = MagicMock()
    user.id          = uuid.uuid4()
    user.email       = "testuser@example.com"
    user.display_name = "Test User"
    user.role        = UserRole.analyst
    user.status      = UserStatus.active
    user.created_at  = datetime.now(timezone.utc)
    user.is_active   = True
    return user


@pytest.fixture
async def override_auth(mock_current_user):
    """
    Override get_current_user to return a fake authenticated user.
    Also overrides optional_user to return the same user.

    WHY both:
        classify.py uses optional_user (returns None for guests).
        history.py uses get_current_user (requires login, raises 401 if missing).
        Tests that exercise authenticated endpoints need both overridden.
    """
    from src.api.auth.deps import get_current_user, optional_user
    from src.api.main import app

    async def _return_user():
        return mock_current_user

    app.dependency_overrides[get_current_user] = _return_user
    app.dependency_overrides[optional_user]    = _return_user
    yield mock_current_user
    del app.dependency_overrides[get_current_user]
    del app.dependency_overrides[optional_user]


@pytest.fixture
async def override_guest():
    """
    Override optional_user to return None (unauthenticated / guest request).
    Used for classify tests where the user is not logged in.
    """
    from src.api.auth.deps import optional_user
    from src.api.main import app

    async def _return_none():
        return None

    app.dependency_overrides[optional_user] = _return_none
    yield
    del app.dependency_overrides[optional_user]


@pytest.fixture
async def client(override_db, override_guest) -> AsyncGenerator[AsyncClient, None]:
    """
    httpx AsyncClient targeting the DeepCoin ASGI app.

    WHY ASGITransport (not a real server):
        ASGITransport passes HTTP requests directly to the ASGI callable,
        bypassing all network I/O. This means:
          - No port binding →  tests never conflict
          - Full lifespan execution → startup/shutdown hooks run exactly as in
            production; app.state.gk is populated by the lifespan
          - Zero network latency → tests run faster

    WHY base_url="http://test":
        httpx requires an absolute URL for the base. "http://test" is a
        well-known dummy origin used in FastAPI documentation for this purpose.
        It never resolves over the network — all traffic stays in-process.
    """
    from src.api.main import app
    # WHY set app.state.gk directly:
    #   ASGITransport triggers the ASGI lifespan only when the app supports
    #   the lifespan protocol AND the transport calls startup/shutdown events.
    #   In practice this depends on httpx version and event-loop setup.
    #   Setting app.state.gk directly is a more robust approach: it works
    #   regardless of whether the lifespan fires, and the mock Gatekeeper
    #   is already stateless/idempotent.
    app.state.gk = _MockGatekeeper()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
async def auth_client(override_db, override_auth) -> AsyncGenerator[AsyncClient, None]:
    """
    AsyncClient pre-configured for authenticated requests.
    override_auth injects a mock user via dependency override, so no real
    JWT token needs to be issued or validated.
    """
    from src.api.main import app
    app.state.gk = _MockGatekeeper()  # same rationale as `client` fixture above
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
