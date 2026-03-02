"""
tests/unit/test_audit.py
=========================
Unit tests for Phase A3 components:

    write_audit()     â€” src/api/db/audit.py
    client_ip()       â€” src/api/db/audit.py
    user_or_ip_key()  â€” src/api/limiter.py

WHAT each group tests:
    write_audit  â€” correct ORM row construction, guest (None user_id) support
    client_ip    â€” X-Forwarded-For extraction, direct-connection fallback, None client
    user_or_ip_key â€” per-user key on valid JWT, IP fallback on missing/malformed/expired token

WHY asyncio.run() instead of pytest-asyncio:
    pytest-asyncio is listed in pyproject.toml dev dependencies but is not
    currently installed in this environment.  asyncio.run() lets us call async
    functions from synchronous test bodies without any extra plugin, matching
    the same pattern already used in test_auth.py (_run helper).

Test count: 37 existing + 9 new = 46 total
Usage:
    pytest tests/unit/test_audit.py -v
    pytest tests/ -v                     # all 46 tests
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _run(coro):
    """Run a coroutine synchronously — mirrors test_auth.py's _run helper.

    WHY get_event_loop().run_until_complete() and NOT asyncio.run():
        asyncio.run() creates a FRESH event loop, runs the coroutine, then
        CLOSES and destroys the loop.  When a later test file (e.g. test_auth.py)
        calls asyncio.get_event_loop(), it finds no loop and raises RuntimeError.
        run_until_complete() reuses the single main-thread loop shared across all
        test files, so no teardown conflict occurs.
    """
    return asyncio.get_event_loop().run_until_complete(coro)


# â”€â”€ write_audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_write_audit_inserts_row():
    """
    write_audit must add exactly one AuditLog row to the session with the
    correct field values.

    WHY we test db.add() and not a real INSERT:
        write_audit deliberately avoids committing â€” it is the session lifecycle
        (get_db) that commits.  Mocking db.add() verifies the contract without
        a live PostgreSQL connection.
    """
    from src.api.db.audit import write_audit

    db = AsyncMock()
    db.add = MagicMock()

    _run(write_audit(
        db,
        action="coin.classify",
        user_id="abc-123",
        resource_type="classification",
        resource_id="rec-456",
        payload={"route": "historian", "confidence": 0.91},
        ip_address="192.168.1.1",
    ))


    db.add.assert_called_once()
    row = db.add.call_args[0][0]
    assert row.action        == "coin.classify"
    assert row.user_id       == "abc-123"
    assert row.resource_type == "classification"
    assert row.resource_id   == "rec-456"
    assert row.payload["route"] == "historian"
    assert row.ip_address    == "192.168.1.1"


def test_write_audit_guest_nullable_user_id():
    """
    write_audit with user_id=None must succeed without raising.

    WHY important:
        POST /api/classify supports unauthenticated (guest) usage.
        user_id must be nullable at the ORM level AND at the call site.
        A regression here would cause every guest classify to 500.
    """
    from src.api.db.audit import write_audit

    db = AsyncMock()
    db.add = MagicMock()

    _run(write_audit(db, action="coin.classify", user_id=None))

    db.add.assert_called_once()
    row = db.add.call_args[0][0]
    assert row.user_id is None
    assert row.action  == "coin.classify"


def test_write_audit_minimal_fields():
    """
    write_audit must accept only the required `action` argument â€”
    all other fields default to None.
    """
    from src.api.db.audit import write_audit

    db = AsyncMock()
    db.add = MagicMock()

    _run(write_audit(db, action="classification.delete"))

    row = db.add.call_args[0][0]
    assert row.action        == "classification.delete"
    assert row.user_id       is None
    assert row.resource_type is None
    assert row.resource_id   is None
    assert row.payload       is None
    assert row.ip_address    is None


# â”€â”€ client_ip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_client_ip_forwarded_for_single():
    """
    client_ip must return the first entry in X-Forwarded-For.

    WHY test only the first entry:
        Proxies APPEND their IP: "original, proxy1, proxy2".
        The original client is always first.  SlowAPI's get_remote_address
        uses the same convention.
    """
    from src.api.db.audit import client_ip

    request = MagicMock()
    request.headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    request.client.host = "127.0.0.1"

    assert client_ip(request) == "203.0.113.5"


def test_client_ip_forwarded_for_multiple():
    """client_ip must strip whitespace and return only the first IP."""
    from src.api.db.audit import client_ip

    request = MagicMock()
    request.headers = {"X-Forwarded-For": "  1.2.3.4 , 10.0.0.2, 172.16.0.1"}
    request.client = None

    assert client_ip(request) == "1.2.3.4"


def test_client_ip_no_forwarded_header():
    """client_ip must fall back to request.client.host when no X-Forwarded-For."""
    from src.api.db.audit import client_ip

    request = MagicMock()
    request.headers = {}
    request.client.host = "10.20.30.40"

    assert client_ip(request) == "10.20.30.40"


def test_client_ip_no_client():
    """client_ip must return None when request.client is None (Unix socket / test env)."""
    from src.api.db.audit import client_ip

    request = MagicMock()
    request.headers = {}
    request.client = None

    assert client_ip(request) is None


# â”€â”€ user_or_ip_key â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_user_or_ip_key_valid_jwt():
    """
    user_or_ip_key must return 'user:<sub>' for a valid Bearer JWT.

    WHY 'user:' prefix:
        Namespacing prevents a pathological collision where a user UUID
        equals an IP address string (effectively impossible, but defensive).
    """
    from src.api.limiter import user_or_ip_key

    request = MagicMock()
    request.headers = {"Authorization": "Bearer valid.jwt.token"}

    with patch(
        "src.api.limiter.decode_token",
        return_value={"sub": "user-uuid-1234", "type": "access"},
    ):
        key = user_or_ip_key(request)

    assert key == "user:user-uuid-1234"


def test_user_or_ip_key_no_auth_header():
    """user_or_ip_key must return the client IP when no Authorization header is present."""
    from src.api.limiter import user_or_ip_key

    request = MagicMock()
    request.headers = {}

    with patch("src.api.limiter.get_remote_address", return_value="1.2.3.4"):
        key = user_or_ip_key(request)

    assert key == "1.2.3.4"


def test_user_or_ip_key_malformed_token():
    """
    user_or_ip_key must fall back to IP when the JWT is malformed.

    WHY silent fallback (not an error):
        This key function must NEVER raise.  If it raises, SlowAPI crashes
        the request before the route handler can return a helpful HTTP error.
        The route's own auth dependency will reject the bad token with a 401.
    """
    from src.api.limiter import user_or_ip_key

    request = MagicMock()
    request.headers = {"Authorization": "Bearer not.a.real.jwt"}

    with (
        patch("src.api.limiter.decode_token", side_effect=HTTPException(status_code=401, detail="bad")),
        patch("src.api.limiter.get_remote_address", return_value="5.6.7.8"),
    ):
        key = user_or_ip_key(request)

    assert key == "5.6.7.8"


def test_user_or_ip_key_expired_token():
    """user_or_ip_key must fall back to IP when the JWT has expired."""
    from src.api.limiter import user_or_ip_key

    request = MagicMock()
    request.headers = {"Authorization": "Bearer expired.jwt.here"}

    with (
        patch("src.api.limiter.decode_token", side_effect=HTTPException(status_code=401, detail="expired")),
        patch("src.api.limiter.get_remote_address", return_value="9.10.11.12"),
    ):
        key = user_or_ip_key(request)

    assert key == "9.10.11.12"
