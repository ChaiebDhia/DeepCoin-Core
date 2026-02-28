"""
tests/unit/test_auth.py
========================
Unit tests for src/api/auth.py — API-key authentication dependency.

These tests verify:
  1. When no API key is configured (dev mode) → requests pass through
  2. When the wrong key is sent → 401 Unauthorized
  3. When the correct key is sent → no error
  4. When key is missing from request but required → 401
  5. Timing-attack resistance (hmac.compare_digest used, not ==)

Usage:
    pytest tests/unit/test_auth.py -v
"""

import asyncio
import os
import hmac
import pytest
from unittest.mock import patch
from fastapi import HTTPException


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine synchronously (avoid pytest-asyncio dependency)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRequireApiKey:
    """
    WHAT: require_api_key is a FastAPI Depends() function.
          It reads DEEPCOIN_API_KEY from the environment and validates the
          X-API-Key header sent by the client.
    WHY:  Without auth, any client can hit the /api/classify endpoint and
          consume GPU resources (model inference) or flood the history store.
    """

    def test_dev_mode_no_key_configured_passes(self):
        """
        When DEEPCOIN_API_KEY is not set, the server is in dev mode.
        Every request must be allowed regardless of the header value.
        This matches the typical developer workflow where the key is .env optional.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPCOIN_API_KEY", None)   # ensure not set
            # re-import to pick up env state
            from src.api.auth import require_api_key
            # No header provided — dev mode allows it
            _run(require_api_key(None))    # must not raise

    def test_dev_mode_any_key_header_passes(self):
        """Even with a random header value, dev mode must not reject."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPCOIN_API_KEY", None)
            from src.api.auth import require_api_key
            _run(require_api_key("totally-random-value"))   # must not raise

    def test_correct_key_passes(self):
        """A request with the correct API key must be accepted."""
        with patch.dict(os.environ, {"DEEPCOIN_API_KEY": "super-secret-key-123"}):
            from src.api.auth import require_api_key
            _run(require_api_key("super-secret-key-123"))   # must not raise

    def test_wrong_key_raises_401(self):
        """A request with the wrong API key must raise HTTP 401 Unauthorized."""
        with patch.dict(os.environ, {"DEEPCOIN_API_KEY": "correct-secret"}):
            from src.api.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key("wrong-key"))
            assert exc_info.value.status_code == 401

    def test_missing_key_header_raises_401(self):
        """A request with no X-API-Key header (None) must raise HTTP 401 when key is configured."""
        with patch.dict(os.environ, {"DEEPCOIN_API_KEY": "my-key"}):
            from src.api.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(None))
            assert exc_info.value.status_code == 401

    def test_empty_string_key_raises_401(self):
        """An empty string key must be rejected, not treated as 'no key'."""
        with patch.dict(os.environ, {"DEEPCOIN_API_KEY": "my-key"}):
            from src.api.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(""))
            assert exc_info.value.status_code == 401

    def test_401_response_has_www_authenticate_header(self):
        """RFC 7235 requires WWW-Authenticate in 401 responses."""
        with patch.dict(os.environ, {"DEEPCOIN_API_KEY": "correct"}):
            from src.api.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key("wrong"))
            headers = exc_info.value.headers or {}
            assert "WWW-Authenticate" in headers, "401 must include WWW-Authenticate header"


class TestTimingAttackResistance:
    """
    WHAT: hmac.compare_digest() prevents timing oracle attacks.
    WHY:  A naive `key == expected` comparison short-circuits on the first
          differing character. An attacker can measure response times to
          brute-force the key one character at a time. hmac.compare_digest
          always takes the same time regardless of where the strings differ.
    """

    def test_hmac_compare_digest_used(self):
        """
        The auth module source code must use hmac.compare_digest, not ==.
        This is a code-quality / security assertion, not a runtime test.
        """
        import inspect
        import src.api.auth as auth_module

        source = inspect.getsource(auth_module)
        assert "hmac.compare_digest" in source, (
            "auth.py must use hmac.compare_digest() for constant-time comparison, "
            "not the == operator. This prevents timing oracle attacks."
        )
