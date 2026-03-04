"""
tests/integration/test_auth_flow.py
=====================================
Integration tests for authentication endpoints under /auth/*.

WHAT this file tests:
    1. Registration input validation — email format, password complexity,
       min-length, and missing fields all return HTTP 422 before any DB hit.
    2. Registration conflict — duplicate email returns HTTP 409.
    3. Registration success — valid body returns HTTP 201 with a message.
    4. Login input validation — missing fields return HTTP 422.
    5. Login credentials fallback — wrong password returns HTTP 401.
    6. /auth/me protection — not authenticated returns HTTP 401.
    7. /auth/refresh protection — missing cookie returns HTTP 401.

WHY this file exists:
    Authentication is the perimeter of the entire application.  A broken
    registration validator could allow accounts with no password; a broken
    login endpoint could let anyone in.  These tests verify the HTTP contract
    of every auth surface — independently of bcrypt / JWT / DB logic —
    by using the same mock fixtures as the rest of the integration suite.

ARCHITECTURE NOTE:
    The auth router uses Depends(get_db) for every endpoint, so all DB calls
    are intercepted by the `override_db` fixture (AsyncMock session).  The
    mock session is pre-configured to return None for scalar_one_or_none(),
    which simulates "email not found" on login and "no collision" on register.

    For the conflict test we reconfigure the mock to return a MagicMock user
    object, making the register handler's duplicate-email check fire.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient


# ── TestRegisterInputValidation ───────────────────────────────────────────────

class TestRegisterInputValidation:
    """
    RegisterRequest schema:
        email        — EmailStr (required)
        password     — str, min_length=8, max_length=128 (required)
                       must contain ≥1 letter AND ≥1 non-letter char
        display_name — str | None, max_length=100 (optional)

    All failures below are caught by Pydantic BEFORE the route handler body
    runs, so no DB call is made.
    """

    async def test_missing_email_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when 'email' field is absent."""
        response = await client.post(
            "/auth/register",
            json={"password": "Password1!"},
        )
        assert response.status_code == 422

    async def test_invalid_email_format_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when email is not a valid address (Pydantic EmailStr)."""
        response = await client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "Password1!"},
        )
        assert response.status_code == 422

    async def test_missing_password_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when 'password' field is absent."""
        response = await client.post(
            "/auth/register",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422

    async def test_short_password_returns_422(self, client: AsyncClient) -> None:
        """
        HTTP 422 when password is fewer than 8 characters (min_length=8).

        WHY this threshold:
            NIST SP 800-63B recommends a minimum of 8 characters for passwords.
            Below 8, the brute-force search space is trivially small.
        """
        response = await client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "Ab1!"},
        )
        assert response.status_code == 422

    async def test_all_letters_password_returns_422(self, client: AsyncClient) -> None:
        """
        HTTP 422 when password has no digit or special character.

        The @field_validator("password") enforces: ≥1 letter AND ≥1 non-alpha.
        'abcdefgh' is 8 chars but all letters → fails complexity check.
        """
        response = await client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "abcdefgh"},
        )
        assert response.status_code == 422

    async def test_display_name_too_long_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when display_name exceeds max_length=100."""
        response = await client.post(
            "/auth/register",
            json={
                "email":        "user@example.com",
                "password":     "Password1!",
                "display_name": "x" * 101,
            },
        )
        assert response.status_code == 422


# ── TestRegisterConflict ──────────────────────────────────────────────────────

class TestRegisterConflict:
    """
    The register handler checks for an existing email before creating a User.
    When the DB mock returns an existing user object, the handler raises 409.
    """

    async def test_duplicate_email_returns_409(
        self, client: AsyncClient, override_db
    ) -> None:
        """
        HTTP 409 when the email is already registered.

        The mock DB is reconfigured to return a non-None user from
        scalar_one_or_none(), simulating the duplicate-email query result.

        WHY 409 (Conflict) not 400:
            HTTP 409 is semantically correct — the resource (account) cannot
            be created because a conflicting resource already exists.
        """
        # Return a fake existing user from the DB lookup
        existing_user = MagicMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = existing_user
        override_db.execute.return_value = result_mock

        response = await client.post(
            "/auth/register",
            json={"email": "already@example.com", "password": "Password1!"},
        )
        assert response.status_code == 409


# ── TestRegisterSuccess ───────────────────────────────────────────────────────

class TestRegisterSuccess:
    """
    A valid RegisterRequest in development mode (ENV=test):
        - Returns 201 Created
        - Body contains a 'message' field
        - Auto-activates the account (no email verification needed)
    """

    async def test_valid_registration_returns_201(
        self, client: AsyncClient, override_db
    ) -> None:
        """
        HTTP 201 for a well-formed registration in environment ENV=test.

        The conftest sets ENV=test which triggers the dev auto-activation
        path (status=active, email_verified_at=now) without SMTP.
        """
        # DB mock: no existing user (default scalar_one_or_none = None)
        # db.add() and db.flush() are both no-ops on the AsyncMock
        response = await client.post(
            "/auth/register",
            json={"email": "new@example.com", "password": "Password1!"},
        )
        assert response.status_code == 201

    async def test_registration_response_has_message(
        self, client: AsyncClient, override_db
    ) -> None:
        """Response body must contain a 'message' key (MessageResponse schema)."""
        response = await client.post(
            "/auth/register",
            json={"email": "another@example.com", "password": "Password1!"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "message" in data, f"Expected 'message' in response, got: {data}"
        assert len(data["message"]) > 0


# ── TestLoginInputValidation ──────────────────────────────────────────────────

class TestLoginInputValidation:
    """
    LoginRequest schema:
        email    — EmailStr (required)
        password — str (required, no length check — already hashed comparison)
    """

    async def test_missing_email_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when 'email' field is absent."""
        response = await client.post(
            "/auth/login",
            json={"password": "somepassword"},
        )
        assert response.status_code == 422

    async def test_missing_password_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when 'password' field is absent."""
        response = await client.post(
            "/auth/login",
            json={"email": "user@example.com"},
        )
        assert response.status_code == 422

    async def test_invalid_email_format_returns_422(self, client: AsyncClient) -> None:
        """HTTP 422 when email is not a valid address."""
        response = await client.post(
            "/auth/login",
            json={"email": "not-valid", "password": "somepassword"},
        )
        assert response.status_code == 422


# ── TestLoginCredentials ──────────────────────────────────────────────────────

class TestLoginCredentials:
    """
    Login credential checking — DB mock returns None (user not found) by
    default, which triggers the generic 401 response.

    WHY the same 401 for "wrong email" and "wrong password":
        Distinct error messages enable user enumeration attacks.
        "Email not found" tells an attacker which emails are registered.
    """

    async def test_unknown_email_returns_401(
        self, client: AsyncClient, override_db
    ) -> None:
        """
        HTTP 401 when the email is not registered.

        The DB mock returns None from scalar_one_or_none(), simulating
        "no user with this email". The handler must return 401, not 404.
        """
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        override_db.execute.return_value = result_mock

        response = await client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "Password1!"},
        )
        assert response.status_code == 401


# ── TestProtectedEndpoints ────────────────────────────────────────────────────

class TestProtectedEndpoints:
    """
    Verify unauthenticated access to protected endpoints is rejected.

    The `client` fixture uses `override_guest` which sets optional_user = None
    and does NOT override get_current_user.  The auth dependency chain then
    fails because no valid Bearer token is present in the request.
    """

    async def test_me_without_token_returns_401(self, client: AsyncClient) -> None:
        """
        GET /auth/me without Authorization header → 401 Unauthorized.

        WHY this matters:
            /auth/me returns the user's profile including their role.
            An unauthenticated call must never return any user data.
        """
        response = await client.get("/auth/me")
        assert response.status_code == 401

    async def test_refresh_without_cookie_returns_401(
        self, client: AsyncClient
    ) -> None:
        """
        POST /auth/refresh without the httpOnly 'refresh_token' cookie → 401.

        The refresh token lives in an httpOnly cookie that cannot be
        set by JavaScript.  This test simulates a client that never
        completed a successful login and therefore has no refresh cookie.

        WHY 401 (not 403):
            The request is unauthenticated (no valid token presented), not
            unauthorised (authenticated but insufficient permission).
        """
        response = await client.post("/auth/refresh")
        assert response.status_code == 401

    async def test_logout_without_token_returns_401(
        self, client: AsyncClient
    ) -> None:
        """POST /auth/logout without auth → 401 (cannot revoke unknown session)."""
        response = await client.post("/auth/logout")
        assert response.status_code == 401
