"""
src/api/auth/deps.py
====================
FastAPI dependency functions for authentication and authorisation.

DEPENDENCIES AVAILABLE
----------------------
    get_current_user  — resolves the JWT Bearer token → User row (raises 401 if missing/invalid)
    require_role(*roles)  — returns a dependency that raises 403 if the user's role is not in the allowed set
    optional_user     — resolves the JWT if present, returns None for unauthenticated requests
                        (used on routes that support both guest and logged-in usage, like /api/classify)

DESIGN PATTERN: "Depends on Depends"
--------------------------------------
    `require_role("admin", "curator")` returns a FUNCTION, not a value.
    That function is itself a FastAPI dependency that calls `get_current_user`.
    FastAPI resolves this automatically:

        @router.get("/admin/users")
        async def list_users(user: User = Depends(require_role("admin"))):
            ...

    This pattern avoids code duplication: we don't need a separate
    `get_current_admin`, `get_current_curator`, etc.

SECURITY NOTES
--------------
    - Tokens are read from the Authorization: Bearer header only.
      We do NOT support token-in-query-string (leaks into server logs).
    - The User row is fetched from the database on every request to detect
      suspended accounts even if the JWT has not expired yet.
    - status check: suspended users get 403 immediately after the db hit.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.api.db.models import User, UserRole, UserStatus
from src.api.db.session import get_db
from src.api.auth.utils import decode_token

logger = logging.getLogger(__name__)

# OAuth2PasswordBearer extracts the token from "Authorization: Bearer <token>"
# tokenUrl is shown in the /docs Swagger UI for the login button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ── Core dependency ───────────────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency: resolve a Bearer JWT to the authenticated User row.

    FLOW:
        1. oauth2_scheme extracts the raw token from the Authorization header.
           If the header is missing it raises HTTP 401 automatically.
        2. decode_token() verifies signature, expiry, and token type.
           Raises HTTP 401 on any failure.
        3. We look up the user by ID in the database.
           This catches suspended accounts even within the token's validity window.
        4. If the account is suspended, raise HTTP 403 (not 401 — the token
           is valid, but the account is blocked).

    Args:
        token: Extracted by oauth2_scheme from the Authorization header.
        db:    Async SQLAlchemy session injected by get_db.

    Raises:
        HTTPException(401): invalid/expired token or user not found.
        HTTPException(403): account is suspended.

    Returns:
        The User ORM instance for the authenticated account.
    """
    payload = decode_token(token)   # raises 401 on failure
    user_id: str = payload["sub"]

    result = await db.execute(select(User).where(User.id == user_id))
    user: User | None = result.scalar_one_or_none()

    if user is None:
        logger.warning("JWT sub=%s not found in users table", user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.status == UserStatus.suspended:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been suspended. Contact an administrator.",
        )

    return user


# ── Optional authentication ───────────────────────────────────────────────────

async def optional_user(
    token: str | None = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    FastAPI dependency: resolve the Bearer token if present; return None if absent.

    WHY:
        Routes like POST /api/classify should work for both guests (no token) and
        logged-in users (token present). When a user IS authenticated, we can
        attach their user_id to the classification record. When they're not, the
        record is stored with user_id=NULL.

    Returns:
        The User instance if the request carries a valid Bearer token, else None.
    """
    if token is None:
        return None
    try:
        return await get_current_user(token=token, db=db)
    except HTTPException:
        return None


# ── Role-based access control ─────────────────────────────────────────────────

def require_role(*allowed_roles: UserRole) -> Callable:
    """
    Factory that returns a FastAPI dependency enforcing a minimum role.

    USAGE:
        @router.get("/admin/users")
        async def list_users(user: User = Depends(require_role(UserRole.admin))):
            ...

        @router.get("/curate")
        async def curate(user: User = Depends(require_role(UserRole.admin, UserRole.curator))):
            ...

    WHY a factory:
        require_role(UserRole.admin) is called ONCE at import time to produce
        `_check`. FastAPI then calls `_check` on every request. This lets us
        declare the allowed set once in the route definition without repeating
        role-check logic in every handler.

    Args:
        *allowed_roles: One or more UserRole values that are permitted.

    Returns:
        An async FastAPI dependency function.
    """
    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied. Required role(s): "
                    f"{', '.join(r.value for r in allowed_roles)}. "
                    f"Your role: {current_user.role.value}."
                ),
            )
        return current_user
    return _check


# ── Convenience role shortcuts ────────────────────────────────────────────────

def require_admin() -> Callable:
    """Shortcut: require admin role."""
    return require_role(UserRole.admin)


def require_curator_or_above() -> Callable:
    """Shortcut: require curator or admin role."""
    return require_role(UserRole.admin, UserRole.curator)
