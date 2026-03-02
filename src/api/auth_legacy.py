"""
src/api/auth.py
================
API key authentication dependency for FastAPI routes.

DESIGN
------
Uses FastAPI's dependency injection system.  A route that requires
authentication adds `Depends(require_api_key)` to its signature.

BEHAVIOUR
---------
Two modes depending on the DEEPCOIN_API_KEY environment variable:

  Mode 1 — Key set (production / staging):
    Client MUST send header:  X-API-Key: <value>
    Wrong or missing key  →   401 Unauthorized
    Correct key           →   request proceeds

  Mode 2 — Key NOT set (local development):
    All requests pass through with no check.
    A DEBUG log line reminds the developer that auth is disabled.
    WHY: During local development you don't want to copy-paste keys into
    every curl command.  Disabling auth at the env level is explicit and
    conscious, not accidental.

HEADER CHOICE  (X-API-Key vs Authorization: Bearer)
----------------------------------------------------
Both are valid.  X-API-Key is the convention for service-to-service
authentication (Stripe, Twilio, OpenAI all use it).  Authorization: Bearer
is typically an OAuth 2.0 JWT token.  Since we issue a simple shared key
(not a user token), X-API-Key is the semantically correct choice.

USAGE
-----
    from src.api.auth import require_api_key
    from fastapi import Depends

    @router.post("/classify", dependencies=[Depends(require_api_key)])
    async def classify(...):
        ...
"""

from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

# FastAPI security scheme — shows the "Authorize" button in /docs Swagger UI
_api_key_scheme = APIKeyHeader(
    name        = "X-API-Key",
    auto_error  = False,          # we handle the 401 ourselves for a cleaner message
    description = "API key sent via the X-API-Key request header.",
)


async def require_api_key(
    api_key: str | None = Security(_api_key_scheme),
) -> None:
    """
    FastAPI dependency — enforces X-API-Key authentication.

    WHAT:
        Reads DEEPCOIN_API_KEY from the environment.
        If set, the incoming X-API-Key header must match it exactly
        (constant-time comparison to prevent timing attacks).
        If not set, auth is disabled (development shortcut).

    WHY constant-time comparison (hmac.compare_digest):
        A naive `if api_key == expected` comparison leaks timing information.
        An attacker can measure how long the comparison takes to deduce how
        many characters they got right — a timing oracle attack.
        hmac.compare_digest always takes the same time regardless.

    Args:
        api_key: Injected by FastAPI from the X-API-Key header (or None).

    Raises:
        HTTPException 401: if DEEPCOIN_API_KEY is set and the key is wrong.
    """
    import hmac

    expected = os.getenv("DEEPCOIN_API_KEY", "")

    # Development mode — no key configured, all requests pass
    if not expected:
        logger.debug("Auth: DEEPCOIN_API_KEY not set — open access (dev mode)")
        return

    # Production mode — key must be present and correct
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not hmac.compare_digest(api_key, expected):
        logger.warning("Auth: rejected request with invalid API key (first 4 chars: %s...)", api_key[:4])
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug("Auth: API key accepted")
