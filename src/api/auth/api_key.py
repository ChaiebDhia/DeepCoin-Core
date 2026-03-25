"""
src/api/auth/api_key.py
========================
X-API-Key header authentication for service-to-service routes.

This is the ORIGINAL auth dependency (pre-JWT auth system).
It gates admin routes like /api/metrics that are called by monitoring
systems (Prometheus, Grafana) that use a shared service key, not a user JWT.

DESIGN
------
Two modes depending on the DEEPCOIN_API_KEY environment variable:

  Mode 1 — Key set (production / staging):
    Client MUST send header:  X-API-Key: <value>
    Wrong or missing key  →   401 Unauthorized
    Correct key           →   request proceeds

  Mode 2 — Key NOT set (local development):
    All requests pass through with no check.
    A DEBUG log line reminds the developer that auth is disabled.

HEADER CHOICE  (X-API-Key vs Authorization: Bearer)
----------------------------------------------------
X-API-Key is the convention for service-to-service authentication
(Stripe, Twilio, OpenAI all use it).  Authorization: Bearer is for user JWTs.
For monitoring tools calling /api/metrics, X-API-Key is correct.

USAGE
-----
    from src.api.auth import require_api_key

    @router.get("/api/metrics", dependencies=[Depends(require_api_key)])
    async def metrics(): ...
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_scheme = APIKeyHeader(
    name       = "X-API-Key",
    auto_error = False,
    description = "Service-to-service API key (monitoring, CI).",
)


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_scheme),
) -> None:
    """
    FastAPI dependency — enforces X-API-Key header authentication.

    WHAT:
        Reads DEEPCOIN_API_KEY from the environment.
        If set, the incoming X-API-Key header must match (constant-time compare).
        If not set, auth is disabled (dev shortcut).

    WHY hmac.compare_digest:
        Prevents timing oracle attacks — a naive `==` comparison leaks
        information about how many characters match through timing differences.

    Raises:
        HTTPException 401: if DEEPCOIN_API_KEY is set and the key is wrong.
    """
    expected = os.getenv("DEEPCOIN_API_KEY", "")

    if not expected:
        logger.debug("Auth: DEEPCOIN_API_KEY not set — open access (dev mode)")
        return

    # Check for Bearer token fallback (e.g. from Prometheus scrape_configs)
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header.replace("Bearer ", "", 1)

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide 'X-API-Key' header or 'Authorization: Bearer'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not hmac.compare_digest(api_key, expected):
        logger.warning("Auth: rejected request with invalid API key (first 4: %s...)", api_key[:4])
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug("Auth: API key accepted")
