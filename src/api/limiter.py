"""
src/api/limiter.py
==================
Shared SlowAPI rate limiter singleton.

WHY a separate module (not defined inline in main.py or classify.py):
    SlowAPI's @limiter.limit decorator is applied at class/function definition
    time — before the app object exists.  If we defined the limiter inside
    main.py, any router that imports it would need to import main.py, creating
    a circular import (main <- router <- main).

    The singleton pattern here mirrors how SQLAlchemy engines and Celery apps
    are typically structured: one module, one object, imported everywhere.

RATE LIMIT POLICY
-----------------
    POST /api/classify  — 10 requests/minute per authenticated user (or IP for guests)
        WHY 10/min: The GPU pipeline takes 3–20 seconds.  10 concurrent jobs
        saturate the RTX 3050 Ti.  Legitimate museum users won't hit this; scrapers
        and abuse will.

        WHY key by user_id when authenticated:
            A museum may run 10 analysts behind one NAT gateway.  IP-keying would
            give them a shared 10/min pool — effectively 1 request/min each.
            User-keying gives each authenticated account its own independent bucket.

    Other endpoints (health, history) — unlimited.
        Read endpoints are cheap and should never be rate-limited.
        Health checks in particular MUST always respond for load balancers.
"""
from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.auth.utils import decode_token


def user_or_ip_key(request: Request) -> str:
    """
    SlowAPI key function: key by authenticated user ID when possible, IP otherwise.

    WHY decode JWT manually instead of using Depends(get_current_user):
        SlowAPI calls this key function *synchronously* (not via FastAPI's async
        dependency injection system) before the route handler starts.  We cannot
        call `await` here, and we cannot inject FastAPI dependencies.

        decode_token() only verifies the JWT signature and decodes the payload —
        no database round-trip, no async I/O.  This is safe to call synchronously.

    WHY we silently fall back to IP on any JWT error:
        This key function must NEVER raise.  If it raises, SlowAPI crashes the
        request before the route handler can return a friendly HTTP error.
        Invalid/expired tokens will be rejected by the route's own auth dependency
        (require_api_key or get_current_user) — no need to handle them here.

    WHY "user:<uuid>" prefix:
        Namespacing prevents a collision where a user's UUID happens to equal
        an IP address string (astronomically unlikely but correct to avoid).

    Args:
        request: The incoming FastAPI Request.

    Returns:
        "user:<uuid>" for authenticated requests, or the client IP for guests.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        try:
            payload = decode_token(token)
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            # Any error (expired, malformed, wrong type, etc.) — fall through to IP.
            pass
    return get_remote_address(request)


# In production behind Nginx, set FORWARDED_ALLOW_IPS=* in uvicorn so
# X-Forwarded-For is trusted and the real client IP is used for the IP path.
#
# IMPORTANT: point config_filename to a file that does not exist so SlowAPI does
# not auto-read the real .env on import, which avoids Windows cp1252 decoding
# errors when the file contains UTF-8 chars.
limiter = Limiter(key_func=user_or_ip_key, config_filename=".slowapi.env")
