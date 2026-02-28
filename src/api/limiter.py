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
    POST /api/classify  — 10 requests/minute per IP
        WHY 10/min: The GPU pipeline takes 3–20 seconds.  10 concurrent jobs
        from a single IP saturates the RTX 3050 Ti.  Legitimate museum users
        won't hit this limit; scrapers / abuse will.

    Other endpoints (health, history) — unlimited.
        Read endpoints are cheap and should never be rate-limited.
        Health checks in particular MUST always respond for load balancers.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Key function: rate-limit by client IP address.
# In production behind Nginx, set FORWARDED_ALLOW_IPS in uvicorn so
# X-Forwarded-For is trusted and the real client IP is used.
limiter = Limiter(key_func=get_remote_address)
