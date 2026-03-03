"""
routes/subscribers.py
=====================
POST /api/subscribers — waitlist / newsletter email capture.

WHAT:
    Accepts an email address and appends it to data/subscribers.json.
    Returns a confirmation message. Idempotent — re-submitting the same
    address is silently accepted (no double-entry, same response).

WHY JSON file instead of a DB table:
    The subscriber list is operational data (outreach), not application
    data. A separate JSON file keeps it out of the main DB schema, makes
    it trivially exportable, and requires zero migration to add or drop.
    The threading lock makes concurrent writes safe.

NOTE — no email is sent:
    SMTP is not configured in this environment. The endpoint stores the
    address for future outreach. The frontend must communicate this
    honestly to the user (no "we sent you an email" messaging).
"""

import json
import re
import threading
from datetime import datetime, timezone
from pathlib  import Path

from fastapi            import APIRouter, Depends
from pydantic           import BaseModel, field_validator
from src.api.auth       import require_api_key

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/subscribers", tags=["Subscribers"])

# ── Thread-safe file access ───────────────────────────────────────────────────

_lock      = threading.Lock()
_DATA_FILE = Path("data/subscribers.json")

# ── Schema ────────────────────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    """
    Request body for POST /api/subscribers.

    FIELDS:
        email — must match a basic email pattern after stripping whitespace
                and lowercasing. No strict RFC-5322 validation: we want to
                accept edge-case valid addresses without rejecting real users.
    """
    email: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        """
        Normalise and validate the email field.

        WHAT: strips whitespace, lowercases, rejects strings that don't
              contain an @ with non-empty local and domain parts.
        WHY lowercase: de-duplicates FOO@bar.com vs foo@bar.com — the same
              inbox, but different strings without normalisation.
        """
        v = v.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Please enter a valid email address.")
        return v


class SubscriberRecord(BaseModel):
    """A single subscriber entry returned by the admin list endpoint."""
    email:         str
    subscribed_at: str


class SubscribeResponse(BaseModel):
    ok:      bool
    message: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=list[SubscriberRecord],
    summary="List all subscribers (admin-only)",
    description="Returns the full subscriber list. Requires X-API-Key header.",
    dependencies=[Depends(require_api_key)],
)
async def list_subscribers() -> list[SubscriberRecord]:
    """
    Return all email subscribers in chronological order.

    WHAT: Reads data/subscribers.json and returns the full list.
    WHY protected: subscriber emails are PII — only admins should access them.
    Auth: X-API-Key header (same key as /api/metrics).
    """
    with _lock:
        if not _DATA_FILE.exists():
            return []
        try:
            records = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return [
        SubscriberRecord(
            email=r.get("email", ""),
            subscribed_at=r.get("subscribed_at", ""),
        )
        for r in records
        if r.get("email")
    ]


@router.post("", response_model=SubscribeResponse, status_code=200)
async def subscribe(req: SubscribeRequest) -> SubscribeResponse:
    """
    Add an email to the waitlist.

    WHAT: Reads data/subscribers.json (or creates it), checks for a
          duplicate, appends the new entry with a UTC timestamp, writes back.

    WHY thread lock:
        FastAPI handles concurrent requests. Without a lock, two POSTs
        arriving simultaneously could both read before either writes,
        causing one entry to be silently dropped.

    IDEMPOTENCY:
        If the email already exists, the response is identical to a new
        subscription — both return ok=True. This prevents leaking
        "your email is already registered" information and avoids
        confusing the user with error messages on accidental re-submits.

    RETURNS:
        200 always (validation errors produce 422 automatically via Pydantic).
    """
    with _lock:
        _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

        if _DATA_FILE.exists():
            try:
                records = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                records = []
        else:
            records = []

        already_present = any(r.get("email") == req.email for r in records)

        if not already_present:
            records.append({
                "email":         req.email,
                "subscribed_at": datetime.now(timezone.utc).isoformat(),
            })
            _DATA_FILE.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return SubscribeResponse(
        ok=True,
        message="You're on the list! We'll reach out when there's news.",
    )
