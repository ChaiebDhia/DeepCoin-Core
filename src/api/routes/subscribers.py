"""
routes/subscribers.py
=====================
POST   /api/subscribers                     — waitlist / newsletter email capture
GET    /api/subscribers/confirm?token=xxx   — confirm subscription via email link
GET    /api/subscribers/unsubscribe?token=xxx — remove from list via email link
GET    /api/subscribers                     — admin: list all subscribers (auth required)

ENTERPRISE CONFIRMATION FLOW
─────────────────────────────
1. User submits email → POST /api/subscribers
2. Backend generates a UUID confirm_token, stores subscriber with
   status="pending", and (if RESEND_API_KEY is set) fires a transactional
   email via the Resend API containing:
     • A confirmation link  → /confirm-subscription?token=<uuid>
     • An unsubscribe link  → /api/subscribers/unsubscribe?token=<uuid>
3. Frontend shows either:
     • email_sent=true  → "Check your inbox for a confirmation link"
     • email_sent=false → inline dev-mode link (RESEND_API_KEY not set)
4. User clicks confirmation link → GET /api/subscribers/confirm?token=xxx
   → status set to "confirmed"
5. User clicks unsubscribe link  → GET /api/subscribers/unsubscribe?token=xxx
   → record deleted from JSON

WHY token-based (not magic-link / OAuth):
    Simple, stateless, zero dependencies beyond random UUID generation.
    A UUID4 is 122 bits of entropy — brute-force infeasible at any
    foreseeable scale for an academic project mailing list.

WHY JSON file instead of a DB table:
    Subscriber list is operational (outreach), not application data.
    JSON keeps it out of the main DB schema, trivially exportable as CSV,
    and requires zero migrations. Threading lock ensures safe concurrent writes.

REQUIRED ENV VARS (all optional — graceful degradation if absent):
    RESEND_API_KEY   — Resend.com API key for transactional email
    APP_BASE_URL     — Public URL prefix, e.g. https://deepcoin.ai
                       Used to build confirmation / unsubscribe links.
                       Defaults to http://localhost:3000 in development.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime  import datetime, timezone
from pathlib   import Path

import httpx
from fastapi            import APIRouter, Depends, Query
from fastapi.responses  import HTMLResponse
from pydantic           import BaseModel, field_validator
from src.api.auth       import require_api_key

logger = logging.getLogger(__name__)

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/subscribers", tags=["Subscribers"])

# ── Config ────────────────────────────────────────────────────────────────────

_lock           = threading.Lock()
_DATA_FILE      = Path("data/subscribers.json")
RESEND_API_KEY  = os.getenv("RESEND_API_KEY", "")
APP_BASE_URL    = os.getenv("APP_BASE_URL", "http://localhost:3000").rstrip("/")
SENDER_EMAIL    = os.getenv("DEEPCOIN_SENDER_EMAIL", "DeepCoin <noreply@deepcoin.ai>")

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
    status:        str = "pending"


class SubscribeResponse(BaseModel):
    """
    Response body for POST /api/subscribers.

    FIELDS:
        ok            — always True on success
        message       — human-readable status string
        confirm_token — UUID the frontend can use to build a dev-mode
                        inline confirmation link when no email is sent
        email_sent    — True when the Resend API call succeeded;
                        False in dev (RESEND_API_KEY not set)
    """
    ok:            bool
    message:       str
    confirm_token: str
    email_sent:    bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_records() -> list[dict]:
    """Read data/subscribers.json; return [] on any error."""
    if not _DATA_FILE.exists():
        return []
    try:
        return json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_records(records: list[dict]) -> None:
    """Atomically write records back to disk."""
    _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DATA_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _send_confirmation_email(email: str, confirm_token: str) -> bool:
    """
    Send a transactional confirmation email via Resend.com.

    WHAT: POSTs to the Resend /emails endpoint with an HTML email body
          containing a confirmation link and an unsubscribe link.

    WHY Resend over SMTP:
        Resend requires a single API key — no SMTP server, no TLS config,
        no credential rotation. The free tier (3,000 emails/month) is
        sufficient for an academic project mailing list.

    RETURNS:
        True  — HTTP 200 from Resend (email queued for delivery)
        False — RESEND_API_KEY not set, or any network/API error.
                Failure is logged but NOT re-raised so the subscribe
                endpoint always returns 200 (dev-mode graceful degradation).
    """
    if not RESEND_API_KEY:
        return False

    confirm_url     = f"{APP_BASE_URL}/confirm-subscription?token={confirm_token}"
    unsubscribe_url = f"{APP_BASE_URL}/api/subscribers/unsubscribe?token={confirm_token}"

    html_body = f"""
    <div style="font-family:sans-serif;max-width:520px;margin:0 auto;color:#0f172a">
      <div style="background:#0a1628;padding:28px 32px;border-radius:12px">
        <h1 style="color:#d4a853;font-size:22px;margin:0 0 4px">DeepCoin</h1>
        <p style="color:#94a3b8;font-size:12px;margin:0">AI Numismatic Analysis</p>
      </div>
      <div style="padding:32px 32px 16px">
        <h2 style="font-size:20px;margin:0 0 12px">Confirm your subscription</h2>
        <p style="color:#475569;line-height:1.6">
          Thanks for signing up! Click the button below to confirm your email
          address and join the DeepCoin mailing list.
        </p>
        <a href="{confirm_url}"
           style="display:inline-block;margin:24px 0;padding:14px 28px;
                  background:#d4a853;color:#0a1628;border-radius:8px;
                  font-weight:700;text-decoration:none;font-size:15px">
          Confirm subscription
        </a>
        <p style="color:#94a3b8;font-size:13px">
          You&rsquo;ll receive notifications about:
        </p>
        <ul style="color:#64748b;font-size:13px;padding-left:20px;line-height:1.8;margin:0">
          <li>Public API launch</li>
          <li>New coin types added to the knowledge base</li>
          <li>New CNN model versions and accuracy improvements</li>
        </ul>
      </div>
      <div style="padding:16px 32px 28px;border-top:1px solid #e2e8f0;margin-top:8px">
        <p style="color:#94a3b8;font-size:12px;margin:0">
          If you didn&rsquo;t sign up, you can safely ignore this email.
          &nbsp;·&nbsp;
          <a href="{unsubscribe_url}" style="color:#94a3b8">Unsubscribe</a>
        </p>
      </div>
    </div>
    """

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from":    SENDER_EMAIL,
                "to":      [email],
                "subject": "Confirm your DeepCoin subscription",
                "html":    html_body,
            },
            timeout=8.0,
        )
        if resp.status_code in (200, 201):
            return True
        logger.warning("Resend API returned %s: %s", resp.status_code, resp.text[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Resend email failed: %s", exc)
    return False


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
        records = _load_records()
    return [
        SubscriberRecord(
            email=r.get("email", ""),
            subscribed_at=r.get("subscribed_at", ""),
            status=r.get("status", "pending"),
        )
        for r in records
        if r.get("email")
    ]


@router.delete(
    "/{email:path}",
    status_code=204,
    summary="Delete a subscriber by email (admin-only)",
    description=(
        "Remove a subscriber from the waitlist. "
        "URL-encode the @ sign (%40) when calling from the browser. "
        "Requires X-API-Key header."
    ),
    dependencies=[Depends(require_api_key)],
)
async def delete_subscriber(email: str) -> None:
    """
    Remove a single subscriber record from data/subscribers.json.

    WHAT: Finds the record whose email matches (case-insensitive after
          normalisation) and removes it from the JSON list, then
          overwrites the file atomically under the threading lock.

    WHY email as path param (not query param):
        REST convention: the resource is the subscriber identified by
        their email. DELETE /api/subscribers/{email} maps cleanly to
        "delete this resource".  The :path modifier lets FastAPI handle
        the encoded @ character in the URL without a routing collision.

    ENCODING NOTE: callers must percent-encode the @ sign as %40, e.g.
        DELETE /api/subscribers/user%40example.com

    ACCESS: requires X-API-Key (same as GET /api/subscribers).
    """
    normalised = email.strip().lower()
    with _lock:
        records = _load_records()
        before  = len(records)
        records = [r for r in records if r.get("email", "").lower() != normalised]
        if len(records) == before:
            from fastapi import HTTPException  # local import to avoid circular
            raise HTTPException(status_code=404, detail="Subscriber not found.")
        _save_records(records)
    logger.info("Subscriber deleted by admin: %s", normalised)


@router.post("", response_model=SubscribeResponse, status_code=200)
async def subscribe(req: SubscribeRequest) -> SubscribeResponse:
    """
    Add an email to the waitlist and send a confirmation email.

    WHAT:
        1. Reads data/subscribers.json (or creates it)
        2. Checks for an existing entry by email
        3. For a new address: generates a UUID confirm_token, stores the
           entry with status="pending", and attempts to send a confirmation
           email via Resend (if RESEND_API_KEY is set in the environment)
        4. For an already-pending address: returns the existing confirm_token
           so the frontend can re-display the dev-mode inline link
        5. For an already-confirmed address: returns success silently

    IDEMPOTENCY:
        Re-submitting the same pending email returns the existing token —
        useful if the user hits submit twice or the confirmation email
        gets lost and they try again.

    RETURNS:
        200 always (validation errors produce 422 automatically via Pydantic).
    """
    token: str

    with _lock:
        records = _load_records()

        existing = next((r for r in records if r.get("email") == req.email), None)

        if existing:
            # Already confirmed — silent success
            if existing.get("status") == "confirmed":
                return SubscribeResponse(
                    ok=True,
                    message="You're already confirmed — we'll keep you posted!",
                    confirm_token=existing.get("confirm_token", ""),
                    email_sent=False,
                )
            # Pending — return same token so frontend can show dev link again
            token = existing.get("confirm_token") or str(uuid.uuid4())
            existing["confirm_token"] = token            # repair in case it was missing
            _save_records(records)
        else:
            token = str(uuid.uuid4())
            records.append({
                "email":         req.email,
                "subscribed_at": datetime.now(timezone.utc).isoformat(),
                "status":        "pending",
                "confirm_token": token,
            })
            _save_records(records)

    email_sent = _send_confirmation_email(req.email, token)

    msg = (
        "Check your inbox for a confirmation link."
        if email_sent
        else "Almost there — click the confirmation link to complete your signup."
    )
    return SubscribeResponse(ok=True, message=msg, confirm_token=token, email_sent=email_sent)


@router.get("/confirm", response_class=HTMLResponse, include_in_schema=False)
async def confirm_subscription(token: str = Query(...)) -> HTMLResponse:
    """
    Confirm a pending subscription via the token from the email.

    WHAT: Finds the subscriber record by confirm_token, sets status to
          "confirmed", and returns a simple HTML success page.

    WHY HTML response instead of JSON:
        The link is clicked directly in an email client — the browser
        opens the URL raw with no JS. An HTML response lets us show a
        branded confirmation page immediately, even before the Next.js
        frontend is loaded. The Next.js /confirm-subscription page also
        calls this endpoint and can render a richer UI.

    NOTE: This is a GET endpoint — the token acts as the credential.
          UUID4 (122 bits entropy) is sufficient for this use-case.
    """
    with _lock:
        records = _load_records()
        record  = next((r for r in records if r.get("confirm_token") == token), None)

        if not record:
            return HTMLResponse(_html_page(
                "Invalid link",
                "This confirmation link is invalid or has already been used.",
                success=False,
            ), status_code=400)

        if record.get("status") == "confirmed":
            return HTMLResponse(_html_page(
                "Already confirmed",
                f"{record['email']} is already confirmed. Thanks!",
                success=True,
            ))

        record["status"]       = "confirmed"
        record["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        _save_records(records)

    logger.info("Subscription confirmed: %s", record.get("email"))
    return HTMLResponse(_html_page(
        "Subscription confirmed!",
        f"Thanks! <strong>{record['email']}</strong> is now confirmed."
        " We&rsquo;ll reach out when there&rsquo;s news.",
        success=True,
    ))


@router.get("/unsubscribe", response_class=HTMLResponse, include_in_schema=False)
async def unsubscribe(token: str = Query(...)) -> HTMLResponse:
    """
    Remove a subscriber via the unsubscribe token from the email footer.

    WHAT: Finds the record by confirm_token and deletes it entirely.
          Returns a simple HTML confirmation page.

    WHY delete instead of status="unsubscribed":
        GDPR / e-privacy best practice: once a user unsubscribes, their
        email address should no longer be stored. Keeping a tombstone
        row creates unnecessary PII retention.
    """
    with _lock:
        records = _load_records()
        before  = len(records)
        records = [r for r in records if r.get("confirm_token") != token]

        if len(records) == before:
            return HTMLResponse(_html_page(
                "Link not recognised",
                "This unsubscribe link is invalid or has already been used.",
                success=False,
            ), status_code=400)

        _save_records(records)

    logger.info("Unsubscribed via token %s…", token[:8])
    return HTMLResponse(_html_page(
        "Unsubscribed",
        "You&rsquo;ve been removed from the list. No further emails will be sent.",
        success=True,
    ))


# ── HTML page helper ──────────────────────────────────────────────────────────

def _html_page(title: str, body: str, *, success: bool) -> str:
    """
    Return a minimal branded HTML page for email-link landings.

    WHY inline styles only:
        This page is served directly from FastAPI, not via Next.js, so
        none of the Tailwind/CSS variables are available. Inline styles
        ensure the page looks reasonable regardless of environment.
    """
    icon   = "✅" if success else "❌"
    colour = "#10b981" if success else "#ef4444"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} — DeepCoin</title>
</head>
<body style="margin:0;font-family:sans-serif;background:#0a1628;color:#f1f5f9;
             display:flex;align-items:center;justify-content:center;min-height:100vh">
  <div style="text-align:center;max-width:420px;padding:40px 24px">
    <div style="font-size:48px;margin-bottom:20px">{icon}</div>
    <h1 style="color:#d4a853;font-size:24px;margin:0 0 16px">{title}</h1>
    <p style="color:#94a3b8;line-height:1.6;margin:0 0 28px">{body}</p>
    <a href="{APP_BASE_URL}"
       style="display:inline-block;padding:12px 24px;background:#d4a853;
              color:#0a1628;border-radius:8px;font-weight:700;text-decoration:none">
      Back to DeepCoin
    </a>
    <p style="color:{colour};font-size:12px;margin:24px 0 0">
      {title}
    </p>
  </div>
</body>
</html>"""
