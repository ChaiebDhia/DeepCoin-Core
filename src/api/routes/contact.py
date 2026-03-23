"""
src/api/routes/contact.py
==========================
Contact message inbox â€” stores messages from the /contact form in a JSON file.

WHAT:
    POST /api/contact  â€” any visitor submits a message; stored in
                         data/contact_messages.json
    GET  /api/admin/contact â€” admin/curator reads all messages

WHY file-based (not a DB table):
    Contact messages are very low-volume (tens per year, not millions).
    Adding a DB migration just for a simple inbox is disproportionate overhead.
    A JSON file is easy to inspect, back up, and migrate later if needed.
    Thread-safety is handled by a module-level Lock (same pattern as subscribers.py).

STORAGE FORMAT (each element in the list):
    {
        "id":         UUID string,
        "name":       str,
        "email":      str,
        "subject":    str,
        "message":    str,
        "created_at": ISO 8601 string,
        "read":       bool
    }
"""


import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib  import Path
from typing   import Any

from fastapi             import APIRouter, Depends, HTTPException, Response
from pydantic            import BaseModel, EmailStr, Field

from src.api.auth.deps   import get_current_user
from src.api.db.models   import User, UserRole

# â”€â”€ Storage path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_STORE_PATH = Path("data/contact_messages.json")
_LOCK        = threading.Lock()

# â”€â”€ Thread-safe file helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load() -> list[dict[str, Any]]:
    """Read the JSON file; return empty list if it doesn't exist yet."""
    if not _STORE_PATH.exists():
        return []
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(messages: list[dict[str, Any]]) -> None:
    """Write the message list back to disk (pretty-printed, atomic enough for our scale)."""
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# â”€â”€ Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ContactRequest(BaseModel):
    """Incoming contact form payload â€” validated by Pydantic before storing."""

    name:    str = Field(..., min_length=1, max_length=120)
    email:   str = Field(..., min_length=3, max_length=254)
    subject: str = Field(default="General", min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=4000)


# â”€â”€ Router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

router = APIRouter(tags=["Contact"])


@router.post(
    "/api/contact",
    status_code=201,
    summary="Submit a contact message (public)",
)
async def submit_contact(body: ContactRequest) -> dict[str, str]:
    """
    Store a contact message from any visitor (no auth required).

    WHAT: Appends a new message record to data/contact_messages.json.
    WHY  no auth: The contact form is visible to unauthenticated visitors â€”
         potential collaborators, recruiters, and museum partners who haven't
         registered yet.  Requiring auth would defeat the purpose.
    SPAM: Rate-limiting is inherited from the SlowAPI middleware applied at
          the app level. No per-user key stored, so abusers can be filtered
          by IP at the reverse-proxy layer (Nginx in Layer 6).
    """
    record: dict[str, Any] = {
        "id":         str(uuid.uuid4()),
        "name":       body.name.strip(),
        "email":      body.email.strip().lower(),
        "subject":    body.subject.strip(),
        "message":    body.message.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read":       False,
    }

    with _LOCK:
        messages = _load()
        messages.append(record)
        _save(messages)

    return {"id": record["id"], "status": "received"}


@router.get(
    "/api/admin/contact",
    summary="List all contact messages (admin/curator only)",
)
async def list_contact_messages(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Return all contact messages, newest first.

    ACCESS: admin or curator role â€” regular analysts cannot read other people's
            contact messages.
    """
    if current_user.role not in (UserRole.admin, UserRole.curator):
        raise HTTPException(status_code=403, detail="Requires admin or curator role.")

    with _LOCK:
        messages = _load()

    # Newest first
    messages.sort(key=lambda m: m.get("created_at", ""), reverse=True)

    unread = sum(1 for m in messages if not m.get("read", False))
    return {
        "items":  messages,
        "total":  len(messages),
        "unread": unread,
    }


@router.patch(
    "/api/admin/contact/{message_id}/read",
    summary="Mark a contact message as read (admin/curator only)",
)
async def mark_read(
    message_id:   str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Mark a specific message as read so the unread badge clears."""
    if current_user.role not in (UserRole.admin, UserRole.curator):
        raise HTTPException(status_code=403, detail="Requires admin or curator role.")

    with _LOCK:
        messages = _load()
        updated  = False
        for m in messages:
            if m["id"] == message_id:
                m["read"] = True
                updated   = True
                break
        if not updated:
            raise HTTPException(status_code=404, detail="Message not found.")
        _save(messages)

    return {"status": "ok"}


@router.delete(
    "/api/admin/contact/{message_id}",
    status_code=204, response_class=Response,
    summary="Delete a contact message (admin only)",
)
async def delete_contact_message(
    message_id:   str,
    current_user: User = Depends(get_current_user),
) -> None:
    """Permanently delete a contact message."""
    if current_user.role not in (UserRole.admin, UserRole.curator):
        raise HTTPException(status_code=403, detail="Requires admin or curator role.")

    with _LOCK:
        messages = _load()
        filtered  = [m for m in messages if m["id"] != message_id]
        if len(filtered) == len(messages):
            raise HTTPException(status_code=404, detail="Message not found.")
        _save(filtered)

