"""
src/api/db/audit.py
====================
Shared helpers for writing to the audit_log table and extracting the real
client IP from a FastAPI request.

WHY a separate module (not inlined in auth/router.py):
    Phase A2 defined _write_audit() and _client_ip() as private functions
    inside auth/router.py, which was fine when only auth endpoints needed
    audit rows.  Phase A3 requires classify.py and history.py to also write
    audit entries.  Moving the helpers here breaks a circular import that
    would otherwise arise:

        Before Phase A3 (private helpers):
            auth/router.py  defines _write_audit (private)
            classify.py     CANNOT import it without importing all of auth/router

        After Phase A3 (shared module):
            db/audit.py     defines write_audit  (public)
            auth/router.py  imports write_audit from db.audit
            classify.py     imports write_audit from db.audit  ← no cycle
            history.py      imports write_audit from db.audit  ← no cycle

AUDIT LOG DESIGN PRINCIPLES
----------------------------
    Immutability:
        The audit_log table has no UPDATE route and no DELETE route.  Rows are
        appended only.  The SQLAlchemy model has no onupdate hook.

    Same-transaction writes:
        write_audit() does NOT call db.commit().  The session is owned by the
        request lifecycle (get_db()).  The audit row is committed atomically
        with any other writes in that request.  If the transaction rolls back,
        the audit row rolls back too — no orphaned entries for failed operations.

    Guest support:
        user_id is nullable.  Unauthenticated classify calls (guest mode) still
        produce audit rows with user_id=NULL so we can track usage volume.

Action naming convention: "{resource}.{verb}"
    "coin.classify"           — a coin photo was classified
    "classification.delete"   — a history record was deleted
    "user.register"           — new account created  (auth/router)
    "user.login"              — successful authentication  (auth/router)
    "user.login_failed"       — wrong password attempt  (auth/router)
    "user.logout"             — refresh token revoked  (auth/router)
    "user.email_verified"     — email confirmation link clicked  (auth/router)
    "user.password_reset"     — password changed via reset token  (auth/router)
"""
from __future__ import annotations

import logging

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import AuditLog

logger = logging.getLogger(__name__)


async def write_audit(
    db: AsyncSession,
    *,
    action: str,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    payload: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """
    Insert one row into the audit_log table.

    WHAT:
        Creates an AuditLog ORM instance and adds it to the active SQLAlchemy
        session.  The session is committed (or rolled back) by get_db() at the
        end of the request lifecycle — this function never commits directly.

    WHY fire-and-forget within the transaction:
        We do NOT call await db.commit() here.  The audit entry is an atomic
        part of the same database transaction that performs the main mutation.
        If the wider transaction rolls back (e.g. an unhandled exception in
        the route handler), the audit row is also rolled back — preventing
        orphaned entries that describe failed operations as if they succeeded.

    Args:
        db:            Active AsyncSession for the current request.
        action:        Short identifier string, e.g. "coin.classify".
        user_id:       UUID string of the acting user, or None for guest requests.
        resource_type: Category of the affected object, e.g. "classification".
        resource_id:   UUID or string ID of the specific object.
        payload:       Optional dict of additional context (stored as JSONB).
        ip_address:    Client IP string or None.

    Example:
        await write_audit(
            db,
            action="coin.classify",
            user_id=current_user.id if current_user else None,
            resource_type="classification",
            resource_id=record_id,
            payload={"route": route, "label": label, "confidence": conf},
            ip_address=client_ip(request),
        )
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        payload=payload,
        ip_address=ip_address,
    )
    db.add(entry)
    # Do NOT commit — the session lifecycle is owned by get_db().


def client_ip(request: Request) -> str | None:
    """
    Extract the real client IP address from a FastAPI Request.

    WHY X-Forwarded-For first:
        When deployed behind Nginx (Layer 6), FastAPI's request.client.host
        is the Nginx process IP (127.0.0.1 or ::1), not the user's IP.
        Nginx injects the original client IP via the X-Forwarded-For header.
        We take the first value in the (possibly comma-separated) list because
        downstream proxies append their own IPs — the original sender is first.

    WHY a shared helper (not duplicated per route):
        auth/router.py, classify.py, and history.py each write audit entries.
        A single helper guarantees consistent IP extraction across all routes.
        If the extraction logic ever needs updating (e.g. to support
        X-Real-IP for a CDN), one change here fixes all routes.

    Args:
        request: The FastAPI Request object (injected by FastAPI into every route).

    Returns:
        The best-available client IP string, or None if it cannot be determined.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else None
