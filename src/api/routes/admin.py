"""
src/api/routes/admin.py
========================
Admin-only endpoints — privileged access required (admin or curator role).

WHAT: Administrative data views not accessible to regular analysts.
      Includes the user corrections (feedback) log and a paginated
      full-history view for administrators.

WHY a separate file (not inside history.py):
    history.py handles per-user history. Admin views need a different
    access model: they SELECT across ALL users, join auxiliary tables
    (Feedback, User), and must only be accessible to privileged roles.
    Mixing these into history.py would make the auth reasoning hard to
    follow. A dedicated admin.py makes the boundary obvious.

AUTH MODEL:
    All routes in this file use Depends(get_current_user) + an explicit
    role check at the start of each handler.  If current_user.role is
    not "admin" or "curator", we raise 403 immediately.  This matches
    the same pattern already used in the history routes for privileged
    list views.

    WHY not a middleware or decorator:
        A per-endpoint check is explicit and readable. A decorator would
        hide the auth logic, making code reviews harder.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi             import APIRouter, Depends, HTTPException, Query
from sqlalchemy          import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm      import joinedload

from src.api.auth.deps      import get_current_user
from src.api.db.models      import Classification, Feedback, User, UserRole, UserStatus
from src.api.db.session     import get_db

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ── auth guard helper ─────────────────────────────────────────────────────────

def _require_privileged(current_user: User) -> None:
    """
    Raise HTTP 403 if the current user is not admin or curator.

    WHY a helper instead of a FastAPI dependency:
        We need the current_user object for business logic AFTER the check.
        Injecting a second dependency that raises for non-privileged users
        works, but returning the user *and* checking role in two separate
        dependencies couples them awkwardly.  A straightforward if-check at
        the top of each handler is explicit and easy to audit.
    """
    if current_user.role not in (UserRole.admin, UserRole.curator):
        raise HTTPException(
            status_code = 403,
            detail      = "Requires admin or curator role.",
        )


# ── GET /api/admin/feedback ───────────────────────────────────────────────────

@router.get(
    "/feedback",
    summary = "List user corrections (admin / curator only)",
)
async def list_feedback(
    skip:  int          = Query(0,  ge=0),
    limit: int          = Query(20, ge=1, le=100),
    db:    AsyncSession  = Depends(get_db),
    current_user: User   = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Return a paginated list of all "mark as wrong" feedback submissions.

    WHAT: Queries the `feedback` table, joins to `classifications` and `users`
    to surface the coin label, submitter email, the suggested correct
    CN type, and any free-text note.

    WHY admins need this:
        Every time a user clicks "Mark as Wrong" on an analysis, a Feedback
        row is created.  These corrections are the raw material for:
          - Active learning (retraining the CNN on disputed records)
          - Quality assurance (finding systematic misclassifications)
          - User trust (showing users their feedback was recorded)

        Without an admin view, corrections disappear into PostgreSQL with
        no way to act on them.

    FIELDS RETURNED per item:
        id               — Feedback UUID
        created_at       — When the correction was submitted
        classification_id — Which analysis was flagged
        coin_label       — CNN label from the flagged analysis (coin type)
        confidence       — CNN confidence from the flagged analysis
        route_taken      — Agent route of the flagged analysis
        correct_type_id  — What the user believes the correct CN type is
        note             — Optional free-text explanation
        submitted_by     — Email of the user who submitted the correction
    """
    _require_privileged(current_user)

    # ── paginated feedback rows ─────────────────────────────────────────────
    rows_q = (
        select(Feedback)
        .options(
            joinedload(Feedback.classification),
            joinedload(Feedback.user),
        )
        .order_by(desc(Feedback.created_at))
        .offset(skip)
        .limit(limit)
    )
    rows = (await db.execute(rows_q)).scalars().unique().all()

    # ── total count ─────────────────────────────────────────────────────────
    total = (await db.execute(select(func.count()).select_from(Feedback))).scalar_one()

    items = []
    for fb in rows:
        cls  = fb.classification
        user = fb.user
        items.append({
            "id":                fb.id,
            "created_at":        fb.created_at.isoformat() if fb.created_at else None,
            "classification_id": fb.classification_id,
            "coin_label":        cls.label       if cls  else None,
            "confidence":        cls.confidence  if cls  else None,
            "route_taken":       cls.route_taken if cls  else None,
            "correct_type_id":   fb.correct_type_id,
            "note":              fb.note,
            "submitted_by":      user.email if user else "guest",
        })

    return {
        "items": items,
        "total": total,
        "skip":  skip,
        "limit": limit,
        "pages": math.ceil(total / limit) if limit else 1,
    }


# ── GET /api/admin/analyses ───────────────────────────────────────────────────

@router.get(
    "/analyses",
    summary = "Paginated full analyses list (admin / curator only)",
)
async def list_all_analyses(
    skip:   int         = Query(0,  ge=0),
    limit:  int         = Query(20, ge=1, le=100),
    route:  str | None  = Query(None, description="Filter by route"),
    search: str | None  = Query(None, description="Partial match on label"),
    db:     AsyncSession = Depends(get_db),
    current_user: User   = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Return a paginated view of ALL analyses across ALL users.

    WHAT: Admin-only equivalent of /api/history.  Returns every classification
    in the system, newest first, with optional route and label filters.

    WHY admins get to see all records:
        The admin/curator role is the platform operator — they need a global
        view to spot systematic issues (e.g. a coin type that is consistently
        misclassified, or a user who is spamming with nonsense images).

    FIELDS RETURNED:
        id, created_at, label, confidence, route_taken, pdf_url, user_email
    """
    _require_privileged(current_user)

    # ── build base query ────────────────────────────────────────────────────
    base = (
        select(Classification)
        .options(joinedload(Classification.user))
        .order_by(desc(Classification.timestamp))
    )
    if route in ("historian", "validator", "investigator"):
        base = base.where(Classification.route_taken == route)
    if search and search.strip():
        base = base.where(Classification.label.ilike(f"%{search.strip()}%"))

    rows_q = base.offset(skip).limit(limit)
    rows   = (await db.execute(rows_q)).scalars().unique().all()

    # ── total count ─────────────────────────────────────────────────────────
    count_q = select(func.count()).select_from(Classification)
    if route in ("historian", "validator", "investigator"):
        count_q = count_q.where(Classification.route_taken == route)
    if search and search.strip():
        count_q = count_q.where(Classification.label.ilike(f"%{search.strip()}%"))
    total = (await db.execute(count_q)).scalar_one()

    items = []
    for row in rows:
        # WHY Path(row.pdf_path).name instead of rsplit("/"):
        #   On Windows, pdf_path uses backslashes (C:\...\report.pdf).
        #   rsplit("/", 1) only splits on forward slashes — with backslashes the
        #   entire path is returned as the "name", producing a broken URL like
        #   /api/reports/C:\Users\...\report.pdf.
        #   pathlib.Path.name handles BOTH forward and backward slashes.
        pdf_name = Path(row.pdf_path).name if row.pdf_path else None
        items.append({
            "id":          row.id,
            "created_at":  row.timestamp.isoformat() if row.timestamp else None,
            "label":       row.label,
            "confidence":  row.confidence,
            "route_taken": row.route_taken,
            "pdf_url":     f"/api/reports/{pdf_name}" if pdf_name else None,
            "user_email":  row.user.email if row.user else "guest",
        })

    return {
        "items": items,
        "total": total,
        "skip":  skip,
        "limit": limit,
        "pages": math.ceil(total / limit) if limit else 1,
    }

# ── GET /api/admin/users ──────────────────────────────────────────────────────

@router.get(
    "/users",
    summary = "Paginated user list (admin only)",
)
async def list_users(
    skip:   int         = Query(0,  ge=0),
    limit:  int         = Query(20, ge=1, le=100),
    search: str | None  = Query(None, description="Partial match on email or display name"),
    db:     AsyncSession = Depends(get_db),
    current_user: User   = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Return a paginated list of all registered users with their role, status,
    email, display name, created date, and analysis count.

    WHAT: Admins need a single view to manage all platform accounts — promote
    an analyst to curator, suspend a spammer, or verify how many analyses a
    specific user submitted.

    WHY analyses_count as a sub-select:
        Loading all classifications for each user (via joinedload) would be a
        full table scan.  A correlated COUNT sub-query is O(log n) on the
        FK index and never materialises all rows.

    ACCESS: admin-only (not curator — user management is a high-privilege action).
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code = 403,
            detail      = "Requires admin role.",
        )

    # ── build query ─────────────────────────────────────────────────────────
    base = select(User).order_by(desc(User.created_at))
    if search and search.strip():
        term = f"%{search.strip()}%"
        base = base.where(
            User.email.ilike(term) | User.display_name.ilike(term)
        )

    rows_q = base.offset(skip).limit(limit)
    rows   = (await db.execute(rows_q)).scalars().unique().all()

    # ── total count ─────────────────────────────────────────────────────────
    count_q = select(func.count()).select_from(User)
    if search and search.strip():
        term = f"%{search.strip()}%"
        count_q = count_q.where(
            User.email.ilike(term) | User.display_name.ilike(term)
        )
    total = (await db.execute(count_q)).scalar_one()

    # ── per-user analysis counts (one query, not N+1) ──────────────────────
    user_ids = [r.id for r in rows]
    counts: dict[str, int] = {}
    if user_ids:
        cnt_q = (
            select(Classification.user_id, func.count(Classification.id).label("n"))
            .where(Classification.user_id.in_(user_ids))
            .group_by(Classification.user_id)
        )
        for row_id, n in (await db.execute(cnt_q)).all():
            counts[row_id] = n

    items = [
        {
            "id":           u.id,
            "email":        u.email,
            "display_name": u.display_name,
            "role":         u.role.value,
            "status":       u.status.value,
            "created_at":   u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            "analyses_count": counts.get(u.id, 0),
        }
        for u in rows
    ]

    return {
        "items": items,
        "total": total,
        "skip":  skip,
        "limit": limit,
        "pages": math.ceil(total / limit) if limit else 1,
    }


# ── PATCH /api/admin/users/{user_id}/role ─────────────────────────────────────

class RoleUpdateBody(dict):
    """Thin wrapper — validated by FastAPI from the JSON body."""
    pass


@router.patch(
    "/users/{user_id}/role",
    summary = "Change a user's role (admin only)",
)
async def update_user_role(
    user_id:      str,
    body:         dict[str, Any],
    db:           AsyncSession  = Depends(get_db),
    current_user: User          = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Promote or demote a user's RBAC role.

    WHAT: Changes user.role to one of admin | curator | analyst.
          Returns the updated user record.

    SAFETY GUARD: An admin cannot demote their own account (to prevent
    accidentally locking themselves out of the admin panel).  A second
    admin can do it.

    ACCESS: admin-only.
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Requires admin role.")

    new_role_str = body.get("role", "").strip().lower()
    valid_roles  = {r.value for r in UserRole}
    if new_role_str not in valid_roles:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid role '{new_role_str}'. Must be one of: {', '.join(valid_roles)}",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    target: User | None = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Prevent self-demotion
    if target.id == current_user.id and new_role_str != UserRole.admin.value:
        raise HTTPException(
            status_code = 409,
            detail      = "You cannot demote your own admin account.",
        )

    target.role = UserRole(new_role_str)
    await db.commit()
    await db.refresh(target)

    return {
        "id":    target.id,
        "email": target.email,
        "role":  target.role.value,
    }


# ── PATCH /api/admin/users/{user_id}/status ───────────────────────────────────

@router.patch(
    "/users/{user_id}/status",
    summary = "Suspend or reactivate a user account (admin only)",
)
async def update_user_status(
    user_id:      str,
    body:         dict[str, Any],
    db:           AsyncSession  = Depends(get_db),
    current_user: User          = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Toggle a user account between active and suspended.

    WHAT: Sets user.status to 'active' or 'suspended'.
          Suspended users receive HTTP 403 on every authenticated request.

    USE CASE: Suspend a spam account without deleting its history.

    ACCESS: admin-only.
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Requires admin role.")

    new_status_str = body.get("status", "").strip().lower()
    valid_statuses = {s.value for s in UserStatus}
    if new_status_str not in valid_statuses:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid status '{new_status_str}'. Must be one of: {', '.join(valid_statuses)}",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    target: User | None = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if target.id == current_user.id:
        raise HTTPException(status_code=409, detail="You cannot change your own status.")

    target.status = UserStatus(new_status_str)
    await db.commit()
    await db.refresh(target)

    return {
        "id":     target.id,
        "email":  target.email,
        "status": target.status.value,
    }


# ── DELETE /api/admin/users/{user_id} ─────────────────────────────────────────

@router.delete(
    "/users/{user_id}",
    status_code = 204,
    summary     = "Permanently delete a user account (admin only)",
)
async def delete_user(
    user_id:      str,
    db:           AsyncSession  = Depends(get_db),
    current_user: User          = Depends(get_current_user),
) -> None:
    """
    Hard-delete a user account from the database.

    WHAT: Removes the User row.  Because all FK columns on
    Classifications / Feedback / AuditLog use ON DELETE SET NULL,
    the user's historical analyses are preserved but unlinked
    (user_id becomes NULL, shown as "guest" in admin views).

    SAFETY: Admin cannot delete their own account.

    ACCESS: admin-only.
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Requires admin role.")

    result = await db.execute(select(User).where(User.id == user_id))
    target: User | None = result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if target.id == current_user.id:
        raise HTTPException(status_code=409, detail="You cannot delete your own account.")

    await db.delete(target)
    await db.commit()