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
from typing import Any

from fastapi             import APIRouter, Depends, HTTPException, Query
from sqlalchemy          import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm      import joinedload

from src.api.auth.deps      import get_current_user
from src.api.db.models      import Classification, Feedback, User, UserRole
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
        pdf_name = row.pdf_path.rsplit("/", 1)[-1] if row.pdf_path else None
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
