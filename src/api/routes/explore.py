"""
src/api/routes/explore.py
==========================
Public gallery endpoint — no authentication required.

WHAT: Returns recent coin analyses for the /explore page.

WHY a separate endpoint (not /api/history):
    /api/history is auth-gated: it returns only the current user's records
    (or all records for admins). Anonymous visitors have no session, so
    calling /api/history returns 401 and the gallery shows nothing.

    /api/explore is intentionally public. It returns:
        - Only non-sensitive fields (id, label, confidence, route, timestamp)
        - NO user identity, NO file paths
    This is the "museum showcase" of the platform — real analyses that
    demonstrate DeepCoin's capabilities to visitors before they sign up.

WHY user info is excluded:
    Users did not consent to having their analyses publicly attributed.
    We surface the coin data (label, route, confidence) but strip user_id
    and any PII. This is GDPR-safe by design.
"""
from __future__ import annotations

from typing import Any

from fastapi        import APIRouter, Depends, Query
from sqlalchemy     import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models  import Classification
from src.api.db.session import get_db

router = APIRouter(prefix="/api", tags=["Explore"])


# ── response helpers ──────────────────────────────────────────────────────────

def _public_row(row: Classification) -> dict[str, Any]:
    """
    Convert a Classification ORM row into a safe public dict.

    WHAT: Extracts only non-sensitive fields — coin identity and pipeline
    metadata, no user attributes.

    WHY strip user_id and pdf_path:
        user_id is a UUID that could be used to track a person across
        records.  pdf_path includes a server filesystem path.  Neither is
        needed for the gallery view.
    """
    return {
        "id":          row.id,
        "created_at":  row.timestamp.isoformat() if row.timestamp else None,
        "route_taken": row.route_taken,
        "label":       row.label,
        "confidence":  row.confidence,
    }


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/explore",
    summary    = "Browse recent analyses (public, no auth required)",
    tags       = ["Explore"],
)
async def list_public_analyses(
    skip:  int          = Query(0,  ge=0,         description="Pagination offset"),
    limit: int          = Query(12, ge=1, le=50,  description="Page size (max 50)"),
    route: str | None   = Query(None,              description="Filter by route: historian | validator | investigator"),
    db:    AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Return a paginated list of recent coin analyses for the public gallery.

    NO authentication required — this endpoint is intentionally public.

    Fields returned per item:
        id          — UUID of the classification record
        created_at  — ISO-8601 timestamp
        route_taken — "historian" | "validator" | "investigator"
        label       — CN type ID / coin name
        confidence  — Top-1 softmax probability (0.0 – 1.0)

    Fields NOT returned:
        user_id, user email, image_filename, pdf_path, full payload — all stripped.

    HOW IT WORKS:
        1. Build a SELECT with optional route filter.
        2. ORDER BY timestamp DESC (newest first — most relevant for a gallery).
        3. Run a parallel COUNT(*) for the total pagination indicator.
        4. Return {items, total, skip, limit}.
    """
    # ── build query ────────────────────────────────────────────────────────
    base_q = select(Classification).order_by(desc(Classification.timestamp))
    if route in ("historian", "validator", "investigator"):
        base_q = base_q.where(Classification.route_taken == route)

    # ── paginated rows ──────────────────────────────────────────────────────
    rows_q  = base_q.offset(skip).limit(limit)
    rows    = (await db.execute(rows_q)).scalars().all()

    # ── total count ─────────────────────────────────────────────────────────
    count_base = select(func.count()).select_from(Classification)
    if route in ("historian", "validator", "investigator"):
        count_base = count_base.where(Classification.route_taken == route)
    total = (await db.execute(count_base)).scalar_one()

    return {
        "items": [_public_row(r) for r in rows],
        "total": total,
        "skip":  skip,
        "limit": limit,
    }
