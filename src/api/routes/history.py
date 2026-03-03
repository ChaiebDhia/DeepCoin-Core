"""
src/api/routes/history.py
==========================
GET    /api/history              — paginated list of the authenticated user's classifications
GET    /api/history/{id}         — full record for one classification
DELETE /api/history/{id}         — remove a classification (owner or admin only)
POST   /api/history/{id}/feedback — submit a "mark as wrong" correction

Business logic:
    Every call to POST /api/classify writes one row to the `classifications`
    table in PostgreSQL (replacing the old SQLite _store). These endpoints
    expose that history to the frontend.

Authentication policy:
    - List + detail + delete + feedback:  require authenticated user
    - Users only see their OWN records (user_id = current user's id)
    - Admins and curators see ALL records (role-based override)
    - Guest classifications (user_id IS NULL) are admin-visible only

Pagination (skip / limit):
    WHY skip/limit instead of cursor:
        Single-user / small-team deployment; skip/limit is simpler and the
        frontend table component already uses it. Cursor pagination needed at
        100 000+ rows per user only.
    Default: skip=0, limit=20  |  Max limit: 100

WHY pure async SQLAlchemy (no asyncio.to_thread):
    Unlike the old SQLite _store (synchronous file I/O wrapped in to_thread),
    SQLAlchemy's async engine uses asyncpg — fully non-blocking. We can await
    DB calls directly in the async route without blocking the event loop.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth.deps  import get_current_user
from src.api.db.audit   import client_ip, write_audit
from src.api.db.models  import Classification, Feedback, User, UserRole
from src.api.db.session import get_db
from src.api.schemas import ClassifyResponse, CnnResult, Top5Item, HistoryListResponse, HistorySummary


# ── feedback request body ─────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """
    Body for POST /api/history/{id}/feedback.

    correct_type_id: CN type ID the user believes is correct (e.g. "1015").
    note:            Optional free-text explanation.
    """
    correct_type_id: str
    note:            str = ""


logger = logging.getLogger(__name__)
router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_privileged(user: User) -> bool:
    """Return True if the user can see ALL classifications (admin or curator)."""
    return user.role in (UserRole.admin, UserRole.curator)


def _row_to_summary(row: Classification) -> HistorySummary:
    """Convert a Classification ORM row to the compact HistorySummary schema."""
    payload = row.payload or {}
    pdf_url = None
    if row.pdf_path:
        pdf_url = f"/api/reports/{row.pdf_path.rsplit('/', 1)[-1]}"
    elif payload.get("pdf_url"):
        pdf_url = payload["pdf_url"]
    return HistorySummary(
        id             = row.id,
        timestamp      = row.timestamp.isoformat(),
        image_filename = row.image_filename or "",
        route_taken    = row.route_taken,
        label          = row.label,
        confidence     = row.confidence,
        pdf_url        = pdf_url,
    )


def _row_to_response(row: Classification) -> ClassifyResponse:
    """Re-hydrate a Classification ORM row into the full ClassifyResponse schema."""
    payload = row.payload or {}
    cnn_raw = payload.get("cnn", {})

    cnn = CnnResult(
        class_id          = cnn_raw.get("class_id", 0),
        label             = str(cnn_raw.get("label", row.label)),
        confidence        = float(cnn_raw.get("confidence", row.confidence)),
        top5              = [Top5Item(**t) for t in cnn_raw.get("top5", [])],
        inference_time_ms = cnn_raw.get("inference_time_ms", 0),
        tta_used          = cnn_raw.get("tta_used", False),
        vote_fraction     = cnn_raw.get("vote_fraction"),
        tta_passes        = cnn_raw.get("tta_passes", 1),
        temperature       = float(cnn_raw.get("temperature", 1.0)),
    )

    pdf_url = None
    if row.pdf_path:
        pdf_url = f"/api/reports/{row.pdf_path.rsplit('/', 1)[-1]}"
    elif payload.get("pdf_url"):
        pdf_url = payload["pdf_url"]

    return ClassifyResponse(
        id                   = row.id,
        timestamp            = row.timestamp.isoformat(),
        image_filename       = row.image_filename or "",
        route_taken          = row.route_taken,
        cnn                  = cnn,
        narrative            = payload.get("narrative"),
        mint                 = payload.get("mint"),
        region               = payload.get("region"),
        date_range           = payload.get("date_range"),
        material             = payload.get("material"),
        denomination         = payload.get("denomination"),
        material_status      = payload.get("material_status"),
        material_confidence  = payload.get("material_confidence"),
        visual_description   = payload.get("visual_description"),
        kb_match_count       = payload.get("kb_match_count"),
        pdf_url              = pdf_url,
        processing_time_s    = float(payload.get("processing_time_s", 0.0)),
    )


# ── list ──────────────────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="List the authenticated user's past coin classifications",
)
async def list_history(
    skip:         int          = Query(0,  ge=0,         description="Records to skip"),
    limit:        int          = Query(20, ge=1, le=100, description="Max records (1–100)"),
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> HistoryListResponse:
    """
    GET /api/history

    Returns the caller's past classifications (newest first), paginated.
    Admins and curators see ALL records; analysts see only their own.

    WHY require authentication:
        History is personal data. Returning anyone's history to an anonymous
        caller would be a privacy violation.
    """
    if _is_privileged(current_user):
        count_stmt = select(func.count()).select_from(Classification)
        rows_stmt  = (
            select(Classification)
            .order_by(Classification.timestamp.desc())
            .offset(skip).limit(limit)
        )
    else:
        count_stmt = (
            select(func.count())
            .select_from(Classification)
            .where(Classification.user_id == current_user.id)
        )
        rows_stmt = (
            select(Classification)
            .where(Classification.user_id == current_user.id)
            .order_by(Classification.timestamp.desc())
            .offset(skip).limit(limit)
        )

    total = (await db.execute(count_stmt)).scalar_one()
    rows  = (await db.execute(rows_stmt)).scalars().all()

    return HistoryListResponse(
        items=[_row_to_summary(r) for r in rows],
        total=total,
        skip=skip,
        limit=limit,
    )


# ── detail ────────────────────────────────────────────────────────────────────

@router.get(
    "/history/{record_id}",
    response_model=ClassifyResponse,
    summary="Get one past classification by ID",
)
async def get_history_item(
    record_id:    str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> ClassifyResponse:
    """
    GET /api/history/{id}

    Returns the full ClassifyResponse for one past classification.
    404 if the id does not exist or does not belong to the caller.

    WHY ownership check:
        Without it, user A could enumerate UUIDs to read user B's records.
        Admin/curator bypass lets support staff investigate any record.
    """
    result = await db.execute(
        select(Classification).where(Classification.id == record_id)
    )
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    if not _is_privileged(current_user) and row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    try:
        return _row_to_response(row)
    except Exception as exc:
        logger.error("Failed to deserialise classification %s: %s", record_id, exc)
        raise HTTPException(status_code=500, detail="Stored record is malformed.")


# ── delete ─────────────────────────────────────────────────────────────────────

@router.delete(
    "/history/{record_id}",
    status_code=204,
    summary="Delete one past classification by ID",
)
async def delete_history_item(
    record_id:    str,
    request:      Request,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> Response:
    """
    DELETE /api/history/{id}

    Permanently removes one classification (and its feedback rows via CASCADE).
    Returns 204 No Content on success, 404 if not found / not owned by caller.
    """
    result = await db.execute(
        select(Classification).where(Classification.id == record_id)
    )
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    if not _is_privileged(current_user) and row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    await db.delete(row)
    await db.commit()

    # Audit log (non-fatal — delete has already succeeded)
    try:
        await write_audit(
            db,
            action="classification.delete",
            user_id=current_user.id,
            resource_type="classification",
            resource_id=record_id,
            ip_address=client_ip(request),
        )
    except Exception as audit_exc:
        logger.warning("audit write failed on delete (non-fatal): %s", audit_exc)

    return Response(status_code=204)


# ── feedback ───────────────────────────────────────────────────────────────────

@router.post(
    "/history/{record_id}/feedback",
    summary="Mark a classification as wrong and supply the correct CN type",
)
async def submit_feedback(
    record_id:    str,
    body:         FeedbackRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> JSONResponse:
    """
    POST /api/history/{id}/feedback

    Inserts a row into the `feedback` table. Returns 200 {"status": "ok"} on
    success, 404 if the classification does not exist or belong to the caller.

    WHY a proper Feedback table (not embedded JSON payload):
        - Queryable independently for active learning exports
        - Joined to users for attribution
        - ON DELETE CASCADE: deleting classification cleans up feedback
        - Payload JSONB is immutable after classify; feedback lives separately

    WHY 200 (not 204):
        Frontend reads response body to show a "Correction saved" toast.
    """
    result = await db.execute(
        select(Classification).where(Classification.id == record_id)
    )
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    if not _is_privileged(current_user) and row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    fb = Feedback(
        id                = str(uuid.uuid4()),
        classification_id = record_id,
        user_id           = current_user.id,
        correct_type_id   = body.correct_type_id,
        note              = body.note or None,
    )
    db.add(fb)
    await db.commit()
    logger.info("Feedback: classification=%s correct_type=%s user=%s",
                record_id, body.correct_type_id, current_user.id)

    return JSONResponse({"status": "ok"})
