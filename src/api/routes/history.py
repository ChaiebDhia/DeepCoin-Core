"""
src/api/routes/history.py
==========================
GET /api/history        — paginated list of past classifications
GET /api/history/{id}   — full record for one classification

Business logic:
    Every call to POST /api/classify saves a record to history.json.
    These endpoints expose that history to the frontend so the user can
    see past results, open old PDF reports, and track their collection.

Pagination design (skip / limit):
    WHY skip/limit instead of cursor-based pagination:
        Our dataset is small (single user, single museum use case).
        Skip/limit is simpler and the frontend's table component expects it.
        Cursor pagination would be needed at 100,000+ records. We're nowhere near that.
    Default: skip=0, limit=20 (one page of a table).
    Max limit: 100 (prevents someone requesting all history in one payload).

WHY these are async routes even though the store is synchronous:
    asyncio.to_thread() wraps the blocking file I/O so the event loop stays free.
    A history read takes ~1 ms but it's still a disk operation; wrapping it is
    the correct habit to build for when you migrate to PostgreSQL (purely async).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api._store  import get_by_id, load_all
from src.api.schemas import ClassifyResponse, CnnResult, Top5Item, HistoryListResponse, HistorySummary

logger = logging.getLogger(__name__)

router = APIRouter()


# ── list ──────────────────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="List past coin classifications",
)
async def list_history(
    skip:  int = Query(0,  ge=0,         description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return (1–100)"),
) -> HistoryListResponse:
    """
    GET /api/history

    Returns a paginated list of past classifications (newest first).
    Each item is a HistorySummary — compact for table display.
    Use GET /api/history/{id} to retrieve the full record.
    """
    records = await asyncio.to_thread(load_all)
    # Newest first — the history file appends, so reverse it
    records = list(reversed(records))

    total   = len(records)
    page    = records[skip : skip + limit]

    items = []
    for r in page:
        cnn = r.get("cnn", {})
        items.append(HistorySummary(
            id             = r.get("id", ""),
            timestamp      = r.get("timestamp", ""),
            image_filename = r.get("image_filename", ""),
            route_taken    = r.get("route_taken", ""),
            label          = cnn.get("label", str(r.get("label", ""))),
            confidence     = float(cnn.get("confidence", 0.0)),
            pdf_url        = r.get("pdf_url"),
        ))

    return HistoryListResponse(items=items, total=total, skip=skip, limit=limit)


# ── detail ────────────────────────────────────────────────────────────────────

@router.get(
    "/history/{record_id}",
    response_model=ClassifyResponse,
    summary="Get one past classification by ID",
)
async def get_history_item(record_id: str) -> ClassifyResponse:
    """
    GET /api/history/{id}

    Returns the full ClassifyResponse for one past classification.
    The id is the UUID returned in the POST /api/classify response.

    404 if the id does not exist.
    """
    record = await asyncio.to_thread(get_by_id, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    # Re-hydrate the stored dict back into the response model.
    # WHY not store ClassifyResponse directly as JSON:
    #   Pydantic v2 model_dump() produces plain Python types that json.dumps()
    #   can serialise. Storing the full model object isn't possible in JSON.
    #   Here we reverse the process: dict → validated Pydantic model.
    try:
        cnn_raw = record.get("cnn", {})
        cnn = CnnResult(
            class_id          = cnn_raw.get("class_id", 0),
            label             = str(cnn_raw.get("label", "")),
            confidence        = float(cnn_raw.get("confidence", 0.0)),
            top5              = [Top5Item(**t) for t in cnn_raw.get("top5", [])],
            inference_time_ms = cnn_raw.get("inference_time_ms", 0),
            tta_used          = cnn_raw.get("tta_used", False),
        )

        return ClassifyResponse(
            id                   = record["id"],
            timestamp            = record["timestamp"],
            image_filename       = record.get("image_filename", ""),
            route_taken          = record.get("route_taken", ""),
            cnn                  = cnn,
            narrative            = record.get("narrative"),
            mint                 = record.get("mint"),
            region               = record.get("region"),
            date_range           = record.get("date_range"),
            material             = record.get("material"),
            denomination         = record.get("denomination"),
            material_status      = record.get("material_status"),
            material_confidence  = record.get("material_confidence"),
            visual_description   = record.get("visual_description"),
            kb_match_count       = record.get("kb_match_count"),
            pdf_url              = record.get("pdf_url"),
            processing_time_s    = float(record.get("processing_time_s", 0.0)),
        )
    except Exception as exc:
        logger.error("Failed to deserialise history record %s: %s", record_id, exc)
        raise HTTPException(status_code=500, detail="Stored record is malformed.")
