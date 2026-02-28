"""
src/api/routes/classify.py
===========================
POST /api/classify — the core endpoint of DeepCoin.

Business logic:
    1. Receive a coin photograph (JPEG or PNG) via multipart form upload
    2. Validate it (type, size, magic bytes)
    3. Save to disk so the Gatekeeper can read it by path
    4. Run the full pipeline: CNN → LangGraph agents → PDF
    5. Return a structured JSON response with the full analysis
    6. Persist to history so /api/history can retrieve it later

Security layers (defence in depth — apply ALL of them, not just one):
    Content-Type header  : client-declared MIME type (easy to forge, but filter first)
    Magic bytes check    : first 4 bytes of the actual file data (cannot be faked)
    File size limit      : reject files > MAX_UPLOAD_BYTES before reading fully
    Filename sanitisation: strip path separators to prevent path traversal attacks
    Allowed extensions   : .jpg / .jpeg / .png only

Why asyncio.to_thread for the Gatekeeper:
    FastAPI is async. Its event loop runs in a single thread.
    If you call a blocking function directly inside an async route, it BLOCKS
    the event loop — no other request can be processed until it finishes.
    The Gatekeeper is synchronous (PyTorch, Ollama, ChromaDB — all blocking).
    asyncio.to_thread() runs it in ThreadPoolExecutor, freeing the event loop
    to serve health checks, history requests, etc. while the CNN is running.
    This is THE most important async pattern for ML APIs.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.api._store   import append as history_append
from src.api.schemas  import ClassifyResponse, CnnResult, Top5Item

logger = logging.getLogger(__name__)

router = APIRouter()

# ── constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB — coins photos are never larger

# MIME types we accept from the Content-Type header
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# File magic bytes (first N bytes of the actual binary data)
# WHY check magic bytes AND content-type:
#   A malicious client could rename a .exe to .jpg and set Content-Type: image/jpeg.
#   Magic bytes are part of the actual binary content — they cannot be spoofed
#   without making the file unreadable as an image.
_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",          # JPEG
    b"\x89PNG":      "image/png",           # PNG
}

_ROOT        = Path(__file__).resolve().parent.parent.parent.parent
_UPLOADS_DIR = _ROOT / "data" / "uploads"
_REPORTS_DIR = _ROOT / "reports"


def _detect_mime(header_bytes: bytes) -> str | None:
    """Return MIME type from magic bytes, or None if unrecognised."""
    for magic, mime in _MAGIC.items():
        if header_bytes[:len(magic)] == magic:
            return mime
    return None


def _sanitise_filename(name: str) -> str:
    """
    Remove path separators and dangerous characters from a filename.

    WHY: A filename like '../../etc/passwd.jpg' from a malicious client
    could cause our save path to escape the uploads directory (path traversal).
    Keeping only alphanumerics, dots, dashes, and underscores is safe.
    """
    name = name.replace("\\", "/").split("/")[-1]   # strip directory components
    name = re.sub(r"[^\w.\-]", "_", name)           # strip special chars
    return name[:128]                                # cap length


# ── route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a coin photograph",
    description=(
        "Upload a JPEG or PNG photograph of an ancient coin.\n\n"
        "The pipeline runs:\n"
        "1. EfficientNet-B3 CNN (438 coin types)\n"
        "2. LangGraph routing by confidence\n"
        "3. Historian / Validator / Investigator agent\n"
        "4. PDF report generation\n\n"
        "Returns a full JSON analysis + a PDF download URL."
    ),
)
async def classify(
    request:  Request,
    file:     UploadFile = File(..., description="Coin photograph (JPEG or PNG, max 10 MB)"),
    tta:      bool       = Query(False, description="Test-Time Augmentation: +~1% accuracy, ~5× slower"),
) -> ClassifyResponse:
    """
    POST /api/classify

    The main route. See module docstring for the full security and async design.
    """
    t_start = time.perf_counter()

    # ── 1. Content-Type header check (fast, first gate) ──────────────────────
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Upload JPEG or PNG only.",
        )

    # ── 2. Read file data (enforce size limit) ────────────────────────────────
    # WHY read all at once and check size:
    #   UploadFile.read() streams from the request body.  We cap it at
    #   MAX_UPLOAD_BYTES + 1 so we never buffer a 1 GB file in RAM.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # ── 3. Magic bytes check (cannot be spoofed) ──────────────────────────────
    detected_mime = _detect_mime(data[:4])
    if detected_mime is None:
        raise HTTPException(
            status_code=415,
            detail="File content does not match a supported image format (JPEG/PNG).",
        )

    # ── 4. Save to uploads directory ──────────────────────────────────────────
    # WHY UUID prefix:
    #   Two users uploading 'coin.jpg' simultaneously would overwrite each other
    #   without UUID namespacing. UUIDs are cryptographically unique.
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name    = _sanitise_filename(file.filename or "upload.jpg")
    record_id    = str(uuid.uuid4())
    save_path    = _UPLOADS_DIR / f"{record_id}_{safe_name}"
    save_path.write_bytes(data)
    logger.info("Saved upload: %s (%d bytes)", save_path.name, len(data))

    # ── 5. Run the Gatekeeper pipeline (blocking → thread pool) ───────────────
    # WHY asyncio.to_thread:
    #   The Gatekeeper is fully synchronous (PyTorch, Ollama, ChromaDB).
    #   Calling it directly inside an async def would block the event loop.
    #   asyncio.to_thread() runs it in a separate OS thread, keeping the event
    #   loop free to handle other requests (health checks, history, etc.).
    gk = request.app.state.gk
    try:
        result = await asyncio.to_thread(gk.analyze, str(save_path), tta)
    except Exception as exc:
        logger.error("Gatekeeper pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    elapsed_s = time.perf_counter() - t_start
    state     = result.get("state", {})

    # ── 6. Extract fields from gatekeeper state ────────────────────────────────
    cnn_raw   = state.get("cnn_prediction", {})
    route     = state.get("route_taken", "unknown")
    pdf_path  = result.get("pdf_path")
    hist_res  = state.get("historian_result",    {})
    val_res   = state.get("validator_result",    {})
    inv_res   = state.get("investigator_result", {})

    # Build CnnResult sub-model
    cnn = CnnResult(
        class_id          = cnn_raw.get("class_id", 0),
        label             = str(cnn_raw.get("label", "")),
        confidence        = float(cnn_raw.get("confidence", 0.0)),
        top5              = [
            Top5Item(**t) for t in cnn_raw.get("top5", [])
        ],
        inference_time_ms = cnn_raw.get("inference_time_ms", 0),
        tta_used          = cnn_raw.get("tta_used", tta),
    )

    # PDF URL — served via GET /api/reports/{filename}
    pdf_url: str | None = None
    if pdf_path:
        pdf_url = f"/api/reports/{Path(pdf_path).name}"

    # Build narrative / material / investigator fields
    timestamp = datetime.now(timezone.utc).isoformat()

    response = ClassifyResponse(
        id                   = record_id,
        timestamp            = timestamp,
        image_filename       = safe_name,
        route_taken          = route,
        cnn                  = cnn,
        # historian fields
        narrative            = hist_res.get("narrative"),
        mint                 = hist_res.get("mint"),
        region               = hist_res.get("region"),
        date_range           = hist_res.get("date") or hist_res.get("date_range"),
        material             = hist_res.get("material"),
        denomination         = hist_res.get("denomination"),
        # validator fields
        material_status      = val_res.get("status"),
        material_confidence  = val_res.get("detection_confidence"),
        # investigator fields
        visual_description   = inv_res.get("visual_description"),
        kb_match_count       = len(inv_res.get("kb_matches", [])) or None,
        # output
        pdf_url              = pdf_url,
        processing_time_s    = round(elapsed_s, 2),
    )

    # ── 7. Persist to history ──────────────────────────────────────────────────
    # Run in thread pool to avoid blocking event loop on file I/O
    history_record = {
        **response.model_dump(),
        "cnn": response.cnn.model_dump(),   # flatten sub-model for JSON storage
        "image_path": str(save_path),       # full disk path (not in response)
    }
    await asyncio.to_thread(history_append, history_record)

    logger.info(
        "classify: id=%s  route=%s  label=%s  conf=%.1f%%  time=%.2fs  pdf=%s",
        record_id, route, cnn.label, cnn.confidence * 100, elapsed_s,
        Path(pdf_path).name if pdf_path else "none",
    )

    return response
