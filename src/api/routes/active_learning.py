"""
src/api/routes/active_learning.py
===================================
REST endpoints for the Active Learning admin panel.

These endpoints are restricted to admin/curator roles.  They expose the
active-learning pipeline to the admin dashboard so that curators can trigger
exports from the browser without SSH access.

ENDPOINTS
---------
    GET  /api/admin/active-learning/candidates
        Returns count + summary of correction records pending export.

    POST /api/admin/active-learning/export
        Triggers the full export and returns statistics.
        Idempotent — safe to call multiple times.

    GET  /api/admin/active-learning/report
        Returns the most recent EXPORT_REPORT.txt content as plain text.

DESIGN: WHY NO AUTO-TRIGGER
----------------------------
Retraining takes ~12 minutes even in fine-tune mode.  Auto-triggering it on
every feedback submission would block the server (or a Celery worker) for 12
minutes per submission.  The correct design:

    Human decision boundary:
        "We have N new corrections — is it worth a retraining run?"
        (answer: yes when N ≥ 10, or monthly by policy)

    Admin clicks "Export + Schedule Retrain" in the dashboard.
    Export completes instantly (<1s).
    Retraining is kicked off as a background task (Celery/subprocess) or
    scheduled as a nightly cron job.

    This module handles the "Export" part.  Retraining scheduling is Gap 4
    (Celery task queue in the Docker stack).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.auth import require_api_key
from src.api._store import get_feedback_candidates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/active-learning", tags=["Active Learning"])

_ROOT       = Path(__file__).resolve().parent.parent.parent.parent
_OUTPUT_DIR = _ROOT / "data" / "active_learning"
_REPORT_PATH = _OUTPUT_DIR / "EXPORT_REPORT.txt"


# ── Pydantic response models ──────────────────────────────────────────────────

class ALCandidate(BaseModel):
    """
    Summary of one correction record pending active-learning export.

    Fields deliberately minimal — the full record is retrievable via
    GET /api/history/{id} if needed.
    """
    record_id:     str
    original_label: str
    correct_label: str
    confidence:    float
    route_taken:   str
    note:          str | None = None
    timestamp:     str


class ALCandidatesResponse(BaseModel):
    """
    Response shape for GET /api/admin/active-learning/candidates.

    total:      Number of unexported correction records.
    candidates: List of candidate summaries (up to 50, newest-first).

    WHY cap at 50:
        The full list can be large.  The dashboard only needs to show
        "there are N corrections waiting" and a sample of them.
        For a full export, POST /export is the correct endpoint.
    """
    total:      int
    candidates: list[ALCandidate]


class ALExportResponse(BaseModel):
    """
    Response shape for POST /api/admin/active-learning/export.
    """
    candidates: int
    exported:   int
    skipped:    int
    output_dir: str
    message:    str


# ── routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/candidates",
    response_model=ALCandidatesResponse,
    summary="List pending active-learning correction candidates",
    dependencies=[Depends(require_api_key)],
)
def get_candidates() -> ALCandidatesResponse:
    """
    Return all classification records with user corrections not yet exported.

    WHAT THIS ANSWERS:
        "How many curator corrections are waiting to improve the model?"

    Full list is returned for the admin dashboard table view.
    """
    records = get_feedback_candidates()
    candidates: list[ALCandidate] = []
    for r in records[:50]:   # cap at 50 for the response payload
        cnn      = r.get("cnn", {})
        feedback = r.get("feedback", {})
        candidates.append(ALCandidate(
            record_id      = r.get("id", ""),
            original_label = str(cnn.get("label", r.get("label", ""))),
            correct_label  = feedback.get("correct_type_id", "unknown"),
            confidence     = float(cnn.get("confidence", r.get("confidence", 0.0))),
            route_taken    = r.get("route_taken", ""),
            note           = feedback.get("note") or None,
            timestamp      = r.get("timestamp", ""),
        ))
    return ALCandidatesResponse(total=len(records), candidates=candidates)


@router.post(
    "/export",
    response_model=ALExportResponse,
    summary="Export correction records as labelled training data",
    dependencies=[Depends(require_api_key)],
)
def export_candidates() -> ALExportResponse:
    """
    Trigger the active learning export pipeline.

    WHAT:
        Calls `scripts.active_learning.run_export()` synchronously.
        Copies images to data/active_learning/{correct_label}/.
        Writes MANIFEST.csv + EXPORT_REPORT.txt.
        Marks all exported records as used_for_training=True.

    IDEMPOTENT:
        Running this endpoint twice will only export NEW corrections added
        since the last run.  Previously exported records are skipped.

    RETURNS:
        Export statistics including count exported, skipped, and output_dir.

    NOTE ON LATENCY:
        This call is synchronous.  For <1000 corrections it completes in
        <2 seconds (file copies are the bottleneck, not DB reads).
        For large batches, this should be moved to a Celery background task.
    """
    try:
        # Import here to avoid circular imports at module load time
        from scripts.active_learning import run_export
        stats = run_export(output_dir=_OUTPUT_DIR, dry_run=False)
        message = (
            f"Exported {stats['exported']} correction(s) to {_OUTPUT_DIR}. "
            f"Run 'python scripts/train.py --active-learning-dir data/active_learning/' to retrain."
            if stats["exported"] > 0
            else "No new corrections to export. Ask curators to use the 'mark as wrong' feature."
        )
        return ALExportResponse(**stats, message=message)
    except Exception as exc:
        logger.error("Active learning export error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc


@router.get(
    "/report",
    response_class=None,
    summary="Get the latest active-learning export report",
    dependencies=[Depends(require_api_key)],
)
def get_report() -> dict:
    """
    Return the most recent EXPORT_REPORT.txt as a JSON dict.

    WHAT:
        Reads data/active_learning/EXPORT_REPORT.txt and returns its
        contents as `{"report": "<text>"}`.  Returns a placeholder message
        if no export has been run yet.
    """
    if not _REPORT_PATH.exists():
        return {
            "report": (
                "No export has been run yet.\n"
                "POST /api/admin/active-learning/export to generate the first report."
            )
        }
    try:
        content = _REPORT_PATH.read_text(encoding="utf-8")
        return {"report": content}
    except Exception as exc:
        logger.error("Failed to read export report: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read report file") from exc
