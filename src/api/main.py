"""
src/api/main.py
================
FastAPI application factory — Layer 4

Run (development):
    uvicorn src.api.main:app --reload --port 8000

Run (production):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
                                                           ^^^^^^^^^^^
                                                           REQUIRED — see note below

═══════════════════════════════════════════════════════════════════════════

workers=1 constraint — WHY:
    The Gatekeeper loads a 79 MB EfficientNet-B3 onto CUDA VRAM at startup.
    Uvicorn workers = separate OS processes. Each process loads the model.
    On our 4.3 GB RTX 3050 Ti:
        1 worker  = 79 MB model + OS overhead = fine
        2 workers = 158 MB + 2× ChromaDB = possible OOM crash
    Concurrency inside ONE worker is handled by asyncio + asyncio.to_thread()
    for the blocking pipeline. For a single-GPU ML service, this is correct.
    Horizontal scaling (multiple GPUs / machines) is handled by a load balancer
    in front of N single-worker pods — not N workers in one process.

CORS policy — WHY not allow_origins=["*"]:
    CORS wildcard + allow_credentials=True allows any website to send
    credentialed requests (cookies, auth headers) to this API from a user's
    browser. This is the definition of a CSRF vulnerability.
    We read the allowed origins from the ALLOWED_ORIGINS environment variable.
    In .env: ALLOWED_ORIGINS=http://localhost:3000 (Next.js dev server)
    In production: ALLOWED_ORIGINS=https://deepcoin.yebni.com

Lifespan pattern — WHY @asynccontextmanager instead of @app.on_event:
    @app.on_event("startup") is deprecated in FastAPI ≥ 0.93.
    The lifespan context manager is the current recommended pattern.
    Code before yield = startup. Code after yield = shutdown.
    We store the Gatekeeper in app.state.gk so it lives exactly as long as
    the app. This is testable: tests can inject a mock gk into app.state.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from src.api._store          import ensure_store, count as history_count
from src.api.auth            import require_api_key
from src.api.auth.router     import router as auth_router
from src.api.limiter         import limiter
from src.api.logging_config  import configure_logging
from src.api.routes.classify     import router as classify_router
from src.api.routes.history      import router as history_router
from src.api.routes.subscribers  import router as subscribers_router
from src.api.routes.explore      import router as explore_router
from src.api.routes.admin        import router as admin_router
from src.api.routes.admin_coins  import router as admin_coins_router
from src.api.routes.chat         import router as chat_router
from src.api.routes.chat_sessions import router as chat_sessions_router
from src.api.routes.kb            import router as kb_router
from src.api.routes.contact       import router as contact_router
from src.api.routes.active_learning import router as active_learning_router

from src import __version__

logger = logging.getLogger(__name__)

# ── paths (used by health + PDF serving) ──────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent.parent
# Load environment file now that _ROOT is defined
load_dotenv(str(_ROOT / ".env"), encoding='utf-8')
_MODEL_PATH   = _ROOT / "models" / "best_model.pth"
_MAPPING_PATH = _ROOT / "models" / "class_mapping.pth"
_CHROMA_DIR   = _ROOT / "data" / "metadata" / "chroma_db_rag"
_REPORTS_DIR  = _ROOT / "reports"
_UPLOADS_DIR  = _ROOT / "data" / "uploads"

# Process start time — used by /api/metrics uptime counter
_START_TIME = time.time()


# ── file cleanup helper ────────────────────────────────────────────────────────

def _cleanup_old_files(
    uploads_max_age_hours: int = 24,
    reports_max_age_hours: int = 720,   # 30 days — PDF links stay valid for a month
) -> None:
    """
    Delete uploaded images and generated PDFs older than their respective TTLs.

    WHAT: Iterates uploads/ and reports/ directories, removes files whose
    last-modified time is older than the cutoff for that directory.

    WHY separate TTLs:
        - Uploaded coin images are temporary (used during analysis only).
          24 hours is plenty. 10,000 uploads × ~200 KB = ~2 GB — clean fast.
        - Generated PDF reports must stay available so users can download
          them from /history/{id}. Deleting after 24h caused the "report not
          found" JSON error. 30 days (720h) keeps disk usage bounded while
          ensuring any report generated in the past month is still accessible.

    WHY called at startup (not scheduled):
        A cron job or APScheduler adds a dependency and complexity.
        Cleaning at startup is simple, deterministic, and runs exactly when
        the admin is watching the logs. Sufficient for current scale.
    """
    import datetime
    deleted = 0
    dir_ttls = [
        (_UPLOADS_DIR, uploads_max_age_hours),
        (_REPORTS_DIR, reports_max_age_hours),
    ]
    for directory, max_age_hours in dir_ttls:
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if not f.is_file():
                continue
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
    if deleted:
        logger.info("Startup cleanup: removed %d stale file(s) older than %dh", deleted, max_age_hours)


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle manager.

    STARTUP (before yield):
      - Create required directories (reports/, data/uploads/, data/history.db)
      - Clean up uploads and reports older than 24 hours
      - Load the Gatekeeper (EfficientNet-B3 + ChromaDB + LangGraph graph)
        stored in app.state.gk — loaded ONCE, reused for every request

    SHUTDOWN (after yield):
      - Python GC handles VRAM / RAM release
      - Log the shutdown so ops engineers see a clean stop in the logs
    """
    logger.info("DeepCoin API v%s starting up...", __version__)

    # P11 — Configure structured logging first so all subsequent log lines
    # are formatted correctly (JSON in prod, human-readable text in dev).
    configure_logging()

    # Ensure directories and history store exist before any request arrives
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_store()

    # Clean up stale files from previous runs (uploads + reports > 24h old)
    _cleanup_old_files(uploads_max_age_hours=24, reports_max_age_hours=720)

    # Load the full pipeline once
    from src.agents.gatekeeper import Gatekeeper
    logger.info("Loading Gatekeeper (CNN + ChromaDB + LangGraph)...")
    app.state.gk = Gatekeeper()
    logger.info("Gatekeeper ready. API is now accepting requests.")

    yield   # ← application runs here

    logger.info("DeepCoin API shutting down.")


# ── app factory ────────────────────────────────────────────────────────────────

_env = os.getenv("ENV", "development")

app = FastAPI(
    title       = "DeepCoin API",
    description = (
        "Archaeological coin classification and historical analysis.\n\n"
        "**Pipeline**: `EfficientNet-B3 CNN` → `LangGraph agents` → `PDF report`\n\n"
        "**Coverage**: Corpus Nummorum (9,716 coin types in KB, 438 in CNN)\n\n"
        "**Institution**: ESPRIT School of Engineering × YEBNI, Tunisia"
    ),
    version     = __version__,
    lifespan    = lifespan,
    # P4 — disable interactive docs in production so internal API surface is
    # not publicly browsable. Set ENV=production in your .env for deployment.
    # docs_url uses /api/docs so the Next.js proxy rewrite
    # (/api/* → http://127.0.0.1:8000/api/*) forwards it correctly.
    # Using /docs would be blocked because Next.js handles /docs itself.
    docs_url      = None if _env == "production" else "/api/docs",
    redoc_url     = None if _env == "production" else "/api/redoc",
    # openapi_url must share the /api/ prefix so the Next.js proxy forwards it.
    # Default is /openapi.json — the Swagger UI would fail to fetch it because
    # Next.js serves /openapi.json itself (returns 404).  /api/openapi.json is
    # forwarded to FastAPI by the afterFiles rewrite in next.config.ts.
    openapi_url   = None if _env == "production" else "/api/openapi.json",
)

Instrumentator().instrument(app).expose(app, endpoint="/api/metrics_prometheus")

# \u2500\u2500 SlowAPI rate-limit exception handler \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS middleware ────────────────────────────────────────────────────────────
#
# Read from environment. Comma-separated list for multiple origins.
# .env default:  ALLOWED_ORIGINS=http://localhost:3000
# Production:    ALLOWED_ORIGINS=https://deepcoin.yebni.com
#
_raw_origins     = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
logger.debug("CORS allowed origins: %s", _allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _allowed_origins,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "DELETE"],
    allow_headers     = ["Content-Type", "Authorization", "X-API-Key"],
)

# P7 — GZip compress responses ≥ 500 bytes.
# JSON responses from /api/history (lists of records) compress ~8× savings.
# minimum_size=500 avoids overhead on tiny payloads (health, root).
from starlette.types import ASGIApp, Receive, Scope, Send
class ConditionalGZipMiddleware:
    def __init__(self, app: ASGIApp, minimum_size: int = 500):
        self.app_to_wrap = app
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size)
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/chat/stream"):
            await self.app_to_wrap(scope, receive, send)
            return
        await self.gzip(scope, receive, send)

app.add_middleware(ConditionalGZipMiddleware, minimum_size=500)

# P15 — X-Request-ID middleware.
# WHAT: Reads the incoming X-Request-ID header (set by load balancer or client);
#       generates a UUID4 if absent. Echoes the ID in every response header.
# WHY:  Lets you correlate a front-end error report ("request id: abc-123") with
#       the exact log line in Loki/Datadog without grepping through timestamps.
@app.middleware("http")
async def add_request_id(request, call_next):
    import uuid as _uuid
    req_id   = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ── routers ────────────────────────────────────────────────────────────────────
#
# WHY prefix="/api":
#   All API routes live under /api so Nginx can route:
#       /api/*      → uvicorn (FastAPI)
#       /*          → Next.js frontend
#   This is the standard reverse-proxy pattern for monorepo deployments.
#
app.include_router(auth_router)                                                # /auth/*
app.include_router(classify_router,   prefix="/api", tags=["Classification"])   # /api/classify
app.include_router(history_router,    prefix="/api", tags=["History"])           # /api/history
app.include_router(subscribers_router)                                          # /api/subscribers
app.include_router(explore_router)                                              # /api/explore  (public)
app.include_router(admin_router)                                                # /api/admin/*  (privileged)
app.include_router(admin_coins_router)                                          # /api/admin/coins/* (privileged)
app.include_router(chat_router)                                                 # /api/chat     (AI Q&A)
app.include_router(chat_sessions_router)                                        # /api/chat/sessions (history)
app.include_router(kb_router)                                                   # /api/kb/types      (KB browser)
app.include_router(contact_router)                                               # /api/contact + /api/admin/contact
app.include_router(active_learning_router)                                       # /api/admin/active-learning/* (admin only)


# ── PDF report serving ────────────────────────────────────────────────────────

@app.get(
    "/api/reports/{filename}",
    tags=["Files"],
    summary="Download a generated PDF report",
    response_class=FileResponse,
)
async def serve_report(filename: str):
    """
    Serve a generated PDF report by filename.

    The filename is returned in the `pdf_url` field of POST /api/classify.
    Only serves files from the reports/ directory.

    Security: strips all path separators from filename to prevent directory
    traversal (e.g. a caller trying '../../etc/passwd' gets a 404).
    """
    # Sanitise filename — never allow path traversal
    safe = Path(filename).name   # strips any directory components
    if not safe.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be served here.")

    report_path = _REPORTS_DIR / safe
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{safe}' not found.")

    return FileResponse(
        path         = report_path,
        media_type   = "application/pdf",
        filename     = safe,
        headers      = {"Content-Disposition": f'attachment; filename="{safe}"'},
    )


# ── Grad-CAM heatmap serving ──────────────────────────────────────────────────

@app.get(
    "/api/gradcam/{filename}",
    tags=["Files"],
    summary="Serve a Grad-CAM heatmap overlay PNG",
    response_class=FileResponse,
)
async def serve_gradcam(filename: str):
    """
    Serve a Grad-CAM heatmap overlay PNG by filename.

    The URL is returned in the ``cnn.gradcam_url`` field of POST /api/classify
    and is also stored in the history payload so the history detail page can
    render it.  Files are retained for 30 days alongside PDFs in reports/.

    Security: the filename is stripped of any directory components to prevent
    path traversal (the same pattern as serve_report above).
    """
    safe = Path(filename).name          # strip any directory components
    if not safe.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG files can be served here.")

    gradcam_path = _REPORTS_DIR / safe
    if not gradcam_path.exists():
        raise HTTPException(status_code=404, detail=f"Grad-CAM file '{safe}' not found.")

    return FileResponse(
        path       = gradcam_path,
        media_type = "image/png",
        headers    = {"Cache-Control": "max-age=2592000"},   # 30 days, matches TTL
    )


# ── health endpoint ────────────────────────────────────────────────────────────

@app.get(
    "/api/health",
    tags=["System"],
    summary="Readiness probe — checks actual system state",
)
async def health():
    """
    GET /api/health

    Returns real component status. Used by:
      - Docker HEALTHCHECK
      - Kubernetes readiness probe
      - Load balancer health checks

    WHY real checks matter:
        A stub that always returns 200 sends traffic to broken pods.
        Load balancers use this endpoint to decide routing.
        If the model file is missing and this returns 200, every classify
        request returns 500 — a silent failure invisible to the load balancer.

    Components checked:
        model_file   : models/best_model.pth exists on disk
        mapping_file : models/class_mapping.pth exists on disk
        chroma_db    : data/metadata/chroma_db_rag/ has content
        gatekeeper   : app.state.gk is initialised (model in VRAM)
        llm_provider : at least one LLM env var is set
    """
    model_ok   = _MODEL_PATH.exists()
    mapping_ok = _MAPPING_PATH.exists()
    chroma_ok  = _CHROMA_DIR.exists() and any(_CHROMA_DIR.iterdir())
    gk_ok      = hasattr(app.state, "gk") and app.state.gk is not None
    llm_ok     = any([
        os.getenv("GITHUB_TOKEN"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("OLLAMA_HOST"),
    ])

    components = {
        "model_file":   "ok" if model_ok   else "MISSING — models/best_model.pth not found",
        "mapping_file": "ok" if mapping_ok else "MISSING — models/class_mapping.pth not found",
        "chroma_db":    "ok" if chroma_ok  else "MISSING — run scripts/rebuild_chroma.py",
        "gatekeeper":   "ok" if gk_ok      else "not_loaded",
        "llm_provider": "ok" if llm_ok     else "no key set — structured fallback only",
    }

    all_critical = model_ok and mapping_ok and chroma_ok and gk_ok
    return {
        "status":     "healthy" if all_critical else "degraded",
        "version":    __version__,
        "components": components,
    }


# ── metrics endpoint ───────────────────────────────────────────────────────────

@app.get(
    "/api/metrics",
    tags=["System"],
    summary="Prometheus-compatible runtime metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
)
async def metrics():
    """
    GET /api/metrics

    Prometheus text exposition format.
    Scrape with: curl http://localhost:8000/api/metrics

    WHY Prometheus format:
        Standard format understood by Grafana, Datadog, Victoria Metrics,
        and any observability stack.  Even without a Prometheus server, ops
        engineers can curl this endpoint to understand system state instantly.
    """
    import asyncio
    import datetime

    uptime_s      = round(time.time() - _START_TIME, 1)
    reports_total = sum(1 for f in _REPORTS_DIR.iterdir() if f.suffix == ".pdf") if _REPORTS_DIR.exists() else 0
    uploads_total = sum(1 for _ in _UPLOADS_DIR.iterdir()) if _UPLOADS_DIR.exists() else 0
    history_total = await asyncio.to_thread(history_count)
    model_loaded  = 1 if (hasattr(app.state, "gk") and app.state.gk is not None) else 0

    lines = [
        "# HELP deepcoin_uptime_seconds Seconds since API process started",
        "# TYPE deepcoin_uptime_seconds gauge",
        f"deepcoin_uptime_seconds {uptime_s}",
        "# HELP deepcoin_reports_total PDF reports currently on disk",
        "# TYPE deepcoin_reports_total gauge",
        f"deepcoin_reports_total {reports_total}",
        "# HELP deepcoin_history_total Total analyses in history store",
        "# TYPE deepcoin_history_total counter",
        f"deepcoin_history_total {history_total}",
        "# HELP deepcoin_model_loaded 1 if EfficientNet-B3 is loaded in VRAM",
        "# TYPE deepcoin_model_loaded gauge",
        f"deepcoin_model_loaded {model_loaded}",
        "# HELP deepcoin_uploads_total Files in uploads directory",
        "# TYPE deepcoin_uploads_total gauge",
        f"deepcoin_uploads_total {uploads_total}",
    ]
    return "\n".join(lines) + "\n"


# ── root ───────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "DeepCoin API",
        "version": __version__,
        "docs":    "/docs",
        "health":  "/api/health",
        "metrics": "/api/metrics",
    }

