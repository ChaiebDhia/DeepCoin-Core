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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api._store          import ensure_store
from src.api.routes.classify import router as classify_router
from src.api.routes.history  import router as history_router

logger = logging.getLogger(__name__)

# ── paths (used by health + PDF serving) ──────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent.parent
_MODEL_PATH   = _ROOT / "models" / "best_model.pth"
_MAPPING_PATH = _ROOT / "models" / "class_mapping.pth"
_CHROMA_DIR   = _ROOT / "data" / "metadata" / "chroma_db_rag"
_REPORTS_DIR  = _ROOT / "reports"
_UPLOADS_DIR  = _ROOT / "data" / "uploads"


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle manager.

    STARTUP (before yield):
      - Create required directories (reports/, data/uploads/, data/history.json)
      - Load the Gatekeeper (EfficientNet-B3 + ChromaDB + LangGraph graph)
        stored in app.state.gk — loaded ONCE, reused for every request

    SHUTDOWN (after yield):
      - Python GC handles VRAM / RAM release
      - Log the shutdown so ops engineers see a clean stop in the logs
    """
    logger.info("DeepCoin API starting up...")

    # Ensure directories and history file exist before any request arrives
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_store()

    # Load the full pipeline once
    from src.agents.gatekeeper import Gatekeeper
    logger.info("Loading Gatekeeper (CNN + ChromaDB + LangGraph)...")
    app.state.gk = Gatekeeper()
    logger.info("Gatekeeper ready. API is now accepting requests.")

    yield   # ← application runs here

    logger.info("DeepCoin API shutting down.")


# ── app factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DeepCoin API",
    description = (
        "Archaeological coin classification and historical analysis.\n\n"
        "**Pipeline**: `EfficientNet-B3 CNN` → `LangGraph agents` → `PDF report`\n\n"
        "**Coverage**: Corpus Nummorum (9,716 coin types in KB, 438 in CNN)\n\n"
        "**Institution**: ESPRIT School of Engineering × YEBNI, Tunisia"
    ),
    version     = "0.4.0",    # 0.LAYER.patch — Layer 4 = first API release
    lifespan    = lifespan,
    docs_url    = "/docs",    # Swagger UI
    redoc_url   = "/redoc",   # ReDoc (cleaner, good for sharing with clients)
)


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
    allow_methods     = ["GET", "POST"],                      # explicit — never ["*"]
    allow_headers     = ["Content-Type", "Authorization"],    # explicit
)


# ── routers ────────────────────────────────────────────────────────────────────
#
# WHY prefix="/api":
#   All API routes live under /api so Nginx can route:
#       /api/*      → uvicorn (FastAPI)
#       /*          → Next.js frontend
#   This is the standard reverse-proxy pattern for monorepo deployments.
#
app.include_router(classify_router, prefix="/api", tags=["Classification"])
app.include_router(history_router,  prefix="/api", tags=["History"])


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
        "version":    "0.4.0",
        "components": components,
    }


# ── root ───────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "DeepCoin API",
        "version": "0.4.0",
        "docs":    "/docs",
        "health":  "/api/health",
    }

