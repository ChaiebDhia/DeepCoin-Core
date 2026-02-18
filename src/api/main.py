"""
FastAPI Backend - Main Entry Point
Run with: uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DeepCoin API",
    description="Archaeological coin classification and historical analysis",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "DeepCoin API - Running",
        "version": "1.0.0",
        "status": "Phase 1 Complete - Training Pipeline Coming Soon"
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "ml_model": "not_loaded",
            "agents": "not_initialized",
            "database": "not_connected"
        }
    }


# TODO: Add routes in Phase 3
# from src.api.routes import classify, history, validate
