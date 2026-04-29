"""
src/api/schemas.py
==================
Pydantic v2 request/response models for the DeepCoin API.

WHY Pydantic v2 models matter (beyond just type safety):
    FastAPI uses these models for THREE things simultaneously:
      1. INPUT VALIDATION — invalid data is rejected with a 422 before it
         reaches your route function. The CNN and Gatekeeper never see bad input.
      2. OUTPUT SERIALISATION — FastAPI converts the gatekeeper's raw Python
         dicts into these models, then to JSON. Extra keys are stripped.
         Callers always get a contract-compliant response, not an internal dict.
      3. AUTO-DOCUMENTATION — FastAPI reads the models and generates the
         full OpenAPI schema at /docs. This is your free API documentation.

WHY separate from gatekeeper's internal dicts:
    The gatekeeper's internal state (CoinState) contains everything — raw
    tensors, intermediate agent dicts, full LangGraph state. We never want to
    expose that to the network. These schemas are the PUBLIC CONTRACT: what
    clients (Next.js frontend, museum integration) are allowed to see.
    Think of it as the difference between your internal work-in-progress
    notes and the final professional report you hand to the client.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── CNN sub-models ─────────────────────────────────────────────────────────────

class Top5Item(BaseModel):
    """One entry in the CNN top-5 prediction list."""
    rank:       int   = Field(..., ge=1, le=5, description="1 = best match")
    class_id:   int   = Field(..., description="CNN sort-order index (0–437)")
    label:      str   = Field(..., description="CN type ID string, e.g. '1015'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")


class CnnResult(BaseModel):
    """
    The raw output from the EfficientNet-B3 classifier.

    WHY expose this separately from the narrative:
        Users (and future API consumers) need the raw confidence to understand
        how certain the CNN was BEFORE the agents processed the result.
        A 91% confidence with a rich historical narrative is a different level
        of trust than a 22% confidence with a VLM visual description.

    Extra calibration fields:
        vote_fraction  — fraction of TTA passes that independently agreed on
                         the same top-1 class.  Complements temperature-scaled
                         softmax: high vote_fraction = high agreement, not just
                         high softmax magnitude.
        tta_passes     — how many TTA forward passes were used (1 = single-pass)
        temperature    — the T scalar applied before softmax; < 1 means the
                         model was under-confident and the distribution was
                         sharpened by calibration.
    """
    class_id:          int                = Field(..., description="CNN sort-order index")
    label:             str                = Field(..., description="CN type ID, e.g. '1015'")
    confidence:        float              = Field(..., ge=0.0, le=1.0)
    top5:              list[Top5Item]
    inference_time_ms: int                = Field(..., ge=0)
    tta_used:          bool
    vote_fraction:     Optional[float]    = Field(
        None, ge=0.0, le=1.0,
        description="Fraction of TTA passes agreeing on top-1 class (None = no TTA)",
    )
    tta_passes:        int                = Field(
        1, ge=1, description="Number of TTA forward passes performed"
    )
    temperature:       float              = Field(
        1.0, description="Calibration temperature T used in softmax(z/T)"
    )
    gradcam_url:       Optional[str]      = Field(
        None,
        description=(
            "URL to the Grad-CAM heatmap overlay PNG for this prediction. "
            "Served via GET /api/gradcam/{filename}. None when unavailable "
            "(pytorch-grad-cam not installed, or generation failed)."
        ),
    )


# ── Main response model ────────────────────────────────────────────────────────

class ClassifyResponse(BaseModel):
    """
    Full response from POST /api/classify.

    This is the public contract for every consumer of the API.
    All internal dicts (CoinState, historian_result, etc.) are flattened
    or summarised into this schema — nothing internal leaks out.
    """
    id:               str            = Field(..., description="UUID for history retrieval")
    timestamp:        str            = Field(..., description="ISO 8601 UTC timestamp")
    image_filename:   str            = Field(..., description="Original uploaded filename")
    route_taken:      str            = Field(..., description="historian | validator | investigator")
    cnn:              CnnResult
    # ── narrative fields (from historian or investigator) ──────────────────────
    narrative:        Optional[str]  = Field(None, description="LLM-generated historical analysis")
    mint:             Optional[str]  = Field(None)
    region:           Optional[str]  = Field(None)
    date_range:       Optional[str]  = Field(None)
    material:         Optional[str]  = Field(None)
    denomination:     Optional[str]  = Field(None)
    # ── validation fields (from validator) ────────────────────────────────────
    material_status:  Optional[str]  = Field(None, description="consistent | mismatch | unknown")
    material_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    # ── investigator fields (low confidence route) ────────────────────────────
    visual_description: Optional[str] = Field(None, description="VLM visual analysis")
    kb_match_count:   Optional[int]  = Field(None)
    # ── output ────────────────────────────────────────────────────────────────
    pdf_url:          Optional[str]  = Field(None, description="Download URL for the PDF report")
    processing_time_s: float         = Field(..., ge=0.0)


# ── History models ─────────────────────────────────────────────────────────────

class HistorySummary(BaseModel):
    """
    Compact summary for the GET /api/history list endpoint.

    WHY not return the full ClassifyResponse in the list:
        Loading 100 past results with full narratives and base64 images would
        send megabytes of data when the user just wants a table of "what was
        classified when". The list shows summaries; the detail endpoint
        GET /api/history/{id} returns the full ClassifyResponse.
        This is a standard REST pattern: collection vs. resource.
    """
    id:             str
    timestamp:      str
    image_filename: str
    route_taken:    str
    label:          str
    confidence:     float = Field(..., ge=0.0, le=1.0)
    pdf_url:        Optional[str] = None


class HistoryListResponse(BaseModel):
    """Paginated history list."""
    items: list[HistorySummary]
    total: int  = Field(..., description="Total records (before pagination)")
    skip:  int  = Field(..., description="Offset applied")
    limit: int  = Field(..., description="Page size applied")


# ── Admin coin inventory models ──────────────────────────────────────────────

class AdminCoinGalleryImage(BaseModel):
    """One gallery image attached to a curated coin inventory record."""
    filename: str
    url:      str
    caption:  Optional[str] = None
    source:   Optional[str] = None
    is_primary: bool = False


class AdminCoinBase(BaseModel):
    """Shared editable fields for admin coin inventory records."""
    title:           Optional[str] = None
    denomination:    Optional[str] = None
    authority:       Optional[str] = None
    region:          Optional[str] = None
    mint:            Optional[str] = None
    date_range:      Optional[str] = None
    material:        Optional[str] = None
    obverse:         Optional[str] = None
    reverse:         Optional[str] = None
    provenance:      Optional[str] = None
    discoverer_name: Optional[str] = None
    source_name:     Optional[str] = None
    source_url:      Optional[str] = None
    source_type:     str = Field("manual", description="manual | kb | ai | import | museum | publication")
    cartography:     Optional[str] = None
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None
    in_training_set: bool = False
    ai_prefilled:    bool = False
    ai_confidence:   Optional[float] = Field(None, ge=0.0, le=1.0)
    notes:           Optional[str] = None
    gallery_images:  list[AdminCoinGalleryImage] = Field(default_factory=list)


class AdminCoinCreateRequest(AdminCoinBase):
    """Create a new admin coin inventory record."""
    type_id: Optional[str] = Field(None, description="CN type ID or internal coin identifier (auto-generated when empty)")


class AdminCoinUpdateRequest(AdminCoinBase):
    """Patch an existing admin coin inventory record."""
    pass


class AdminCoinItem(AdminCoinBase):
    """Full admin coin inventory record returned by the API."""
    id:                str
    type_id:           str
    created_at:        str
    updated_at:        str
    created_by_email:  Optional[str] = None
    updated_by_email:  Optional[str] = None


class AdminCoinListResponse(BaseModel):
    items: list[AdminCoinItem]
    total: int
    skip:  int
    limit: int
    pages: int


class AdminCoinStatCount(BaseModel):
    label: str
    count: int


class AdminCoinStatPoint(BaseModel):
    layer:     str
    color:     str
    region:    Optional[str] = None
    mint:      Optional[str] = None
    latitude:  Optional[float] = None
    longitude: Optional[float] = None
    count:     int


class AdminCoinStatsResponse(BaseModel):
    total:            int
    kb_total:         int
    kb_training_total: int
    kb_rag_only_total: int
    user_total:       int
    manual_count:     int
    ai_prefilled:     int
    in_training_set:  int
    with_gallery:     int
    by_source_type:   list[AdminCoinStatCount]
    by_region:        list[AdminCoinStatCount]
    by_mint:          list[AdminCoinStatCount]
    map_points:       list[AdminCoinStatPoint]


class AdminCoinPrefillRequest(BaseModel):
    """Request the backend to draft a curated coin record from KB data."""
    type_id: Optional[str] = Field(None, description="Exact CN type ID")
    query:   Optional[str] = Field(None, description="Free-text search query")


class AdminCoinPrefillResponse(BaseModel):
    """Backend-generated prefill suggestion for the admin coin editor."""
    source:          str
    matched_type_id:  Optional[str] = None
    confidence:      Optional[float] = None
    duplicate_exists: bool = False
    warnings:        list[str] = Field(default_factory=list)
    coin:            AdminCoinCreateRequest
