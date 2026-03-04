"""
src/api/routes/kb.py
=====================
Knowledge Base discovery/browse endpoint.

WHAT: Exposes the 9,541-type Corpus Nummorum knowledge base for browsing
      and searching directly from the frontend — independent of coin photos.

WHY THIS ENDPOINT EXISTS:
    The Explore page needs to show VALUE beyond 'user-analysis gallery'.
    The KB contains structured scholarly records for 9,541 CN types.
    Exposing this as a browsable, searchable resource lets users discover
    coins by denomination, dynasty, material, and region — without needing
    to upload a photo.  It showcases the depth of the scholarly dataset.

API:
    GET /api/kb/types
        ?search=      text query (BM25+vector hybrid search if non-empty)
        ?skip=        pagination offset  (default 0)
        ?limit=       page size          (default 20, max 50)
        ?in_training_set=true  filter to CNN-known types only

    Returns:
        { items: KbTypeItem[], total: int, search_used: bool }

    KbTypeItem fields:
        type_id, denomination, region, date_range, material,
        mint, authority, in_training_set, text_snippet
"""
from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["Knowledge Base"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_identity(text: str) -> dict[str, str]:
    """
    Parse a pipe-delimited identity chunk into structured fields.

    Identity chunk format (from rag_engine._prepare_chunks):
        "CN type 1015 | denomination: drachm | mint: Maroneia | region: Thrace |
         date: c.365–330 BC | period: Greek"

    Extracts each labeled field into a dict.  Unknown fields are ignored.
    Always safe — returns empty strings for missing fields.
    """
    result: dict[str, str] = {
        "denomination": "", "mint": "", "region": "",
        "date_range": "", "period": "", "authority": "",
    }
    parts = [p.strip() for p in text.split("|")]
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key in result:
            result[key] = val
        elif key == "date":
            result["date_range"] = val
    return result


def _build_item(type_id: int, record: dict[str, Any], engine: Any) -> dict[str, Any]:
    """
    Build a KbTypeItem dict from a RAG engine record.

    Combines metadata from the full record dict (denomination, region,
    material, etc.) with in_training_set flag and a short text snippet from
    the identity chunk.
    """
    tid_str   = str(type_id)
    chunk_id  = f"{type_id}_identity"
    chunk     = engine._chunk_index.get(chunk_id)
    snippet   = chunk["text"][:250] if chunk else ""

    return {
        "type_id":          tid_str,
        "denomination":     record.get("denomination", ""),
        "region":           record.get("region", ""),
        "date_range":       record.get("date_range", ""),
        "material":         record.get("material", ""),
        "mint":             record.get("mint", ""),
        "authority":        record.get("authority", ""),
        "in_training_set":  bool(chunk and chunk.get("in_training_set")),
        "text_snippet":     snippet,
    }


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get(
    "/types",
    summary = "Browse / search the Corpus Nummorum knowledge base",
)
async def browse_kb_types(
    search:          str  = Query(default="", description="Full-text search query"),
    skip:            int  = Query(default=0,  ge=0,  description="Pagination offset"),
    limit:           int  = Query(default=20, ge=1, le=50, description="Page size"),
    in_training_set: bool = Query(default=False, description="Filter to CNN-known types only"),
) -> dict[str, Any]:
    """
    Browse all 9,541 Corpus Nummorum coin types with optional text search.

    NO authentication required — the Corpus Nummorum database is public
    scholarly data.

    HOW IT WORKS:
        Without search: returns all identity chunks, paginated, newest IDs
            sorted first so freshly-scraped types appear.
        With  search:   calls the hybrid BM25+vector RAG search, deduplicates
            by type_id, and returns structured records for the top matches.

    WHY deduplicate by type_id:
        The RAG engine searches ALL 5 chunk types. A "silver" query may match
        the material chunk AND the identity chunk for the same coin. Without
        deduplication the results list would repeat the same type multiple times.
    """
    from src.core.rag_engine import get_rag_engine

    engine = get_rag_engine()

    if search.strip():
        # ── Text search: hybrid BM25+vector, deduplicated by type_id ────────
        hits = engine.search(search.strip(), n=min(200, (skip + limit) * 3))

        seen_ids: set[int] = set()
        unique: list[dict[str, Any]] = []

        for hit in hits:
            tid_raw = hit.get("type_id", hit.get("id"))
            if tid_raw is None:
                continue
            try:
                tid = int(tid_raw)
            except (ValueError, TypeError):
                continue
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            record = engine.get_by_id(tid) or {}
            if in_training_set and not record.get("in_training_set"):
                continue
            unique.append(_build_item(tid, record, engine))

        total = len(unique)
        items = unique[skip : skip + limit]
        return {"items": items, "total": total, "search_used": True}

    else:
        # ── Browse all: iterate identity chunks, paginated ───────────────────
        all_identity = [
            ch for ch in engine._all_chunks
            if ch["chunk_type"] == "identity"
        ]
        if in_training_set:
            all_identity = [ch for ch in all_identity if ch.get("in_training_set")]

        # Sort by type_id numerically so the browse order is consistent
        all_identity.sort(key=lambda c: int(re.sub(r"\D", "", c["chunk_id"].split("_")[0]) or "0"))

        total   = len(all_identity)
        page    = all_identity[skip : skip + limit]
        items   = []

        for ch in page:
            # extract type_id from chunk_id "<tid>_identity"
            id_part = ch["chunk_id"].rsplit("_", 1)[0]
            try:
                tid = int(id_part)
            except ValueError:
                continue
            record = engine.get_by_id(tid) or {}
            items.append(_build_item(tid, record, engine))

        return {"items": items, "total": total, "search_used": False}
