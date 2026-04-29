"""
src/api/routes/admin_coins.py
=============================
Admin coin inventory endpoints.

WHAT:
    Provides CRUD, prefill, analytics, and gallery image upload for the
    curated admin coin catalogue.

WHY a dedicated router:
    The existing admin router already covers platform moderation and user
    analytics. Coin inventory has different concerns: duplicate prevention,
    AI-assisted prefill, provenance metadata, and gallery assets.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi import Response
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.api.auth.deps import get_current_user
from src.api.db.audit import write_audit
from src.api.db.models import CoinInventory, User
from src.api.db.session import get_db
from src.api.schemas import (
    AdminCoinCreateRequest,
    AdminCoinGalleryImage,
    AdminCoinItem,
    AdminCoinListResponse,
    AdminCoinPrefillRequest,
    AdminCoinPrefillResponse,
    AdminCoinStatCount,
    AdminCoinStatPoint,
    AdminCoinStatsResponse,
    AdminCoinUpdateRequest,
)
from src.core.rag_engine import get_rag_engine
from src.agents.gatekeeper import get_gatekeeper

from .admin import _require_privileged

router = APIRouter(prefix="/api/admin/coins", tags=["Admin Coins"])

_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_DIR = _ROOT / "data" / "admin_coin_gallery"
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
_TMP_DIR = _ROOT / "data" / "admin_coin_tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)

_META_FULL = _ROOT / "data" / "metadata" / "cn_types_metadata_full.json"
_META_FALLBACK = _ROOT / "data" / "metadata" / "cn_types_metadata.json"
_CORPUS_CACHE: list[dict[str, Any]] | None = None

_LAYER_COLORS = {
    "training": "#10b981",
    "rag_only": "#3b82f6",
    "user_added": "#d4a853",
}

_MINT_COORDS = {
    "abdera": (40.963, 24.977),
    "abydos": (40.192, 26.409),
    "adramyttion": (39.250, 26.970),
    "ainos": (40.723, 26.083),
    "alexandria troas": (39.741, 26.157),
    "anchialos": (42.493, 27.474),
    "antandros": (39.576, 26.602),
    "apollonia pontike": (42.424, 27.695),
    "assos": (39.489, 26.337),
    "augusta traiana": (42.425, 25.634),
    "byzantion": (41.008, 28.978),
    "chersonesus thracica": (40.398, 26.665),
    "dardanos": (40.077, 26.386),
    "deultum": (42.430, 27.270),
    "dionysopolis": (43.407, 28.161),
    "hadrianopolis": (41.676, 26.556),
    "ilion": (39.957, 26.238),
    "istros": (44.533, 28.766),
    "kabyle": (42.550, 26.500),
    "kallatis": (43.817, 28.583),
    "kyzikos": (40.389, 27.872),
    "lampsakos": (40.352, 26.904),
    "lysimacheia": (40.766, 26.650),
    "markianopolis": (43.214, 27.914),
    "maroneia": (40.867, 25.506),
    "mesembria": (42.659, 27.736),
    "myrina": (39.818, 26.766),
    "nikopolis ad istrum": (43.150, 25.650),
    "odessos": (43.214, 27.914),
    "parion": (40.380, 26.790),
    "pergamon": (39.120, 27.180),
    "perinthos": (40.978, 27.511),
    "philippopolis": (42.135, 24.745),
    "samothrake": (40.473, 25.523),
    "serdika": (42.697, 23.321),
    "sestos": (40.188, 26.405),
    "smyrna": (38.423, 27.142),
    "thasos": (40.778, 24.709),
    "tomis": (44.173, 28.638),
    "traianopolis": (40.850, 26.000),
}

_MINT_ALIASES = {
    "adramytteion as agathokleia": "adramyttion",
    "bizye": "hadrianopolis",
}

_REGION_CENTROIDS = {
    "thrace": (42.0, 25.0),
    "macedon": (40.6, 22.9),
    "attica": (37.98, 23.72),
    "ionia": (38.4, 27.1),
    "lydia": (38.5, 28.0),
    "caria": (37.3, 27.7),
    "bithynia": (40.6, 30.5),
    "cilicia": (37.0, 35.3),
    "phoenicia": (33.9, 35.5),
    "syria": (34.8, 36.3),
    "egypt": (30.0, 31.2),
    "sicily": (37.6, 14.0),
    "campania": (40.9, 14.5),
    "latium": (41.9, 12.5),
    "rome": (41.9, 12.5),
    "gaul": (46.2, 2.2),
    "hispania": (40.2, -3.7),
    "iberia": (40.2, -3.7),
    "britannia": (52.4, -1.5),
    "greece": (39.0, 22.0),
    "asia minor": (39.0, 31.0),
}


def _load_corpus_records() -> list[dict[str, Any]]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE

    path = _META_FULL if _META_FULL.exists() else _META_FALLBACK
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CORPUS_CACHE = []
        return _CORPUS_CACHE

    _CORPUS_CACHE = [r for r in raw if isinstance(r, dict) and r.get("type_id")]
    return _CORPUS_CACHE


def _region_to_coords(region: str | None) -> tuple[float, float] | None:
    if not region:
        return None
    low = region.lower().strip()
    for key, coords in _REGION_CENTROIDS.items():
        if key in low:
            return coords
    return None


def _normalize_place_name(value: str | None) -> str:
    if not value:
        return ""
    v = value.lower().strip()
    v = v.replace("/", " ")
    v = re.sub(r"\s+", " ", v)
    return v


def _mint_to_coords(mint: str | None) -> tuple[float, float] | None:
    key = _normalize_place_name(mint)
    if not key:
        return None
    key = _MINT_ALIASES.get(key, key)
    return _MINT_COORDS.get(key)


def _clean_string(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalize_source_url(value: str | None) -> str | None:
    cleaned = _clean_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/+$", "", cleaned)
    return cleaned


def _safe_filename(original: str) -> str:
    name = Path(original).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "image.bin"


def _image_url(filename: str) -> str:
    return f"/api/admin/coins/images/{filename}"


def _serialize_images(images: list[dict[str, Any]] | list[Any]) -> list[AdminCoinGalleryImage]:
    out: list[AdminCoinGalleryImage] = []
    for item in images or []:
        if isinstance(item, dict):
            filename = _clean_string(item.get("filename") or item.get("name") or "")
            if not filename:
                continue
            out.append(AdminCoinGalleryImage(
                filename=filename,
                url=_image_url(filename),
                caption=_clean_string(item.get("caption")),
                source=_clean_string(item.get("source")),
                is_primary=bool(item.get("is_primary", False)),
            ))
    return out


def _build_item(row: CoinInventory) -> AdminCoinItem:
    return AdminCoinItem(
        id=row.id,
        type_id=row.type_id,
        title=row.title,
        denomination=row.denomination,
        authority=row.authority,
        region=row.region,
        mint=row.mint,
        date_range=row.date_range,
        material=row.material,
        obverse=row.obverse,
        reverse=row.reverse,
        provenance=row.provenance,
        discoverer_name=row.discoverer_name,
        source_name=row.source_name,
        source_url=row.source_url,
        source_type=row.source_type,
        cartography=row.cartography,
        latitude=row.latitude,
        longitude=row.longitude,
        in_training_set=row.in_training_set,
        ai_prefilled=row.ai_prefilled,
        ai_confidence=row.ai_confidence,
        notes=row.notes,
        gallery_images=_serialize_images(row.gallery_images or []),
        created_at=row.created_at.isoformat() if row.created_at else "",
        updated_at=row.updated_at.isoformat() if row.updated_at else "",
        created_by_email=row.created_by.email if row.created_by else None,
        updated_by_email=row.updated_by.email if row.updated_by else None,
    )


def _build_payload(payload: AdminCoinCreateRequest | AdminCoinUpdateRequest) -> dict[str, Any]:
    data = payload.model_dump()
    data["title"] = _clean_string(data.get("title"))
    data["denomination"] = _clean_string(data.get("denomination"))
    data["authority"] = _clean_string(data.get("authority"))
    data["region"] = _clean_string(data.get("region"))
    data["mint"] = _clean_string(data.get("mint"))
    data["date_range"] = _clean_string(data.get("date_range"))
    data["material"] = _clean_string(data.get("material"))
    data["obverse"] = _clean_string(data.get("obverse"))
    data["reverse"] = _clean_string(data.get("reverse"))
    data["provenance"] = _clean_string(data.get("provenance"))
    data["discoverer_name"] = _clean_string(data.get("discoverer_name"))
    data["source_name"] = _clean_string(data.get("source_name"))
    data["source_url"] = _normalize_source_url(data.get("source_url"))
    data["source_type"] = _clean_string(data.get("source_type")) or "manual"
    data["cartography"] = _clean_string(data.get("cartography"))
    data["notes"] = _clean_string(data.get("notes"))
    images = []
    for img in data.get("gallery_images", []):
        if not isinstance(img, dict):
            continue
        filename = _clean_string(img.get("filename"))
        if not filename:
            continue
        images.append({
            "filename": filename,
            "caption": _clean_string(img.get("caption")),
            "source": _clean_string(img.get("source")),
            "is_primary": bool(img.get("is_primary", False)),
        })
    data["gallery_images"] = images
    return data


def _validate_coordinates(latitude: float | None, longitude: float | None) -> None:
    if latitude is None and longitude is None:
        return
    if latitude is None or longitude is None:
        raise HTTPException(status_code=422, detail="Latitude and longitude must be provided together.")
    if not (-90 <= latitude <= 90):
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180.")


def _draft_from_record(record: dict[str, Any], *, source: str, confidence: float | None, duplicate_exists: bool) -> AdminCoinPrefillResponse:
    type_id = str(record.get("type_id") or record.get("id") or "")
    denomination = _clean_string(record.get("denomination")) or "Uncatalogued coin"
    authority = _clean_string(record.get("authority"))
    region = _clean_string(record.get("region"))
    mint = _clean_string(record.get("mint"))
    date_range = _clean_string(record.get("date_range") or record.get("date"))
    material = _clean_string(record.get("material"))
    title_bits = [denomination]
    if authority:
        title_bits.append(authority)
    elif mint:
        title_bits.append(mint)

    coin = AdminCoinCreateRequest(
        type_id=type_id,
        title=" — ".join(title_bits),
        denomination=denomination,
        authority=authority,
        region=region,
        mint=mint,
        date_range=date_range,
        material=material,
        obverse=_clean_string(record.get("obverse_design") or record.get("obverse")),
        reverse=_clean_string(record.get("reverse_design") or record.get("reverse")),
        provenance=_clean_string(record.get("extra", {}).get("citation") if isinstance(record.get("extra"), dict) else None),
        source_name=_clean_string(record.get("source_name") or "Corpus Nummorum"),
        source_url=_clean_string(record.get("source_url")),
        source_type=source,
        cartography=_clean_string(record.get("cartography") or record.get("findspot") or region),
        latitude=record.get("latitude"),
        longitude=record.get("longitude"),
        in_training_set=bool(record.get("in_training_set", False)),
        ai_prefilled=True,
        ai_confidence=confidence,
        notes=_clean_string(record.get("notes")),
        gallery_images=[],
    )
    warnings: list[str] = []
    if not coin.title:
        warnings.append("Title was empty and has been replaced with a fallback.")
    if not coin.denomination:
        warnings.append("Denomination was missing from the source record.")
    return AdminCoinPrefillResponse(
        source=source,
        matched_type_id=type_id or None,
        confidence=confidence,
        duplicate_exists=duplicate_exists,
        warnings=warnings,
        coin=coin,
    )


async def _ensure_unique_coin(db: AsyncSession, type_id: str, source_url: str | None, coin_id: str | None = None) -> None:
    q = select(CoinInventory).where(CoinInventory.type_id == type_id)
    if coin_id:
        q = q.where(CoinInventory.id != coin_id)
    existing = (await db.execute(q)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Coin type {type_id} already exists in the inventory.")

    if source_url:
        src_q = select(CoinInventory).where(CoinInventory.source_url == source_url)
        if coin_id:
            src_q = src_q.where(CoinInventory.id != coin_id)
        src_match = (await db.execute(src_q)).scalar_one_or_none()
        if src_match is not None:
            raise HTTPException(status_code=409, detail="A coin with the same source URL already exists.")


def _generate_internal_type_id() -> str:
    return f"USR-{uuid4().hex[:10].upper()}"


@router.get("/images/{filename}")
async def get_coin_image(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = _IMAGE_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


@router.get("")
async def list_coin_inventory(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
    region: str | None = Query(None),
    source_type: str | None = Query(None),
    in_training_set: bool | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinListResponse:
    _require_privileged(current_user)

    base = select(CoinInventory).options(joinedload(CoinInventory.created_by), joinedload(CoinInventory.updated_by)).order_by(desc(CoinInventory.updated_at))
    if search and search.strip():
        term = f"%{search.strip()}%"
        base = base.where(
            or_(
                CoinInventory.type_id.ilike(term),
                CoinInventory.title.ilike(term),
                CoinInventory.denomination.ilike(term),
                CoinInventory.authority.ilike(term),
                CoinInventory.region.ilike(term),
                CoinInventory.mint.ilike(term),
                CoinInventory.source_name.ilike(term),
            )
        )
    if region and region.strip():
        base = base.where(CoinInventory.region.ilike(f"%{region.strip()}%"))
    if source_type and source_type.strip():
        base = base.where(CoinInventory.source_type == source_type.strip())
    if in_training_set is not None:
        base = base.where(CoinInventory.in_training_set == in_training_set)

    rows = (await db.execute(base.offset(skip).limit(limit))).scalars().unique().all()

    count_q = select(func.count()).select_from(CoinInventory)
    if search and search.strip():
        term = f"%{search.strip()}%"
        count_q = count_q.where(
            or_(
                CoinInventory.type_id.ilike(term),
                CoinInventory.title.ilike(term),
                CoinInventory.denomination.ilike(term),
                CoinInventory.authority.ilike(term),
                CoinInventory.region.ilike(term),
                CoinInventory.mint.ilike(term),
                CoinInventory.source_name.ilike(term),
            )
        )
    if region and region.strip():
        count_q = count_q.where(CoinInventory.region.ilike(f"%{region.strip()}%"))
    if source_type and source_type.strip():
        count_q = count_q.where(CoinInventory.source_type == source_type.strip())
    if in_training_set is not None:
        count_q = count_q.where(CoinInventory.in_training_set == in_training_set)
    total = (await db.execute(count_q)).scalar_one()

    return AdminCoinListResponse(
        items=[_build_item(row) for row in rows],
        total=total,
        skip=skip,
        limit=limit,
        pages=max(1, math.ceil(total / limit)) if limit else 1,
    )


@router.get("/stats")
async def coin_inventory_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinStatsResponse:
    _require_privileged(current_user)

    corpus_records = _load_corpus_records()
    kb_total = len(corpus_records)
    kb_training_total = sum(1 for r in corpus_records if bool(r.get("in_training_set", False)))
    kb_rag_only_total = max(0, kb_total - kb_training_total)

    total = (await db.execute(select(func.count()).select_from(CoinInventory))).scalar_one()
    user_total = total
    manual_count = (await db.execute(select(func.count()).select_from(CoinInventory).where(CoinInventory.source_type == "manual"))).scalar_one()
    ai_prefilled = (await db.execute(select(func.count()).select_from(CoinInventory).where(CoinInventory.ai_prefilled.is_(True)))).scalar_one()
    in_training_set = (await db.execute(select(func.count()).select_from(CoinInventory).where(CoinInventory.in_training_set.is_(True)))).scalar_one()
    gallery_rows = (await db.execute(select(CoinInventory.gallery_images))).scalars().all()
    with_gallery = sum(1 for images in gallery_rows if isinstance(images, list) and len(images) > 0)

    source_rows = (await db.execute(
        select(CoinInventory.source_type, func.count(CoinInventory.id)).group_by(CoinInventory.source_type).order_by(desc(func.count(CoinInventory.id)))
    )).all()
    region_rows = (await db.execute(
        select(CoinInventory.region, func.count(CoinInventory.id)).where(CoinInventory.region.is_not(None)).group_by(CoinInventory.region).order_by(desc(func.count(CoinInventory.id))).limit(12)
    )).all()
    mint_rows = (await db.execute(
        select(CoinInventory.mint, func.count(CoinInventory.id)).where(CoinInventory.mint.is_not(None)).group_by(CoinInventory.mint).order_by(desc(func.count(CoinInventory.id))).limit(12)
    )).all()
    user_rows = (await db.execute(
        select(CoinInventory.region, CoinInventory.mint, CoinInventory.latitude, CoinInventory.longitude)
        .order_by(desc(CoinInventory.updated_at))
        .limit(250)
    )).all()

    user_point_counts: dict[tuple[str | None, str | None, float, float], int] = defaultdict(int)
    for region, mint, lat, lng in user_rows:
        _lat = float(lat) if lat is not None else None
        _lng = float(lng) if lng is not None else None
        if _lat is None or _lng is None:
            derived = _region_to_coords(region)
            if derived is None:
                continue
            _lat, _lng = derived
        user_point_counts[(region, mint, _lat, _lng)] += 1

    map_points: list[AdminCoinStatPoint] = []
    for (region, mint, lat, lng), count in sorted(user_point_counts.items(), key=lambda kv: kv[1], reverse=True)[:80]:
        map_points.append(
            AdminCoinStatPoint(
                layer="user_added",
                color=_LAYER_COLORS["user_added"],
                region=region,
                mint=mint,
                latitude=lat,
                longitude=lng,
                count=count,
            )
        )

    corpus_point_counts: dict[tuple[str, str | None, str | None, float, float], int] = defaultdict(int)
    for rec in corpus_records:
        region = (rec.get("region") or "").strip() or None
        mint = (rec.get("mint") or "").strip() or None
        coords = _mint_to_coords(mint) or _region_to_coords(region)
        if not coords:
            continue
        layer = "training" if bool(rec.get("in_training_set", False)) else "rag_only"
        corpus_point_counts[(layer, region, mint, coords[0], coords[1])] += 1

    # Use many mint-level points to avoid one central blob and mimic corpus map behavior.
    for (layer, region, mint, lat, lng), count in sorted(corpus_point_counts.items(), key=lambda kv: kv[1], reverse=True)[:500]:
        map_points.append(
            AdminCoinStatPoint(
                layer=layer,
                color=_LAYER_COLORS[layer],
                region=region,
                mint=mint,
                latitude=lat,
                longitude=lng,
                count=count,
            )
        )

    return AdminCoinStatsResponse(
        total=total,
        kb_total=kb_total,
        kb_training_total=kb_training_total,
        kb_rag_only_total=kb_rag_only_total,
        user_total=user_total,
        manual_count=manual_count,
        ai_prefilled=ai_prefilled,
        in_training_set=in_training_set,
        with_gallery=with_gallery,
        by_source_type=[AdminCoinStatCount(label=src or "unknown", count=count) for src, count in source_rows],
        by_region=[AdminCoinStatCount(label=region or "unknown", count=count) for region, count in region_rows],
        by_mint=[AdminCoinStatCount(label=mint or "unknown", count=count) for mint, count in mint_rows],
        map_points=map_points,
    )


@router.post("/prefill")
async def prefill_coin(
    body: AdminCoinPrefillRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinPrefillResponse:
    _require_privileged(current_user)

    if not (body.type_id and body.type_id.strip()) and not (body.query and body.query.strip()):
        raise HTTPException(status_code=422, detail="Provide either type_id or query for prefill.")

    engine = get_rag_engine()
    record: dict[str, Any] | None = None
    source = "kb"
    confidence: float | None = None

    if body.type_id and body.type_id.strip():
        tid = body.type_id.strip()
        try:
            record = engine.get_by_id(int(tid))
        except ValueError:
            record = None
        confidence = 1.0 if record else None
    elif body.query and body.query.strip():
        hits = engine.search(body.query.strip(), n=1)
        if hits:
            hit = hits[0]
            tid_raw = hit.get("type_id")
            if tid_raw is not None and str(tid_raw).isdigit():
                record = engine.get_by_id(int(tid_raw))
            else:
                record = hit
            raw_score = float(hit.get("rrf_score", hit.get("score", 0.0)) or 0.0)
            confidence = min(1.0, raw_score / 2.0)
            source = "search"
    if not record:
        raise HTTPException(status_code=404, detail="No matching CN record found for prefill.")

    duplicate_exists = False
    if record.get("type_id"):
        existing = (await db.execute(select(CoinInventory.id).where(CoinInventory.type_id == str(record.get("type_id"))))).first()
        duplicate_exists = existing is not None

    response = _draft_from_record(record, source=source, confidence=confidence, duplicate_exists=duplicate_exists)
    if body.query and body.query.strip():
        response.warnings.append(f"Prefill based on search query: {body.query.strip()}")
    return response


@router.post("/prefill-image")
async def prefill_coin_from_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinPrefillResponse:
    _require_privileged(current_user)

    original = _safe_filename(file.filename or "coin-image.bin")
    temp_name = f"scan_{uuid4().hex}_{original}"
    temp_path = _TMP_DIR / temp_name
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        gk = get_gatekeeper()
        pred = gk._inference.predict(str(temp_path), tta=True)
        label = str(pred.get("label") or "").strip()
        confidence = float(pred.get("confidence", 0.0) or 0.0)

        record: dict[str, Any] | None = None
        engine = get_rag_engine()
        if label.isdigit():
            record = engine.get_by_id(int(label))

        if not record and pred.get("top5"):
            for candidate in pred.get("top5", []):
                cid = str(candidate.get("label") or "").strip()
                if cid.isdigit():
                    record = engine.get_by_id(int(cid))
                    if record:
                        break

        if not record:
            raise HTTPException(status_code=404, detail="AI could not map this image to a known CN type for prefill.")

        duplicate_exists = False
        if record.get("type_id"):
            existing = (await db.execute(select(CoinInventory.id).where(CoinInventory.type_id == str(record.get("type_id"))))).first()
            duplicate_exists = existing is not None

        response = _draft_from_record(
            record,
            source="image-ai",
            confidence=confidence,
            duplicate_exists=duplicate_exists,
        )
        response.warnings.append(
            f"Image AI draft based on CNN label {pred.get('label')} at {confidence * 100:.1f}% confidence. Verify before saving."
        )
        return response
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/{coin_id}")
async def get_coin(
    coin_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinItem:
    _require_privileged(current_user)
    row = (await db.execute(
        select(CoinInventory)
        .options(joinedload(CoinInventory.created_by), joinedload(CoinInventory.updated_by))
        .where(CoinInventory.id == coin_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Coin inventory record not found.")
    return _build_item(row)


@router.post("")
async def create_coin(
    body: AdminCoinCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinItem:
    _require_privileged(current_user)
    payload = _build_payload(body)
    if not payload["title"] or not payload["denomination"]:
        raise HTTPException(status_code=422, detail="Title and denomination are required.")
    payload_type_id = _clean_string(payload.get("type_id")) or _generate_internal_type_id()
    payload["type_id"] = payload_type_id
    _validate_coordinates(payload.get("latitude"), payload.get("longitude"))
    await _ensure_unique_coin(db, payload_type_id, payload.get("source_url"))
    row = CoinInventory(
        type_id=payload_type_id,
        title=payload["title"],
        denomination=payload["denomination"],
        authority=payload.get("authority"),
        region=payload.get("region"),
        mint=payload.get("mint"),
        date_range=payload.get("date_range"),
        material=payload.get("material"),
        obverse=payload.get("obverse"),
        reverse=payload.get("reverse"),
        provenance=payload.get("provenance"),
        discoverer_name=payload.get("discoverer_name"),
        source_name=payload.get("source_name"),
        source_url=payload.get("source_url"),
        source_type=payload.get("source_type", "manual"),
        cartography=payload.get("cartography"),
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        in_training_set=bool(payload.get("in_training_set", False)),
        ai_prefilled=bool(payload.get("ai_prefilled", False)),
        ai_confidence=payload.get("ai_confidence"),
        notes=payload.get("notes"),
        gallery_images=payload.get("gallery_images", []),
        created_by_user_id=current_user.id,
        updated_by_user_id=current_user.id,
    )
    db.add(row)
    await db.flush()
    await write_audit(
        db,
        action="coin_inventory.create",
        user_id=current_user.id,
        resource_type="coin_inventory",
        resource_id=row.id,
        payload={"type_id": row.type_id, "source_type": row.source_type},
    )
    await db.refresh(row)
    return _build_item(row)


@router.patch("/{coin_id}")
async def update_coin(
    coin_id: str,
    body: AdminCoinUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinItem:
    _require_privileged(current_user)
    row = (await db.execute(select(CoinInventory).where(CoinInventory.id == coin_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Coin inventory record not found.")

    payload = _build_payload(body)
    _validate_coordinates(payload.get("latitude"), payload.get("longitude"))
    await _ensure_unique_coin(db, row.type_id, payload.get("source_url"), coin_id=row.id)

    for field in (
        "title", "denomination", "authority", "region", "mint", "date_range", "material", "obverse", "reverse",
        "provenance", "discoverer_name", "source_name", "source_url", "source_type", "cartography", "latitude", "longitude",
        "in_training_set", "ai_prefilled", "ai_confidence", "notes", "gallery_images",
    ):
        setattr(row, field, payload.get(field))
    row.updated_by_user_id = current_user.id
    await write_audit(
        db,
        action="coin_inventory.update",
        user_id=current_user.id,
        resource_type="coin_inventory",
        resource_id=row.id,
        payload={"type_id": row.type_id, "source_type": row.source_type},
    )
    await db.flush()
    await db.refresh(row)
    return _build_item(row)


@router.delete("/{coin_id}", status_code=204, response_class=Response)
async def delete_coin(
    coin_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    _require_privileged(current_user)
    row = (await db.execute(select(CoinInventory).where(CoinInventory.id == coin_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Coin inventory record not found.")
    await write_audit(
        db,
        action="coin_inventory.delete",
        user_id=current_user.id,
        resource_type="coin_inventory",
        resource_id=row.id,
        payload={"type_id": row.type_id},
    )
    await db.delete(row)
    return Response(status_code=204)


@router.post("/{coin_id}/images")
async def upload_coin_image(
    coin_id: str,
    file: UploadFile = File(...),
    caption: str = Form(default=""),
    source: str = Form(default=""),
    primary: bool = Form(default=False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AdminCoinItem:
    _require_privileged(current_user)
    row = (await db.execute(select(CoinInventory).where(CoinInventory.id == coin_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Coin inventory record not found.")

    original = _safe_filename(file.filename or "coin-image.bin")
    stored_name = f"{coin_id}_{uuid4().hex}_{original}"
    path = _IMAGE_DIR / stored_name
    content = await file.read()
    path.write_bytes(content)

    images = list(row.gallery_images or [])
    if primary:
        for item in images:
            if isinstance(item, dict):
                item["is_primary"] = False
    auto_primary = not images
    images.append({
        "filename": stored_name,
        "caption": _clean_string(caption),
        "source": _clean_string(source),
        "is_primary": bool(primary or auto_primary),
    })
    row.gallery_images = images
    row.updated_by_user_id = current_user.id
    await write_audit(
        db,
        action="coin_inventory.image_upload",
        user_id=current_user.id,
        resource_type="coin_inventory",
        resource_id=row.id,
        payload={"type_id": row.type_id, "filename": stored_name},
    )
    await db.flush()
    await db.refresh(row)
    return _build_item(row)
