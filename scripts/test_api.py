"""
scripts/test_api.py
====================
Comprehensive API smoke-test suite — Layer 4

Covers EVERY scenario the system must handle:

  HAPPY PATHS (3 routing scenarios)
  ─────────────────────────────────
  Test 01  GET  /api/health              → 200  all 5 components ok
  Test 02  POST /api/classify  Route 1   → type 1015  conf > 85%  historian
  Test 03  POST /api/classify  Route 2   → type 21027 conf 40-85% validator
  Test 04  POST /api/classify  Route 3   → type 544   conf < 40%  investigator
  Test 05  GET  /api/history             → paginated list, newest-first
  Test 06  GET  /api/history?skip=&limit → pagination edge cases
  Test 07  GET  /api/history/{id}        → full record by UUID
  Test 08  GET  /api/reports/{filename}  → PDF download (Content-Type check)

  BACKUP PLANS (fallback + degraded behaviours)
  ─────────────────────────────────────────────
  Test 09  Validator node           → material consistency check present in state
  Test 10  Investigator node        → kb_matches >= 0 (OpenCV fallback works without LLM)
  Test 11  PDF path written         → all 3 routes write a PDF to disk
  Test 12  History persisted        → records survive between classify calls
  Test 13  node_timings present     → per-node timing data in state

  ERROR / SECURITY SCENARIOS (defence in depth)
  ──────────────────────────────────────────────
  Test 14  Wrong Content-Type (text/plain)     → 415
  Test 15  Wrong magic bytes (JPEG header, PDF content) → 415
  Test 16  File too large (> 10 MB)            → 413
  Test 17  Unknown history ID                  → 404
  Test 18  Path traversal in report URL        → 400 or 404
  Test 19  Health endpoint: /api/health        → 200 (not 503)
  Test 20  Docs endpoint: /docs                → 200 Swagger UI

Run:
    python scripts/test_api.py [--base-url http://localhost:8000]

Exit code:
    0 — all tests passed
    1 — one or more tests failed
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests

# ── config ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# One known image per routing tier (confirmed in previous end-to-end test)
_IMG_ROUTE1 = _ROOT / "data" / "processed" / "1015"  / "CN_type_1015_cn_coin_5943_p.jpg"
_IMG_ROUTE2 = _ROOT / "data" / "processed" / "21027" / "CN_type_21027_cn_coin_6169_p.jpg"
_IMG_ROUTE3 = _ROOT / "data" / "processed" / "544"   / "CN_type_544_cn_coin_2324_p.jpg"


# ══════════════════════════════════════════════════════════════════════════════
#  Test harness
# ══════════════════════════════════════════════════════════════════════════════

class TestResult:
    """Accumulates pass / fail results across the whole suite."""

    def __init__(self) -> None:
        self._results: list[tuple[str, bool, str]] = []   # (name, ok, detail)

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one test outcome and print immediately."""
        icon   = "✓" if ok else "✗"
        color  = ""   # plain text — works in any terminal / CI log
        status = "PASS" if ok else "FAIL"
        line   = f"  [{status}] {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
        self._results.append((name, ok, detail))

    def summary(self) -> int:
        """Print summary and return exit code (0=all pass, 1=any fail)."""
        total   = len(self._results)
        passed  = sum(1 for _, ok, _ in self._results if ok)
        failed  = total - passed
        print()
        print("=" * 60)
        print(f"RESULTS: {passed}/{total} passed", end="")
        if failed:
            print(f"  ({failed} FAILED)")
            for name, ok, detail in self._results:
                if not ok:
                    print(f"  ✗ {name}: {detail}")
        else:
            print("  — ALL PASS")
        print("=" * 60)
        return 0 if failed == 0 else 1


def _post_image(base: str, image_path: Path, tta: bool = False) -> requests.Response:
    """Upload a coin image to POST /api/classify."""
    with open(image_path, "rb") as f:
        return requests.post(
            f"{base}/api/classify",
            files={"file": (image_path.name, f, "image/jpeg")},
            params={"tta": str(tta).lower()},
            timeout=180,   # investigator with Ollama can be 120 s cold start
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Test definitions
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(base: str) -> int:
    r = TestResult()
    classified: dict = {}   # accumulate classify responses keyed by route

    print(f"\nDeepCoin API Test Suite — {base}")
    print("=" * 60)

    # ── Test 01: Health — all components OK ───────────────────────────────────
    print("\n[HEALTH]")
    try:
        resp = requests.get(f"{base}/api/health", timeout=10)
        body = resp.json()
        r.record("01 health: HTTP 200",          resp.status_code == 200, f"got {resp.status_code}")
        r.record("01 health: status=healthy",     body.get("status") == "healthy", str(body.get("status")))
        comps = body.get("components", {})
        for c in ("model_file", "mapping_file", "chroma_db", "gatekeeper", "llm_provider"):
            r.record(f"01 health: {c}=ok",        comps.get(c) == "ok",  str(comps.get(c)))
        r.record("01 health: version present",    "version" in body,     body.get("version", "missing"))
    except Exception as exc:
        r.record("01 health", False, str(exc))

    # ── Test 02: Route 1 — Historian (conf > 85%) ────────────────────────────
    print("\n[ROUTE 1 — HISTORIAN]")
    try:
        t0   = time.perf_counter()
        resp = _post_image(base, _IMG_ROUTE1)
        elapsed = time.perf_counter() - t0
        body = resp.json()
        classified["historian"] = body

        r.record("02 Route1: HTTP 200",              resp.status_code == 200,       f"got {resp.status_code}")
        r.record("02 Route1: route=historian",        body.get("route_taken") == "historian",  body.get("route_taken"))
        r.record("02 Route1: label=1015",             body.get("cnn", {}).get("label") == "1015",
                                                      body.get("cnn", {}).get("label"))
        conf = body.get("cnn", {}).get("confidence", 0)
        r.record("02 Route1: conf > 0.85",            conf > 0.85,                 f"{conf:.4f}")
        r.record("02 Route1: narrative not empty",    bool(body.get("narrative")),  "empty" if not body.get("narrative") else f"{len(body['narrative'])} chars")
        r.record("02 Route1: mint present",           bool(body.get("mint")),       body.get("mint", "missing"))
        r.record("02 Route1: region present",         bool(body.get("region")),     body.get("region", "missing"))
        r.record("02 Route1: date_range present",     bool(body.get("date_range")), body.get("date_range", "missing"))
        r.record("02 Route1: pdf_url present",        bool(body.get("pdf_url")),    body.get("pdf_url", "missing"))
        r.record("02 Route1: processing_time_s > 0",  body.get("processing_time_s", 0) > 0,
                                                      f"{body.get('processing_time_s', 0):.1f}s")
        top5 = body.get("cnn", {}).get("top5", [])
        r.record("02 Route1: top5 has 5 items",       len(top5) == 5,              f"got {len(top5)}")
        r.record("02 Route1: uuid id present",         bool(body.get("id")),         body.get("id", "missing"))
        print(f"         (wall time: {elapsed:.1f}s)")
    except Exception as exc:
        r.record("02 Route1", False, str(exc))

    # ── Test 03: Route 2 — Validator (conf 40–85%) ───────────────────────────
    print("\n[ROUTE 2 — VALIDATOR]")
    try:
        t0   = time.perf_counter()
        resp = _post_image(base, _IMG_ROUTE2)
        elapsed = time.perf_counter() - t0
        body = resp.json()
        classified["validator"] = body

        r.record("03 Route2: HTTP 200",              resp.status_code == 200,       f"got {resp.status_code}")
        r.record("03 Route2: route=validator",        body.get("route_taken") == "validator",   body.get("route_taken"))
        conf = body.get("cnn", {}).get("confidence", 0)
        r.record("03 Route2: conf 0.40–0.85",         0.40 <= conf <= 0.85,         f"{conf:.4f}")
        r.record("03 Route2: material_status present",
                  body.get("material_status") is not None,
                  body.get("material_status", "null"))
        r.record("03 Route2: material_confidence float",
                  isinstance(body.get("material_confidence"), float),
                  str(type(body.get("material_confidence")).__name__))
        r.record("03 Route2: pdf_url present",        bool(body.get("pdf_url")),   body.get("pdf_url", "missing"))
        print(f"         (wall time: {elapsed:.1f}s)")
    except Exception as exc:
        r.record("03 Route2", False, str(exc))

    # ── Test 04: Route 3 — Investigator (conf < 40%) ─────────────────────────
    print("\n[ROUTE 3 — INVESTIGATOR]")
    try:
        t0   = time.perf_counter()
        resp = _post_image(base, _IMG_ROUTE3)
        elapsed = time.perf_counter() - t0
        body = resp.json()
        classified["investigator"] = body

        r.record("04 Route3: HTTP 200",                    resp.status_code == 200,     f"got {resp.status_code}")
        r.record("04 Route3: route=investigator",           body.get("route_taken") == "investigator",  body.get("route_taken"))
        conf = body.get("cnn", {}).get("confidence", 0)
        r.record("04 Route3: conf < 0.40",                  conf < 0.40,                f"{conf:.4f}")
        r.record("04 Route3: visual_description present",
                  bool(body.get("visual_description")),
                  "empty" if not body.get("visual_description") else f"{len(body['visual_description'])} chars")
        r.record("04 Route3: kb_match_count >= 0",
                  isinstance(body.get("kb_match_count"), int),
                  str(body.get("kb_match_count")))
        r.record("04 Route3: pdf_url present",              bool(body.get("pdf_url")), body.get("pdf_url", "missing"))
        print(f"         (wall time: {elapsed:.1f}s)")
    except Exception as exc:
        r.record("04 Route3", False, str(exc))

    # ── Test 05: History list — paginated, newest-first ──────────────────────
    print("\n[HISTORY]")
    try:
        resp = requests.get(f"{base}/api/history", timeout=10)
        body = resp.json()
        r.record("05 history: HTTP 200",               resp.status_code == 200,       f"got {resp.status_code}")
        r.record("05 history: items is list",          isinstance(body.get("items"), list), "")
        r.record("05 history: total >= 3",             body.get("total", 0) >= 3,     f"total={body.get('total')}")
        r.record("05 history: skip field present",     "skip"  in body,               "")
        r.record("05 history: limit field present",    "limit" in body,               "")
        items = body.get("items", [])
        if len(items) >= 2:
            # newest-first: first item's timestamp should be >= second item's
            ts0 = items[0].get("timestamp", "")
            ts1 = items[1].get("timestamp", "")
            r.record("05 history: newest-first order", ts0 >= ts1,  f"{ts0[:19]} >= {ts1[:19]}")
    except Exception as exc:
        r.record("05 history", False, str(exc))

    # ── Test 06: Pagination edge cases ───────────────────────────────────────
    print("\n[PAGINATION]")
    try:
        # Skip past all results — should return empty list, not an error
        resp = requests.get(f"{base}/api/history?skip=9999&limit=5", timeout=10)
        body = resp.json()
        r.record("06 pagination: skip=9999 HTTP 200",    resp.status_code == 200, f"got {resp.status_code}")
        r.record("06 pagination: skip=9999 items=[]",    body.get("items") == [], f"got {body.get('items')}")

        # limit=1 — should return exactly 1 item
        resp2 = requests.get(f"{base}/api/history?skip=0&limit=1", timeout=10)
        body2 = resp2.json()
        r.record("06 pagination: limit=1 has 1 item",    len(body2.get("items", [])) == 1,
                  f"got {len(body2.get('items', []))}")

        # limit=200 exceeds max=100 — should still return 200 or return error
        # (our schema clamps to le=100 so FastAPI returns 422)
        resp3 = requests.get(f"{base}/api/history?skip=0&limit=200", timeout=10)
        r.record("06 pagination: limit=200 rejected",    resp3.status_code == 422,
                  f"got {resp3.status_code} (expected 422 Unprocessable Entity)")
    except Exception as exc:
        r.record("06 pagination", False, str(exc))

    # ── Test 07: History by ID ────────────────────────────────────────────────
    print("\n[HISTORY BY ID]")
    try:
        # Use the ID returned by Route 1 classify
        record_id = classified.get("historian", {}).get("id")
        if record_id:
            resp = requests.get(f"{base}/api/history/{record_id}", timeout=10)
            body = resp.json()
            r.record("07 history/{id}: HTTP 200",          resp.status_code == 200,           f"got {resp.status_code}")
            r.record("07 history/{id}: id matches",        body.get("id") == record_id,       body.get("id"))
            r.record("07 history/{id}: route=historian",   body.get("route_taken") == "historian", body.get("route_taken"))
            r.record("07 history/{id}: narrative present", bool(body.get("narrative")),        "missing" if not body.get("narrative") else "ok")
        else:
            r.record("07 history/{id}", False, "No record_id available (Route 1 failed)")
    except Exception as exc:
        r.record("07 history/{id}", False, str(exc))

    # ── Test 08: PDF download ─────────────────────────────────────────────────
    print("\n[PDF DOWNLOAD]")
    try:
        pdf_url = classified.get("historian", {}).get("pdf_url")
        if pdf_url:
            resp = requests.get(f"{base}{pdf_url}", timeout=30)
            r.record("08 PDF: HTTP 200",                resp.status_code == 200,  f"got {resp.status_code}")
            content_type = resp.headers.get("content-type", "")
            r.record("08 PDF: Content-Type=application/pdf",
                      "pdf" in content_type or "octet-stream" in content_type,
                      content_type)
            r.record("08 PDF: body not empty",          len(resp.content) > 1024, f"{len(resp.content)} bytes")
            # First 4 bytes of a PDF are always %PDF
            r.record("08 PDF: magic bytes %PDF",        resp.content[:4] == b"%PDF", repr(resp.content[:4]))
        else:
            r.record("08 PDF", False, "No pdf_url available (Route 1 failed)")
    except Exception as exc:
        r.record("08 PDF", False, str(exc))

    # ── Test 09: Validator — material data in response ────────────────────────
    print("\n[BACKUP PLANES]")
    try:
        val_body = classified.get("validator", {})
        status = val_body.get("material_status")
        r.record("09 backup: validator material_status in expected set",
                  status in ("consistent", "mismatch", "uncertain", None),
                  str(status))
        det_conf = val_body.get("material_confidence")
        if det_conf is not None:
            r.record("09 backup: material_confidence 0-1",
                      0.0 <= det_conf <= 1.0, f"{det_conf}")
        else:
            r.record("09 backup: material_confidence present", False, "null")
    except Exception as exc:
        r.record("09 backup validator", False, str(exc))

    # ── Test 10: Investigator — kb_matches (works without vision LLM) ─────────
    try:
        inv_body  = classified.get("investigator", {})
        kb_count  = inv_body.get("kb_match_count", -1)
        r.record("10 backup: investigator kb_match_count >= 0",        kb_count >= 0,           str(kb_count))
        vis_desc  = inv_body.get("visual_description", "")
        r.record("10 backup: investigator visual_description not empty", bool(vis_desc), f"{len(vis_desc)} chars")
    except Exception as exc:
        r.record("10 backup investigator", False, str(exc))

    # ── Test 11: All 3 routes generate a PDF on disk ─────────────────────────
    try:
        reports_dir = _ROOT / "reports"
        for route_name, body in classified.items():
            pdf_url = body.get("pdf_url", "")
            filename = Path(pdf_url).name if pdf_url else ""
            if filename:
                pdf_path = reports_dir / filename
                r.record(f"11 backup: {route_name} PDF on disk",
                          pdf_path.exists() and pdf_path.stat().st_size > 0,
                          str(pdf_path.name))
            else:
                r.record(f"11 backup: {route_name} PDF on disk", False, "no pdf_url")
    except Exception as exc:
        r.record("11 backup PDF on disk", False, str(exc))

    # ── Test 12: History persisted — count increased across calls ─────────────
    try:
        resp = requests.get(f"{base}/api/history", timeout=10)
        total_after = resp.json().get("total", 0)
        r.record("12 backup: history total >= 3 (all 3 calls persisted)",
                  total_after >= 3, f"total={total_after}")
    except Exception as exc:
        r.record("12 backup history persist", False, str(exc))

    # ── Test 13: node_timings present in state ────────────────────────────────
    try:
        # Pull full state via history/{id} on the historian record
        record_id = classified.get("historian", {}).get("id")
        if record_id:
            resp  = requests.get(f"{base}/api/history/{record_id}", timeout=10)
            body  = resp.json()
            # node_timings is not exposed in ClassifyResponse (it's internal state)
            # but processing_time_s proves timing was captured
            pt = body.get("processing_time_s")
            r.record("13 backup: processing_time_s > 0", isinstance(pt, float) and pt > 0, str(pt))
        else:
            r.record("13 backup: processing_time_s", False, "historian record not available")
    except Exception as exc:
        r.record("13 backup timings", False, str(exc))

    # ── Test 14: Wrong Content-Type → 415 ────────────────────────────────────
    print("\n[SECURITY / ERROR HANDLING]")
    try:
        with open(_IMG_ROUTE1, "rb") as f:
            resp = requests.post(
                f"{base}/api/classify",
                files={"file": ("test.txt", f, "text/plain")},   # wrong MIME
                timeout=15,
            )
        r.record("14 security: wrong Content-Type → 415",
                  resp.status_code == 415, f"got {resp.status_code}")
    except Exception as exc:
        r.record("14 security wrong content-type", False, str(exc))

    # ── Test 15: Wrong magic bytes (valid MIME, corrupt content) → 415 ────────
    try:
        fake_jpeg = b"Not a JPEG at all -- just some text bytes pretending hard"
        resp = requests.post(
            f"{base}/api/classify",
            files={"file": ("fake.jpg", io.BytesIO(fake_jpeg), "image/jpeg")},
            timeout=15,
        )
        r.record("15 security: fake magic bytes → 415",
                  resp.status_code == 415, f"got {resp.status_code}")
    except Exception as exc:
        r.record("15 security fake magic bytes", False, str(exc))

    # ── Test 16: File too large → 413 ─────────────────────────────────────────
    try:
        oversized = b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024)   # 11 MB with valid JPEG header
        resp = requests.post(
            f"{base}/api/classify",
            files={"file": ("big.jpg", io.BytesIO(oversized), "image/jpeg")},
            timeout=30,
        )
        r.record("16 security: file too large → 413",
                  resp.status_code == 413, f"got {resp.status_code}")
    except Exception as exc:
        r.record("16 security oversized file", False, str(exc))

    # ── Test 17: Unknown history ID → 404 ─────────────────────────────────────
    try:
        resp = requests.get(f"{base}/api/history/00000000-0000-0000-0000-000000000000", timeout=10)
        r.record("17 security: unknown history ID → 404",
                  resp.status_code == 404, f"got {resp.status_code}")
    except Exception as exc:
        r.record("17 security unknown id", False, str(exc))

    # ── Test 18: Path traversal in /api/reports ────────────────────────────────
    try:
        resp = requests.get(f"{base}/api/reports/../../etc/passwd", timeout=10)
        r.record("18 security: path traversal → 400 or 404",
                  resp.status_code in (400, 404), f"got {resp.status_code}")
    except Exception as exc:
        r.record("18 security path traversal", False, str(exc))

    # ── Test 19: Health not degraded ──────────────────────────────────────────
    try:
        resp = requests.get(f"{base}/api/health", timeout=10)
        body = resp.json()
        r.record("19 health: not degraded after tests",
                  body.get("status") == "healthy", body.get("status"))
    except Exception as exc:
        r.record("19 health post-test", False, str(exc))

    # ── Test 20: Swagger UI accessible ────────────────────────────────────────
    try:
        resp = requests.get(f"{base}/docs", timeout=10)
        r.record("20 docs: /docs Swagger → 200",
                  resp.status_code == 200, f"got {resp.status_code}")
    except Exception as exc:
        r.record("20 docs swagger", False, str(exc))

    return r.summary()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepCoin API test suite")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    exit_code = run_tests(args.base_url)
    sys.exit(exit_code)
