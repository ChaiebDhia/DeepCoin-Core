"""
End-to-end test of the full Layer 3 pipeline.
Tests all three routing paths:

  Route 1 — historian    (conf > 85%)  : type 1015, conf ~ 91%
  Route 2 — validator    (40-85%)      : type 21027 image, conf ~ 43%
  Route 3 — investigator (conf < 40%)  : type 544 image, conf ~ 21%

Usage: python scripts/test_pipeline.py
"""
import sys
import warnings
import io
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
# Fix Windows console encoding for Greek/Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.agents.gatekeeper import Gatekeeper

# ── test cases — one per routing path ───────────────────────────────────────────
TESTS = [
    {
        "name":           "Route 1 — HISTORIAN (high confidence)",
        "image":          "data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg",
        "expected_route": "historian",
        "min_conf":        0.85,
    },
    {
        "name":           "Route 2 — VALIDATOR (medium confidence)",
        "image":          "data/processed/21027/CN_type_21027_cn_coin_6169_p.jpg",
        "expected_route": "validator",
        "min_conf":        0.40,
        "max_conf":        0.85,
    },
    {
        "name":           "Route 3 — INVESTIGATOR (low confidence)",
        "image":          "data/processed/544/CN_type_544_cn_coin_2324_p.jpg",
        "expected_route": "investigator",
        "max_conf":        0.40,
    },
]

# ── load gatekeeper once for all tests ──────────────────────────────────────────
print("Loading Gatekeeper (CNN + KB + all agents)...")
gk = Gatekeeper(save_pdf=True)
print("Gatekeeper ready.\n")

# ── run all three routes ─────────────────────────────────────────────────────────
passed = 0
failed = 0

for i, tc in enumerate(TESTS, 1):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  TEST {i}/3 — {tc['name']}")
    print(sep)
    print(f"  Image: {tc['image']}")

    result = gk.analyze(tc["image"], tta=False)
    state  = result["state"]

    route  = state.get("route_taken", "?")
    conf   = state["cnn_prediction"]["confidence"]
    label  = state["cnn_prediction"]["label"]
    pdf    = result.get("pdf_path")

    print(f"\n  CNN label    : {label}")
    print(f"  CNN conf     : {conf:.1%}")
    print(f"  Route taken  : {route}")
    print(f"  PDF saved    : {pdf}")

    # ── timing summary ──────────────────────────────────────────────────────
    timings = state.get("node_timings", {})
    total   = sum(timings.values())
    print(f"  Total time   : {total:.1f}s  "
          + "  ".join(f"{k}={v:.1f}s" for k, v in timings.items()))

    # ── route-specific checks ───────────────────────────────────────────────
    ok = True
    # route matches expected
    if route != tc["expected_route"]:
        print(f"  [WARN] expected route={tc['expected_route']} got={route}")
        ok = False
    # report and PDF present
    if not result.get("report"):
        print("  [FAIL] report is empty")
        ok = False
    if pdf is None:
        print("  [WARN] PDF was not saved")

    # route 2: validator checks
    if tc["expected_route"] == "validator":
        vr = state.get("validator_result", {})
        status = vr.get("status", "?")
        dconf  = vr.get("detection_confidence", 0.0)
        unc    = vr.get("uncertainty", "?")
        print(f"  Material     : status={status}  det_conf={dconf:.2f}  uncertainty={unc}")
        if "status" not in vr:
            print("  [FAIL] validator_result missing status")
            ok = False

    # route 3: investigator checks
    if tc["expected_route"] == "investigator":
        ir = state.get("investigator_result", {})
        n_matches = len(ir.get("kb_matches", []))
        llm_used  = ir.get("llm_used", False)
        print(f"  KB matches   : {n_matches}  llm_used={llm_used}")
        desc = ir.get("visual_description", "")
        if not desc:
            print("  [FAIL] visual_description is empty")
            ok = False
        else:
            print(f"  VLM desc     : {desc[:120]}...")

    if ok:
        print(f"\n  [PASS] {tc['name']}")
        passed += 1
    else:
        print(f"\n  [FAIL] {tc['name']}")
        failed += 1

# ── final summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"  RESULTS: {passed}/{len(TESTS)} passed"
      + (f"  ({failed} FAILED)" if failed else "  — all routes OK"))
print("=" * 62)
sys.exit(0 if failed == 0 else 1)

