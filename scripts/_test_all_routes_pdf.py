"""
Generate one PDF for each of the 3 routing paths and open them all.

Route 1  conf > 85%  → Historian   : type 1015, ~91%
Route 2  40-85%      → Validator   : type 21027, ~43%
Route 3  conf < 40%  → Investigator: type 544,  ~21%
"""
import os
import subprocess
import sys
import time

import requests

CASES = [
    ("Route 1 — Historian   (conf > 85%)", "data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg"),
    ("Route 2 — Validator   (40-85%)",     "data/processed/21027/CN_type_21027_cn_coin_6169_p.jpg"),
    ("Route 3 — Investigator(conf < 40%)", "data/processed/544/CN_type_544_cn_coin_2324_p.jpg"),
]

BASE = "http://localhost:8000"
pdfs = []

for label, img_path in CASES:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Image: {img_path}")
    t0 = time.time()
    with open(img_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/classify",
            files={"file": (os.path.basename(img_path), f, "image/jpeg")},
            timeout=300,
        )
    elapsed = time.time() - t0

    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text[:300]}")
        sys.exit(1)

    data   = r.json()
    route  = data.get("route_taken", "?")
    label_ = data["cnn"]["label"]
    conf   = data["cnn"]["confidence"]
    narr   = data.get("narrative", "")
    pdf_url = data.get("pdf_url", "")
    fname  = pdf_url.split("/")[-1]
    local  = os.path.join("reports", fname)

    print(f"  Route   : {route}")
    print(f"  CN Type : {label_}   conf={conf:.1%}   time={elapsed:.1f}s")
    print(f"  PDF     : {local}")

    # Show first 300 chars of narrative
    if narr:
        preview = narr[:300].replace("\n", " ")
        print(f"  Narrative preview: {preview}...")
    else:
        print("  Narrative: (none — fallback route)")

    # Sanity checks
    assert route in ("historian", "validator", "investigator"), f"Unexpected route: {route}"
    assert os.path.isfile(local), f"PDF not on disk: {local}"
    assert os.path.getsize(local) > 2000, f"PDF suspiciously small: {os.path.getsize(local)} bytes"
    print("  CHECK OK")
    pdfs.append(local)

# Open all 3 PDFs
print(f"\n{'═'*60}")
print("Opening all 3 PDFs...")
for p in pdfs:
    subprocess.Popen(["start", "", p], shell=True)
    time.sleep(0.5)

print(f"\nRESULTS: {len(pdfs)}/3 PDFs generated and opened")
