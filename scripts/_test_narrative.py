"""Quick smoke test: classify Route 1 and verify the narrative is clean."""
import requests
import time

IMG = "data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg"
print(f"Sending: {IMG}")
t0 = time.time()
with open(IMG, "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/classify",
        files={"file": ("coin.jpg", f, "image/jpeg")},
        timeout=180,
    )
elapsed = time.time() - t0
print(f"Status : {r.status_code}  Time: {elapsed:.1f}s")

if r.status_code != 200:
    print("ERROR:", r.text[:600])
    raise SystemExit(1)

data      = r.json()
route     = data.get("route_taken", "")
label     = data.get("cnn", {}).get("label", "")
conf      = data.get("cnn", {}).get("confidence", 0)
narrative = data.get("narrative", "")   # top-level in ClassifyResponse

print(f"Route  : {route}")
print(f"Type   : {label}  conf={conf:.1%}")
print()
print("─" * 60)
print("NARRATIVE:")
print(narrative)
print("─" * 60)

# ── Forbidden-token checks ────────────────────────────────────────────────────
issues = []
if "[CONTEXT" in narrative.upper():
    issues.append("  FAIL  [CONTEXT N] markers still present")
if "**" in narrative:
    issues.append("  FAIL  ** bold markers still present")
if "##" in narrative:
    issues.append("  FAIL  ## heading markers still present")
if "`" in narrative:
    issues.append("  FAIL  backtick markers still present")
# ? coming from latin-1 replacement is chr(63) — check against obvious artefacts
suspicious = [narrative[i-2:i+2] for i, c in enumerate(narrative) if c == "?"]
if suspicious:
    issues.append(f"  WARN  '?' chars found near: {suspicious[:3]}")

print()
if issues:
    for i in issues:
        print(i)
    raise SystemExit(1)
else:
    print("  PASS  narrative is clean: no [CONTEXT N], no Markdown, no bad chars")
