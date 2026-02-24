"""
scripts/build_knowledge_base.py
================================
Layer 2 — Knowledge Base Builder

Scrapes https://www.corpus-nummorum.eu/types/{id} for all 438 trained classes,
extracts structured metadata from the HTML, saves to:
  data/metadata/cn_types_metadata.json

Then calls knowledge_base.py to build the ChromaDB vector index.

Usage:
    python scripts/build_knowledge_base.py              # scrape + build index
    python scripts/build_knowledge_base.py --scrape-only  # only scrape, save JSON
    python scripts/build_knowledge_base.py --index-only   # only (re)build index from JSON
    python scripts/build_knowledge_base.py --resume       # skip already scraped types

Engineering rules:
  - 1 second delay between requests (polite scraping)
  - SSL verification disabled (lab environment workaround)
  - Resume support: checks existing JSON before re-fetching
  - Progress bar so you can see it working
"""

import sys
import ssl
import time
import json
import argparse
import urllib.request
import html as html_module
import re
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
MAPPING_PATH = ROOT / "models" / "class_mapping.pth"
METADATA_OUT = ROOT / "data" / "metadata" / "cn_types_metadata.json"
CHROMA_DIR   = ROOT / "data" / "metadata" / "chroma_db"

# ── SSL context (bypass cert check for lab environments) ───────────────────────
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE

# Fields we skip in the final output (noise, not useful for RAG)
SKIP_FIELDS = {"editions", "cite_this_record:", "license:", "contact", "edition"}


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD CLASS IDs
# ══════════════════════════════════════════════════════════════════════════════

def load_class_ids() -> list[int]:
    """Return the 438 CN type IDs from class_mapping.pth."""
    import torch
    mapping = torch.load(MAPPING_PATH, map_location="cpu", weights_only=False)
    class_to_idx: dict = mapping["class_to_idx"]      # {"1015": 0, ...}
    return sorted(int(k) for k in class_to_idx.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  2. SCRAPER — one page → dict
# ══════════════════════════════════════════════════════════════════════════════

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s).strip()

def _clean(s: str) -> str:
    s = html_module.unescape(_strip_tags(s))
    # Remove leftover emoji / icon characters from CN navigation buttons
    s = re.sub(r"[🔍❐✤]", "", s)
    return " ".join(s.split()).strip()


def scrape_cn_type(type_id: int) -> dict:
    """
    Fetch https://www.corpus-nummorum.eu/types/{type_id} and extract metadata.

    Returns a dict with keys:
        type_id, mint, region, date, period, obverse_legend, obverse_design,
        reverse_legend, reverse_design, persons, denomination, material,
        diameter_mm, weight_g, source_url, raw_fields (everything else)
    """
    url = f"https://www.corpus-nummorum.eu/types/{type_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "DeepCoin-Scraper/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {"type_id": type_id, "error": f"HTTP {e.code}", "source_url": url}
    except Exception as e:
        return {"type_id": type_id, "error": str(e), "source_url": url}

    # Extract <dl>…</dl> blocks — all structured data lives here
    dl_blocks = re.findall(r"<dl[^>]*>(.*?)</dl>", raw_html, re.DOTALL)

    raw_fields: dict[str, str] = {}
    for block in dl_blocks:
        dt_list = re.findall(r"<dt[^>]*>(.*?)</dt>", block, re.DOTALL)
        dd_list = re.findall(r"<dd[^>]*>(.*?)</dd>", block, re.DOTALL)
        if not dt_list:
            continue
        label  = _clean(dt_list[0]).lower().replace(" ", "_")
        values = [_clean(dd) for dd in dd_list if _clean(dd)]
        if label and values and label not in SKIP_FIELDS:
            raw_fields[label] = " | ".join(values[:4])

    # ── map raw_fields → clean structured record ──────────────────────────────
    def get(*keys: str) -> str:
        for k in keys:
            if k in raw_fields:
                return raw_fields.pop(k)
        return ""

    # Mint field often looks like "Maroneia  Region: Thrace  Typology"
    mint_raw   = get("mint")
    mint_parts = re.split(r"Region:", mint_raw, maxsplit=1)
    mint       = mint_parts[0].strip()
    region_raw = mint_parts[1].strip() if len(mint_parts) > 1 else ""
    region     = re.sub(r"\s+Typology.*", "", region_raw).strip()

    # Date field often looks like "c. 365-330 BC Classical Period"
    date_raw   = get("date")
    period_match = re.search(
        r"(Classical|Hellenistic|Roman Imperial|Byzantine|Medieval|"
        r"Republican|Imperial|Modern|Other)\s*Period", date_raw, re.IGNORECASE
    )
    period = period_match.group(0).strip() if period_match else ""
    date   = re.sub(r"\s+" + re.escape(period), "", date_raw).strip() if period else date_raw.strip()

    # Obverse / reverse — each has a "legend" and a "design" in separate dl blocks
    # The HTML order is: Obverse header → legend dd → Design header → design dd
    # We grab them in order from raw_fields
    obverse_legend = get("obverse")
    obverse_design = ""
    reverse_legend = get("reverse")
    reverse_design = ""

    # "design" key might appear twice (obverse+reverse); we consumed obverse above
    # so whatever remains in "design" is the reverse design
    design_val = get("design")
    # If obverse has no separate design, treat the single design as obverse
    if not obverse_design:
        obverse_design = design_val

    persons      = get("persons")
    denomination = ""
    metrology_raw = get("metrology")
    if metrology_raw:
        denom_match = re.search(r"Denomination\s+(.+?)(?:\s+🔍|$)", metrology_raw)
        denomination = denom_match.group(1).strip() if denom_match else metrology_raw

    material  = get("material").replace("🔍", "").replace("❐", "").strip()
    diameter  = get("diameter").replace("Max:", "").replace("Min:", "").strip()
    weight    = get("average_weight")

    result = {
        "type_id":        type_id,
        "source_url":     url,
        "mint":           mint,
        "region":         region,
        "date":           date,
        "period":         period,
        "obverse_legend": obverse_legend,
        "obverse_design": obverse_design,
        "reverse_legend": reverse_legend,
        "reverse_design": reverse_design,
        "persons":        persons,
        "denomination":   denomination,
        "material":       material,
        "diameter_mm":    diameter,
        "weight_g":       weight,
    }
    # Keep any remaining fields that might be useful
    leftover = {k: v for k, v in raw_fields.items()
                if k not in SKIP_FIELDS and v}
    if leftover:
        result["extra"] = leftover

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. SCRAPE ALL 438 TYPES
# ══════════════════════════════════════════════════════════════════════════════

def scrape_all(class_ids: list[int], resume: bool = False) -> list[dict]:
    """
    Scrape all class IDs with a 1-second delay between requests.
    If resume=True, skip any type_id already in the existing metadata file.
    """
    # Load existing data if resuming
    existing: dict[int, dict] = {}
    if resume and METADATA_OUT.exists():
        with open(METADATA_OUT, encoding="utf-8") as f:
            for record in json.load(f):
                existing[record["type_id"]] = record
        print(f"  Resume mode: {len(existing)} already scraped, "
              f"{len(class_ids) - len(existing)} remaining")

    results: list[dict] = list(existing.values())
    todo = [cid for cid in class_ids if cid not in existing]

    for i, type_id in enumerate(todo, start=1):
        pct = (i / len(todo)) * 100
        bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {i}/{len(todo)}  type {type_id:<6}", end="", flush=True)

        record = scrape_cn_type(type_id)
        results.append(record)

        # Save incrementally every 50 types (crash protection)
        if i % 50 == 0:
            _save_metadata(results)

        if i < len(todo):          # no delay after last request
            time.sleep(1.0)

    print()  # newline after progress bar
    return results


def _save_metadata(records: list[dict]) -> None:
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  4. BUILD CHROMADB INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_index() -> None:
    """
    Load cn_types_metadata.json → build ChromaDB collection.
    Called by knowledge_base.py but can also be run standalone here.
    """
    # Import here so the scrape step works even if chromadb isn't installed yet
    from src.core.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(chroma_dir=str(CHROMA_DIR))
    kb.build_from_metadata(str(METADATA_OUT))
    print(f"  ChromaDB index built → {CHROMA_DIR}")
    print(f"  Collection size: {kb.count()} documents")


# ══════════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DeepCoin knowledge base from Corpus Nummorum website"
    )
    parser.add_argument("--scrape-only", action="store_true",
                        help="Only scrape and save JSON, skip ChromaDB build")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip scraping, only (re)build ChromaDB from existing JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Skip types already in existing metadata JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  DeepCoin — Knowledge Base Builder  (Layer 2)")
    print("=" * 60)

    if not args.index_only:
        print(f"\n[1/2] Loading class IDs from {MAPPING_PATH.name} ...")
        class_ids = load_class_ids()
        print(f"  Found {len(class_ids)} classes to scrape")

        print(f"\n[2/2] Scraping corpus-nummorum.eu (1 req/sec) ...")
        records = scrape_all(class_ids, resume=args.resume)
        _save_metadata(records)

        # Report
        ok      = [r for r in records if "error" not in r]
        errors  = [r for r in records if "error" in r]
        print(f"\n  Done!  {len(ok)} OK  |  {len(errors)} errors")
        if errors:
            print("  Errors:")
            for e in errors[:10]:
                print(f"    type {e['type_id']}: {e['error']}")
        print(f"  Saved → {METADATA_OUT}")

    if not args.scrape_only:
        print(f"\n[ChromaDB] Building vector index ...")
        build_index()

    print("\nLayer 2 complete.")


if __name__ == "__main__":
    main()
