"""
scripts/rebuild_chroma.py
==========================
STEP 2 of the enterprise RAG upgrade.

WHAT this script does:
    1. Loads cn_types_metadata_full.json (9,541 scraped coin records)
    2. Splits every record into 5 semantic text chunks:
           identity | obverse | reverse | material | context
    3. Embeds all chunks with all-MiniLM-L6-v2 (384-dim, CPU, 22 MB)
    4. Upserts everything into ChromaDB at data/metadata/chroma_db_rag/

WHY 5 chunks instead of 1 blob (the old approach):
    The old KB stored each coin as ONE 200-word paragraph.
    Problem: "Find silver coins" would match a sentence buried mid-paragraph,
    making the vector imprecise — the whole blob gets a mediocre score.

    With 5 focused chunks, each vector encodes ONE concept:
      - identity chunk  → type_id, denomination, authority, date
      - obverse chunk   → portrait, symbols, legend on front
      - reverse chunk   → symbols, legend on back
      - material chunk  → metal, weight, diameter, mint city
      - context chunk   → persons, references, notes

    "silver coin" now hits material chunks with a very high cosine score.
    "eagle reverse" now hits reverse chunks precisely.
    The Historian gets back exactly the facts it needs, not a blob.

WHY all-MiniLM-L6-v2:
    22 MB, CPU inference, 384-dim vectors, cosine similarity.
    Fast enough for 47,705 chunks (~15-20 min one-time build).
    Already installed as part of ChromaDB's default embedding function.

Output:
    data/metadata/chroma_db_rag/   ← persisted to disk, never rebuilt again
                                     unless you re-scrape the metadata JSON

Run time:   ~15–20 minutes on CPU (one-time operation)
Disk space: ~180 MB

Usage:
    & C:\\Users\\Administrator\\deepcoin\\venv\\Scripts\\python.exe scripts/rebuild_chroma.py
"""

import sys
import time
from pathlib import Path

# ── make sure project root is on the path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.rag_engine import get_rag_engine   # noqa: E402


def main() -> None:
    print("=" * 62)
    print("  DeepCoin — STEP 2: Rebuild ChromaDB (5-chunk RAG index)")
    print("=" * 62)

    # ── Load the RAG engine (parses JSON + builds BM25 in memory) ─────────────
    print("\n[1/3] Loading RAG engine and metadata JSON …")
    t0 = time.perf_counter()
    engine = get_rag_engine()
    t1 = time.perf_counter()

    n_records = engine.corpus_size()
    n_chunks  = n_records * 5      # always 5 chunks per record

    print(f"      Records loaded  : {n_records:,}")
    print(f"      Total chunks    : {n_chunks:,}  (5 × {n_records:,})")
    print(f"      BM25 build time : {t1 - t0:.1f}s")

    # ── Check if ChromaDB is already populated ─────────────────────────────────
    if engine.is_chroma_built():
        existing = engine._collection.count()
        print(f"\n[!] ChromaDB already contains {existing:,} vectors.")
        answer = input("    Wipe and rebuild from scratch? [y/N]: ").strip().lower()
        if answer != "y":
            print("    Aborted — existing index kept.")
            sys.exit(0)
        # Delete the collection and recreate it
        engine._chroma_client.delete_collection(engine._collection.name)
        engine._collection = engine._chroma_client.get_or_create_collection(
            name=engine._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("    Old collection wiped.")

    # ── Embed and upsert ───────────────────────────────────────────────────────
    print(f"\n[2/3] Embedding {n_chunks:,} chunks with all-MiniLM-L6-v2 …")
    print("      This takes ~15-20 minutes on CPU.  Do not interrupt.\n")

    t2 = time.perf_counter()
    engine.populate_chroma(batch_size=200)
    t3 = time.perf_counter()

    elapsed   = t3 - t2
    per_chunk = elapsed / n_chunks if n_chunks else 0

    # ── Verify ─────────────────────────────────────────────────────────────────
    print("\n[3/3] Verifying …")
    final_count = engine._collection.count()

    print(f"\n{'=' * 62}")
    print(f"  ChromaDB rebuild complete")
    print(f"  Vectors stored : {final_count:,}")
    print(f"  Time elapsed   : {elapsed / 60:.1f} min")
    print(f"  Per chunk      : {per_chunk * 1000:.1f} ms")
    print(f"  DB location    : data/metadata/chroma_db_rag/")
    print(f"{'=' * 62}\n")

    if final_count < n_chunks * 0.95:
        print(f"  [WARNING] Expected ~{n_chunks:,} vectors but only got {final_count:,}.")
        print("  You may need to re-run this script.")
        sys.exit(1)

    print("  STEP 2 complete. Ready for STEP 3 (historian.py upgrade).")


if __name__ == "__main__":
    main()
