"""
src/core/knowledge_base.py
===========================
Layer 2 — Knowledge Base

Wraps a ChromaDB collection of 438 CN coin type records.
Each document = one coin type, with:
  - text:     natural-language description for semantic search
  - metadata: structured fields for keyword/filter search

Public API used by the Historian agent (Layer 3):
    kb = KnowledgeBase()
    results = kb.search("silver drachm Thrace 4th century BC", n=5)
    results = kb.search_by_id(1015)
    count   = kb.count()

Engineering rules:
  - Load once per process (singleton pattern via module-level instance)
  - ChromaDB is local on disk — no network required at query time
  - Embedding model: all-MiniLM-L6-v2 (fast, 384-dim, 22MB, runs on CPU)
  - Build once offline; query thousands of times at inference
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── default paths ──────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent.parent
_CHROMA_DIR   = _ROOT / "data" / "metadata" / "chroma_db"
_METADATA_JSON = _ROOT / "data" / "metadata" / "cn_types_metadata.json"

COLLECTION_NAME = "cn_coin_types"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — build the text document for a single coin type
# ══════════════════════════════════════════════════════════════════════════════

def build_document_text(record: dict) -> str:
    """
    Convert a scraped metadata dict into a single natural-language string
    that will be embedded and stored in ChromaDB.

    The text is designed so that semantic search phrases like:
      "silver drachm Thrace 4th century"
      "horse prancing obverse Maroneia"
      "bronze coin Roman Imperial Dionysus"
    …will all surface the right coin.

    Example output for type 1015:
      "CN type 1015. Mint: Maroneia (Region: Thrace).
       Date: c. 365-330 BC (Classical Period).
       Obverse: Legend MAR. Prancing horse, right.
       Reverse: Legend EPI ZINONOS. Bunch of grapes on vine branch.
       Material: silver. Denomination: drachm.
       Persons: Magistrate Zenon (Maroneia).
       Diameter: Max: 17 mm / Min: 13 mm. Weight: 2.44 g."
    """
    parts: list[str] = [f"CN type {record['type_id']}."]

    if record.get("mint"):
        loc = record["mint"]
        if record.get("region"):
            loc += f" (Region: {record['region']})"
        parts.append(f"Mint: {loc}.")

    if record.get("date"):
        d = record["date"]
        if record.get("period"):
            d += f" ({record['period']})"
        parts.append(f"Date: {d}.")

    if record.get("obverse_legend") or record.get("obverse_design"):
        ob = " ".join(filter(None, [record.get("obverse_legend"), record.get("obverse_design")]))
        # Strip "go to the NLP result…" trailing noise
        ob = re.sub(r"go to the NLP result.*", "", ob).strip()
        if ob:
            parts.append(f"Obverse: {ob}.")

    if record.get("reverse_legend") or record.get("reverse_design"):
        rev = " ".join(filter(None, [record.get("reverse_legend"), record.get("reverse_design")]))
        rev = re.sub(r"go to the NLP result.*", "", rev).strip()
        if rev:
            parts.append(f"Reverse: {rev}.")

    if record.get("material"):
        parts.append(f"Material: {record['material']}.")

    if record.get("denomination"):
        parts.append(f"Denomination: {record['denomination']}.")

    if record.get("persons"):
        parts.append(f"Persons: {record['persons']}.")

    if record.get("diameter_mm"):
        parts.append(f"Diameter: {record['diameter_mm']}.")

    if record.get("weight_g"):
        parts.append(f"Weight: {record['weight_g']}.")

    return " ".join(parts)


def build_metadata_dict(record: dict) -> dict[str, Any]:
    """
    Build the ChromaDB metadata dict for a record.
    ChromaDB metadata values must be str | int | float | bool.
    """
    def s(key: str) -> str:
        return str(record.get(key, "") or "")

    return {
        "type_id":      int(record["type_id"]),
        "mint":         s("mint"),
        "region":       s("region"),
        "date":         s("date"),
        "period":       s("period"),
        "material":     s("material"),
        "denomination": s("denomination"),
        "source_url":   s("source_url"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    ChromaDB-backed knowledge base for Corpus Nummorum coin types.

    Usage:
        kb = KnowledgeBase()          # loads existing index from disk
        results = kb.search("silver drachm Thrace", n=5)

    Build (offline, run once):
        kb = KnowledgeBase()
        kb.build_from_metadata("data/metadata/cn_types_metadata.json")
    """

    def __init__(
        self,
        chroma_dir: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        Parameters
        ----------
        chroma_dir      Path to the ChromaDB directory on disk.
                        Defaults to data/metadata/chroma_db/
        collection_name ChromaDB collection name (default: "cn_coin_types")
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._dir        = Path(chroma_dir) if chroma_dir else _CHROMA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._client     = chromadb.PersistentClient(path=str(self._dir))
        self._embed_fn   = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"   # 22 MB, fast CPU inference
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    # ── build ─────────────────────────────────────────────────────────────────

    def build_from_metadata(
        self,
        metadata_path: str | None = None,
        batch_size: int = 50,
        reset: bool = False,
    ) -> None:
        """
        Read cn_types_metadata.json and upsert all documents into ChromaDB.

        Parameters
        ----------
        metadata_path   Path to the JSON file (default: data/metadata/cn_types_metadata.json)
        batch_size      How many docs to upsert per ChromaDB call
        reset           If True, delete + recreate the collection first
        """
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        path = Path(metadata_path) if metadata_path else _METADATA_JSON
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {path}\n"
                "Run: python scripts/build_knowledge_base.py --scrape-only"
            )

        with open(path, encoding="utf-8") as f:
            records: list[dict] = json.load(f)

        # Filter out error records
        good = [r for r in records if "error" not in r]
        bad  = [r for r in records if "error" in r]
        print(f"  Loading {len(good)} records ({len(bad)} skipped — scrape errors)")

        if reset:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

        # Upsert in batches
        ids       = [str(r["type_id"]) for r in good]
        documents = [build_document_text(r) for r in good]
        metadatas = [build_metadata_dict(r) for r in good]

        for start in range(0, len(good), batch_size):
            end = min(start + batch_size, len(good))
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            pct = (end / len(good)) * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {end}/{len(good)} embedded", end="", flush=True)

        print()

    # ── query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search: embed query → find n closest coin type documents.

        Parameters
        ----------
        query   Natural language search string, e.g. "silver drachm Thrace"
        n       Number of results to return
        where   Optional ChromaDB metadata filter, e.g. {"material": "silver"}

        Returns
        -------
        list of dicts, each with:
            type_id, label, score, mint, region, date, period,
            material, denomination, source_url, document (full text)
        """
        kwargs: dict[str, Any] = {
            "query_texts":  [query],
            "n_results":    min(n, self._collection.count()),
            "include":      ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            score = float(1.0 - dist)      # cosine distance → similarity
            results.append({
                "type_id":      meta["type_id"],
                "label":        f"CN_{meta['type_id']}",
                "score":        round(score, 4),
                "mint":         meta.get("mint", ""),
                "region":       meta.get("region", ""),
                "date":         meta.get("date", ""),
                "period":       meta.get("period", ""),
                "material":     meta.get("material", ""),
                "denomination": meta.get("denomination", ""),
                "source_url":   meta.get("source_url", ""),
                "document":     doc,
            })
        return results

    def search_by_id(self, type_id: int) -> dict | None:
        """
        Fetch the stored document for a specific CN type ID.
        Returns None if the type is not in the collection.
        """
        raw = self._collection.get(
            ids=[str(type_id)],
            include=["documents", "metadatas"],
        )
        if not raw["ids"]:
            return None
        meta = raw["metadatas"][0]
        doc  = raw["documents"][0]
        return {
            "type_id":      meta["type_id"],
            "label":        f"CN_{meta['type_id']}",
            "mint":         meta.get("mint", ""),
            "region":       meta.get("region", ""),
            "date":         meta.get("date", ""),
            "period":       meta.get("period", ""),
            "material":     meta.get("material", ""),
            "denomination": meta.get("denomination", ""),
            "source_url":   meta.get("source_url", ""),
            "document":     doc,
        }

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def is_built(self) -> bool:
        """Return True if the collection has at least 1 document."""
        return self._collection.count() > 0


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON (lazy-loaded)
# ══════════════════════════════════════════════════════════════════════════════

_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """
    Return the shared KnowledgeBase instance (created on first call).
    Safe to call multiple times — only one ChromaDB connection is opened.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
