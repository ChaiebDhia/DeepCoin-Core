"""
src/core/rag_engine.py
=======================
Layer 2 — RAG Engine  (Hybrid BM25 + Vector Search with RRF)

WHY THIS FILE EXISTS
--------------------
The original KnowledgeBase (knowledge_base.py) stored each of the 438 CNN
training types as ONE text "blob" and searched with vector similarity only.

Two problems:
  1. One blob = diluted embeddings.  A query "silver coin" has to compete
     against obverse legends, persons, and dates all mixed in the same vector.

  2. Vector-only search misses exact legend fragments ("ΣΕΒΑΣΤΟΣ") that BM25
     would catch instantly.

This module replaces that with a two-stage retrieval architecture:

    Query
      │
      ├─► BM25 (rank-bm25)     — keyword match, exact legend fragments
      │     ranks all chunks
      │
      └─► ChromaDB             — semantic/embedding match
            ranks same chunks
              │
              └─► RRF merge    — Reciprocal Rank Fusion
                    score(d) = Σ 1 / (60 + rank)
                    deduplicates by type_id → top-N coin records

CHUNKING STRATEGY (5 focused chunks per coin)
----------------------------------------------
  identity  : type_id, denomination, mint, region, date, period
  obverse   : obverse legend + obverse design (NLP noise stripped)
  reverse   : reverse legend + reverse design (NLP noise stripped)
  material  : material, weight, diameter, mint
  context   : persons, literature, citation notes

WHY 5 CHUNKS NOT 1 BLOB
------------------------
Each chunk embeds ONE aspect of the coin.  A query "silver coin" now lands
directly on the material chunk — not on a mixed blob that dilutes the match
with horse designs and magistrate names.  Precision improves significantly.

PUBLIC API
----------
    engine = get_rag_engine()                # singleton, lazy-loaded
    results = engine.search("silver drachm Thrace", n=5)
    results = engine.search("silver", n=3, chunk_filter="material")
    record  = engine.get_by_id(1015)
    blocks  = engine.get_context_blocks(1015)   # for LLM prompt injection
    count   = engine.corpus_size()

INTEGRATION NOTES
-----------------
- populate_chroma() must be called ONCE (from scripts/rebuild_chroma.py, STEP 2)
  to embed all chunks into ChromaDB.
- BM25 index is rebuilt from the JSON on every process start (< 2 s for 9,716).
- If ChromaDB is empty, search() falls back to BM25-only (no crash, no error).
- Module-level singleton ensures a single index load per process.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ── default paths ──────────────────────────────────────────────────────────────
_ROOT                   = Path(__file__).resolve().parent.parent.parent
_CHROMA_DIR             = _ROOT / "data" / "metadata" / "chroma_db_rag"
_METADATA_JSON          = _ROOT / "data" / "metadata" / "cn_types_metadata_full.json"
_METADATA_JSON_FALLBACK = _ROOT / "data" / "metadata" / "cn_types_metadata.json"

COLLECTION_NAME = "cn_rag_chunks"
ChunkType = Literal["identity", "obverse", "reverse", "material", "context"]

# Noise injected by the CN website into scraped design fields
_NLP_NOISE = re.compile(r"\s*go to the NLP result.*", re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNKING — split one coin record into 5 focused text chunks
# ══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """
    Strip website navigation noise from a scraped text field.

    WHY: The CN website appends "go to the NLP result of this description"
    to every design field.  This phrase is meaningless for retrieval and
    would pollute every obverse/reverse embedding with identical noise tokens.
    """
    return _NLP_NOISE.sub("", text or "").strip()


def chunk_record(record: dict) -> list[dict]:
    """
    Split one scraped coin record into 5 semantically focused chunks.

    WHAT: Produces 5 focused text strings — one per topic — instead of one
          long mixed blob.  Each string becomes a separate embedding vector
          in ChromaDB and a separate BM25 document.

    WHY 5 chunks not 1 blob:
        A query "silver coin" should match the material chunk strongly.
        A query "prancing horse" should match the obverse chunk strongly.
        Mixing all fields into one blob halves effective retrieval precision
        because unrelated tokens dilute the semantic distance calculation.

    Parameters
    ----------
    record : dict
        A single entry from cn_types_metadata_full.json.
        Must contain "type_id".  Other fields default to empty string if absent.

    Returns
    -------
    list[dict]  Five dicts, each with keys:
        chunk_id        : str  "<type_id>_<chunk_type>"  (unique ChromaDB ID)
        type_id         : int
        chunk_type      : str  one of identity|obverse|reverse|material|context
        text            : str  the focused natural-language text to embed
        in_training_set : bool  True if this type was in the CNN training set
    """
    tid = int(record["type_id"])
    its = bool(record.get("in_training_set", False))

    # ── 1. identity chunk ──────────────────────────────────────────────────────
    identity_parts = [f"CN type {tid}"]
    if record.get("denomination"):
        identity_parts.append(f"denomination: {record['denomination']}")
    if record.get("mint"):
        identity_parts.append(f"mint: {record['mint']}")
    if record.get("region"):
        identity_parts.append(f"region: {record['region']}")
    if record.get("date"):
        dt = record["date"]
        if record.get("period"):
            dt += f" ({record['period']})"
        identity_parts.append(f"date: {dt}")

    # ── 2. obverse chunk ───────────────────────────────────────────────────────
    ob_legend = _clean(record.get("obverse_legend") or "")
    ob_design = _clean(record.get("obverse_design") or "")
    ob_parts  = []
    if ob_legend:
        ob_parts.append(f"obverse: {ob_legend}")
    if ob_design:
        ob_parts.append(f"design: {ob_design}")
    obverse_text = (
        " | ".join(ob_parts)
        if ob_parts
        else f"CN type {tid} obverse: no description available"
    )

    # ── 3. reverse chunk ───────────────────────────────────────────────────────
    rv_legend = _clean(record.get("reverse_legend") or "")
    rv_design = _clean(record.get("reverse_design") or "")
    rv_parts  = []
    if rv_legend:
        rv_parts.append(f"reverse: {rv_legend}")
    if rv_design:
        rv_parts.append(f"design: {rv_design}")
    reverse_text = (
        " | ".join(rv_parts)
        if rv_parts
        else f"CN type {tid} reverse: no description available"
    )

    # ── 4. material chunk ──────────────────────────────────────────────────────
    mat_parts = []
    if record.get("material"):
        mat_parts.append(f"material: {record['material']}")
    if record.get("weight_g"):
        mat_parts.append(f"weight: {record['weight_g']}")
    if record.get("diameter_mm"):
        mat_parts.append(f"diameter: {record['diameter_mm']}")
    if record.get("mint"):
        mat_parts.append(f"mint: {record['mint']}")
    material_text = (
        " | ".join(mat_parts)
        if mat_parts
        else f"CN type {tid} material: no data available"
    )

    # ── 5. context chunk ───────────────────────────────────────────────────────
    ctx_parts = []
    if record.get("persons"):
        ctx_parts.append(f"persons: {record['persons']}")
    extra = record.get("extra") or {}
    if isinstance(extra, dict):
        # Cap citation/literature length to avoid overwhelming the embedding
        if extra.get("citation"):
            ctx_parts.append(f"citation: {extra['citation'][:300]}")
        if extra.get("literature"):
            ctx_parts.append(f"literature: {extra['literature'][:300]}")
    context_text = (
        " | ".join(ctx_parts)
        if ctx_parts
        else f"CN type {tid} context: no additional data"
    )

    # ── assemble the 5 chunk dicts ─────────────────────────────────────────────
    chunk_defs = [
        ("identity", " | ".join(identity_parts)),
        ("obverse",  obverse_text),
        ("reverse",  reverse_text),
        ("material", material_text),
        ("context",  context_text),
    ]
    return [
        {
            "chunk_id":         f"{tid}_{ctype}",
            "type_id":          tid,
            "chunk_type":       ctype,
            "text":             text,
            "in_training_set":  its,
        }
        for ctype, text in chunk_defs
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  RRF — Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def _rrf_merge(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    WHAT: Combines ranked results from BM25 and ChromaDB into one unified
          ranking without needing a cross-encoder model.

    WHY RRF not score averaging:
        BM25 TF-IDF scores and ChromaDB cosine similarities live on
        completely different numeric scales.  You cannot average 0.87 cosine
        and 14.3 BM25 — the scales are incomparable.  Rank positions (1st,
        2nd, 3rd…) ARE comparable across any retrieval method.

    Formula: score(d) = Σ_r  1 / (k + rank_r(d) + 1)
        k=60 is the standard smoothing constant from Cormack et al. 2009.
        It prevents top-1 results from dominating too aggressively.

    Parameters
    ----------
    ranked_lists : list of ranked lists of chunk_id strings
    k            : RRF smoothing constant (default 60)

    Returns
    -------
    list of (chunk_id, rrf_score) sorted by descending score
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Hybrid BM25 + ChromaDB retrieval engine for Corpus Nummorum coin types.

    Architecture
    ------------
    Two parallel retrieval paths, merged with RRF:

      BM25     — fast keyword search over pre-tokenised chunk texts.
                 Built in memory at startup from the metadata JSON (~1-2 s).
                 No GPU, no network — pure Python.

      ChromaDB — semantic vector search over the same 5-chunk corpus.
                 Requires offline index build via populate_chroma() (STEP 2).
                 Falls back to BM25-only if the collection is empty.

    Usage
    -----
        engine = get_rag_engine()
        results = engine.search("silver drachm Thrace", n=5)
        blocks  = engine.get_context_blocks(1015)   # inject into Gemini prompt
    """

    def __init__(
        self,
        metadata_path: Path | None = None,
        chroma_dir:    Path | None = None,
    ) -> None:
        """
        Load coin metadata, build BM25 index, and connect to ChromaDB.

        Parameters
        ----------
        metadata_path   JSON metadata file path.  Defaults to
                        cn_types_metadata_full.json; falls back to
                        cn_types_metadata.json (438 types) if full not found.
        chroma_dir      ChromaDB directory for the chunked index.
                        Defaults to data/metadata/chroma_db_rag/
        """
        # ── 1. resolve metadata path ──────────────────────────────────────────
        self._meta_path = metadata_path or (
            _METADATA_JSON if _METADATA_JSON.exists() else _METADATA_JSON_FALLBACK
        )
        if not self._meta_path.exists():
            raise FileNotFoundError(
                f"No metadata JSON found at {self._meta_path}.\n"
                "Run: python scripts/build_knowledge_base.py --scrape-only"
            )
        logger.info("RAGEngine: loading metadata from %s", self._meta_path.name)

        # ── 2. load successful records only ───────────────────────────────────
        with open(self._meta_path, encoding="utf-8") as f:
            raw: list[dict] = json.load(f)

        self._records: dict[int, dict] = {
            int(r["type_id"]): r
            for r in raw
            if "error" not in r
        }
        logger.info("RAGEngine: %d coin records loaded", len(self._records))

        # ── 3. build all chunks in memory (5 per coin) ────────────────────────
        # WHY in memory: 9,716 × 5 = ~48,580 short strings ≈ 20 MB RAM.
        # Rebuilding at startup is faster than an extra serialisation layer.
        self._all_chunks: list[dict] = []
        self._chunk_index: dict[str, dict] = {}  # chunk_id → chunk dict
        for record in self._records.values():
            for ch in chunk_record(record):
                self._all_chunks.append(ch)
                self._chunk_index[ch["chunk_id"]] = ch
        logger.info("RAGEngine: %d chunks prepared", len(self._all_chunks))

        # ── 4. build BM25 index ───────────────────────────────────────────────
        self._bm25 = self._build_bm25()

        # ── 5. connect to ChromaDB ────────────────────────────────────────────
        self._chroma_dir = chroma_dir or _CHROMA_DIR
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._collection = self._connect_chroma()

        chroma_count = self._collection.count()
        if chroma_count == 0:
            logger.warning(
                "RAGEngine: ChromaDB collection '%s' is empty. "
                "Run scripts/rebuild_chroma.py (STEP 2) to populate it. "
                "Falling back to BM25-only search.",
                COLLECTION_NAME,
            )
        else:
            logger.info("RAGEngine: ChromaDB has %d chunk vectors", chroma_count)

    # ── internal builders ──────────────────────────────────────────────────────

    def _build_bm25(self):
        """
        Build a BM25Okapi index over all chunk texts.

        WHAT: Tokenises every chunk text (lowercase, keeps Latin + Greek chars)
              and builds an in-memory inverted index.

        WHY BM25Okapi:  normalises term frequency by document length, so a
        short material chunk is not penalised vs a long citation chunk.

        WHY Greek chars included: CN legends contain Greek text (e.g. ΣΕΒΑΣΤΟΣ).
        The regex [a-zA-Z0-9\u0370-\u03FF]+ preserves those tokens.
        """
        from rank_bm25 import BM25Okapi
        tokenised = [
            re.findall(r"[a-zA-Z0-9\u0370-\u03FF]+", ch["text"].lower())
            for ch in self._all_chunks
        ]
        return BM25Okapi(tokenised)

    def _connect_chroma(self):
        """Connect to (or create) the ChromaDB collection for chunked vectors."""
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        client   = chromadb.PersistentClient(path=str(self._chroma_dir))
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query:           str,
        n:               int = 5,
        chunk_filter:    ChunkType | None = None,
        type_id_filter:  int | None = None,
    ) -> list[dict]:
        """
        Hybrid BM25 + vector search, merged with RRF.

        WHAT: Runs BM25 and ChromaDB in parallel on the same query, then
              merges the two ranked lists using Reciprocal Rank Fusion.
              Returns the top-N unique coin records (deduplicated by type_id).

        WHY both BM25 and ChromaDB:
            BM25 is strong at exact legend fragments ("ΣΕΒΑΣΤΟΣ", "Maroneia").
            ChromaDB is strong at semantic description ("helmeted portrait right"
            matching "head with helmet right").  Together they cover both cases.

        Parameters
        ----------
        query           Natural language or keyword search string.
        n               Number of distinct coin types to return (default 5).
        chunk_filter    Restrict search to one chunk type only.
                        "material"  → only material chunks (used by Validator).
                        "obverse"   → only obverse chunks, etc.
        type_id_filter  If set, only consider chunks for this specific type.
                        Used by Historian: fetch the 5 known-type chunks only.

        Returns
        -------
        list[dict]  Each dict is the full coin record from the JSON plus:
            rrf_score   : float  merged rank score (higher = better)
            chunk_types : list[str]  which chunk type produced the top hit
        """
        # Determine candidate chunks
        candidates = [
            ch for ch in self._all_chunks
            if (type_id_filter is None or ch["type_id"] == type_id_filter)
            and (chunk_filter  is None or ch["chunk_type"] == chunk_filter)
        ]
        if not candidates:
            logger.warning("RAGEngine.search: no candidates after filtering")
            return []

        candidate_ids = {ch["chunk_id"] for ch in candidates}

        # BM25 retrieval
        bm25_ranked = self._bm25_search(query, candidates, top_k=n * 10)

        # ChromaDB retrieval
        chroma_ranked = self._chroma_search(query, candidate_ids, top_k=n * 10)

        # RRF merge
        if chroma_ranked:
            merged = _rrf_merge([bm25_ranked, chroma_ranked])
        else:
            # BM25-only fallback when ChromaDB is empty
            merged = [
                (cid, 1.0 / (60 + i + 1))
                for i, cid in enumerate(bm25_ranked)
            ]

        # Deduplicate by type_id and collect top-N coin records
        seen:    set[int]   = set()
        results: list[dict] = []
        for chunk_id, rrf_score in merged:
            if chunk_id not in self._chunk_index:
                continue
            ch  = self._chunk_index[chunk_id]
            tid = ch["type_id"]
            if tid in seen:
                continue
            seen.add(tid)
            record = dict(self._records[tid])
            record["rrf_score"]   = round(rrf_score, 6)
            record["chunk_types"] = [ch["chunk_type"]]
            results.append(record)
            if len(results) >= n:
                break

        return results

    def _bm25_search(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int,
    ) -> list[str]:
        """
        Score candidates with BM25 and return top-k chunk_ids.

        The BM25 index was built over ALL chunks, so we must map each
        candidate back to its global index position before scoring.
        """
        tokens = re.findall(r"[a-zA-Z0-9\u0370-\u03FF]+", query.lower())
        if not tokens:
            return []

        # Map chunk_id → global position in self._all_chunks
        global_idx = {ch["chunk_id"]: i for i, ch in enumerate(self._all_chunks)}
        all_scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            (
                (ch["chunk_id"], float(all_scores[global_idx[ch["chunk_id"]]]))
                for ch in candidates
            ),
            key=lambda x: -x[1],
        )
        return [cid for cid, _ in ranked[:top_k]]

    def _chroma_search(
        self,
        query:         str,
        candidate_ids: set[str],
        top_k:         int,
    ) -> list[str]:
        """
        Query ChromaDB for semantic matches, filtered to candidate chunk_ids.
        Returns an empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            return []
        try:
            # Over-request to allow post-filter candidate intersection
            n_req = min(top_k * 3, self._collection.count())
            raw   = self._collection.query(
                query_texts=[query],
                n_results=n_req,
                include=["distances"],
            )
            ranked = [cid for cid in raw["ids"][0] if cid in candidate_ids]
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("RAGEngine: ChromaDB query failed (%s) — BM25 only", exc)
            return []

    def get_by_id(self, type_id: int) -> dict | None:
        """
        Return the full coin record for a specific CN type ID.
        Returns None if the type was not successfully scraped.
        """
        return self._records.get(int(type_id))

    def get_context_blocks(self, type_id: int) -> str:
        """
        Format the 5 semantic chunks of a coin as labeled [CONTEXT N] blocks
        for direct injection into a Gemini / LLM prompt.

        WHAT: Produces a structured multi-line string like:

            [CONTEXT 1 — Identity]  CN type 1015 | denomination: drachm | ...
            [CONTEXT 2 — Obverse]   obverse: Prancing horse right | legend MAR
            [CONTEXT 3 — Reverse]   reverse: Bunch of grapes | legend EPI ZINONOS
            [CONTEXT 4 — Material]  material: silver | weight: 2.44 g | mint: ...
            [CONTEXT 5 — Context]   persons: Magistrate Zenon

        WHY this format:
            The Historian/Investigator prompt instructs Gemini:
            "Use ONLY the contexts above and cite [CONTEXT N] for each fact."
            Labeled blocks make citations natural and verifiable — if Gemini
            writes [CONTEXT 3], the reviewer can check exactly what reverse
            description was used.  This eliminates hallucination on facts.

        Returns empty string if the type_id is not in the corpus.
        """
        tid    = int(type_id)
        labels = [
            ("identity", "Identity"),
            ("obverse",  "Obverse"),
            ("reverse",  "Reverse"),
            ("material", "Material"),
            ("context",  "Context"),
        ]
        lines = []
        for i, (ctype, label) in enumerate(labels, start=1):
            chunk_id = f"{tid}_{ctype}"
            ch       = self._chunk_index.get(chunk_id)
            text     = ch["text"] if ch else f"(no {ctype} data for type {tid})"
            lines.append(f"[CONTEXT {i} \u2014 {label}]  {text}")
        return "\n".join(lines)

    def populate_chroma(self, batch_size: int = 200) -> None:
        """
        Embed and upsert all chunks into the ChromaDB collection.

        WHAT: Iterates over all chunks (up to 48,580), embeds each one
              with all-MiniLM-L6-v2, and upserts into ChromaDB in batches.

        WHEN TO CALL: Once, from scripts/rebuild_chroma.py (STEP 2),
              or whenever the metadata JSON has been re-scraped.

        WHY batches of 200: ChromaDB builds its HNSW index incrementally.
              Batches allow progress reporting and avoid OOM on large upserts.
              200 is a practical balance between write efficiency and feedback.

        Parameters
        ----------
        batch_size  Chunks per ChromaDB upsert call (default 200).
        """
        total     = len(self._all_chunks)
        ids       = [ch["chunk_id"]  for ch in self._all_chunks]
        documents = [ch["text"]      for ch in self._all_chunks]
        metadatas = [
            {
                "type_id":         ch["type_id"],
                "chunk_type":      ch["chunk_type"],
                "in_training_set": ch["in_training_set"],
            }
            for ch in self._all_chunks
        ]

        logger.info("RAGEngine.populate_chroma: upserting %d chunks …", total)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            pct = (end / total) * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {end}/{total} chunks embedded", end="", flush=True)
        print()
        logger.info(
            "RAGEngine.populate_chroma: done — %d vectors in ChromaDB",
            self._collection.count(),
        )

    def corpus_size(self) -> int:
        """Return the number of coin records loaded (not chunks)."""
        return len(self._records)

    def is_chroma_built(self) -> bool:
        """Return True if the ChromaDB collection has been populated."""
        return self._collection.count() > 0


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_engine_instance: RAGEngine | None = None


def get_rag_engine(
    metadata_path: Path | None = None,
    chroma_dir:    Path | None = None,
) -> RAGEngine:
    """
    Return the shared RAGEngine instance (created on first call, then cached).

    WHY singleton:
        Building the BM25 index takes 1-2 seconds for ~48,580 chunks.
        Instantiating per-request inside a FastAPI route would be a
        serious performance regression.  The singleton ensures zero overhead
        on every call after the first one in the process lifetime.

    Thread safety:
        Safe for concurrent read-only search() calls.
        NOT safe for concurrent populate_chroma() — run that offline only.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(
            metadata_path=metadata_path,
            chroma_dir=chroma_dir,
        )
    return _engine_instance
