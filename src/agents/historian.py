"""
src/agents/historian.py
========================
Layer 3 — Historian Agent

Triggered when CNN confidence >= 0.40.
Pipeline:
  1. search_by_id(class_id) → pull structured record from ChromaDB
  2. Build a rich prompt with that context
  3. Send to LLM (priority: GitHub Models → Google AI Studio → Ollama → fallback)
  4. Return a structured narrative dict

Engineering notes:
  - KB is a singleton — loaded once per process
  - LLM provider resolved via _get_llm() priority chain (see that function)
  - If no key set → returns the raw KB data without LLM narrative
  - .env is loaded at import time via python-dotenv
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv
load_dotenv()   # reads .env from project root — sets OLLAMA_HOST, GITHUB_TOKEN, etc.

from src.core.knowledge_base import get_knowledge_base
from src.core.rag_engine     import get_rag_engine

# ── LLM clients (lazy singletons, one per capability) ─────────────────────────
# WHY two caches instead of one:
#   Text tasks (Historian) and vision tasks (Investigator) need different models.
#   gemma3:4b is a text-only model — it cannot process images.
#   qwen3-vl:4b is a Vision-Language model — it can process both.
#   We cache them separately so both can coexist in the same process.
_text_client  = None
_text_model   = None
_vision_client = None
_vision_model  = None


def _get_llm(capability: str = "text"):
    """
    Return (client, model_name) for the best available LLM provider.

    Parameters
    ----------
    capability : "text" | "vision"
        "text"   — for Historian narrative generation (gemma3:4b locally)
        "vision" — for Investigator image analysis  (qwen3-vl:4b locally)

    Priority chain (same for both capabilities):
      1. GITHUB_TOKEN   → GitHub Models API  (Gemini 2.5 Flash — handles both)
      2. GOOGLE_API_KEY → Google AI Studio   (Gemini 2.5 Flash — handles both)
      3. OLLAMA_HOST    → Local Ollama       (model selected by capability)
                            text   → OLLAMA_MODEL        (default: gemma3:4b)
                            vision → OLLAMA_VISION_MODEL (default: qwen3-vl:4b)
      4. None           → structured fallback (no LLM, never crashes)

    WHY qwen3-vl:4b for vision:
        qwen3-vl is a Vision-Language model: it accepts images as input.
        At 4b parameters (Q4 quantized ~2.5 GB VRAM), it fits in the
        RTX 3050 Ti's 4.3 GB budget alongside the OS overhead.
        It understands numismatic vocabulary and can describe coin obverse,
        reverse, legends, and metal type from a raw photograph — offline.

    WHY gemma3:4b for text:
        Fast, 3.3 GB VRAM, strong instruction following for structured prompts.
        Text-only — lighter than a VL model for pure narrative generation.

    WHY OpenAI client for Ollama:
        Ollama exposes an OpenAI-compatible REST API at /v1.
        Same client object, different base_url — zero extra dependencies.
    """
    global _text_client, _text_model, _vision_client, _vision_model

    # Return cached client if already resolved for this capability
    if capability == "vision" and _vision_client is not None:
        return _vision_client, _vision_model
    if capability == "text" and _text_client is not None:
        return _text_client, _text_model

    github_token = os.getenv("GITHUB_TOKEN")
    google_key   = os.getenv("GOOGLE_API_KEY")
    ollama_host  = os.getenv("OLLAMA_HOST")   # e.g. http://localhost:11434

    client: Any = None
    model:  str = ""

    if github_token:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_token,
        )
        model = "gemini-2.5-flash"   # Gemini handles both text and vision

    elif google_key:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=google_key,
        )
        model = "gemini-2.5-flash"   # Gemini handles both text and vision

    elif ollama_host:
        # Ollama exposes an OpenAI-compatible API at /v1.
        # api_key="ollama" is a required dummy value — Ollama ignores it.
        from openai import OpenAI
        base = ollama_host.rstrip("/")
        client = OpenAI(base_url=f"{base}/v1", api_key="ollama")
        if capability == "vision":
            # qwen3-vl:4b — Vision-Language model, understands images
            model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b")
        else:
            # gemma3:4b — text-only, faster and lighter for narrative generation
            model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    # Store in the correct cache slot
    if capability == "vision":
        _vision_client, _vision_model = client, model
    else:
        _text_client, _text_model = client, model

    return client, model


# ══════════════════════════════════════════════════════════════════════════════

class Historian:
    """
    Historian agent — RAG + LLM narrative for a classified coin.

    Input  (from Gatekeeper state):
        class_id   : int    — CNN predicted class
        confidence : float  — CNN confidence score
        label      : str    — e.g. "CN_1015"

    Output dict:
        {
          "type_id"         : int,
          "mint"            : str,
          "region"          : str,
          "date"            : str,
          "period"          : str,
          "material"        : str,
          "denomination"    : str,
          "obverse"         : str,
          "reverse"         : str,
          "persons"         : str,
          "source_url"      : str,
          "narrative"       : str,   # Gemini-generated paragraph
          "llm_used"        : bool,
        }
    """

    def __init__(self) -> None:
        self._kb  = get_knowledge_base()   # legacy KB — 438 types, used only as last-resort fallback
        self._rag = get_rag_engine()        # new RAG engine — 9,541 types, 47,705 chunks

    # ── public ────────────────────────────────────────────────────────────────

    def research(self, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        WHAT: Looks up the coin in the RAG engine (9,541 types), builds
              structured [CONTEXT N] blocks, and passes them to the LLM
              with strict grounding instructions.

        WHY RAG engine instead of legacy KB:
            The legacy KB has only 438 types (the CNN training subset).
            The RAG engine covers all 9,541 scraped types — 21× more coverage.
            For coins the CNN classifies correctly (conf > 40%), the record
            is almost always present in the RAG engine.

        Parameters
        ----------
        cnn_prediction : dict with keys class_id, label, confidence, top5
        """
        class_id   = int(cnn_prediction["class_id"])   # raw tensor index (0-437)
        label_str  = cnn_prediction.get("label", "")  # folder name = CN type ID e.g. "1015"
        confidence = float(cnn_prediction["confidence"])

        # WHY use label_str not class_id for the lookup:
        #   class_id is the raw softmax index (0, 1, 2 ... 437) — NOT the CN type number.
        #   label_str is the folder name, which IS the CN type ID (e.g. "1015").
        #   The RAG engine keys records by CN type ID, so we must use label_str.
        cn_type_id = int(label_str) if label_str.isdigit() else class_id

        # 1. Pull structured record — RAG engine first, legacy KB as fallback
        record = self._rag.get_by_id(cn_type_id)
        if record is None:
            record = self._kb.search_by_id(cn_type_id)
        if record is None:
            results = self._kb.search(label_str or str(class_id), n=1)
            record  = results[0] if results else {}

        # 2. Build [CONTEXT N] blocks for grounded LLM prompt
        #    WHY: The LLM is instructed to cite [CONTEXT N] for every fact —
        #         eliminates hallucination on structured historical data.
        type_id        = record.get("type_id", class_id)
        context_blocks = self._rag.get_context_blocks(type_id)

        # 3. Generate grounded narrative
        narrative, llm_used = self._generate_narrative(record, confidence, context_blocks)

        # 4. Extract obverse/reverse from structured fields (clean, no blob parsing)
        obverse_parts = []
        if record.get("obverse_design"): obverse_parts.append(record["obverse_design"])
        if record.get("obverse_legend"): obverse_parts.append(f"legend {record['obverse_legend']}")
        obverse = " | ".join(obverse_parts) or record.get("obverse", "")

        reverse_parts = []
        if record.get("reverse_design"): reverse_parts.append(record["reverse_design"])
        if record.get("reverse_legend"): reverse_parts.append(f"legend {record['reverse_legend']}")
        reverse = " | ".join(reverse_parts) or record.get("reverse", "")

        return {
            "type_id":      type_id,
            "mint":         record.get("mint", ""),
            "region":       record.get("region", ""),
            "date":         record.get("date", ""),
            "period":       record.get("period", ""),
            "material":     record.get("material", ""),
            "denomination": record.get("denomination", ""),
            "obverse":      obverse,
            "reverse":      reverse,
            "persons":      record.get("persons", ""),
            "source_url":   record.get("source_url", ""),
            "narrative":    narrative,
            "llm_used":     llm_used,
        }

    def search(self, query: str, n: int = 5) -> list[dict]:
        """
        Hybrid BM25+vector search over the full CN corpus (9,541 types).
        Used by Investigator for cross-referencing visual descriptions.
        WHY RAG engine: covers 9,541 types vs the legacy KB's 438.
        """
        return self._rag.search(query, n=n)

    # ── private ───────────────────────────────────────────────────────────────

    def _generate_narrative(
        self,
        record:         dict,
        confidence:     float,
        context_blocks: str,
    ) -> tuple[str, bool]:
        """
        Generate a grounded historical narrative via [CONTEXT N] injection.

        WHAT: Injects 5 structured context blocks into the prompt and instructs
              the LLM to write ONLY from those blocks, citing [CONTEXT N].

        WHY [CONTEXT N] instead of a raw blob:
            The old approach sent one 200-word paragraph to the LLM.
            The LLM could misread fields or invent plausible-sounding facts.
            With labeled blocks, every sentence the LLM writes can be traced
            back to a specific chunk — verifiable, auditable, zero hallucination
            on structured facts (dates, weights, mint names).

        WHY max_tokens=800:
            Reasoning models (gemma3, deepseek) generate internal chain-of-thought
            tokens before writing the visible content.  With max_tokens=300,
            the thinking budget was consumed and content came back empty.
            800 gives enough headroom for both thinking and prose output.

        Parameters
        ----------
        record         : structured coin record from RAG engine
        confidence     : CNN classification confidence (shown to LLM)
        context_blocks : [CONTEXT 1—Identity] ... string from get_context_blocks()
        """
        client, model = _get_llm(capability="text")
        if client is None or not record:
            return _fallback_narrative(record), False

        if not context_blocks:
            context_blocks = f"CN type {record.get('type_id', 'unknown')}: limited data available."

        prompt = (
            "You are a professional numismatist and archaeologist specialising in ancient coins.\n"
            "You have been given structured data from the Corpus Nummorum academic database.\n\n"
            "CONTEXT BLOCKS (these are the ONLY facts you may use):\n"
            f"{context_blocks}\n\n"
            f"CNN image classification confidence: {confidence:.1%}\n\n"
            "TASK: Write a concise expert commentary of 2-3 paragraphs about this ancient coin.\n"
            "STRICT OUTPUT FORMAT RULES — violating any rule makes the output unusable:\n"
            "  1. Use ONLY facts present in the context blocks above.\n"
            "  2. Do NOT invent dates, weights, mint names, or historical events not in the context.\n"
            "  3. Synthesise the facts into flowing professional prose — do not list fields.\n"
            "  4. If confidence is below 85%, note that the classification should be verified.\n"
            "  5. OUTPUT PLAIN TEXT ONLY. No Markdown. No asterisks, no bold, no italics,\n"
            "     no headers, no bullet points, no backticks, no underscores for emphasis.\n"
            "  6. Do NOT include any citation markers such as [CONTEXT N] or [CONTEXT CNN]\n"
            "     in your response. The context blocks are for your internal reference only.\n"
            "  7. Write in complete sentences. No special characters except standard punctuation.\n\n"
            "Write the commentary now:"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4,
            )
            narrative = resp.choices[0].message.content.strip()
            # Reasoning models park output in .reasoning when max_tokens is tight
            if not narrative:
                narrative = getattr(resp.choices[0].message, "reasoning", "") or ""
            if not narrative:
                return _fallback_narrative(record), False
            return _clean_narrative(narrative), True
        except Exception as e:
            return f"{_fallback_narrative(record)} [LLM unavailable: {e}]", False


# ── helpers ────────────────────────────────────────────────────────────────────

# Compiled once at module level — same logic as synthesis._s() but returns a
# clean Python str (no latin-1 encode) so the state dict holds clean text.
_RE_CTX   = re.compile(r"\[CONTEXT\s*(?:\d+|CNN|N)?[^\]]*\]", re.I)
_MD_PATS  = [
    (re.compile(r"\*{3}(.+?)\*{3}"),    r"\1"),
    (re.compile(r"\*{2}(.+?)\*{2}"),    r"\1"),
    (re.compile(r"\*(.+?)\*"),          r"\1"),
    (re.compile(r"_{2}(.+?)_{2}"),      r"\1"),
    (re.compile(r"_(.+?)_"),            r"\1"),
    (re.compile(r"`{1,3}(.+?)`{1,3}"), r"\1"),
    (re.compile(r"(?m)^\s*#{1,6}\s+"),  ""),
    (re.compile(r"#{2,}\s*"),           ""),
]


def _clean_narrative(text: str) -> str:
    """
    Sanitise the raw LLM narrative before it enters the state dict.

    WHY at source (here) AND in synthesis._s():
        Cleaning here means the plain-text report (synthesize()) and any
        downstream consumers (API response JSON, history store) also get
        clean text.  synthesis._s() stays as a final safety net for the PDF.

    Steps applied:
      1. Strip [CONTEXT N] / [CONTEXT CNN] citation markers.
      2. Strip Markdown formatting (**, *, ##, backticks).
      3. Collapse excess whitespace.
      4. NFD decompose + strip combining marks (o-with-macron → o, etc.)
    """
    t = _RE_CTX.sub("", text)
    for pat, repl in _MD_PATS:
        t = pat.sub(repl, t)
    t = re.sub(r"  +", " ", t).strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t


def _fallback_narrative(record: dict) -> str:
    """Plain-English summary when LLM is unavailable."""
    if not record:
        return "No historical record found for this coin type."
    parts = []
    if record.get("mint"):
        parts.append(f"Minted at {record['mint']}")
        if record.get("region"):
            parts[-1] += f" ({record['region']})"
    if record.get("date"):
        parts.append(f"dated {record['date']}")
    if record.get("period"):
        parts.append(f"({record['period']})")    
    if record.get("material") and record.get("denomination"):
        parts.append(f"{record['material']} {record['denomination']}")
    elif record.get("denomination"):
        parts.append(record["denomination"])
    return ". ".join(parts) + "." if parts else "Historical data retrieved from Corpus Nummorum."


if __name__ == "__main__":
    print("📚 Historian Agent - Ready for Phase 5")
