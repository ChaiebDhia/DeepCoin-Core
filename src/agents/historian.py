"""
src/agents/historian.py
========================
Layer 3 — Historian Agent

Triggered when CNN confidence >= 0.40.
Pipeline:
  1. search_by_id(class_id) → pull structured record from ChromaDB
  2. Build a rich prompt with that context
  3. Send to Gemini 2.5 Flash (GitHub Models API)
  4. Return a structured narrative dict

Engineering notes:
  - KB is a singleton — loaded once per process
  - Gemini key from GITHUB_TOKEN env var (GitHub Models, free for students)
  - Falls back to Google AI Studio key if GITHUB_TOKEN not set
  - If both missing → returns the raw KB data without LLM narrative
"""

from __future__ import annotations

import os
from typing import Any

from src.core.knowledge_base import get_knowledge_base

# ── LLM client (lazy, loaded once) ────────────────────────────────────────────
_llm_client = None
_llm_model   = None

def _get_llm():
    """Return (client, model_name) — GitHub Models preferred, Google AI fallback."""
    global _llm_client, _llm_model
    if _llm_client is not None:
        return _llm_client, _llm_model

    github_token = os.getenv("GITHUB_TOKEN")
    google_key   = os.getenv("GOOGLE_API_KEY")

    if github_token:
        from openai import OpenAI
        _llm_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_token,
        )
        _llm_model = "gemini-2.5-flash"
    elif google_key:
        from openai import OpenAI          # google AI studio is also OpenAI-compatible
        _llm_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=google_key,
        )
        _llm_model = "gemini-2.5-flash"
    else:
        _llm_client = None
        _llm_model  = None

    return _llm_client, _llm_model


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
        self._kb = get_knowledge_base()   # singleton — loaded once

    # ── public ────────────────────────────────────────────────────────────────

    def research(self, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        Parameters
        ----------
        cnn_prediction : dict  with keys  class_id, label, confidence
        """
        class_id   = int(cnn_prediction["class_id"])
        confidence = float(cnn_prediction["confidence"])

        # 1. Pull record from ChromaDB
        record = self._kb.search_by_id(class_id)
        if record is None:
            # class not in KB (one of the 4 deleted types) — semantic fallback
            label   = cnn_prediction.get("label", f"CN_{class_id}")
            results = self._kb.search(label, n=1)
            record  = results[0] if results else {}

        # 2. Generate Gemini narrative
        narrative, llm_used = self._generate_narrative(record, confidence)

        return {
            "type_id":      record.get("type_id", class_id),
            "mint":         record.get("mint", ""),
            "region":       record.get("region", ""),
            "date":         record.get("date", ""),
            "period":       record.get("period", ""),
            "material":     record.get("material", ""),
            "denomination": record.get("denomination", ""),
            "obverse":      _extract_obverse(record.get("document", "")),
            "reverse":      _extract_reverse(record.get("document", "")),
            "persons":      _extract_persons(record.get("document", "")),
            "source_url":   record.get("source_url", ""),
            "narrative":    narrative,
            "llm_used":     llm_used,
        }

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Semantic KB search — used by Investigator for cross-referencing."""
        return self._kb.search(query, n=n)

    # ── private ───────────────────────────────────────────────────────────────

    def _generate_narrative(self, record: dict, confidence: float) -> tuple[str, bool]:
        """
        Send KB record to Gemini → get a 2-3 sentence archaeological narrative.
        Returns (narrative_str, llm_was_used).
        """
        client, model = _get_llm()
        if client is None or not record:
            return _fallback_narrative(record), False

        doc = record.get("document", "No data available.")
        prompt = (
            "You are a professional numismatist and archaeologist.\n"
            "Based on the following Corpus Nummorum record, write a concise "
            "2-3 sentence expert commentary about this ancient coin. "
            "Focus on its historical significance, the issuing authority, "
            "and what makes it numismatically interesting. "
            "Do NOT repeat the raw data fields verbatim — synthesize them into prose.\n\n"
            f"Record:\n{doc}\n\n"
            f"CNN classification confidence: {confidence:.1%}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4,
            )
            narrative = resp.choices[0].message.content.strip()
            return narrative, True
        except Exception as e:
            return f"{_fallback_narrative(record)} [LLM unavailable: {e}]", False


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_field(document: str, label: str) -> str:
    """Pull a field value from the pre-built document text."""
    import re
    m = re.search(rf"{label}:\s*(.+?)(?:\.|$)", document, re.IGNORECASE)
    return m.group(1).strip() if m else ""

def _extract_obverse(doc: str) -> str:
    return _extract_field(doc, "Obverse")

def _extract_reverse(doc: str) -> str:
    return _extract_field(doc, "Reverse")

def _extract_persons(doc: str) -> str:
    return _extract_field(doc, "Persons")

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
