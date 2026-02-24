"""
src/agents/investigator.py
===========================
Layer 3 — Investigator Agent

Triggered when CNN confidence < 0.40 (model is unsure).
Pipeline:
  1. Load coin image → base64 encode
  2. Send to Gemini 2.5 Flash Vision with a structured numismatic prompt
  3. Parse response into structured attributes
  4. Optional: cross-reference the description against the KB
     to find the closest matching type

Engineering notes:
  - Same LLM client as Historian (GitHub Models / Google AI Studio)
  - Image is sent as base64 data URL, not a file upload
  - KB cross-reference gives a "best guess" type even at low CNN confidence
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from src.core.knowledge_base import get_knowledge_base

# Reuse the same LLM loader as Historian
from src.agents.historian import _get_llm


# ══════════════════════════════════════════════════════════════════════════════

class Investigator:
    """
    Visual Investigator — Gemini Vision analysis for low-confidence coins.

    Input  (from Gatekeeper state):
        image_path : str   — path to the coin image
        class_id   : int   — CNN best guess (low confidence)
        confidence : float — CNN score (< 0.40)
        top5       : list  — CNN top-5 predictions

    Output dict:
        {
          "visual_description" : str,    # Gemini free-text description
          "detected_features"  : {
              "metal_color"        : str,
              "profile_direction"  : str,
              "inscriptions"       : list[str],
              "symbols"            : list[str],
              "condition"          : str,
          },
          "kb_matches"         : list[dict],  # top-3 KB hits from Gemini description
          "suggested_type_id"  : int | None,  # best KB match type_id
          "llm_used"           : bool,
        }
    """

    def __init__(self) -> None:
        self._kb = get_knowledge_base()

    # ── public ────────────────────────────────────────────────────────────────

    def investigate(self, image_path: str, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        Parameters
        ----------
        image_path     : absolute path to coin image
        cnn_prediction : dict with keys class_id, label, confidence, top5
        """
        confidence = float(cnn_prediction["confidence"])
        top5       = cnn_prediction.get("top5", [])

        # 1. Gemini Vision description
        description, features, llm_used = self._describe_image(image_path, confidence, top5)

        # 2. KB cross-reference: use the visual description as a semantic query
        kb_matches: list[dict] = []
        suggested_type_id: int | None = None
        if description:
            kb_matches = self._kb.search(description, n=3)
            if kb_matches:
                suggested_type_id = kb_matches[0]["type_id"]

        return {
            "visual_description":  description,
            "detected_features":   features,
            "kb_matches":          kb_matches,
            "suggested_type_id":   suggested_type_id,
            "llm_used":            llm_used,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _describe_image(self, image_path: str, confidence: float, top5: list) -> tuple:
        """
        Encode image as base64 and call Gemini Vision.
        Returns (description_str, features_dict, llm_was_used).
        """
        client, model = _get_llm()
        if client is None:
            return _fallback_description(), _empty_features(), False

        # Read and base64-encode the image
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            # Detect MIME type from extension
            ext = Path(image_path).suffix.lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
        except Exception as e:
            return f"Could not read image: {e}", _empty_features(), False

        # Build the CNN context hints
        cnn_hint = f"CNN confidence is low ({confidence:.1%})."
        if top5:
            candidates = ", ".join(
                f"{t['label']} ({t['confidence']:.1%})" for t in top5[:3]
            )
            cnn_hint += f" Top CNN candidates: {candidates}."

        prompt = (
            "You are an expert numismatist and archaeologist specialising in ancient Greek and Roman coins.\n"
            f"{cnn_hint}\n"
            "Carefully examine this coin image and provide a structured analysis:\n\n"
            "1. METAL/MATERIAL: What metal does the coin appear to be? (silver / bronze / gold / unknown)\n"
            "2. OBVERSE: Describe everything visible on the front face (portrait, symbols, legend text).\n"
            "3. REVERSE: Describe everything visible on the reverse (symbols, legend text).\n"
            "4. INSCRIPTIONS: List any readable text fragments.\n"
            "5. CONDITION: Brief condition assessment (well-preserved / worn / corroded).\n"
            "6. IDENTIFICATION: Based on what you see, what type of ancient coin is this likely to be? "
            "   Include period, issuing authority, and region if you can determine them.\n\n"
            "Be specific and use numismatic terminology. If something is not visible, say so."
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                max_tokens=600,
                temperature=0.3,
            )
            description = resp.choices[0].message.content.strip()
            features    = _parse_features(description)
            return description, features, True
        except Exception as e:
            return f"Gemini Vision unavailable: {e}", _empty_features(), False


# ── helpers ──────────────────────────────────────────────────────────────────────

def _empty_features() -> dict:
    return {
        "metal_color":       "unknown",
        "profile_direction": "unknown",
        "inscriptions":      [],
        "symbols":           [],
        "condition":         "unknown",
    }

def _fallback_description() -> str:
    return (
        "Visual analysis unavailable (LLM not configured). "
        "Set GITHUB_TOKEN or GOOGLE_API_KEY environment variable."
    )

def _parse_features(description: str) -> dict:
    """
    Light parsing of Gemini's free-text response into structured fields.
    Best-effort — if a field can’t be extracted, returns 'unknown'.
    """
    import re

    def _extract(pattern: str) -> str:
        m = re.search(pattern, description, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().split("\n")[0].strip()
        return "unknown"

    # Metal
    metal = "unknown"
    for m in ("silver", "bronze", "gold", "copper", "billon"):
        if m in description.lower():
            metal = m
            break

    # Inscriptions — look for capital letter runs that look like legends
    inscriptions = re.findall(r"[A-ZΑ-Ω]{3,}", description)
    inscriptions = list(dict.fromkeys(inscriptions))[:5]   # deduplicate, keep 5

    # Condition
    condition = "unknown"
    for cond in ("well-preserved", "well preserved", "worn", "corroded", "good", "poor", "fair"):
        if cond in description.lower():
            condition = cond
            break

    return {
        "metal_color":       metal,
        "profile_direction": _extract(r"(?:facing|portrait|head)\s+(left|right)"),
        "inscriptions":      inscriptions,
        "symbols":           [],   # too complex to parse reliably from free text
        "condition":         condition,
    }
