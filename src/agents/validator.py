"""
src/agents/validator.py
========================
Layer 3 — Validator Agent

Triggered when CNN confidence is between 0.40 and 0.85.
Pipeline:
  1. Load coin image with OpenCV
  2. Convert to HSV → compute hue/saturation histogram
  3. Classify dominant metal color (silver / bronze / gold / unknown)
  4. Look up expected material for the predicted class from the KB
  5. Flag a mismatch if they disagree

Engineering notes:
  - No internet calls — pure local OpenCV analysis
  - HSV thresholds calibrated on ancient coin images
  - Returns structured dict that Synthesis agent reads
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.knowledge_base import get_knowledge_base


# ══════════════════════════════════════════════════════════════════════════════
#  HSV metal thresholds (tuned empirically for ancient coin photography)
# ══════════════════════════════════════════════════════════════════════════════
#
# The logic:
#   ancient coins are photographed under controlled white light.
#   HSV Hue channel:  0–179 in OpenCV (maps to 0–360°)
#   Gold   → warm yellow-orange  hue ~15-35, S > 80
#   Bronze → reddish-brown       hue ~5-20,  S 50-150
#   Silver → desaturated grey    S < 40 (any hue, low saturation)
#
_METAL_THRESHOLDS = {
    "gold":   {"h_min": 15, "h_max": 35,  "s_min": 80,  "s_max": 255},
    "bronze": {"h_min": 5,  "h_max": 25,  "s_min": 50,  "s_max": 180},
    # silver = everything with low saturation (S < 40)
}


# ══════════════════════════════════════════════════════════════════════════════

class Validator:
    """
    Forensic Validator — cross-checks CNN material prediction against image.

    Input  (from Gatekeeper state):
        image_path : str   — path to the coin image
        class_id   : int   — CNN predicted class
        confidence : float

    Output dict:
        {
          "status"           : "consistent" | "mismatch" | "unknown",
          "detected_material": str,          # what the image actually looks like
          "expected_material": str,          # what the KB says it should be
          "match"            : bool,
          "warning"          : str,          # human-readable message
          "hsv_stats"        : dict,         # raw numbers for transparency
        }
    """

    def __init__(self) -> None:
        self._kb = get_knowledge_base()

    # ── public ────────────────────────────────────────────────────────────────

    def validate(self, image_path: str, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        Parameters
        ----------
        image_path     : absolute path to coin image
        cnn_prediction : dict with keys class_id, label, confidence
        """
        class_id = int(cnn_prediction["class_id"])

        # 1. Detect material from image pixel data
        detected, hsv_stats = self._detect_material(image_path)

        # 2. Expected material from KB
        record = self._kb.search_by_id(class_id)
        expected = ""
        if record:
            mat = record.get("material", "").lower()
            # Normalise KB material strings: "silver ", "bronze ", "gold " etc.
            for metal in ("gold", "silver", "bronze", "copper", "billon"):
                if metal in mat:
                    expected = metal
                    break

        # 3. Compare
        match   = _materials_match(detected, expected)
        status  = "consistent" if match else ("mismatch" if expected else "unknown")
        warning = _build_warning(detected, expected, match, cnn_prediction["confidence"])

        return {
            "status":            status,
            "detected_material": detected,
            "expected_material": expected,
            "match":             match,
            "warning":           warning,
            "hsv_stats":         hsv_stats,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _detect_material(self, image_path: str) -> tuple[str, dict]:
        """
        Load image, compute HSV stats, return (metal_name, stats_dict).
        """
        img = cv2.imread(image_path)
        if img is None:
            return "unknown", {"error": "cannot read image"}

        # Crop to centre 60% to avoid background bias
        h, w = img.shape[:2]
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        crop = img[y1:y2, x1:x2]

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        # Pixel masks
        total = crop.shape[0] * crop.shape[1]

        gold_mask   = cv2.inRange(hsv,
            np.array([_METAL_THRESHOLDS["gold"]["h_min"],   _METAL_THRESHOLDS["gold"]["s_min"],   0]),
            np.array([_METAL_THRESHOLDS["gold"]["h_max"],   _METAL_THRESHOLDS["gold"]["s_max"],   255]))
        bronze_mask = cv2.inRange(hsv,
            np.array([_METAL_THRESHOLDS["bronze"]["h_min"], _METAL_THRESHOLDS["bronze"]["s_min"], 0]),
            np.array([_METAL_THRESHOLDS["bronze"]["h_max"], _METAL_THRESHOLDS["bronze"]["s_max"], 255]))
        silver_mask = cv2.inRange(hsv,
            np.array([0,   0,   80]),
            np.array([179, 40, 255]))   # low saturation = silver/grey

        pct_gold   = float(np.count_nonzero(gold_mask))   / total
        pct_bronze = float(np.count_nonzero(bronze_mask)) / total
        pct_silver = float(np.count_nonzero(silver_mask)) / total

        stats = {
            "mean_hue":       round(mean_h, 1),
            "mean_saturation":round(mean_s, 1),
            "mean_value":     round(mean_v, 1),
            "pct_gold":       round(pct_gold,   3),
            "pct_bronze":     round(pct_bronze, 3),
            "pct_silver":     round(pct_silver, 3),
        }

        # Decision: highest percentage wins, with a 15% minimum threshold
        scores = {"gold": pct_gold, "bronze": pct_bronze, "silver": pct_silver}
        best   = max(scores, key=scores.get)
        if scores[best] < 0.15:
            return "unknown", stats
        return best, stats


# ── helpers ──────────────────────────────────────────────────────────────────────

_MATERIAL_GROUPS = {
    "silver": {"silver", "ar", "electrum"},
    "gold":   {"gold",   "au", "electrum"},
    "bronze": {"bronze", "copper", "ae", "billon", "orichalcum"},
}

def _materials_match(detected: str, expected: str) -> bool:
    """True if detected and expected belong to the same metal group."""
    if not detected or not expected or detected == "unknown":
        return True   # can't disprove — don't flag
    for group in _MATERIAL_GROUPS.values():
        if detected in group and expected in group:
            return True
    return detected == expected

def _build_warning(detected: str, expected: str, match: bool, confidence: float) -> str:
    if not expected or detected == "unknown":
        return ""
    if match:
        return f"Material consistent: image shows {detected}, KB records {expected}."
    return (
        f"Material mismatch: image appears {detected} but this type is recorded as {expected}. "
        f"CNN confidence was {confidence:.1%}. Manual review recommended."
    )
