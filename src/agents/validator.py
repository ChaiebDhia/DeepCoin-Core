"""
src/agents/validator.py
========================
Layer 3 — Validator Agent

Triggered when CNN confidence is between 0.40 and 0.85.
Pipeline:
  1. Load coin image with OpenCV
  2. Run multi-scale HSV analysis (3 crop sizes: 40% / 60% / 80%)
  3. Classify dominant metal color with a confidence score (0.0–1.0)
  4. Look up expected material for the predicted type from the RAG engine
  5. Compare results: consistent / mismatch / uncertain

Engineering notes:
  - No internet calls — pure local OpenCV analysis
  - HSV thresholds calibrated on ancient coin images
  - Multi-scale analysis reduces false positives from edge shadows / patina
  - Detection confidence = weighted average pixel coverage across scales
  - Uncertainty flag (high/medium/low) reflects scale agreement
  - Returns structured dict that Synthesis agent reads
  - Uses RAG engine (9,541 types) for material lookup, not legacy KB
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.knowledge_base import get_knowledge_base
from src.core.rag_engine     import get_rag_engine


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
          "status"             : "consistent" | "mismatch" | "uncertain",
          "detected_material"  : str,    # what the image actually looks like
          "expected_material"  : str,    # what the KB records as the material
          "detection_confidence": float, # 0.0–1.0 — how sure the HSV detector is
          "uncertainty"        : str,    # "high" | "medium" | "low"
          "match"              : bool,
          "warning"            : str,    # human-readable message
          "hsv_stats"          : dict,   # raw numbers across all 3 scales
        }
    """

    def __init__(self) -> None:
        self._kb  = get_knowledge_base()   # legacy fallback (438 types)
        self._rag = get_rag_engine()        # primary lookup (9,541 types)

    # ── public ────────────────────────────────────────────────────────────────

    def validate(self, image_path: str, cnn_prediction: dict) -> dict:
        """
        Main entry point called by Gatekeeper.

        Parameters
        ----------
        image_path     : absolute path to coin image
        cnn_prediction : dict with keys class_id, label, confidence
        """
        # WHY label_str not class_id:
        #   class_id is the raw softmax index (0-437), not the CN type number.
        #   label_str is the folder name = CN type ID (e.g. "1015").
        #   The RAG engine keys records by CN type ID.
        label_str = cnn_prediction.get("label", "")
        cn_type_id = int(label_str) if label_str.isdigit() else int(cnn_prediction["class_id"])

        # 1. Multi-scale material detection from image pixels
        detected, detection_confidence, uncertainty, hsv_stats = self._detect_material(image_path)

        # 2. Expected material from RAG engine (9,541 types), fall back to legacy KB
        record = self._rag.get_by_id(cn_type_id)
        if record is None:
            record = self._kb.search_by_id(cn_type_id)
        expected = ""
        if record:
            mat = record.get("material", "").lower()
            for metal in ("gold", "silver", "bronze", "copper", "billon"):
                if metal in mat:
                    expected = metal
                    break

        # 3. Compare
        match  = _materials_match(detected, expected)
        status = "consistent" if match else ("mismatch" if expected else "uncertain")
        warning = _build_warning(detected, expected, match,
                                 detection_confidence, uncertainty,
                                 cnn_prediction["confidence"])

        return {
            "status":              status,
            "detected_material":   detected,
            "expected_material":   expected,
            "detection_confidence": round(detection_confidence, 3),
            "uncertainty":         uncertainty,
            "match":               match,
            "warning":             warning,
            "hsv_stats":           hsv_stats,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _detect_material(self, image_path: str) -> tuple[str, float, str, dict]:
        """
        Multi-scale HSV metal detection.

        WHAT: Runs the same HSV pixel mask analysis at 3 crop sizes:
              crop_40%: tight central crop (design only, less background)
              crop_60%: standard crop (default in legacy code)
              crop_80%: loose crop (includes outer rim, patina)

        WHY multi-scale:
            Ancient coin photos vary in framing and lighting.
            A single 60% crop may hit a dark patina band and call the coin
            "unknown". By averaging across 3 scales:
              - Stable reading across all 3   → high confidence
              - 2 out of 3 agree              → medium confidence
              - Only 1 agrees or all unknown  → low confidence
            This prevents a single outlier crop from flipping the decision.

        Returns (metal_str, detection_confidence_0_to_1, uncertainty_str, stats_dict).
        """
        img = cv2.imread(image_path)
        if img is None:
            return "unknown", 0.0, "low", {"error": "cannot read image"}

        h, w = img.shape[:2]
        # Three crop scales: (margin_fraction)
        scales = [
            ("40pct", 0.30, 0.70),   # tight centre
            ("60pct", 0.20, 0.80),   # standard
            ("80pct", 0.10, 0.90),   # loose
        ]

        all_stats: dict = {}
        per_scale_winner: list[str] = []
        per_scale_score:  list[float] = []

        for name, margin, limit in scales:
            y0, y1 = int(h * margin), int(h * limit)
            x0, x1 = int(w * margin), int(w * limit)
            crop   = img[y0:y1, x0:x1]
            hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            total  = crop.shape[0] * crop.shape[1]

            gold_mask   = cv2.inRange(hsv,
                np.array([_METAL_THRESHOLDS["gold"]["h_min"],   _METAL_THRESHOLDS["gold"]["s_min"],   0]),
                np.array([_METAL_THRESHOLDS["gold"]["h_max"],   _METAL_THRESHOLDS["gold"]["s_max"],   255]))
            bronze_mask = cv2.inRange(hsv,
                np.array([_METAL_THRESHOLDS["bronze"]["h_min"], _METAL_THRESHOLDS["bronze"]["s_min"], 0]),
                np.array([_METAL_THRESHOLDS["bronze"]["h_max"], _METAL_THRESHOLDS["bronze"]["s_max"], 255]))
            silver_mask = cv2.inRange(hsv,
                np.array([0,   0,   80]),
                np.array([179, 40, 255]))

            pct_gold   = float(np.count_nonzero(gold_mask))   / total
            pct_bronze = float(np.count_nonzero(bronze_mask)) / total
            pct_silver = float(np.count_nonzero(silver_mask)) / total

            mean_h = float(np.mean(hsv[:, :, 0]))
            mean_s = float(np.mean(hsv[:, :, 1]))
            mean_v = float(np.mean(hsv[:, :, 2]))

            all_stats[name] = {
                "mean_hue":    round(mean_h,    1),
                "mean_sat":    round(mean_s,    1),
                "mean_val":    round(mean_v,    1),
                "pct_gold":    round(pct_gold,  3),
                "pct_bronze":  round(pct_bronze,3),
                "pct_silver":  round(pct_silver,3),
            }

            scores = {"gold": pct_gold, "bronze": pct_bronze, "silver": pct_silver}
            best   = max(scores, key=scores.get)
            if scores[best] < 0.15:
                per_scale_winner.append("unknown")
                per_scale_score.append(0.0)
            else:
                per_scale_winner.append(best)
                per_scale_score.append(scores[best])

        # ── Aggregate across scales ────────────────────────────────────────
        non_unknown = [m for m in per_scale_winner if m != "unknown"]
        if not non_unknown:
            return "unknown", 0.0, "low", all_stats

        # Majority vote
        from collections import Counter
        vote    = Counter(non_unknown)
        winner, vote_count = vote.most_common(1)[0]

        # Detection confidence = mean pixel score for the winning metal
        winning_scores = [
            s for m, s in zip(per_scale_winner, per_scale_score) if m == winner
        ]
        det_confidence = float(np.mean(winning_scores))

        # Uncertainty: based on how many scales agreed
        if vote_count == 3:
            uncertainty = "low"     # all scales agree → high certainty → low uncertainty
        elif vote_count == 2:
            uncertainty = "medium"
        else:
            uncertainty = "high"    # only 1 scale voted → unreliable

        all_stats["winner"]           = winner
        all_stats["vote_count"]       = vote_count
        all_stats["det_confidence"]   = round(det_confidence, 3)

        return winner, det_confidence, uncertainty, all_stats


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

def _build_warning(detected: str, expected: str, match: bool,
                   detection_confidence: float, uncertainty: str,
                   cnn_confidence: float) -> str:
    """
    Build a human-readable warning message from the validation result.

    WHAT: Combines material match/mismatch with the HSV detection confidence
          and the uncertainty flag into one professional sentence.

    WHY include detection_confidence in the message:
        A mismatch with det_confidence=0.92 (high certainty) is a strong
        red flag.  A mismatch with det_confidence=0.17 (barely over threshold)
        may just be a difficult photograph.  The reader needs both to decide
        whether manual review is really warranted.
    """
    if not expected or detected == "unknown":
        return (
            f"Material detection inconclusive (uncertainty: {uncertainty}). "
            "Could not compare against expected material."
        ) if detected == "unknown" else ""
    conf_note = f"(detection confidence {detection_confidence:.0%}, uncertainty: {uncertainty})"
    if match:
        return (
            f"Material consistent: image shows {detected} {conf_note}, "
            f"KB records {expected}."
        )
    return (
        f"Material mismatch {conf_note}: image appears {detected} but this "
        f"type is recorded as {expected}. "
        f"CNN confidence was {cnn_confidence:.1%}. Manual review recommended."
    )
