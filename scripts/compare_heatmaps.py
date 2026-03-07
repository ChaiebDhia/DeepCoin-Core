"""
scripts/compare_heatmaps.py
============================
Generates a 3-panel Grad-CAM++ comparison figure for jury presentation.

Panel layout:
  Left   — HIGH CONFIDENCE (>90%): expert case — model should look at coin features
  Centre — LOW CONFIDENCE, in-distribution (BNF 1966 scan of trained class): uncertain case
  Right  — OUT-OF-DISTRIBUTION (type not in training set): stranger case

WHY this matters for the jury / encadrant
-----------------------------------------
This single figure answers the key PFE evaluation question:
"How do we know the model learned the right features and not shortcuts?"

If the left panel shows heatmap on faces/legends and the right panel shows heatmap
on the rim/background, the model is HEALTHY — it uses real numismatic features for
confident predictions and honestly signals uncertainty on unknown inputs by attending
to the safest shared feature (the circular rim).

If even the left panel shows rim attention → shortcut learning → must retrain.

USAGE
-----
    python scripts/compare_heatmaps.py

Output:
    reports/heatmap_comparison.png   (1500×550 px, 150 dpi — journal quality)

The script auto-selects the 3 representative images; override via CLI flags if
you want to test specific coins:
    --high   path/to/high_conf.jpg
    --low    path/to/low_conf.jpg
    --ood    path/to/ood_coin.jpg
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.inference import CoinInference


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT IMAGE PATHS
# These were chosen from diagnostic runs (scripts/_debug_confidence.py).
# ─────────────────────────────────────────────────────────────────────────────

# HIGH confidence — standard _p composite photograph (80-97% on these)
_DEFAULT_HIGH = "data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg"

# LOW confidence, IN-DISTRIBUTION — BNF 1966.453 catalog scan of a TRAINED type.
# CN 220 is in the training set (11 images) but this specific image is a 1966
# Bibliothèque nationale de France archive scan → model scores it 28%.
_DEFAULT_LOW  = "data/processed/220/CN_type_220_BNF_1966.453_cn_coin_12683_o.jpg"

# OUT-OF-DISTRIBUTION — a coin type NOT in the 438-class training set.
# CN 10111 is in the raw dataset but was excluded (fewer than 10 training images).
# Uses the MK_ museum photo style (similar to standard composite) so any
# rim attention is purely from model uncertainty, not photograph style.
_DEFAULT_OOD  = "data/raw/CN_dataset_v1/dataset_types/10111/CN_type_10111_MK_18249936_cn_coin_6792_o.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_bgr_resized(path: str, size: int = 299) -> np.ndarray:
    """
    Load an image at its ORIGINAL (pre-CLAHE) scale for display.
    We load it through OpenCV and resize to (size × size) for uniform panels.
    NOT passed through CoinInference._load_image() — we call that internally
    via predict() for the actual inference.  This copy is display-only.

    WHY display at 299×299:
    - inference.py internally preprocesses to 299×299 anyway
    - using the same size means the Grad-CAM overlay aligns perfectly with
      the displayed coin image (no geometric scaling artefacts)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    # Aspect-preserving resize with black padding — same as prep_engine.py
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _read_heatmap_png(png_path: str) -> np.ndarray:
    """
    Read the saved Grad-CAM++ heatmap PNG and return as RGB numpy array.
    The PNG was saved by generate_gradcam() at 299×299.
    """
    img = cv2.imread(png_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load heatmap PNG: {png_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_PANEL_LABELS = {
    "high": ("HIGH CONFIDENCE", "#10b981"),   # green
    "low":  ("LOW CONFIDENCE\n(IN TRAINING SET)", "#f59e0b"),  # amber
    "ood":  ("OUT-OF-DISTRIBUTION\n(NOT IN TRAINING)", "#8b5cf6"),  # purple
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(high_path: str, low_path: str, ood_path: str, out_path: str) -> None:
    """
    Run inference + Grad-CAM++ for the three images and produce the comparison
    figure.

    Parameters
    ----------
    high_path : absolute or relative path to the high-confidence test image
    low_path  : path to the low-confidence but in-distribution image
    ood_path  : path to the out-of-distribution image
    out_path  : where to save the final PNG comparison figure
    """
    import torch as _torch
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"Loading CoinInference on {_device}…")
    ci = CoinInference(device=_device)

    samples = [
        ("high", high_path),
        ("low",  low_path),
        ("ood",  ood_path),
    ]

    results   = {}
    heatmaps  = {}
    originals = {}

    for key, path in samples:
        print(f"\n[{key.upper()}] {path}")
        r = ci.predict(path, tta=False, gradcam=True)
        results[key]   = r
        originals[key] = _load_bgr_resized(path)
        if r.get("gradcam_path") and os.path.exists(r["gradcam_path"]):
            heatmaps[key] = _read_heatmap_png(r["gradcam_path"])
        else:
            print(f"  WARNING: no heatmap generated for {key}")
            heatmaps[key] = originals[key].copy()

        trained = key != "ood"
        print(f"  predicted : {r['label']} | conf = {r['confidence']:.1%}")
        print(f"  in training set: {trained}")
        if r.get("gradcam_path"):
            print(f"  heatmap   : {r['gradcam_path']}")

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("#0f172a")   # dark navy background (brand colour)

    col_order = ["high", "low", "ood"]

    for col_idx, key in enumerate(col_order):
        r      = results[key]
        label, color = _PANEL_LABELS[key]
        conf   = r["confidence"]
        pred   = r["label"]
        trained_text = "✔ In training set" if key != "ood" else "✘ Not in training set"

        # ── TOP ROW: original photo ──────────────────────────────────────────
        ax_orig = axes[0, col_idx]
        ax_orig.imshow(originals[key])
        ax_orig.set_facecolor("#0f172a")
        ax_orig.set_xticks([]); ax_orig.set_yticks([])
        for spine in ax_orig.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax_orig.set_title("Original Photograph", color="#94a3b8", fontsize=9,
                           pad=4, fontfamily="monospace")

        # ── BOTTOM ROW: heatmap overlay ──────────────────────────────────────
        ax_heat = axes[1, col_idx]
        ax_heat.imshow(heatmaps[key])
        ax_heat.set_facecolor("#0f172a")
        ax_heat.set_xticks([]); ax_heat.set_yticks([])
        for spine in ax_heat.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax_heat.set_title("Grad-CAM++ Heatmap", color="#94a3b8", fontsize=9,
                           pad=4, fontfamily="monospace")

        # ── Per-column annotation box ────────────────────────────────────────
        ax_orig.text(
            0.5, 1.22,
            label,
            transform=ax_orig.transAxes,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color,
            fontfamily="monospace",
        )
        ax_orig.text(
            0.5, 1.10,
            f"Predicted: CN {pred}  |  Confidence: {conf:.1%}",
            transform=ax_orig.transAxes,
            ha="center", va="bottom",
            fontsize=9, color="#e2e8f0",
        )
        ax_orig.text(
            0.5, 1.01,
            trained_text,
            transform=ax_orig.transAxes,
            ha="center", va="bottom",
            fontsize=8, color=color,
        )

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "DeepCoin · Grad-CAM++ Comparison: Expert vs Uncertain vs Stranger",
        color="#f8fafc", fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Colour scale legend (blue=cold → red=hot) ────────────────────────────
    cbar_ax = fig.add_axes([0.35, 0.01, 0.30, 0.015])
    import matplotlib.colorbar as mcbar
    import matplotlib.cm as cm
    cb = mcbar.ColorbarBase(
        cbar_ax, cmap=cm.jet, orientation="horizontal",
        norm=plt.Normalize(0, 1),
    )
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low attention (blue)", "Medium", "High attention (red)"])
    cbar_ax.tick_params(colors="#94a3b8", labelsize=7)
    cb.outline.set_edgecolor("#475569")

    plt.subplots_adjust(
        top=0.88, bottom=0.06, left=0.02, right=0.98,
        hspace=0.08, wspace=0.06,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n✔  Comparison figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-panel Grad-CAM++ comparison: high / low / OOD"
    )
    parser.add_argument("--high", default=_DEFAULT_HIGH)
    parser.add_argument("--low",  default=_DEFAULT_LOW)
    parser.add_argument("--ood",  default=_DEFAULT_OOD)
    parser.add_argument("--out",  default="reports/heatmap_comparison.png")
    args = parser.parse_args()

    for attr in ("high", "low", "ood"):
        p = getattr(args, attr)
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found — {p}")

    run(args.high, args.low, args.ood, args.out)
