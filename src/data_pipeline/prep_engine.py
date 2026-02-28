"""
src/data_pipeline/prep_engine.py
==================================
Layer 0 — Preprocessing Engine

Transforms raw ancient coin photographs from the Corpus Nummorum dataset into
ML-ready images. This is the FIRST step in the entire pipeline — garbage in,
garbage out. The quality of this preprocessing directly impacts CNN accuracy.

Two operations are applied to every image, in this order:
  1. CLAHE contrast enhancement  (LAB colour space, L-channel only)
  2. Aspect-preserving resize     (longest edge = TARGET_SIZE, zero-padded to square)

The output of this module feeds directly into:
  - scripts/train.py   (training the EfficientNet-B3 model)
  - src/core/inference.py (production inference)

Both consumers call get_val_transforms() which applies the same
ImageNet normalisation that EfficientNet expects.

Usage (CLI):
    python -m src.data_pipeline.prep_engine

Usage (programmatic):
    from src.data_pipeline.prep_engine import process_image, run_pipeline
    run_pipeline(min_images=10)


Design decisions (WHY these exact parameters):
-----------------------------------------------

CLAHE clipLimit=2.0:
    CLAHE stands for Contrast-Limited Adaptive Histogram Equalization.
    It divides the image into tiles and equalises each tile's histogram.
    clipLimit controls the maximum allowed amplification in each tile.
    Too high (e.g. 8.0) → amplifies sensor noise into visible artefacts.
    Too low  (e.g. 0.5) → barely improves contrast.
    2.0 is the empirically established default for natural images; it gives
    clear enhancement of worn coin inscriptions without introducing noise.

CLAHE tileGridSize=(8,8):
    Divides the 299×299 image into an 8×8 grid = 64 tiles, each ~37×37 px.
    Smaller tiles = more localised enhancement (better for fine details).
    Larger tiles = global behaviour, loses local detail.
    8×8 is the standard for numismatic images (used in academic literature).

LAB colour space, L-channel only:
    LAB separates luminance (L) from colour (A=green-red, B=blue-yellow).
    Applying CLAHE to RGB directly would shift all three channels — this
    distorts the metal patina (the green/brown oxidation layer that proves
    archaeological authenticity). Numismatists use patina as a dating clue.
    By enhancing ONLY the L channel and leaving A and B untouched, we
    sharpen coin features without destroying colour information.

Aspect-preserving resize vs. simple resize:
    Coins are circular objects. A simple cv2.resize(img, (299, 299)) would
    stretch the coin into an ellipse if the source was not already square.
    EfficientNet-B3 learns from SHAPE — a stretched coin looks different from
    a round coin even if the content is identical. Stretching would silently
    degrade model accuracy by introducing geometric distortion that does not
    exist in the real world.

    Correct approach: scale so the LONGEST edge = TARGET_SIZE, then pad the
    shorter edge symmetrically with black pixels to reach a square.

    Interpolation selection:
      - INTER_AREA  (downscaling): reduces moiré pattern artefacts
      - INTER_CUBIC (upscaling):   smooth interpolation, no pixelation

TARGET_SIZE = 299:
    EfficientNet-B3 expects 299×299 input. This is hardcoded in the compound
    scaling formula. Any other size would require re-designing the model.

min_images threshold = 10:
    After applying this filter, the 9,716 CN type folders reduce to 438
    classes. Below 10 images, transfer learning cannot generalise — the model
    memorises. With 10+, EfficientNet-B3's ImageNet pretrained features are
    sufficient to learn class-specific patterns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
# WHY __name__ ("src.data_pipeline.prep_engine"):
#   Every log line shows which module it came from. When running a pipeline
#   with many steps, you can filter with grep or logging level.


# ── Pipeline constants ─────────────────────────────────────────────────────────
#
# WHY constants at module level, not hardcoded in function bodies:
#   - Searchable: one `grep TARGET_SIZE` finds every place the number is used
#   - Testable: tests can monkeypatch these names
#   - Self-documenting: the name explains the intent, the value explains the detail
#
_ROOT        = Path(__file__).resolve().parent.parent.parent   # project root
RAW_PATH     = _ROOT / "data" / "raw" / "CN_dataset_v1" / "dataset_types"
PROCESSED_PATH = _ROOT / "data" / "processed"
TARGET_SIZE  = 299      # EfficientNet-B3 required input size (pixels)
CLAHE_CLIP   = 2.0      # CLAHE amplification limit (see module docstring)
CLAHE_TILE   = (8, 8)   # CLAHE tile grid (8×8 = 64 tiles over 299×299)
VALID_EXTS   = {".jpg", ".jpeg", ".png"}   # extensions accepted by OpenCV


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — CLAHE contrast enhancement
# ══════════════════════════════════════════════════════════════════════════════

def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L channel of the LAB representation.

    WHAT:
        1. Convert BGR → LAB   (separates luminance from colour)
        2. Apply CLAHE to L    (boost contrast without touching A, B)
        3. Merge L, A, B back
        4. Convert LAB → BGR

    WHY LAB and not BGR/RGB:
        CLAHE on RGB shifts all three colour channels simultaneously,
        which destroys the metal patina (the oxidation colours that are
        archaeological evidence). LAB isolates the brightness component,
        so we sharpen the coin's features while leaving colour intact.

    Args:
        img_bgr: uint8 BGR image from cv2.imread()

    Returns:
        Contrast-enhanced uint8 BGR image of the same size.
    """
    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE object is cheap to create; creating per-image avoids
    # state contamination between different preprocessing runs.
    clahe  = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_eq   = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Aspect-preserving resize with zero padding
# ══════════════════════════════════════════════════════════════════════════════

def _resize_and_pad(img_bgr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Resize the longest edge to `size` and zero-pad to a `size × size` square.

    WHAT:
        1. Determine which edge (height or width) is longer
        2. Scale the image so that long edge = `size`
           - preserves aspect ratio (no stretching)
        3. Pad the short edge symmetrically with black (0, 0, 0)
        4. Result is always exactly `size × size`

    WHY not simple cv2.resize(img, (size, size)):
        That would stretch a tall coin into a wide square, or vice versa.
        EfficientNet-B3 learns that coins are ROUND. A stretched ellipse
        activates completely different learned features, silently degrading
        classification accuracy.

    Interpolation selection:
        INTER_AREA  — downscaling: averages neighbouring pixels → no moiré
        INTER_CUBIC — upscaling:   bicubic smooth interpolation → no jagged edges

    Args:
        img_bgr: Input image (any size, any aspect ratio)
        size:    Target square side length (default: TARGET_SIZE = 299)

    Returns:
        uint8 BGR image of shape (size, size, 3).
    """
    h, w = img_bgr.shape[:2]

    # Choose interpolation based on whether we're downscaling or upscaling
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC

    aspect = w / h

    if aspect > 1:
        # Wider than tall → constrain width
        new_w  = size
        new_h  = int(np.round(new_w / aspect))
        pad_h  = size - new_h
        pad_top, pad_bot = pad_h // 2, pad_h - pad_h // 2   # integer floor/ceil
        pad_left, pad_right = 0, 0

    elif aspect < 1:
        # Taller than wide → constrain height
        new_h  = size
        new_w  = int(np.round(new_h * aspect))
        pad_w  = size - new_w
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bot = 0, 0

    else:
        # Already square
        new_h, new_w = size, size
        pad_top = pad_bot = pad_left = pad_right = 0

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)

    # Symmetric zero-padding (black pixels, [0, 0, 0] in BGR)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def process_image(
    img_path:    str | Path,
    output_path: str | Path,
    size:        int = TARGET_SIZE,
) -> bool:
    """
    Apply the full preprocessing pipeline to a single coin image.

    Steps applied:
        1. Load image with OpenCV
        2. CLAHE contrast enhancement (LAB L-channel)
        3. Aspect-preserving resize + zero padding to `size × size`
        4. Save to output_path as JPEG

    Args:
        img_path:    Source image path (JPEG or PNG)
        output_path: Destination path (will be created by run_pipeline)
        size:        Output square size in pixels (default: 299)

    Returns:
        True  — image processed and saved successfully
        False — image could not be loaded (corrupt file, wrong format)

    Why return bool instead of raising:
        During bulk pipeline runs, a handful of corrupt files are expected.
        Raising would stop the entire pipeline. Returning False lets
        run_pipeline() skip the bad file and report it in the log summary.
    """
    img_path    = Path(img_path)
    output_path = Path(output_path)

    img = cv2.imread(str(img_path))
    if img is None:
        # Covers: file does not exist, corrupt binary, unsupported format
        logger.warning("Could not read image (skipping): %s", img_path.name)
        return False

    img = _apply_clahe(img)
    img = _resize_and_pad(img, size=size)

    # imwrite returns False if the path is unwritable; treat as failure
    success = cv2.imwrite(str(output_path), img)
    if not success:
        logger.error("cv2.imwrite failed: %s", output_path)
        return False

    return True


def run_pipeline(
    min_images:   int  = 10,
    raw_dir:      Path = RAW_PATH,
    output_dir:   Path = PROCESSED_PATH,
    size:         int  = TARGET_SIZE,
    dry_run:      bool = False,
) -> dict:
    """
    Batch-process the entire Corpus Nummorum raw dataset.

    WHY it lives here and not in a script:
        Scripts should be thin shells. The logic belongs in the module so it
        can be imported, tested, and called programmatically. The
        `if __name__ == "__main__"` block at the bottom calls this function
        with default arguments — it is the CLI entry point.

    Algorithm:
        For each coin_type folder in raw_dir:
            1. Count the images in the folder
            2. If count < min_images → skip (not enough data to train from)
               WHY 10: below this threshold, transfer learning memorises rather
               than generalises. This threshold filters 9,716 → 438 classes.
            3. For each image in the folder:
               a. Skip non-image files (hidden files, .DS_Store, etc.)
               b. Call process_image(src, dst)
               c. Count success / failure

        Skipped classes are logged at DEBUG level (not WARNING) because the
        majority of CN types have <10 images — this is EXPECTED behaviour, not
        an error condition. Engineers can turn DEBUG on to see the skip list.

    Args:
        min_images:  Minimum images per class to include (default: 10)
        raw_dir:     Path to the raw dataset directory
        output_dir:  Where to write processed images
        size:        Output image size (default: 299 for EfficientNet-B3)
        dry_run:     If True, count images without writing anything.
                     Useful to audit the dataset before a full 103-minute run.

    Returns:
        Summary statistics dict:
            {
              "classes_processed": int,   # classes with >= min_images
              "classes_skipped":   int,   # classes below threshold
              "images_processed":  int,   # successfully written
              "images_failed":     int,   # corrupt / unreadable files
            }
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_dir}\n"
            "Expected: data/raw/CN_dataset_v1/dataset_types/"
        )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    classes_all    = sorted(raw_dir.iterdir())   # sorted → deterministic
    classes_kept   = [d for d in classes_all if d.is_dir()]

    classes_processed = 0
    classes_skipped   = 0
    images_processed  = 0
    images_failed     = 0

    logger.info(
        "Preprocessing pipeline starting | raw_dir=%s | min_images=%d | dry_run=%s",
        raw_dir, min_images, dry_run,
    )

    for class_dir in tqdm(classes_kept, desc="Preprocessing", unit="class"):
        # Collect all supported image files in this class folder
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in VALID_EXTS
        ]

        if len(image_files) < min_images:
            # Expected for most of the 9,716 CN types — not an error
            logger.debug(
                "Skipping class %s: only %d image(s) (threshold=%d)",
                class_dir.name, len(image_files), min_images,
            )
            classes_skipped += 1
            continue

        # This class passes the threshold — process it
        target_dir = output_dir / class_dir.name
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            dst = target_dir / img_file.name
            if dry_run:
                images_processed += 1   # count without writing
                continue
            ok = process_image(img_file, dst, size=size)
            if ok:
                images_processed += 1
            else:
                images_failed += 1

        classes_processed += 1

    summary = {
        "classes_processed": classes_processed,
        "classes_skipped":   classes_skipped,
        "images_processed":  images_processed,
        "images_failed":     images_failed,
    }
    logger.info(
        "Pipeline complete: classes_processed=%d  skipped=%d  "
        "images_processed=%d  failed=%d",
        classes_processed, classes_skipped, images_processed, images_failed,
    )
    return summary


# ── CLI entry point ────────────────────────────────────────────────────────────
#
# WHY __main__ block at file bottom:
#   Keeps the module importable without side effects.
#   `python -m src.data_pipeline.prep_engine` triggers this block.
#   Calling `from src.data_pipeline.prep_engine import process_image`
#   does NOT trigger this block — the module is importable cleanly.
#
if __name__ == "__main__":
    import sys

    # Minimal CLI logging so the terminal shows progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run_pipeline(min_images=10)
    print(
        f"\nDone.\n"
        f"  Classes kept    : {summary['classes_processed']}\n"
        f"  Classes skipped : {summary['classes_skipped']}\n"
        f"  Images written  : {summary['images_processed']}\n"
        f"  Images failed   : {summary['images_failed']}\n"
    )
    sys.exit(0 if summary["images_failed"] == 0 else 1)