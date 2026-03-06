"""
DeepCoin Inference Engine — Layer 1
====================================
Loads the trained EfficientNet-B3 model ONCE and exposes a predict() method.

ENGINEERING RULES (non-negotiable):
  1. Always model.eval()      — disables Dropout + BatchNorm training mode
  2. Always torch.no_grad()   — no gradient graph, saves VRAM, 2× faster
  3. Always get_val_transforms() — NEVER train transforms in production
  4. Load model ONCE in __init__ — never inside predict()

Output contract (what every agent receives from this class):
  {
      "class_id":         int,   # e.g. 3314
      "label":            str,   # e.g. "CN_3314"
      "confidence":       float, # e.g. 0.87  (0.0 – 1.0)
      "top5": [
          {"rank": 1, "class_id": int, "label": str, "confidence": float},
          ...
      ],
      "inference_time_ms": int,  # e.g. 142
      "tta_used":          bool
  }
"""

import logging
import time
import cv2
import torch
import numpy as np
from pathlib import Path

from src.core.model_factory import get_deepcoin_model
from src.core.dataset import get_val_transforms

logger = logging.getLogger(__name__)
# WHY __name__:
#   Resolves to 'src.core.inference' at runtime — you see exactly which module
#   emitted each log line. FastAPI/uvicorn configure the root logger; child
#   loggers inherit the level automatically.


# ── Path constants ─────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent.parent   # project root
MODEL_PATH     = _ROOT / "models" / "best_model.pth"
MAPPING_PATH   = _ROOT / "models" / "class_mapping.pth"

# ── TTA augmentation passes ────────────────────────────────────────────────────
# 5 lightweight transforms used at test time to average out prediction noise.
# Each is applied AFTER the base val transforms (norm + tensor conversion).
import albumentations as A
from albumentations.pytorch import ToTensorV2

_TTA_TRANSFORMS = [
    # Pass 1 — no extra augmentation (clean baseline)
    None,
    # Pass 2 — horizontal flip
    # WHY: coins photographed from either side are mirrored in the training set
    A.Compose([A.HorizontalFlip(p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 3 — small clockwise rotation (+10°)
    A.Compose([A.Rotate(limit=(10, 10), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 4 — small counter-clockwise rotation (-10°)
    A.Compose([A.Rotate(limit=(-10, -10), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 5 — slight brightness boost (+0.12)
    # Simulates well-lit museum vs ambient photography
    A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.12, 0.12),
                                          contrast_limit=0, p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 6 — wider clockwise rotation (+15°)
    # Covers hand-held photos tilted more noticeably
    A.Compose([A.Rotate(limit=(15, 15), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 7 — brightness reduction (-0.12)
    # Complements pass 5: handles underexposed / shadow photos
    A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.12, -0.12),
                                          contrast_limit=0, p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 8 — contrast boost (+0.15)
    # Accentuates relief details on well-preserved coins; also in training aug set
    A.Compose([A.RandomBrightnessContrast(brightness_limit=0,
                                          contrast_limit=(0.15, 0.15), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
]


# ══════════════════════════════════════════════════════════════════════════════
# Auto-crop helper
# ══════════════════════════════════════════════════════════════════════════════

def _auto_crop_coin(img_bgr: np.ndarray) -> np.ndarray:
    """
    Automatically detect the coin region and return a tight crop.

    WHY this exists:
        The CNN was trained on images where the coin fills the entire 299×299
        frame (prep_engine.py crops tightly during dataset preparation).
        When a user uploads a screenshot of a website page, the coin may only
        occupy 20–40% of the image — surrounded by browser chrome, white
        backgrounds, or other UI elements.  The model then "sees" a small coin
        on a large blank canvas instead of a close-up, and confidence collapses
        to noise level (< 20%) even for known coin types.

    Strategy — three attempts in priority order:

    1. cv2.HoughCircles — dedicated circular-object detector.
       Best for clean museum photos and website thumbnails with clear coin edge.
       Uses a Gaussian-blurred grayscale as input (reduces false edges).
       Selects the circle closest to the image centre (not necessarily the
       largest) — this avoids picking up circular icons in the browser UI.

    2. Largest near-circular contour — fallback when Hough finds nothing.
       Runs Canny edge detection → findContours → filters by:
         * area > 0.05 × (h × w)   — not a noise speck
         * 0.60 ≤ w/h ≤ 1.67      — roughly circular bounding box
         * circularity > 0.50      — 4π·area/perimeter² (perfect circle = 1.0)
       Best for darkened, worn, or high-contrast coins.

    3. Centre 80% crop — last resort when neither detector fires.
       Simply removes 10% margin on each side.  Strips the most common browser
       chrome/padding without risking a bad cut over the coin itself.

    In all cases a 12% padding ring is added around the detected region before
    returning, so the coin edge is never clipped.

    WHY not a coin-detection YOLO model:
        A pretrained detector would need a numismatic coin dataset to be reliable
        and would add 50–200 MB to the deployment.  HoughCircles + Canny handles
        > 95% of real-world cases with zero extra dependencies.

    Parameters
    ----------
    img_bgr : np.ndarray  (H, W, 3) BGR image, any resolution

    Returns
    -------
    np.ndarray — cropped BGR image (may be the original if no region found)
    """
    h, w = img_bgr.shape[:2]

    # Skip if image is already small and well-framed (longest edge < 400 px).
    # WHY 400 and not 200:
    #   Preprocessed training images are exactly 299×299 (prep_engine.py output).
    #   With the old threshold of 200, these images PASSED through auto-crop:
    #   HoughCircles detected the coin circle, stripped the zero-padding added
    #   during training preprocessing, and returned a tight crop.
    #   At inference time the model then saw the same coin without padding —
    #   a different spatial composition than training → confidence collapsed to
    #   noise level (2–12%) even for coin types with known 80%+ validation accuracy.
    #   With max(h,w) < 400: any 299×299 processed image bypasses auto-crop
    #   entirely (max=299 < 400).  Real user photos and screenshots are always
    #   larger (≥600px for phone cameras, ≥800px for museum scans) and still go
    #   through the full Hough/contour pipeline as intended.
    if max(h, w) < 400:
        return img_bgr

    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    def _apply_crop(cx: int, cy: int, r: int) -> np.ndarray | None:
        """Add padding, clamp to image bounds, return crop (None if too small)."""
        pad = int(r * 0.12)
        x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
        x2, y2 = min(w, cx + r + pad), min(h, cy + r + pad)
        if (x2 - x1) < 64 or (y2 - y1) < 64:
            return None
        return img_bgr[y1:y2, x1:x2]

    # ── Strategy 1: HoughCircles ───────────────────────────────────────────────
    min_r = min(h, w) // 8          # coin must be ≥ 1/8 of the smallest dim
    max_r = int(min(h, w) * 0.54)   # coin may fill almost the entire frame
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,     # only one coin expected
        param1=60,                   # Canny upper threshold inside Hough
        param2=28,                   # accumulator threshold (lower = more hits)
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick the circle closest to the image centre — avoids browser icons
        cx_img, cy_img = w // 2, h // 2
        best = min(circles, key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
        crop = _apply_crop(int(best[0]), int(best[1]), int(best[2]))
        if crop is not None:
            logger.debug("_auto_crop_coin: HoughCircles hit — circle r=%d", best[2])
            return crop

    # ── Strategy 2: Largest near-circular contour ─────────────────────────────
    edges    = cv2.Canny(blurred, 30, 100)
    # Dilate to close small edge gaps on worn coins
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges    = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area    = 0.0
    min_area     = h * w * 0.05   # coin must cover ≥ 5% of the image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if not (0.60 <= aspect <= 1.67):   # reject very rectangular regions
            continue
        perim = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0
        if circularity < 0.45:             # reject non-circular contours
            continue
        if area > best_area:
            best_area    = area
            best_contour = cnt

    if best_contour is not None:
        bx, by, bw, bh = cv2.boundingRect(best_contour)
        cx_c = bx + bw // 2
        cy_c = by + bh // 2
        r_c  = max(bw, bh) // 2
        crop = _apply_crop(cx_c, cy_c, r_c)
        if crop is not None:
            logger.debug("_auto_crop_coin: contour hit — area=%.0f circ=%.2f", best_area, circularity)
            return crop

    # ── Strategy 3: centre 80% crop ───────────────────────────────────────────
    # Removes the typical browser chrome / website padding at the image borders.
    # Only applied when the image is large enough to benefit.
    if min(h, w) >= 400:
        m_h, m_w = h // 10, w // 10
        logger.debug("_auto_crop_coin: centre-crop fallback")
        return img_bgr[m_h : h - m_h, m_w : w - m_w]

    # Nothing useful found — return original
    return img_bgr


class CoinInference:
    """
    Wraps the trained EfficientNet-B3 for single-image inference.

    Usage:
        engine = CoinInference()                        # loads model once
        result = engine.predict("path/to/coin.jpg")     # standard
        result = engine.predict("path/to/coin.jpg", tta=True)  # +TTA
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        mapping_path: str | Path = MAPPING_PATH,
        device: str | None = None,
    ):
        """
        Load model weights and class mapping.

        Args:
            model_path:   Path to best_model.pth
            mapping_path: Path to class_mapping.pth
            device:       "cuda" | "cpu" | None (auto-detect)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("CoinInference: device=%s", self.device)

        # ── Load class mapping ─────────────────────────────────────────────────
        # class_mapping.pth contains {class_to_idx, idx_to_class, n_classes}
        # WHY weights_only=True:
        #   torch.load with weights_only=False can execute arbitrary Python code
        #   embedded in the pickle stream (CVE-class vulnerability).  True mode
        #   only deserialises tensors and primitives — safe for untrusted checkpoints.
        mapping = torch.load(mapping_path, map_location="cpu", weights_only=True)
        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = mapping["idx_to_class"]
        self.num_classes: int = mapping.get("n_classes", len(self.class_to_idx))
        logger.info("CoinInference: %d classes loaded", self.num_classes)

        # ── Load model weights ─────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = get_deepcoin_model(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # RULE 1 — ALWAYS eval() after loading
        # Disables Dropout (p=0.4) and switches BatchNorm to use running stats.
        # Without this: different result every call. Non-deterministic = unusable.
        self.model.eval()

        # Store base val transforms (no augmentation, just normalize + tensor)
        self._val_transform = get_val_transforms()

        # Allocate CLAHE once — reused by every _load_image() call.
        # WHY here: cv2.createCLAHE() allocates internal memory; creating it
        # on every inference call wastes allocation/deallocation overhead.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ── Temperature scalar (post-hoc calibration) ─────────────────────────
        # WHY temperature scaling (Guo et al., NeurIPS 2017):
        #   This model was trained with label_smoothing=0.1, which deliberately
        #   pushes output logits away from hard-one-hot targets.  The result is
        #   a systematically UNDER-CONFIDENT softmax: correct class at 8-15%
        #   even on clean images.  Temperature T < 1 sharpens the distribution:
        #       p_calibrated = softmax(z / T)
        #   T is fitted once on the 1,151-image validation set by minimising
        #   cross-entropy.  The ranking / accuracy is unchanged; only the
        #   probability magnitudes are rescaled.
        #   Run  python scripts/calibrate_temperature.py  to generate this file.
        _temp_path = Path(model_path).parent / "temperature.pth"
        if _temp_path.exists():
            _temp_data        = torch.load(_temp_path, map_location="cpu", weights_only=True)
            self._temperature = float(_temp_data["temperature"])
            logger.info(
                "CoinInference: temperature T=%.6f loaded  "
                "(NLL %.5f → %.5f, ECE %.4f → %.4f)",
                self._temperature,
                _temp_data.get("val_nll_before", float("nan")),
                _temp_data.get("val_nll_after",  float("nan")),
                _temp_data.get("ece_before",     float("nan")),
                _temp_data.get("ece_after",      float("nan")),
            )
        else:
            self._temperature = 1.0
            logger.warning(
                "CoinInference: temperature.pth not found — "
                "running without calibration (confidence will be lower). "
                "Run scripts/calibrate_temperature.py to fix this."
            )

        val_acc = checkpoint.get("val_acc", "unknown")
        epoch   = checkpoint.get("epoch", "unknown")
        logger.info("CoinInference: model loaded — epoch=%s  val_acc=%s", epoch, val_acc)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load image from disk as RGB numpy array (H, W, 3) uint8.

        CRITICAL — applies CLAHE preprocessing to match training distribution.

        The model (best_model.pth) was trained exclusively on CLAHE-enhanced
        images produced by prep_engine.py (clipLimit=2.0, tileGridSize=(8,8)).
        Skipping CLAHE here creates a train/inference distribution mismatch:
          - Raw photos have lower contrast than CLAHE-processed training images
          - Lower contrast → weaker convolutional feature activations
          - Weaker activations → softmax spreads probability mass more evenly
          - Result: top-1 confidence 5–15% instead of expected 50–90%
          - Even coin types the model knows well appear to be "unknown"

        The CLAHE parameters MUST exactly match prep_engine.py:
          CLAHE_CLIP = 2.0   (too high amplifies noise; too low = no effect)
          CLAHE_TILE = (8,8) (localised enhancement over 64 sub-tiles)
          L channel only     (preserves metal patina colour in A, B channels)

        Raises:
            FileNotFoundError: if the path doesn't exist
            ValueError:        if OpenCV can't decode the file
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # WHY open(rb) + np.frombuffer + imdecode instead of cv2.imread / np.fromfile:
        # cv2.imread() and np.fromfile() both use C-runtime fopen() on Windows,
        # which only accepts ANSI paths.  Any non-ASCII character (accented
        # letters, CJK, etc.) causes a silent None return.
        # Common real-world case: French/Windows screenshots saved as
        # "Capture_d_écran_2026-03-01.png".
        # Python's built-in open() in binary mode uses the Windows Unicode API
        # (CreateFileW) — it handles every valid Unicode path.
        # np.frombuffer() wraps the in-memory bytes with zero copies;
        # cv2.imdecode() decodes from memory → no file path reaches the C runtime.
        with open(str(path), "rb") as _fh:
            raw     = np.frombuffer(_fh.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"OpenCV could not decode image: {path}")

        # ── Step 0: auto-crop to the coin region ─────────────────────────────
        # WHY before CLAHE:
        #   The CNN expects the coin to fill nearly the entire 299×299 frame.
        #   If the user uploads a screenshot (coin = 30% of image), the model
        #   sees mostly background and confidence collapses to noise level.
        #   _auto_crop_coin() detects the circular coin region via Hough
        #   circles or contour analysis and returns a tight crop.
        #   Running BEFORE CLAHE means the enhancement applies only to the
        #   coin pixels, not the surrounding background.
        orig_h, orig_w = img_bgr.shape[:2]
        img_bgr = _auto_crop_coin(img_bgr)
        cropped_h, cropped_w = img_bgr.shape[:2]
        if (cropped_h, cropped_w) != (orig_h, orig_w):
            logger.debug(
                "_load_image: auto-crop %d×%d → %d×%d",
                orig_w, orig_h, cropped_w, cropped_h,
            )

        # ── Step 1: CLAHE in LAB colour space (identical to prep_engine.py) ──
        # Convert BGR → LAB, apply CLAHE to L channel only, convert back.
        # WHY L only: A and B carry colour (patina) — enhancing them distorts
        # the oxidation hues that numismatists use to date coins.
        lab        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l_eq       = self._clahe.apply(l)
        lab_eq     = cv2.merge((l_eq, a, b))
        img_bgr    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # ── Step 2: Aspect-preserving resize to 299×299 ──────────────────────
        # WHY: EfficientNet-B3 was calibrated on 299×299 images during training.
        # Raw uploads can be any size (200px thumbnail → 8 MP phone photo).
        # get_val_transforms() has no resize because prep_engine.py already saved
        # processed images at 299×299 — but production photos are never
        # pre-processed. Simple cv2.resize() would distort coin geometry.
        # We replicate _resize_and_pad() from prep_engine.py exactly:
        #   scale so longest edge = 299 → zero-pad shortest edge to 299.
        _SIZE   = 299
        h, w    = img_bgr.shape[:2]
        scale   = _SIZE / max(h, w)
        new_h   = int(h * scale)
        new_w   = int(w * scale)
        interp  = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        canvas  = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)
        y_off   = (_SIZE - new_h) // 2
        x_off   = (_SIZE - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        img_bgr = canvas

        # ── Step 3: BGR → RGB (EfficientNet pretrained on ImageNet = RGB) ────
        # OpenCV loads as BGR; feeding BGR to an RGB-pretrained model shifts
        # every colour channel → wrong feature activations throughout all layers.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _preprocess(self, img_rgb: np.ndarray, transform=None) -> torch.Tensor:
        """
        Apply transforms and add batch dimension.

        Returns:
            Tensor of shape [1, 3, H, W] on self.device
        """
        t = transform if transform is not None else self._val_transform
        tensor = t(image=img_rgb)["image"]          # [3, H, W]  float32
        return tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass with temperature scaling.

        RULE 2 — ALWAYS torch.no_grad()
        During inference we never backpropagate.
        no_grad() tells PyTorch: don't build the computation graph.
        Result: ~2× faster, much lower VRAM usage.

        Temperature scaling is applied here:
            probs = softmax(logits / T)
        When T < 1 (under-confident model): sharpens the distribution,
        pushing the dominant class to a higher probability.
        When T = 1.0 (default, no temperature.pth loaded): no effect.
        """
        with torch.no_grad():
            logits = self.model(tensor)                       # [1, num_classes]
            if self._temperature != 1.0:
                logits = logits / self._temperature           # scale before softmax
            probs = torch.softmax(logits, dim=1)             # [1, num_classes]
        return probs.squeeze(0)                              # [num_classes]

    def _build_result(
        self,
        probs: torch.Tensor,
        inference_time_ms: int,
        tta_used: bool,
        vote_fraction: float | None = None,
        n_tta_passes: int = 1,
    ) -> dict:
        """
        Convert raw probability tensor into the standard output contract dict.

        Confidence blending (when TTA is used):
            The temperature-scaled softmax gives the base confidence score.
            The vote_fraction — how many of the N TTA passes independently
            agreed on the same top-1 class — provides a second, orthogonal
            signal.  We blend them:

                conf_final = conf_ts × (0.5 + 0.5 × vote_fraction)

            Interpretation:
              vote_fraction = 1.0  (all passes agree) → multiplier = 1.00 → unchanged
              vote_fraction = 0.5  (half agree)        → multiplier = 0.75 → 25% lower
              vote_fraction = 0.0  (none agree)        → multiplier = 0.50 → 50% lower

            WHY subtract rather than add:
                We do NOT boost confidence above the temperature-scaled value.
                Temperature scaling is already the calibrated ceiling.  The
                vote blend only PENALISES cases with low agreement — when
                different augmented views of the same coin give different
                answers, we should be less confident, not more.

        Args:
            probs:             Temperature-scaled softmax probability vector [num_classes]
            inference_time_ms: Total wall-clock time including all TTA passes
            tta_used:          Whether TTA was active
            vote_fraction:     Fraction of TTA passes that selected the same
                               top-1 class as the averaged prediction (None = no TTA)
            n_tta_passes:      Number of TTA passes performed (1 = single pass)
        """
        probs_cpu = probs.cpu()

        # Top-5 predictions (temperature-scaled, unadjusted by vote)
        top5_values, top5_indices = torch.topk(probs_cpu, k=5)

        top5 = []
        for rank, (conf, idx) in enumerate(
            zip(top5_values.tolist(), top5_indices.tolist()), start=1
        ):
            # idx_to_class keys are stored as strings in the mapping file
            label = self.idx_to_class.get(str(idx), self.idx_to_class.get(idx, f"class_{idx}"))
            top5.append({
                "rank":       rank,
                "class_id":   idx,
                "label":      label,
                "confidence": round(conf, 4),   # pure temperature-scaled value
            })

        best = top5[0]
        base_confidence = best["confidence"]

        # Reported confidence = pure temperature-scaled softmax top-1 value.
        # WHY no vote_fraction blending here:
        #   The temperature-scaled softmax is already the calibrated, honest
        #   measure of the model's certainty on this input.  The vote_fraction
        #   is exposed as a separate field so the Gatekeeper can use it as a
        #   routing signal WITHOUT distorting the displayed confidence number.
        #   Blending them would make the confidence less interpretable:
        #       - It would drop a clear 91% prediction to ~79% if only 6/8 passes
        #         agreed, potentially changing its route from historian to validator.
        #       - Users would see a confidence that reflects two different signals
        #         in ways that are hard to explain.
        #   The clean architecture: confidence = model certainty,
        #                           vote_fraction = TTA agreement (separate concern).
        return {
            "class_id":          best["class_id"],
            "label":             best["label"],
            "confidence":        base_confidence,
            "top5":              top5,
            "inference_time_ms": inference_time_ms,
            "tta_used":          tta_used,
            "vote_fraction":     round(vote_fraction, 4) if vote_fraction is not None else None,
            "tta_passes":        n_tta_passes,
            "temperature":       round(self._temperature, 6),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image_path : str | Path,
        tta        : bool = False,
        gradcam    : bool = False,
    ) -> dict:
        """
        Classify a single coin image, optionally adding a Grad-CAM heatmap.

        RULE 3 — ALWAYS val transforms (never train transforms).
        The 79.08% / 80.03% accuracy was measured with val transforms.
        Using train transforms in production would SILENTLY degrade accuracy.

        Args:
            image_path : Path to the coin image (jpg/png).
            tta        : If True, run 8-pass Test-Time Augmentation.  Each pass
                         applies a different lightweight augmentation (flip,
                         rotation ±10°/±15°, brightness ±0.12, contrast +0.15).
                         The 8 softmax vectors are averaged before computing
                         top-1.  The fraction of passes that independently agree
                         on the top-1 class (vote_fraction) is used as a
                         supplementary confidence signal.  ~8× inference time.
            gradcam    : If True, generate a Grad-CAM heatmap overlay PNG after
                         the prediction.  The heatmap is saved alongside the
                         image (same directory, ``_gradcam.png`` suffix) and its
                         path is included in the returned dict as ``gradcam_path``.
                         Does NOT affect prediction values — purely visual.
                         Silently skipped if pytorch-grad-cam is not installed.

        Returns:
            dict following the output contract (see module docstring), with
            additional fields: vote_fraction, tta_passes, temperature,
            gradcam_path (str | None).
        """
        t_start = time.time()

        # RULE 4 — model already loaded in __init__, just use it
        img_rgb = self._load_image(image_path)

        vote_fraction: float | None = None

        if tta:
            # ── 8-pass TTA with vote tracking ────────────────────────────────
            # WHY vote tracking:
            #   Averaging softmax vectors reduces variance but does not tell us
            #   HOW MANY passes independently reached the same conclusion.
            #   If 8/8 pass agree on CN 1015 → genuinely confident result.
            #   If only 2/8 passes agree → the average may look OK numerically
            #   but the individual passes disagreed, signalling real ambiguity.
            #   vote_fraction feeds into _build_result() confidence blending.
            all_probs:  list[torch.Tensor] = []
            top1_votes: list[int]          = []

            for transform in _TTA_TRANSFORMS:
                tensor = self._preprocess(img_rgb, transform=transform)
                p      = self._forward(tensor)
                all_probs.append(p)
                top1_votes.append(int(p.argmax().item()))  # per-pass top-1

            probs           = torch.stack(all_probs).mean(dim=0)   # averaged
            predicted_top1  = int(probs.argmax().item())           # averaged top-1
            vote_fraction   = top1_votes.count(predicted_top1) / len(top1_votes)

            logger.debug(
                "predict: TTA vote %d/%d for class %s (vote_fraction=%.3f)",
                top1_votes.count(predicted_top1), len(top1_votes),
                self.idx_to_class.get(str(predicted_top1), str(predicted_top1)),
                vote_fraction,
            )
        else:
            tensor = self._preprocess(img_rgb)
            probs  = self._forward(tensor)

        elapsed_ms = int((time.time() - t_start) * 1000)
        result = self._build_result(
            probs,
            elapsed_ms,
            tta_used      = tta,
            vote_fraction = vote_fraction,
            n_tta_passes  = len(_TTA_TRANSFORMS) if tta else 1,
        )

        # ── Optional Grad-CAM heatmap ─────────────────────────────────────────
        # WHY after _build_result():
        #   Grad-CAM needs the predicted class_id from the result dict.
        #   Running it here (not inside _build_result) keeps the probability
        #   computation path clean — Grad-CAM is a purely visual enhancement.
        #
        # WHY single-pass tensor for Grad-CAM even when TTA was used:
        #   TTA averages 8 tensors.  GradCAM hooks work on a single forward pass.
        #   Using the un-augmented val-transform tensor gives the cleanest,
        #   most spatially faithful heatmap — TTA transforms (flips, crops)
        #   would produce averaged/rotated heat activations that are harder
        #   to interpret visually.
        if gradcam:
            try:
                from src.core.gradcam import generate_gradcam
                import cv2 as _cv2
                gradcam_tensor   = self._preprocess(img_rgb)          # [1,3,299,299]
                original_bgr     = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)
                gcam_save_path   = str(Path(image_path).with_suffix("")) + "_gradcam.png"
                gcam_path        = generate_gradcam(
                    model        = self.model,
                    image_tensor = gradcam_tensor,
                    original_bgr = original_bgr,
                    class_idx    = result["class_id"],
                    save_path    = gcam_save_path,
                    device       = self.device,
                )
                result["gradcam_path"] = str(gcam_path) if gcam_path else None
            except Exception as _gcam_err:
                logger.warning("predict: Grad-CAM failed (non-fatal): %s", _gcam_err)
                result["gradcam_path"] = None
        else:
            result["gradcam_path"] = None

        return result
