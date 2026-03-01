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
    A.Compose([A.HorizontalFlip(p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 3 — small clockwise rotation
    A.Compose([A.Rotate(limit=(10, 10), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 4 — small counter-clockwise rotation
    A.Compose([A.Rotate(limit=(-10, -10), p=1.0),
               A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               ToTensorV2()]),
    # Pass 5 — slight brightness boost (simulates different lighting)
    A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1),
                                          contrast_limit=0, p=1.0),
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

    # Skip if image is already tight (< 200 px) — nothing meaningful to crop away
    if min(h, w) < 200:
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
        Single forward pass.

        RULE 2 — ALWAYS torch.no_grad()
        During inference we never backpropagate.
        no_grad() tells PyTorch: don't build the computation graph.
        Result: ~2× faster, much lower VRAM usage.
        """
        with torch.no_grad():
            logits = self.model(tensor)             # [1, num_classes]
            probs  = torch.softmax(logits, dim=1)   # [1, num_classes]
        return probs.squeeze(0)                     # [num_classes]

    def _build_result(
        self,
        probs: torch.Tensor,
        inference_time_ms: int,
        tta_used: bool,
    ) -> dict:
        """
        Convert raw probability tensor into the standard output contract dict.
        """
        probs_cpu = probs.cpu()

        # Top-5 predictions
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
                "confidence": round(conf, 4),
            })

        best = top5[0]
        return {
            "class_id":          best["class_id"],
            "label":             best["label"],
            "confidence":        best["confidence"],
            "top5":              top5,
            "inference_time_ms": inference_time_ms,
            "tta_used":          tta_used,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str | Path, tta: bool = False) -> dict:
        """
        Classify a single coin image.

        RULE 3 — ALWAYS val transforms (never train transforms).
        The 79.08% / 80.03% accuracy was measured with val transforms.
        Using train transforms in production would SILENTLY degrade accuracy.

        Args:
            image_path: Path to the coin image (jpg/png)
            tta:        If True, run 5-pass Test-Time Augmentation (+0.78% accuracy,
                        measured on the CN test set).  ~5× inference time overhead.

        Returns:
            dict following the output contract (see module docstring)
        """
        t_start = time.time()

        # RULE 4 — model already loaded in __init__, just use it
        img_rgb = self._load_image(image_path)

        if tta:
            # Average probabilities across all 5 TTA passes
            all_probs = []
            for transform in _TTA_TRANSFORMS:
                tensor = self._preprocess(img_rgb, transform=transform)
                all_probs.append(self._forward(tensor))
            probs = torch.stack(all_probs).mean(dim=0)
        else:
            tensor = self._preprocess(img_rgb)
            probs  = self._forward(tensor)

        elapsed_ms = int((time.time() - t_start) * 1000)
        return self._build_result(probs, elapsed_ms, tta_used=tta)
