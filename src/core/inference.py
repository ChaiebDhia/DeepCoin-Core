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

        # WHY np.fromfile + imdecode instead of cv2.imread:
        # cv2.imread() on Windows uses the C runtime fopen() which only supports
        # ANSI paths.  Any non-ASCII character (accented letters, CJK, spaces
        # encoded differently, etc.) makes imread() silently return None.
        # Common real-world case: Windows screenshots are saved as
        # "Capture_d_écran_2026-03-01.png" (French locale).
        # np.fromfile() delegates to Python's own I/O layer which handles full
        # Unicode; the resulting byte array is then decoded in memory by
        # cv2.imdecode() — no file path ever reaches the C runtime.
        raw     = np.fromfile(str(path), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"OpenCV could not decode image: {path}")

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
