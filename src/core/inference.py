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

import time
import cv2
import torch
import numpy as np
from pathlib import Path

from src.core.model_factory import get_deepcoin_model
from src.core.dataset import get_val_transforms


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
        print(f"[CoinInference] device = {self.device}")

        # ── Load class mapping ─────────────────────────────────────────────────
        # class_mapping.pth contains {class_to_idx, idx_to_class, n_classes}
        mapping = torch.load(mapping_path, map_location="cpu", weights_only=False)
        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = mapping["idx_to_class"]
        self.num_classes: int = mapping.get("n_classes", len(self.class_to_idx))
        print(f"[CoinInference] classes loaded: {self.num_classes}")

        # ── Load model weights ─────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = get_deepcoin_model(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # RULE 1 — ALWAYS eval() after loading
        # Disables Dropout (p=0.4) and switches BatchNorm to use running stats.
        # Without this: different result every call. Non-deterministic = unusable.
        self.model.eval()

        # Store base val transforms (no augmentation, just normalize + tensor)
        self._val_transform = get_val_transforms()

        val_acc = checkpoint.get("val_acc", "unknown")
        epoch   = checkpoint.get("epoch", "unknown")
        print(f"[CoinInference] model loaded — epoch {epoch}, val_acc {val_acc:.2f}%")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load image from disk as RGB numpy array (H, W, 3) uint8.

        Raises:
            FileNotFoundError: if the path doesn't exist
            ValueError:        if OpenCV can't decode the file
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # OpenCV loads as BGR — convert to RGB immediately
        # WHY? EfficientNet was pretrained on ImageNet (RGB).
        # Feeding BGR shifts every colour channel → garbage features.
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"OpenCV could not decode image: {path}")

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
            tta:        If True, run 5-pass Test-Time Augmentation (+~1% accuracy)
                        at the cost of 5× inference time.

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
