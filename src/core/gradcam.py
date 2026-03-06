"""
src/core/gradcam.py
====================
Grad-CAM feature attribution for EfficientNet-B3.

WHAT this module does
---------------------
Given a coin image (already preprocessed to 299×299) and the trained
EfficientNet-B3, produces a coloured heatmap showing which spatial regions
of the coin drove the classification decision.  The overlay image is:
  - saved to disk as a PNG
  - returned to the caller by path
  - embedded in the PDF report (synthesis.py) as a dedicated section

WHY this exists
---------------
When the CNN says "type 1015 with 91% confidence," the natural follow-up is
"what did it look at?"  Grad-CAM answers this by producing a spatial map of
pixel importance.  This is critical for three reasons:

1. TRUST — museum professionals and jury members see what the AI "saw,"
   not just a number.  A heatmap on the portrait/legend is convincing;
   a heatmap on the background would reveal a model that learned wrong features.

2. DIAGNOSIS — if the heatmap highlights photo backgrounds or watermarks,
   the model suffers from shortcut learning and needs better augmentation or
   dataset cleaning.

3. EU AI ACT COMPLIANCE — the EU AI Act (2024) requires "meaningful
   explanation" for high-stakes AI decisions.  A Grad-CAM heatmap in the
   PDF report provides exactly that.

HOW Grad-CAM works (mathematical intuition)
-------------------------------------------
EfficientNet-B3 has 18 convolutional blocks.  The last block (features[-1])
produces a feature map of shape [B, C, H', W']:
  - B = batch size
  - C = number of channels (each encodes a different visual concept)
  - H', W' = spatial grid (~8×8 for 299×299 input after 5 maxpool stages)

Step 1 — Select the target class:
    y_c = the logit for predicted class c (e.g., "type 1015")

Step 2 — Backpropagate to the last conv layer:
    α_k = (1/Z) × Σ_{i,j}  ∂y_c / ∂A^k_{i,j}
    where A^k is the k-th feature map channel.
    α_k is the "importance weight" of feature map k.
    Channels that caused large positive gradients got large α_k.

Step 3 — Weighted sum + ReLU:
    L = ReLU( Σ_k  α_k × A^k )
    ReLU discards channels that DECREASED confidence (negative contribution).
    The result is a low-resolution heatmap (e.g., 8×8).

Step 4 — Upsample + colorise:
    Bilinear interpolation scales 8×8 → 299×299.
    Normalise to [0, 1].
    Apply COLORMAP_JET: blue=cold/unimportant → cyan → green → yellow → red=hot.

Step 5 — Overlay:
    overlay = 0.6 × original_coin + 0.4 × colormap(heatmap)
    Enough color to be clearly visible; enough coin to still read the inscription.

REFERENCE
---------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017.  https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing   import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    import torch.nn as nn


logger = logging.getLogger(__name__)


# ── availability guard ────────────────────────────────────────────────────────
# pytorch-grad-cam is an optional dependency.  If not installed, the module
# degrades gracefully: generate_gradcam() logs a warning and returns None.
# Installation: pip install grad-cam>=1.4.8
try:
    from pytorch_grad_cam                    import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image        import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False
    logger.warning(
        "pytorch-grad-cam not installed — Grad-CAM heatmaps disabled. "
        "Install with: pip install grad-cam>=1.4.8"
    )


class GradCAMExtractor:
    """
    Thin wrapper around pytorch-grad-cam for EfficientNet-B3.

    WHY a class (not just a function)
    ----------------------------------
    GradCAM() registers forward and backward hooks on the target layer.
    Hooks hold references to tensors, which prevents garbage collection.
    By keeping GradCAMExtractor as a short-lived object (created, used,
    then discarded), the hooks are cleaned up automatically when the instance
    goes out of scope.  Creating a new instance per prediction ensures no
    hook accumulation across requests.

    Target layer selection
    ----------------------
    ``model.features[-4]`` is EfficientNet-B3 stage-5 (features[5]):
        - High semantic content (after 13 convolutional blocks of abstraction)
        - Still has spatial dimensions (not yet flattened by adaptive avg pool)
        - Spatial resolution: 19×19 (361 cells) vs features[-1] which is 10×10
          (100 cells).  Each 19×19 cell covers ~16×16 pixels of the coin image
          instead of ~30×30 — enables the heatmap to distinguish between the
          coin face features (portrait, legend, reverse iconography) and the
          coin rim/background border.

    WHY features[-4] and NOT features[-1] (the standard choice):
        The standard guidance is to use the LAST conv feature map for maximum
        semantic content.  For clean, single-object images this is optimal.
        For Corpus Nummorum (every training image = two coin faces side-by-side
        in one 299×299 square), the last 10×10 feature map is too coarse:
          - Each of the 100 cells covers a ~30×30 pixel area
          - The coin rim (maximum contrast transition: bright coin → black
            background) dominates neighbouring cells because its gradient is
            always large and consistent
          - For OOD or low-confidence inputs, gradients are diffuse and the
            biggest gradient signal is at the high-contrast rim, not the
            numismatic content (portrait, legend)
        At 19×19, the rim occupies 1–2 cells; the coin FACE occupies 12–15
        cells — the heatmap can now show WHERE ON THE FACE the model looked.

    WHY GradCAMPlusPlus (not vanilla GradCAM):
        GradCAM computes α_k = mean(∂y_c/∂A^k) — the mean gradient across
        all spatial positions.  When the coin has TWO sub-objects (obverse +
        reverse), this mean gradient washes out between both, often yielding
        a bright ring at the boundary between them rather than on either face.
        GradCAM++ replaces the simple mean with a second-order gradient term
        that up-weights spatial positions where the gradient is largest — it
        correctly localises to the more discriminative face (usually obverse)
        even in the presence of the second coin face and background.
    """

    def __init__(self, model: "nn.Module", device: str = "cpu") -> None:
        """
        Parameters
        ----------
        model  : Trained EfficientNet-B3 in eval mode.  Must NOT be wrapped
                 in torch.no_grad() — Grad-CAM needs gradients.
        device : "cuda" or "cpu".  Grad-CAM runs on the same device as the
                 model (avoids expensive tensor moves mid-computation).
        """
        if not _GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam is required for GradCAMExtractor. "
                "Install with: pip install grad-cam"
            )
        self._device = device
        self._model  = model.eval()

        # target_layers must be a list — pytorch-grad-cam supports multi-layer
        # averaging, but we use a single layer (features[-4], stage-5, 19×19).
        # See class docstring for why features[-4] outperforms features[-1]
        # on composite numismatic images.
        target_layers   = [model.features[-4]]
        self._cam       = GradCAMPlusPlus(model=model, target_layers=target_layers)

    def explain(
        self,
        image_tensor : torch.Tensor,   # shape [1, 3, 299, 299], normalised
        original_bgr : np.ndarray,     # shape [299, 299, 3], uint8, CLAHE-enhanced
        class_idx    : int,            # softmax index of the predicted class (0-437)
    ) -> np.ndarray:
        """
        Produce a Grad-CAM heatmap overlaid on the original coin image.

        Parameters
        ----------
        image_tensor : The same preprocessed, normalised tensor used for the
                       prediction forward pass.  Shape [1, 3, 299, 299].
                       MUST be the normalised tensor (not raw pixel values).
        original_bgr : CLAHE-enhanced, 299×299 BGR image BEFORE normalisation.
                       Used as the background layer for the coloured overlay.
                       WHY before normalisation: normalised values have mean~0
                       — converting them back to uint8 for display requires
                       denormalisation, which introduces rounding error. Using
                       the pre-normalisation BGR avoids all that complexity.
        class_idx    : Predicted class index (0-437) to explain.
                       Grad-CAM is class-discriminative: it shows which pixels
                       caused the model to choose THIS class over all others.
                       Passing a different index would show what features
                       "almost" made the model choose that other class.

        Returns
        -------
        np.ndarray : uint8 BGR image of shape [299, 299, 3].
                     60% original coin + 40% COLORMAP_JET heatmap.
                     Red/yellow = important regions.
                     Blue = unimportant for this classification.

        Notes
        -----
        pytorch-grad-cam temporarily enables gradient computation internally.
        The caller does NOT need to disable torch.no_grad() — the library
        handles context management.
        """
        targets       = [ClassifierOutputTarget(class_idx)]
        # ClassifierOutputTarget tells Grad-CAM to compute ∂y_c/∂A^k where
        # y_c is the RAW logit (pre-softmax) for class_idx.
        # WHY pre-softmax: softmax normalization dilutes gradient signal from
        # competing classes.  Raw logits give cleaner, sharper heatmaps.

        # ── Compute the spatial activation mask ──────────────────────────────
        grayscale_cam = self._cam(
            input_tensor = image_tensor.to(self._device),
            targets      = targets,
        )
        # grayscale_cam: float32 array [1, 299, 299], values in [0, 1]
        mask = grayscale_cam[0]   # [299, 299]

        # ── Convert original BGR → float RGB for the overlay function ────────
        original_rgb_f = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        original_rgb_f = original_rgb_f.astype(np.float32) / 255.0
        # show_cam_on_image() requires float32 [0, 1] in RGB channel order.

        # ── Blend heatmap with original image ─────────────────────────────────
        overlay_rgb = show_cam_on_image(
            img          = original_rgb_f,
            mask         = mask,
            use_rgb      = True,
            colormap     = cv2.COLORMAP_JET,
            # COLORMAP_JET: blue=0 (cold/unimportant) → cyan → green →
            # yellow → red=1 (hot/important).  The most universally legible
            # colormap for heatmaps in scientific and medical imaging.
            image_weight = 0.6,
            # 60% coin, 40% colourmap.  Below 0.5: heatmap overwhelms coin.
            # Above 0.75: heatmap becomes hard to see on complex coin surfaces.
        )
        # overlay_rgb: uint8 [299, 299, 3] in RGB

        return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        # Return BGR: all downstream consumers (OpenCV, fpdf2) use BGR natively.


def generate_gradcam(
    model        : "nn.Module",
    image_tensor : torch.Tensor,
    original_bgr : np.ndarray,
    class_idx    : int,
    save_path    : str | Path,
    device       : str = "cpu",
) -> Path | None:
    """
    Generate a Grad-CAM heatmap overlay and save it as a PNG.

    This is the primary entry point for all callers (inference.py, synthesis.py).
    It wraps GradCAMExtractor construction/cleanup into one function call so
    callers cannot forget to release the hook references.

    Parameters
    ----------
    model        : Trained EfficientNet-B3 in eval mode.
    image_tensor : Preprocessed tensor [1, 3, 299, 299].
    original_bgr : CLAHE-enhanced BGR image [299, 299, 3] uint8.
    class_idx    : Predicted class softmax index (0-437).
    save_path    : Output PNG path (parent dirs created automatically).
    device       : "cuda" or "cpu".

    Returns
    -------
    Path if successful, None on any error (caller should check).

    WHY returns None on error (not raise):
        Grad-CAM is an enhancement.  If it fails (library not installed, OOM,
        unusual model state), the rest of the pipeline must still work.
        synthesis.py checks for None and skips the heatmap section if absent.
    """
    if not _GRADCAM_AVAILABLE:
        logger.warning("generate_gradcam: pytorch-grad-cam not installed, skipping.")
        return None

    try:
        extractor = GradCAMExtractor(model=model, device=device)
        overlay   = extractor.explain(image_tensor, original_bgr, class_idx)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), overlay)

        logger.info("generate_gradcam: saved heatmap → %s", save_path)
        return save_path

    except Exception as exc:
        logger.warning("generate_gradcam failed (non-fatal): %s", exc)
        return None
