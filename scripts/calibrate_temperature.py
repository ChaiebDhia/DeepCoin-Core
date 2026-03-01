"""
scripts/calibrate_temperature.py
==================================
Post-training calibration: finds the optimal temperature scalar T for the
trained EfficientNet-B3 via Temperature Scaling (Guo et al., NeurIPS 2017).

WHAT temperature scaling does
──────────────────────────────
The model normally computes confidence as  p = softmax(z)  where z are the
raw output logits.  Temperature scaling inserts a scalar T:

    p_calibrated = softmax(z / T)

T is the ONLY new parameter.  It is fitted on the validation set (never seen
during training) by minimising the cross-entropy (NLL) loss over all 1,151
validation images.  The model's RANKING accuracy is unchanged — only the
probability magnitudes are rescaled.

    T < 1  → sharpens the distribution → higher peak confidence
             Correct for an UNDER-CONFIDENT model (our case:
             the CNN scores CN 1015 at 8.4% even on a screenshot
             where it is the clear best match).
    T > 1  → flattens the distribution → lower peak confidence
             Correct for an OVER-CONFIDENT model.

WHY this model is under-confident (root causes)
────────────────────────────────────────────────
1. label_smoothing=0.1 was used during training (deliberately, to prevent
   overfit).  It pushes the model's output AWAY from hard one-hot targets,
   resulting in systematically lower logit magnitudes at the correct class.
2. 438 output classes: even the most dominant class reaches only ~25–30%
   natural softmax before calibration on clean, preprocessed images.
   On degraded screenshots the signal is weaker and the score drops further.
3. The model learned to be humble — it was optimised on coins in training-
   set lighting/quality.  Real-world screenshots add uncertainty it was
   never rewarded for resolving.

WHY post-hoc calibration instead of retraining
───────────────────────────────────────────────
- Retraining takes 103 minutes and consumes the GPU.
- Temperature scaling takes < 5 minutes on the 1,151-image validation set.
- It is theoretically justified: Guo et al. show it is the best-performing
  single-parameter calibration method and does not hurt accuracy.
- ECE (Expected Calibration Error) consistently drops 30–60%.

OUTPUT
──────
models/temperature.pth — contains:
    {
        "temperature":    float,   # optimal T (< 1 for this model)
        "val_nll_before": float,   # cross-entropy before calibration
        "val_nll_after":  float,   # cross-entropy after calibration
        "ece_before":     float,   # ECE before calibration
        "ece_after":      float,   # ECE after calibration
        "n_val_samples":  int,     # validation set size (sanity check)
    }

Usage
─────
    python scripts/calibrate_temperature.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.dataset      import DeepCoinDataset, get_val_transforms  # noqa: E402
from src.core.model_factory import get_deepcoin_model                   # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = ROOT / "models" / "best_model.pth"
MAPPING_PATH = ROOT / "models" / "class_mapping.pth"
TEMP_PATH    = ROOT / "models" / "temperature.pth"
DATA_DIR     = ROOT / "data"   / "processed"


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_val_loader(batch_size: int = 64) -> DataLoader:
    """
    Reconstruct the EXACT validation split used during training.

    WHY exact replication matters:
        Temperature scaling must be fitted on data the model has NOT seen
        during training.  If we accidentally include training images, the
        calibrated T will be over-fitted and will not generalise to real-
        world uploads.  We replicate the exact same stratified split
        (seed=42, 70/15/15) from train.py, step by step.

    Returns:
        DataLoader over the 1,151-image validation subset (no augmentation).
    """
    log.info("Loading dataset from %s …", DATA_DIR)
    dataset   = DeepCoinDataset(str(DATA_DIR), transform=get_val_transforms())
    all_labels  = [label for _, label in dataset.samples]
    all_indices = list(range(len(dataset)))

    # SPLIT 1 — separate 15% test set (same as train.py)
    train_val_idx, _ = train_test_split(
        all_indices,
        test_size    = 0.15,
        stratify     = all_labels,
        random_state = 42,
    )

    # SPLIT 2 — separate 15% val from train+val
    tv_labels = [all_labels[i] for i in train_val_idx]
    val_frac  = 0.15 / 0.85   # 15% of total expressed as fraction of 85% pool

    _, val_idx = train_test_split(
        train_val_idx,
        test_size    = val_frac,
        stratify     = tv_labels,
        random_state = 42,
    )

    val_subset = Subset(dataset, val_idx)
    log.info("Validation subset: %d images across %d classes",
             len(val_subset), len(dataset.classes))

    return DataLoader(
        val_subset,
        batch_size  = batch_size,
        shuffle     = False,          # deterministic order — we match logits to labels
        num_workers = 0,              # avoid Windows multiprocessing pickle issues
        pin_memory  = torch.cuda.is_available(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Logit collection
# ══════════════════════════════════════════════════════════════════════════════

def _collect_logits(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all validation images through the model and collect raw logits.

    WHY raw logits (not softmax probabilities):
        Temperature scaling is applied to logits: softmax(z / T).
        If we stored softmax probabilities p we would lose the scale
        information — log(p) is not the same as z/T for T ≠ 1.
        We MUST store z, not softmax(z).

    Returns:
        logits_np : shape (N, 438)  — raw unnormalised scores
        labels_np : shape (N,)      — integer ground-truth class indices
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images)                    # [B, num_classes]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            if batch_idx % 5 == 0 or batch_idx == n_batches:
                log.info("  Batch %d / %d", batch_idx, n_batches)

    return np.vstack(all_logits), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration metrics
# ══════════════════════════════════════════════════════════════════════════════

def _nll(T_arr: np.ndarray, logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Negative log-likelihood (cross-entropy) at temperature T.

    This is the objective function minimised by L-BFGS-B.
    scipy.optimize passes the parameter as a 1-element numpy array.

    Returns:
        Scalar NLL value.  Lower is better.
    """
    T = float(T_arr[0])
    if T <= 0.0:
        return 1e9    # hard positivity constraint
    logits_t = torch.tensor(logits_np / T, dtype=torch.float32)
    labels_t = torch.tensor(labels_np,     dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits_t, labels_t).item()
    return float(loss)


def _ece(logits_np: np.ndarray, labels_np: np.ndarray,
         T: float = 1.0, n_bins: int = 15) -> float:
    """
    Expected Calibration Error with equal-width confidence bins.

    ECE measures how closely predicted confidence matches empirical accuracy.
    A perfectly calibrated model has ECE = 0.  Real models have ECE 0.02–0.20.
    """
    probs   = torch.softmax(
        torch.tensor(logits_np / T, dtype=torch.float32), dim=1
    ).numpy()
    conf    = probs.max(axis=1)                       # top-1 predicted confidence
    pred    = probs.argmax(axis=1)
    correct = (pred == labels_np).astype(float)

    ece_val = 0.0
    for b in range(n_bins):
        lo, hi  = b / n_bins, (b + 1) / n_bins
        in_bin  = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        acc_bin  = correct[in_bin].mean()
        conf_bin = conf[in_bin].mean()
        ece_val += (in_bin.sum() / len(conf)) * abs(conf_bin - acc_bin)
    return float(ece_val)


def _confidence_summary(
    logits_np: np.ndarray, labels_np: np.ndarray, T: float, label: str
) -> None:
    """Print confidence distribution statistics for correctly classified images."""
    probs     = torch.softmax(
        torch.tensor(logits_np / T, dtype=torch.float32), dim=1
    ).numpy()
    pred      = probs.argmax(axis=1)
    correct   = pred == labels_np
    top1_conf = probs.max(axis=1)

    log.info(
        "  %-6s | correct=%d/%d (%.1f%%) | conf: mean=%.1f%%  "
        "median=%.1f%%  p10=%.1f%%  p90=%.1f%%",
        label,
        correct.sum(), len(correct), correct.mean() * 100,
        top1_conf[correct].mean()    * 100 if correct.any() else 0,
        np.median(top1_conf[correct]) * 100 if correct.any() else 0,
        np.percentile(top1_conf[correct], 10) * 100 if correct.any() else 0,
        np.percentile(top1_conf[correct], 90) * 100 if correct.any() else 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 62)
    log.info("  DeepCoin — Temperature Scaling Calibration")
    log.info("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading model …")
    mapping    = torch.load(MAPPING_PATH, map_location="cpu", weights_only=True)
    num_classes = mapping.get("n_classes", len(mapping["class_to_idx"]))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model      = get_deepcoin_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    log.info(
        "Model ready — epoch %s  val_acc %s  num_classes %d",
        checkpoint.get("epoch", "?"),
        checkpoint.get("val_acc", "?"),
        num_classes,
    )

    # ── Build validation loader ────────────────────────────────────────────────
    val_loader = _build_val_loader(batch_size=64)

    # ── Collect logits ─────────────────────────────────────────────────────────
    log.info("Collecting logits (this takes ~2–4 min on GPU) …")
    t0 = time.perf_counter()
    logits_np, labels_np = _collect_logits(val_loader, model, device)
    elapsed = time.perf_counter() - t0
    log.info(
        "Collected %d logits × %d classes in %.1f s",
        logits_np.shape[0], logits_np.shape[1], elapsed,
    )

    # ── Baseline metrics (T = 1.0) ────────────────────────────────────────────
    nll_before = _nll(np.array([1.0]), logits_np, labels_np)
    ece_before = _ece(logits_np, labels_np, T=1.0)
    log.info("─" * 50)
    log.info("BEFORE calibration (T = 1.00)")
    log.info("  NLL : %.5f", nll_before)
    log.info("  ECE : %.4f  (0 = perfectly calibrated)", ece_before)
    _confidence_summary(logits_np, labels_np, T=1.0, label="T=1.0")

    # ── Optimise T ────────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("Optimising T with L-BFGS-B …")
    result = minimize(
        _nll,
        x0      = np.array([1.0]),
        args    = (logits_np, labels_np),
        method  = "L-BFGS-B",
        bounds  = [(0.01, 10.0)],
        options = {"maxiter": 500, "ftol": 1e-9, "gtol": 1e-7},
    )
    T_opt = float(result.x[0])
    log.info("Optimisation converged: T = %.6f  (success=%s)", T_opt, result.success)
    if not result.success:
        log.warning("L-BFGS-B did not fully converge: %s", result.message)

    # ── Post-calibration metrics ───────────────────────────────────────────────
    nll_after = _nll(np.array([T_opt]), logits_np, labels_np)
    ece_after = _ece(logits_np, labels_np, T=T_opt)
    log.info("─" * 50)
    log.info("AFTER calibration (T = %.6f)", T_opt)
    log.info("  NLL : %.5f  (Δ = %.5f)", nll_after, nll_before - nll_after)
    log.info("  ECE : %.4f  (Δ = %.4f)", ece_after, ece_before - ece_after)
    _confidence_summary(logits_np, labels_np, T=T_opt, label=f"T={T_opt:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "temperature":    T_opt,
        "val_nll_before": nll_before,
        "val_nll_after":  nll_after,
        "ece_before":     ece_before,
        "ece_after":      ece_after,
        "n_val_samples":  int(len(labels_np)),
    }
    torch.save(payload, TEMP_PATH)
    log.info("─" * 50)
    log.info("Saved  → %s", TEMP_PATH)
    log.info("")
    log.info("✅ Done.  Restart the API server to activate temperature scaling.")
    log.info("   Inference will now compute:  softmax(logits / %.6f)", T_opt)
    if T_opt < 1.0:
        log.info(
            "   Effect: distribution SHARPENED — correct class gets ~%.0f× "
            "boost on clear images.",
            1.0 / T_opt,
        )


if __name__ == "__main__":
    main()
