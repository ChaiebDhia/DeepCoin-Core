"""
DeepCoin — Test-Time Augmentation (TTA) Evaluation
====================================================
Runs inference on the test set 5 times per image with different augmentations,
then averages the probability distributions before making a final prediction.

WHY THIS WORKS:
    Normal inference: model sees 1 version of each image → one set of probabilities
    TTA inference:    model sees 5 versions → 5 probability vectors → averaged

    The averaged result is more stable because:
    - A horizontally flipped coin looks the same to a human expert
    - Slight brightness/contrast variation shouldn't change the answer
    - Averaging 5 "opinions" reduces the effect of any single unlucky view

    Typical gain: +1% to +3% accuracy, with zero additional training.

Usage:
    python scripts/evaluate_tta.py

Compares:
    Standard inference (1 pass) vs TTA (5 passes) on the same test set
"""

import os
import sys
import warnings

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", message=".*Error fetching version info.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image as PILImage

from src.core.dataset import DeepCoinDataset, get_val_transforms
from src.core.model_factory import get_deepcoin_model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'models/best_model.pth'
MAPPING_PATH = 'models/class_mapping.pth'
DATA_DIR     = 'data/processed'
BATCH_SIZE   = 16
NUM_WORKERS  = 2
RANDOM_SEED  = 42

# Number of TTA passes — 5 is the standard sweet spot.
# More passes = marginally better but slower (5 is already 5× inference time)
TTA_N = 5


# ─────────────────────────────────────────────────────────────────────────────
# TTA TRANSFORM SET
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is one "augmented view" of the input image.
# Rule: only use augmentations that preserve coin identity
#   ✅ Horizontal flip — same coin, mirrored
#   ✅ Brightness/contrast — same coin, different lighting
#   ✅ Slight rotation — same coin, slightly tilted in photo
#   ✅ Center crop + resize — same coin, slightly zoomed in
#   ❌ CoarseDropout — hides parts of the coin (bad for TTA)
#   ❌ ElasticTransform — distorts shape (bad for TTA)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 299

def get_tta_transforms():
    """
    Returns a list of 5 transform pipelines.
    Pass 1: clean (no augmentation) — always include the original
    Pass 2: horizontal flip
    Pass 3: brightness & contrast boost
    Pass 4: slight rotation (±10°)
    Pass 5: center crop (95%) then resize back
    """
    base = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]

    return [
        # Pass 1 — original (clean baseline)
        A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),

        # Pass 2 — horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2()
        ]),

        # Pass 3 — slight brightness increase
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2()
        ]),

        # Pass 4 — small rotation
        A.Compose([
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2()
        ]),

        # Pass 5 — center crop 95% then resize back to 299
        A.Compose([
            A.CenterCrop(height=int(IMG_SIZE * 0.95), width=int(IMG_SIZE * 0.95)),
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2()
        ]),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET THAT SUPPORTS MULTIPLE TRANSFORMS (for TTA)
# ─────────────────────────────────────────────────────────────────────────────
class TTADataset(torch.utils.data.Dataset):
    """
    Wraps the base dataset to apply a specific transform.
    We create one TTADataset per TTA pass, each with a different transform.
    """
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices      = indices
        self.transform    = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        abs_idx    = self.indices[i]
        img_path, label = self.base_dataset.samples[abs_idx]

        # Load raw image (numpy array, no normalization yet)
        img = PILImage.open(img_path).convert('RGB')
        img = np.array(img)

        # Apply this TTA pass's transform
        augmented = self.transform(image=img)
        return augmented['image'], label


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DEEPCOIN — TTA EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    mapping     = torch.load(MAPPING_PATH, map_location='cpu')
    num_classes = mapping['num_classes']

    model = get_deepcoin_model(num_classes=num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"   Loaded model: epoch {checkpoint.get('epoch','?')}, "
          f"val_acc={checkpoint.get('val_acc',0):.2f}%")

    # ── Rebuild identical test split ──────────────────────────────────────────
    base_dataset = DeepCoinDataset(root_dir=DATA_DIR, transform=None)
    all_labels   = [label for _, label in base_dataset.samples]
    all_indices  = list(range(len(base_dataset)))

    _, test_idx = train_test_split(
        all_indices, test_size=0.15, stratify=all_labels, random_state=RANDOM_SEED
    )
    print(f"   Test set: {len(test_idx)} images\n")

    # ── Standard inference (1 pass, clean) ────────────────────────────────────
    print("Running standard inference (1 pass, no TTA)...")
    std_dataset = TTADataset(base_dataset, test_idx, get_tta_transforms()[0])
    std_loader  = DataLoader(std_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    all_true = []
    std_preds = []

    with torch.no_grad():
        for images, labels in tqdm(std_loader, desc="Standard"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            _, pred = probs.max(dim=1)
            all_true.extend(labels.numpy())
            std_preds.extend(pred.cpu().numpy())

    all_true  = np.array(all_true)
    std_preds = np.array(std_preds)
    std_acc   = (all_true == std_preds).mean() * 100
    print(f"   Standard accuracy: {std_acc:.2f}%\n")

    # ── TTA inference (5 passes, averaged) ────────────────────────────────────
    print(f"Running TTA inference ({TTA_N} passes per image)...")
    tta_transforms = get_tta_transforms()

    # accumulated_probs[i] = sum of probability vectors across all TTA passes
    # Shape: [num_test_images, num_classes]
    accumulated_probs = np.zeros((len(test_idx), num_classes), dtype=np.float32)

    for pass_num, transform in enumerate(tta_transforms, 1):
        tta_ds     = TTADataset(base_dataset, test_idx, transform)
        tta_loader = DataLoader(tta_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)

        offset = 0
        with torch.no_grad():
            for images, labels in tqdm(tta_loader, desc=f"TTA pass {pass_num}/{TTA_N}"):
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs   = torch.softmax(outputs, dim=1).cpu().numpy()

                batch_size = probs.shape[0]
                accumulated_probs[offset: offset + batch_size] += probs
                offset += batch_size

    # Average the accumulated probabilities
    # accumulated_probs / TTA_N = mean probability across all 5 passes
    avg_probs = accumulated_probs / TTA_N
    tta_preds = np.argmax(avg_probs, axis=1)
    tta_acc   = (all_true == tta_preds).mean() * 100

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TTA RESULTS")
    print("=" * 60)
    print(f"   Standard (1 pass):  {std_acc:.2f}%")
    print(f"   TTA ({TTA_N} passes):       {tta_acc:.2f}%")
    gain = tta_acc - std_acc
    if gain > 0:
        print(f"   Gain:              +{gain:.2f}%  (TTA wins)")
    elif gain == 0:
        print(f"   Gain:               0.00%  (no change)")
    else:
        print(f"   Gain:              {gain:.2f}%  (unusual — check transforms)")

    print(f"\n   Test set size:   {len(test_idx)} images")
    print(f"   Total forward passes: {len(test_idx) * TTA_N:,}")
    print("=" * 60)

    # ── Per-class breakdown: where did TTA help? ──────────────────────────────
    improved   = ((std_preds != all_true) & (tta_preds == all_true)).sum()
    hurt       = ((std_preds == all_true) & (tta_preds != all_true)).sum()
    unchanged  = len(all_true) - improved - hurt

    print(f"\n   TTA fixed {improved} predictions that were wrong")
    print(f"   TTA broke {hurt} predictions that were right")
    print(f"   Net improvement: {improved - hurt} images")
    print(f"   Unchanged: {unchanged} images\n")


if __name__ == '__main__':
    main()
