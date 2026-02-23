"""
DeepCoin — EfficientNet-B3 Training Script  (v3 — Enterprise Grade)
====================================================================
Teaches the model to recognize 438 archaeological coin types.

Usage:
    python scripts/train.py                        # Full training
    python scripts/train.py --fast                 # Smoke test: 500 images, 3 epochs
    python scripts/train.py --resume               # Resume from last checkpoint

New in v3:
    - AMP  (Automatic Mixed Precision): halves VRAM usage, ~2x faster per epoch
    - Mixup augmentation: blends two images → smoother decision boundaries
    - Resume capability: never lose progress on interruption
    - Cosine LR schedule: better convergence than ReduceLROnPlateau
    - Balanced augmentation: realistic, not destructive
"""

import os
import sys
import time
import argparse
import warnings

# Fix Windows terminal encoding (cp1252 can't handle emojis)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress albumentations version-check network noise
warnings.filterwarnings("ignore", message=".*Error fetching version info.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
# AMP (torch.amp) — GradScaler and autocast used directly via torch.amp below
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import numpy as np

from src.core.dataset import DeepCoinDataset, get_train_transforms, get_val_transforms
from src.core.model_factory import get_deepcoin_model


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: DATA SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(dataset, val_size=0.15, test_size=0.15, random_seed=42):
    """
    🎓 WHAT THIS DOES:
    Splits the full dataset into 3 non-overlapping groups:
        - Train (70%): Model learns from this
        - Val   (15%): We check progress after each epoch (never trained on)
        - Test  (15%): Final honest exam, used ONCE at the end

    🔑 KEY CONCEPT — STRATIFIED SPLIT:
    We use stratify=labels to guarantee all 438 classes appear in ALL 3 splits.
    Example: Class 5181 has 5 images → 3 in train, 1 in val, 1 in test.
    Without stratify, all 5 might end up in train → val can't evaluate that class.

    Args:
        dataset: The full DeepCoinDataset
        val_size: Fraction for validation (0.15 = 15%)
        test_size: Fraction for test (0.15 = 15%)
        random_seed: Fixed seed for reproducibility (same split every run)

    Returns:
        train_dataset, val_dataset, test_dataset (all are Subset objects)
    """
    # Extract ALL labels from the dataset (needed for stratify=)
    # This is just a list like: [0, 0, 0, 1, 1, 2, 2, 2, ...]
    # Handle both raw DeepCoinDataset and Subset (used in --fast mode)
    if isinstance(dataset, Subset):
        all_labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        all_labels = [label for _, label in dataset.samples]
    all_indices = list(range(len(dataset)))

    # SPLIT 1: Separate test set from the rest (train+val)
    # test_size=0.15 → 15% goes to test, 85% goes to train+val
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=all_labels,        # ← ensures proportional class distribution
        random_state=random_seed    # ← fixed seed = same split every run
    )

    # SPLIT 2: Separate val from train
    # We need val to be 15% of TOTAL, but we're splitting from 85% remaining.
    # So: val_fraction_of_remainder = 0.15 / 0.85 ≈ 0.176
    train_val_labels = [all_labels[i] for i in train_val_indices]
    val_fraction = val_size / (1 - test_size)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=random_seed
    )

    # Wrap indices in Subset objects
    # Subset is a PyTorch wrapper that makes a "view" of a dataset
    # without copying any data — memory efficient!
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    print(f"\n📊 Dataset Split:")
    print(f"   Total:      {len(dataset):>6} images")
    print(f"   Train:      {len(train_dataset):>6} images  ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"   Validation: {len(val_dataset):>6} images  ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"   Test:       {len(test_dataset):>6} images  ({len(test_dataset)/len(dataset)*100:.1f}%)")

    return train_dataset, val_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: CLASS IMBALANCE FIX — WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────
def get_weighted_sampler(train_dataset):
    """
    🎓 WHAT THIS DOES:
    Fixes the 40:1 class imbalance problem.

    The Problem:
        Class 3987: 204 images → model sees this ALL the time
        Class 5181:   5 images → model almost never sees this
        Result: Model learns to predict common classes, ignores rare ones.

    The Solution — Inverse Frequency Weighting:
        weight_i = 1 / count(class_i)
        Class 5181: weight = 1/5  = 0.200 (picked often)
        Class 3987: weight = 1/204 = 0.005 (picked rarely)

    The sampler then draws images proportionally to their weights,
    so the model sees each class approximately equally often.

    Args:
        train_dataset: The training Subset

    Returns:
        WeightedRandomSampler — a replacement for shuffle=True in DataLoader
    """
    # Get labels for ONLY the training images (not full dataset)
    # Navigate through nested Subsets to reach the raw DeepCoinDataset
    def get_root_dataset(ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    def get_absolute_indices(ds):
        """Resolve indices through nested Subsets to get absolute indices into root dataset."""
        if not isinstance(ds, Subset):
            return list(range(len(ds)))
        parent_indices = get_absolute_indices(ds.dataset)
        return [parent_indices[i] for i in ds.indices]

    root_dataset = get_root_dataset(train_dataset)
    absolute_indices = get_absolute_indices(train_dataset)
    train_labels = [root_dataset.samples[i][1] for i in absolute_indices]

    # Count how many images exist per class in training set
    class_counts = Counter(train_labels)

    # Assign weight to each sample: weight = 1 / class_count
    # If class 3987 has 143 train images → weight = 1/143 = 0.007
    # If class 5181 has  3 train images → weight = 1/3   = 0.333
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # WeightedRandomSampler draws `num_samples` indices with replacement
    # according to the weights — rarer classes get drawn more often
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True    # ← allows drawing same image multiple times per epoch
    )

    print(f"\n⚖️  WeightedRandomSampler created:")
    print(f"   Most common class:  {max(class_counts, key=class_counts.get)} → {max(class_counts.values())} images")
    print(f"   Rarest class:       {min(class_counts, key=class_counts.get)} → {min(class_counts.values())} images")
    print(f"   Imbalance ratio:    {max(class_counts.values()) / min(class_counts.values()):.1f}:1  → now balanced by sampling")

    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(train_dataset, val_dataset, test_dataset, sampler, batch_size=32):
    """
    🎓 WHAT THIS DOES:
    Wraps the datasets in DataLoaders — the conveyor belt that feeds
    the GPU with batches of images.

    Key parameters explained:
    - batch_size=32: Feed 32 images at a time to the GPU
      Why 32? It fits in 4GB VRAM for EfficientNet-B3. (64 would be faster but might OOM)
    - num_workers=4: 4 CPU threads preload the NEXT batch while GPU trains on current batch
      Without this: GPU sits idle waiting for data (major bottleneck)
    - pin_memory=True: Locks data in RAM so GPU can fetch it faster (CUDA optimization)
    - sampler replaces shuffle=True for training (can't use both!)

    Args:
        train_dataset, val_dataset, test_dataset: The 3 split subsets
        sampler: WeightedRandomSampler for balanced class sampling
        batch_size: Number of images per batch

    Returns:
        train_loader, val_loader, test_loader
    """
    # num_workers=2: safe for laptop (4 can cause OOM on 4GB VRAM GPUs)
    # pin_memory=True only for train (val/test are smaller, safer without it)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,        # ← replaces shuffle=True
        num_workers=2,
        pin_memory=True         # ← faster CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # ← same batch size (safer for 4GB VRAM)
        shuffle=False,          # ← never shuffle val/test (reproducible results)
        num_workers=2,
        pin_memory=False        # ← disabled for val to reduce VRAM pressure
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    print(f"\n🔄 DataLoaders created:")
    print(f"   Train batches: {len(train_loader)} × {batch_size} images")
    print(f"   Val batches:   {len(val_loader)} × {batch_size} images")
    print(f"   Test batches:  {len(test_loader)} × {batch_size} images")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(images, labels, num_classes, alpha=0.4):
    """
    🎓 WHAT IS MIXUP?
    Mixup blends TWO training images together with a random ratio (lambda):

        mixed_image = λ × image_A + (1-λ) × image_B
        mixed_label = λ × label_A + (1-λ) × label_B

    Example with λ=0.7:
        - 70% of a Roman denarius + 30% of a Greek drachma
        - Label: [0.7 probability for denarius, 0.3 probability for drachma]

    WHY DOES THIS HELP?
    Without Mixup: The model sees sharp, clean images and memorizes them.
        Decision boundary: "if I see THIS exact pattern → class A"
    With Mixup: The model must handle blended images it never saw before.
        Decision boundary: smooth transitions between classes, not hard edges.

    This is proven to reduce the train/val gap by 3-5% on small datasets.
    It's the single most effective technique for your dataset size.

    Args:
        images: batch tensor [B, C, H, W]
        labels: batch label tensor [B] (integer class indices)
        num_classes: total number of classes (438)
        alpha: controls blend ratio. alpha=0.4 → λ drawn from Beta(0.4, 0.4)
               Higher alpha = more aggressive mixing
               Lower alpha = mostly one image (less aggressive)

    Returns:
        mixed_images: [B, C, H, W] — blended image tensors
        labels_a, labels_b: original one-hot label tensors [B, num_classes]
        lam: the mixing ratio (scalar)
    """
    # Convert integer labels to one-hot vectors for soft label mixing
    # [B] → [B, 438]  e.g., label 3 → [0,0,0,1,0,...,0]
    labels_onehot = torch.zeros(labels.size(0), num_classes, device=images.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

    # Sample lambda from Beta distribution
    # Beta(0.4, 0.4) gives values mostly near 0 or 1 (one image dominates slightly)
    # This prevents images from being blended 50/50 too often (would be unrecognizable)
    lam = np.random.beta(alpha, alpha)

    # Create shuffled index for pairing — image[i] blends with image[perm[i]]
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    # Blend images: λ × original + (1-λ) × shuffled partner
    mixed_images = lam * images + (1 - lam) * images[perm]

    # Blend labels proportionally
    labels_a = labels_onehot
    labels_b = labels_onehot[perm]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(outputs, labels_a, labels_b, lam):
    """
    🎓 LOSS FUNCTION FOR MIXUP
    Since labels are now soft (blended), we can't use standard CrossEntropyLoss directly.
    Instead we compute:
        loss = λ × CE(output, label_A) + (1-λ) × CE(output, label_B)

    This tells the model: "I want you to be λ% right for class A and (1-λ)% right for class B"
    """
    # log_softmax gives log-probabilities from raw logits
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)

    # Cross entropy with soft labels = -sum(soft_label × log_prob)
    loss_a = -(labels_a * log_probs).sum(dim=1).mean()
    loss_b = -(labels_b * log_probs).sum(dim=1).mean()

    return lam * loss_a + (1 - lam) * loss_b


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5: TRAIN ONE EPOCH  (with AMP + Mixup)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, total_epochs, num_classes, use_mixup=True):
    """
    🎓 ONE EPOCH WITH AMP + MIXUP

    AMP (Automatic Mixed Precision):
        Normally PyTorch uses float32 (32 bits per number) for everything.
        AMP uses float16 (16 bits) for most operations → 2x less VRAM, 2x faster.
        For critical operations (loss scaling) it stays in float32 for stability.
        The GradScaler handles the numerical stability automatically.

    Mixup:
        Every batch is blended with a shuffled copy of itself before training.
        See mixup_batch() for full explanation.

    Args:
        scaler: GradScaler for AMP — handles float16/float32 switching
        use_mixup: False during first few epochs to let model build basic features first
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── MIXUP (applied to 80% of batches after warmup) ──────────────────
        if use_mixup and np.random.random() < 0.8:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, num_classes)

            # ── AMP FORWARD PASS ────────────────────────────────────────────
            # autocast: PyTorch automatically chooses float16 or float32
            # for each operation. Fast ops (matmul, conv) → float16.
            # Precision-critical ops (softmax, loss) → float32.
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = mixup_criterion(outputs, labels_a, labels_b, lam)

            # For accuracy tracking with mixup: use the dominant label
            _, predicted = outputs.detach().max(dim=1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a.argmax(1)).sum().item()
                       + (1 - lam) * predicted.eq(labels_b.argmax(1)).sum().item())

        else:
            # Standard forward pass (no mixup) — used for warmup epochs
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, predicted = outputs.detach().max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # ── AMP BACKWARD PASS ────────────────────────────────────────────────
        # scaler.scale(loss): multiplies loss by a scale factor to prevent
        # float16 underflow (very small gradients becoming zero)
        scaler.scale(loss).backward()

        # scaler.unscale_: divides gradients back before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # scaler.step: applies optimizer update, skips if gradients have inf/nan
        scaler.step(optimizer)

        # scaler.update: adjusts the scale factor for next iteration
        scaler.update()

        running_loss += loss.item()
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    return running_loss / len(loader), 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6: VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    """
    🎓 WHAT THIS DOES:
    Evaluates the model on validation data — NO learning, just measurement.

    Key differences from training:
    - model.eval(): Disables Dropout (no random neurons turned off during eval)
                    Freezes BatchNorm stats (uses running averages, not batch stats)
    - torch.no_grad(): Tells PyTorch "don't track gradients" → saves memory + speed

    Think of this as: the student takes a practice exam with no hints.

    Returns:
        avg_loss (float): Validation loss
        accuracy (float): % correct on validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():   # ← disable gradient tracking (saves ~50% memory)
        progress = tqdm(loader, desc="Validation", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Ties everything together (Blocks 4 + 7)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    🎓 THE MAIN TRAINING PIPELINE
    Orchestrates all blocks in order:
    1. Parse arguments
    2. Setup device (GPU or CPU)
    3. Load and split dataset
    4. Create weighted sampler and dataloaders
    5. Build model, optimizer, scheduler, loss
    6. Training loop with early stopping and checkpointing
    """

    # ── ARGUMENT PARSING ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='DeepCoin Training v3')
    parser.add_argument('--fast', action='store_true',
                        help='Smoke test mode: 500 images, 3 epochs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from models/checkpoint_last.pth')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16, safe for 4GB VRAM)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    args = parser.parse_args()

    print("=" * 65)
    print("  DEEPCOIN -- EfficientNet-B3 Training")
    print("=" * 65)

    # ── BLOCK 4A: DEVICE SETUP ───────────────────────────────────────────────
    # 'cuda' = NVIDIA GPU (RTX 3050 Ti) — uses CUDA cores for parallel computation
    # 'cpu'  = fallback if no GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚙️  Device: {device}", end="")
    if device.type == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(" (no GPU detected — training will be slow)")

    # ── BLOCK 1: LOAD + SPLIT DATASET ───────────────────────────────────────
    print(f"\n📂 Loading dataset...")
    full_dataset = DeepCoinDataset(
        root_dir='data/processed',
        transform=get_train_transforms()    # train transforms applied to full dataset
    )                                       # val/test will override this below

    # Override: val and test should use val transforms (no augmentation!)
    # We create a second dataset object with val transforms for this purpose
    full_dataset_val = DeepCoinDataset(
        root_dir='data/processed',
        transform=get_val_transforms()
    )

    # If --fast mode: use only 500 images for quick testing
    if args.fast:
        print("\n⚡ FAST MODE: Using 500 images, 3 epochs")
        fast_indices = list(range(500))
        full_dataset = Subset(full_dataset, fast_indices)
        full_dataset_val = Subset(full_dataset_val, fast_indices)
        args.epochs = 3

    # Split train dataset (with augmentation transforms)
    train_ds, _, _ = split_dataset(full_dataset)

    # Split val/test from the clean dataset (no augmentation)
    # WHY two dataset objects? train needs augmentation, val/test need clean images.
    # We use the SAME random_seed so the split indices are identical.
    _, val_ds, test_ds = split_dataset(full_dataset_val)

    # ── BLOCK 2: WEIGHTED SAMPLER ────────────────────────────────────────────
    sampler = get_weighted_sampler(train_ds)

    # ── BLOCK 3: DATALOADERS ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds,
        sampler=sampler,
        batch_size=args.batch_size
    )

    # ── BLOCK 4B: MODEL ──────────────────────────────────────────────────────
    print(f"\n🧠 Building EfficientNet-B3 model...")
    num_classes = len(full_dataset.dataset.classes) if args.fast else len(full_dataset.classes)
    model = get_deepcoin_model(num_classes=num_classes)
    model = model.to(device)    # ← move all 12M parameters to GPU memory

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # ── BLOCK 4C: LOSS FUNCTION ──────────────────────────────────────────────
    # CrossEntropyLoss = Softmax + NegativeLogLikelihood combined
    # It measures "how wrong" the model is on each batch
    # label_smoothing=0.1: prevents overconfident predictions
    #   Instead of learning "class 3987 = 100% sure", learns "class 3987 = 90% sure"
    #   This improves generalization to real-world imperfect photos
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # increased 0.1→0.15: less overconfident predictions

    # ── BLOCK 4D: OPTIMIZER ──────────────────────────────────────────────────
    # AdamW = Adam + Weight Decay fix
    # - lr=1e-4: learning rate (how big each weight update step is)
    # - weight_decay=0.01: L2 regularization (penalizes large weights → prevents overfitting)
    # WHY AdamW over SGD? AdamW adapts learning rate per parameter — converges faster
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # ── BLOCK 4E: SCHEDULER ──────────────────────────────────────────────────
    # CosineAnnealingLR: smoothly decays LR following a cosine curve.
    # No plateaus, no manual tuning — proven better than ReduceLROnPlateau
    # for fine-tuning pretrained models on image datasets.
    #
    # How it looks: LR starts at args.lr (1e-4), smoothly falls to eta_min (1e-6)
    # by the final epoch. This is the "cosine annealing" shape:
    #
    #   LR
    #   ┐
    #   │ \
    #   │   \
    #   │     \___
    #   └──────────► epochs
    #
    # T_max = total epochs = one full cosine period
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,      # ← full cosine period over entire training run
        eta_min=1e-6            # ← minimum LR floor (never goes below this)
    )

    # ── AMP SCALER ───────────────────────────────────────────────────────────
    # GradScaler: manages float16/float32 switching for AMP.
    # It scales the loss up before backward() to prevent float16 underflow,
    # then scales gradients back down before optimizer.step().
    # enabled=True only when CUDA is available (CPU doesn't benefit from AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if device.type == 'cuda':
        print(f"\n⚡ AMP enabled — float16 training active (saves ~50% VRAM, ~2x faster)")

    # ── RESUME FROM CHECKPOINT ───────────────────────────────────────────────
    # If --resume flag is set, load the last saved checkpoint and continue
    # from where we left off instead of starting from scratch.
    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0

    checkpoint_path = 'models/checkpoint_last.pth'
    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"\n♻️  Resuming from checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch     = ckpt['epoch'] + 1
            best_val_acc    = ckpt['best_val_acc']
            patience_counter = ckpt.get('patience_counter', 0)
            print(f"   Resumed at epoch {start_epoch}, best val so far: {best_val_acc:.2f}%")
        else:
            print(f"\n⚠️  --resume set but no checkpoint found at {checkpoint_path}. Starting fresh.")

    # ── BLOCK 7: TRAINING LOOP WITH EARLY STOPPING ──────────────────────────
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Time':>7}")
    print("-" * 65)

    early_stop_patience = 10    # stop if no improvement for 10 epochs

    os.makedirs('models', exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # Train for one epoch
        # use_mixup: skip Mixup for the first 3 epochs (warmup)
        # Reason: the model needs to first learn basic features before we
        # start blending images. Mixup too early → confused gradients.
        use_mixup = (epoch > 3) and not args.fast
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.epochs, num_classes, use_mixup=use_mixup
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Cosine LR decay — step every epoch (no condition needed)
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        marker = " ✅" if val_acc > best_val_acc else ""
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>10.4f} {val_acc:>8.2f}% {current_lr:>10.2e} {elapsed:>5.0f}s{marker}")

        # ── SAVE LAST CHECKPOINT (for resume) ────────────────────────────────
        # checkpoint_last.pth is overwritten every epoch — always has latest state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
        }, checkpoint_path)

        # ── CHECKPOINTING ────────────────────────────────────────────────────
        # Save model if this is the best val accuracy we've seen
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
            }, 'models/best_model.pth')

            # Also save just class mapping for inference
            torch.save({
                'class_to_idx': full_dataset.class_to_idx if not args.fast else full_dataset.dataset.class_to_idx,
                'idx_to_class': full_dataset.idx_to_class if not args.fast else full_dataset.dataset.idx_to_class,
                'num_classes': num_classes,
            }, 'models/class_mapping.pth')

        else:
            patience_counter += 1

        # ── EARLY STOPPING ───────────────────────────────────────────────────
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} — no improvement for {early_stop_patience} epochs")
            break

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Best model saved to:      models/best_model.pth")

    # ── FINAL TEST EVALUATION ────────────────────────────────────────────────
    # Load the BEST model (not the last epoch!) and evaluate on test set
    # This is the honest number you report to YEBNI/ESPRIT
    print(f"\n🧪 Loading best model for final test evaluation...")
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n📊 FINAL TEST RESULTS (honest, never-seen-during-training):")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"\n   This is the number you present to YEBNI and ESPRIT. 🎓")
    print("=" * 65)


if __name__ == "__main__":
    main()
