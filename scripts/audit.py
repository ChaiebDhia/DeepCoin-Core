"""
DeepCoin — Model Audit Script
==============================
Generates a full diagnostic report on the trained model:

    1. Confusion matrix (top-confused pairs heatmap)
    2. Top 10 worst-performing classes
    3. Top 5 confusion hotspots (class A mistaken for class B N times)
    4. 4×4 gallery of misclassified images (actual vs predicted)
    5. Per-class CSV report (Precision, Recall, F1 for all 438 classes)

Usage:
    python scripts/audit.py

Output files (all saved to reports/):
    reports/confusion_heatmap.png
    reports/misclassified_gallery.png
    reports/class_performance_audit.csv
"""

import os
import sys
import warnings

# Fix Windows encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", message=".*Error fetching version info.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.core.dataset import DeepCoinDataset, get_val_transforms
from src.core.model_factory import get_deepcoin_model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'models/best_model.pth'
MAPPING_PATH = 'models/class_mapping.pth'
DATA_DIR     = 'data/processed'
REPORT_DIR   = 'reports'
BATCH_SIZE   = 16
NUM_WORKERS  = 2
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD MODEL + REBUILD TEST LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_testset():
    """
    Loads the saved model weights and rebuilds the exact same test split
    that was used during training (same random_seed=42 guarantees identical split).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    # Load class mapping
    mapping = torch.load(MAPPING_PATH, map_location='cpu')
    num_classes = mapping['num_classes']
    idx_to_class = mapping['idx_to_class']   # {0: '1015', 1: '1017', ...}

    # Rebuild dataset with val transforms (no augmentation — clean eval)
    full_dataset = DeepCoinDataset(root_dir=DATA_DIR, transform=get_val_transforms())
    all_labels  = [label for _, label in full_dataset.samples]
    all_indices = list(range(len(full_dataset)))

    # Reproduce the exact same split as training (same seed)
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.15, stratify=all_labels, random_state=RANDOM_SEED
    )
    test_dataset = Subset(full_dataset, test_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # Load model weights
    model = get_deepcoin_model(num_classes=num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   Loaded model from epoch {checkpoint.get('epoch', '?')} "
          f"(val_acc={checkpoint.get('val_acc', 0):.2f}%)")
    print(f"   Test set: {len(test_dataset)} images across {num_classes} classes\n")

    return model, test_loader, test_dataset, full_dataset, idx_to_class, device


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RUN INFERENCE — collect all predictions
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, test_loader, test_dataset, full_dataset, device):
    """
    Runs the model on the entire test set.
    Returns:
        all_true    : list of true class indices
        all_pred    : list of predicted class indices
        all_conf    : list of confidence scores (max softmax probability)
        all_img_idx : absolute indices into full_dataset (for image retrieval)
    """
    all_true    = []
    all_pred    = []
    all_conf    = []
    all_img_idx = []   # track which images we're looking at

    print("Running inference on test set...")
    with torch.no_grad():
        offset = 0
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)

            all_true.extend(labels.numpy())
            all_pred.extend(pred.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())

            # Record absolute indices in full_dataset for image retrieval later
            batch_indices = test_dataset.indices[offset: offset + len(labels)]
            all_img_idx.extend(batch_indices)
            offset += len(labels)

    all_true    = np.array(all_true)
    all_pred    = np.array(all_pred)
    all_conf    = np.array(all_conf)
    all_img_idx = np.array(all_img_idx)

    overall_acc = (all_true == all_pred).mean() * 100
    print(f"\nOverall test accuracy: {overall_acc:.2f}%")
    return all_true, all_pred, all_conf, all_img_idx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CONFUSION HEATMAP (top confused pairs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_heatmap(all_true, all_pred, idx_to_class, top_n=30):
    """
    For 438 classes, a full 438×438 matrix is unreadable.
    Strategy: find the top_n classes with the most confusion activity,
    then plot only that sub-matrix.

    This is what Google/Meta engineers actually do for large-class models.
    """
    print(f"\n[1/4] Generating confusion heatmap (top {top_n} most confused classes)...")

    cm = confusion_matrix(all_true, all_pred)

    # Find the top_n classes with the highest off-diagonal confusion count
    # (i.e., classes that are most often confused with something else)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confusion_per_class = cm_no_diag.sum(axis=1) + cm_no_diag.sum(axis=0)
    top_indices = np.argsort(confusion_per_class)[-top_n:]
    top_indices = np.sort(top_indices)

    sub_cm     = cm[np.ix_(top_indices, top_indices)]
    sub_labels = [idx_to_class.get(i, str(i)) for i in top_indices]

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        sub_cm,
        annot=True, fmt='d',
        cmap='YlOrRd',
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        linewidths=0.3,
        ax=ax
    )
    ax.set_title(
        f'Confusion Matrix — Top {top_n} Most Confused Classes\n'
        f'(Diagonal = correct predictions, Off-diagonal = mistakes)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, 'confusion_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TOP 10 WORST CLASSES + TOP 5 CONFUSION HOTSPOTS
# ─────────────────────────────────────────────────────────────────────────────
def print_class_analysis(all_true, all_pred, idx_to_class):
    """
    Prints:
      - Top 10 classes with the lowest F1-score
      - Top 5 confusion pairs (class A mistaken for class B N times)
    """
    num_classes = len(idx_to_class)

    # ── PER-CLASS F1 ─────────────────────────────────────────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_pred,
        labels=list(range(num_classes)),
        zero_division=0,
        average=None
    )

    # Sort by F1 ascending (worst first)
    sorted_by_f1 = np.argsort(f1)

    print("\n" + "="*60)
    print(" TOP 10 WORST-PERFORMING CLASSES (lowest F1-score)")
    print("="*60)
    print(f"{'Rank':<5} {'Class':<12} {'F1':>6} {'Precision':>10} {'Recall':>8} {'Samples':>8}")
    print("-"*60)
    for rank, idx in enumerate(sorted_by_f1[:10], 1):
        class_name = idx_to_class.get(idx, str(idx))
        print(f"{rank:<5} {class_name:<12} {f1[idx]:>6.3f} {precision[idx]:>10.3f} "
              f"{recall[idx]:>8.3f} {support[idx]:>8}")

    # ── TOP 5 CONFUSION HOTSPOTS ──────────────────────────────────────────────
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)   # zero out diagonal (correct predictions)

    # Flatten and find top pairs
    flat_indices = np.argsort(cm.flatten())[-5:][::-1]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    print("\n" + "="*60)
    print(" TOP 5 CONFUSION HOTSPOTS")
    print(" (True class → misidentified as Predicted class)")
    print("="*60)
    print(f"{'Rank':<5} {'True Class':<14} {'Predicted As':<14} {'Times':>6}")
    print("-"*60)
    for rank, (r, c) in enumerate(zip(rows, cols), 1):
        true_cls = idx_to_class.get(r, str(r))
        pred_cls = idx_to_class.get(c, str(c))
        count    = cm[r, c]
        print(f"{rank:<5} {true_cls:<14} {pred_cls:<14} {count:>6}×")

    print()
    return precision, recall, f1, support


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: MISCLASSIFIED IMAGE GALLERY (4×4 grid)
# ─────────────────────────────────────────────────────────────────────────────
def save_misclassified_gallery(all_true, all_pred, all_conf, all_img_idx,
                                full_dataset, idx_to_class):
    """
    Saves a 4×4 grid of images the model got wrong.
    Each tile shows: the image, actual class, predicted class, confidence.

    This is the most impactful asset for the oral defense —
    showing you understand where and WHY the model fails.
    """
    print("[3/4] Generating misclassified image gallery...")

    # Find all misclassified samples
    wrong_mask  = (all_true != all_pred)
    wrong_idx   = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print("   No misclassified images found!")
        return

    # Sample up to 16 random wrong predictions
    n_show = min(16, len(wrong_idx))
    chosen = np.random.choice(wrong_idx, size=n_show, replace=False)
    chosen = np.sort(chosen)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        'Misclassified Images — What the Model Got Wrong\n'
        '(Green = Actual label | Red = Predicted label | % = model confidence)',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.3)

    for plot_pos, sample_idx in enumerate(chosen):
        ax = fig.add_subplot(gs[plot_pos // 4, plot_pos % 4])

        # Get the original image from full_dataset (unnormalized for display)
        img_dataset_idx = all_img_idx[sample_idx]
        img_path, _     = full_dataset.samples[img_dataset_idx]

        # Load raw image for display (not the normalized tensor)
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert('RGB')

        true_cls = idx_to_class.get(int(all_true[sample_idx]), str(all_true[sample_idx]))
        pred_cls = idx_to_class.get(int(all_pred[sample_idx]), str(all_pred[sample_idx]))
        conf     = all_conf[sample_idx] * 100

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f'Actual: {true_cls}\nPredicted: {pred_cls}\nConf: {conf:.1f}%',
            fontsize=8,
            color='black',
            pad=3,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8)
        )

    out_path = os.path.join(REPORT_DIR, 'misclassified_gallery.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: PER-CLASS CSV REPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_per_class_csv(precision, recall, f1, support, idx_to_class):
    """
    Saves a CSV with one row per class:
        class_id, class_name, precision, recall, f1_score, test_samples
    Sorted by F1 ascending (worst classes first — easier to find problems).
    """
    print("[4/4] Saving per-class CSV report...")

    rows = []
    for idx in range(len(f1)):
        rows.append({
            'class_idx':  idx,
            'class_name': idx_to_class.get(idx, str(idx)),
            'precision':  round(float(precision[idx]), 4),
            'recall':     round(float(recall[idx]),    4),
            'f1_score':   round(float(f1[idx]),        4),
            'test_samples': int(support[idx])
        })

    df = pd.DataFrame(rows).sort_values('f1_score', ascending=True)
    out_path = os.path.join(REPORT_DIR, 'class_performance_audit.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"   Saved: {out_path}")

    # Print summary stats
    print(f"\n   Summary:")
    print(f"   Classes with F1 >= 0.9:  {(df.f1_score >= 0.9).sum():>4} / {len(df)}")
    print(f"   Classes with F1 >= 0.7:  {(df.f1_score >= 0.7).sum():>4} / {len(df)}")
    print(f"   Classes with F1 >= 0.5:  {(df.f1_score >= 0.5).sum():>4} / {len(df)}")
    print(f"   Classes with F1 == 0.0:  {(df.f1_score == 0.0).sum():>4} / {len(df)}")
    print(f"   Mean F1 across all classes: {df.f1_score.mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DEEPCOIN MODEL AUDIT")
    print("=" * 60)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load everything
    model, test_loader, test_dataset, full_dataset, idx_to_class, device = \
        load_model_and_testset()

    # Run inference once — reuse results for all 4 reports
    all_true, all_pred, all_conf, all_img_idx = run_inference(
        model, test_loader, test_dataset, full_dataset, device
    )

    # Generate all 4 audit artifacts
    print("\n[1/4] Confusion heatmap...")
    plot_confusion_heatmap(all_true, all_pred, idx_to_class, top_n=30)

    print("\n[2/4] Class analysis...")
    precision, recall, f1, support = print_class_analysis(all_true, all_pred, idx_to_class)

    print("\n[3/4] Misclassified gallery...")
    save_misclassified_gallery(all_true, all_pred, all_conf, all_img_idx,
                                full_dataset, idx_to_class)

    print("\n[4/4] Per-class CSV...")
    save_per_class_csv(precision, recall, f1, support, idx_to_class)

    print("\n" + "=" * 60)
    print("  AUDIT COMPLETE")
    print(f"  All reports saved to: {os.path.abspath(REPORT_DIR)}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
