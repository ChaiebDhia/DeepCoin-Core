"""
eval_by_source.py -- slice DeepCoin test accuracy and confidence by image source.

Put this file in DeepCoin-Core/scripts/ and run from the repo root:
    python scripts/eval_by_source.py

What it does (single pass, no TTA, same stratified 70/15/15 split, seed 42):
  1. Tags every image by source using its filename:
       *_p.jpg                          -> standard_p   (standard composite photo)
       CN_type_<id>_BNF_1966.<n>_...    -> BNF_1966     (the catalogue scans spot-checked earlier)
       CN_type_<id>_<SRC>_...           -> <SRC>        (e.g. MK, BNF)
       *gradcam*                        -> gradcam_artifact (heatmap files that leaked into data/)
  2. Prints how many images of each source are in the FULL set and in the TEST set.
  3. Prints, per source, on the TEST set only: n, accuracy with 95% Wilson interval,
     mean confidence, mean confidence when correct, mean confidence when wrong.
  4. Saves every test prediction to eval_by_source.csv so you can reuse it in the thread.

It does not change any model or data file.
"""
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `src` and sibling scripts importable when run as `python scripts/eval_by_source.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_tta import (  # noqa: E402  (reuse the exact helpers of the original eval)
    BATCH_SIZE, DATA_DIR, MAPPING_PATH, MODEL_PATH, NUM_WORKERS,
    RANDOM_SEED, TTADataset, get_tta_transforms,
)
from src.core.dataset import DeepCoinDataset  # noqa: E402
from src.core.model_factory import get_deepcoin_model  # noqa: E402


_SRC = re.compile(r"^CN_type_\d+_([A-Za-z]+)")
_BNF1966 = re.compile(r"_BNF_(1966\.\d+)_")


def source_of(path: str) -> str:
    """Filename scheme: CN_type_<id>_<SOURCE>_<source id>_cn_coin_<n>_o.jpg  (single-face images)
    or cn_coin_<n>_p.jpg (standard composite photos)."""
    name = os.path.basename(path)
    if "gradcam" in name.lower():
        return "gradcam_artifact"          # heatmap output that ended up in the data folder
    if name.endswith("_p.jpg") or name.endswith("_p.png"):
        return "standard_p"
    if _BNF1966.search(name):
        return "BNF_1966"                  # the catalogue scans I spot-checked earlier
    m = _SRC.match(name)
    if m:
        return m.group(1)                  # e.g. MK, BNF
    return "other"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mapping = torch.load(MAPPING_PATH, map_location="cpu")
    model = get_deepcoin_model(num_classes=mapping["num_classes"])
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    base = DeepCoinDataset(root_dir=DATA_DIR, transform=None)
    labels = [lbl for _, lbl in base.samples]
    all_idx = list(range(len(base)))
    _, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=labels, random_state=RANDOM_SEED
    )

    # ---- 1/2. provenance counts ----
    full_src = Counter(source_of(p) for p, _ in base.samples)
    test_src = Counter(source_of(base.samples[i][0]) for i in test_idx)
    print(f"\nFULL set: {len(base)} images | TEST set: {len(test_idx)} images")
    print("\nSource counts (full / test):")
    for s, c in full_src.most_common(15):
        print(f"  {s:<28} {c:>6} / {test_src.get(s, 0):>5}")

    # ---- 3. single-pass inference on the test set ----
    ds = TTADataset(base, test_idx, get_tta_transforms()[0])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    true, pred, conf = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Test set"):
            p = torch.softmax(model(x.to(device)), dim=1)
            c, k = p.max(dim=1)
            true += y.numpy().tolist()
            pred += k.cpu().numpy().tolist()
            conf += c.cpu().numpy().tolist()
    true, pred, conf = np.array(true), np.array(pred), np.array(conf)
    correct = true == pred

    print(f"\nOverall test accuracy (single pass): {correct.mean() * 100:.2f}%")

    by_src = defaultdict(list)
    for j, i in enumerate(test_idx):
        by_src[source_of(base.samples[i][0])].append(j)

    print("\nPer-source results on the TEST set (single pass):")
    print(f"{'source':<28}{'n':>5}{'acc%':>8}{'95% CI':>16}{'meanConf':>10}{'confOK':>8}{'confWrong':>10}")
    for s, js in sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:12]:
        js = np.array(js)
        k, n = int(correct[js].sum()), len(js)
        lo, hi = wilson(k, n)
        ok = conf[js][correct[js]]
        bad = conf[js][~correct[js]]
        print(
            f"{s:<28}{n:>5}{k / n * 100:>8.1f}{f'[{lo * 100:.0f}-{hi * 100:.0f}]':>16}"
            f"{conf[js].mean() * 100:>10.1f}"
            f"{(ok.mean() * 100 if len(ok) else float('nan')):>8.1f}"
            f"{(bad.mean() * 100 if len(bad) else float('nan')):>10.1f}"
        )

    # ---- 4. save for the thread ----
    with open("eval_by_source.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "true", "pred", "confidence", "correct"])
        for j, i in enumerate(test_idx):
            pth = base.samples[i][0]
            w.writerow([pth, source_of(pth), int(true[j]), int(pred[j]),
                        f"{conf[j]:.4f}", int(correct[j])])
    print("\nSaved eval_by_source.csv")
    print("Note: groups with n < 30 are anecdotes, not measurements.")


if __name__ == "__main__":
    main()