"""Temporary debug script — compare confidence on training vs uploaded images."""
import glob
import sys
sys.path.insert(0, ".")

from src.core.inference import CoinInference

ci = CoinInference(device="cuda")
print(f"Temperature: {ci._temperature}")
print()

# Test on ACTUAL training images
tests = [
    ("220",  glob.glob("data/processed/220/*.jpg")),
    ("1015", glob.glob("data/processed/1015/*.jpg")),
    ("8455", glob.glob("data/processed/8455/*.jpg")[:3]),
    ("325",  glob.glob("data/processed/325/*.jpg")[:3]),
]

for label, imgs in tests:
    print(f"--- CN {label} ({len(imgs)} imgs in processed/) ---")
    for p in imgs[:4]:
        r = ci.predict(p, tta=False)
        print(f"  {p[-40:]}")
        print(f"    pred={r['label']}  conf={r['confidence']:.1%}  correct={'YES' if r['label']==label else 'NO'}")
    print()
