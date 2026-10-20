"""
routing_check.py -- does the confidence score mean what the router assumes?

Reads eval_by_source.csv (written by eval_by_source.py) and prints:
  1. Accuracy by confidence bin (a reliability table) and Expected Calibration Error.
  2. How the 3 routes in the agent graph would split the test set at the current
     thresholds (<0.40 investigator, 0.40-0.85 validator, >0.85 historian), with the
     CNN's own accuracy inside each route.

Run from the repo root:  python scripts/routing_check.py
"""
import csv
import math

rows = []
with open("eval_by_source.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((float(r["confidence"]), int(r["correct"]), r["source"]))
N = len(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0001)]
print(f"\nTest images: {N}  |  overall accuracy {sum(c for _, c, _ in rows) / N * 100:.1f}%\n")
print("Reliability table (does confidence match accuracy?)")
print(f"{'conf bin':<12}{'n':>6}{'share%':>8}{'mean conf%':>12}{'accuracy%':>11}{'95% CI':>12}")
ece = 0.0
for lo, hi in bins:
    sel = [(p, c) for p, c, _ in rows if lo <= p < hi]
    n = len(sel)
    if n == 0:
        continue
    mc = sum(p for p, _ in sel) / n
    acc = sum(c for _, c in sel) / n
    l, h = wilson(sum(c for _, c in sel), n)
    ece += n / N * abs(acc - mc)
    print(f"{f'{lo:.1f}-{min(hi, 1.0):.1f}':<12}{n:>6}{n / N * 100:>8.1f}{mc * 100:>12.1f}{acc * 100:>11.1f}{f'[{l * 100:.0f}-{h * 100:.0f}]':>12}")
print(f"\nExpected Calibration Error: {ece * 100:.1f} points (0 = perfectly calibrated)")

print("\nCurrent routing thresholds (what the agent graph does today)")
print(f"{'route':<34}{'n':>6}{'share%':>8}{'CNN accuracy%':>15}{'95% CI':>12}")
routes = [
    ("investigator  (conf < 0.40)", lambda p: p < 0.40),
    ("validator     (0.40 <= conf <= 0.85)", lambda p: 0.40 <= p <= 0.85),
    ("historian     (conf > 0.85)", lambda p: p > 0.85),
]
for name, test in routes:
    sel = [c for p, c, _ in rows if test(p)]
    n = len(sel)
    if n == 0:
        print(f"{name:<34}{0:>6}")
        continue
    k = sum(sel)
    l, h = wilson(k, n)
    print(f"{name:<34}{n:>6}{n / N * 100:>8.1f}{k / n * 100:>15.1f}{f'[{l * 100:.0f}-{h * 100:.0f}]':>12}")
print("\nRead it like this: if accuracy in a bin is much HIGHER than its mean confidence,")
print("the model is under-confident and a fixed 0.85 / 0.40 threshold sends good answers down the wrong route.")
