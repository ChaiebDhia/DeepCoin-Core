"""
DeepCoin CLI — Single-image coin classification
================================================
Usage:
    python scripts/predict.py --image path/to/coin.jpg
    python scripts/predict.py --image path/to/coin.jpg --tta
    python scripts/predict.py --image path/to/coin.jpg --tta --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure project root is on the Python path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.inference import CoinInference


# ── Confidence thresholds (must match gatekeeper.py) ──────────────────────────
THRESHOLD_HIGH = 0.85   # > 0.85  → Historian Agent (high confidence)
THRESHOLD_LOW  = 0.40   # < 0.40  → Investigator Agent (low confidence)
                        # 0.40–0.85 → Validator Agent (mid confidence)


def get_routing_label(confidence: float) -> str:
    """Return which agent would handle this result in the full pipeline."""
    if confidence > THRESHOLD_HIGH:
        return f"✅ HIGH   → Historian Agent  (conf > {THRESHOLD_HIGH})"
    elif confidence >= THRESHOLD_LOW:
        return f"⚠️  MID    → Validator Agent  ({THRESHOLD_LOW} ≤ conf ≤ {THRESHOLD_HIGH})"
    else:
        return f"❌ LOW    → Investigator Agent (conf < {THRESHOLD_LOW})"


def print_result(result: dict) -> None:
    """Pretty-print the inference result to stdout."""
    sep = "─" * 56

    print(f"\n{sep}")
    print(f"  🪙  DeepCoin Classification Result")
    print(sep)
    print(f"  Predicted class : {result['label']}")
    print(f"  Class ID        : {result['class_id']}")
    print(f"  Confidence      : {result['confidence'] * 100:.2f}%")
    print(f"  Inference time  : {result['inference_time_ms']} ms")
    print(f"  TTA used        : {'Yes (5 passes)' if result['tta_used'] else 'No'}")
    print(f"\n  Agent routing   : {get_routing_label(result['confidence'])}")
    print(f"\n  Top-5 predictions:")
    for entry in result["top5"]:
        bar_len  = int(entry["confidence"] * 30)
        bar      = "█" * bar_len + "░" * (30 - bar_len)
        marker   = " ◀" if entry["rank"] == 1 else ""
        print(f"    {entry['rank']}. [{bar}] {entry['confidence']*100:5.2f}%  {entry['label']}{marker}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a single archaeological coin image using DeepCoin-Core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict.py --image data/processed/1015/coin_001.jpg
  python scripts/predict.py --image coin.jpg --tta
  python scripts/predict.py --image coin.jpg --tta --device cpu
        """,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the coin image (jpg or png)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable Test-Time Augmentation (5 passes, ~+1%% accuracy, 5× slower)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force device (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override path to model checkpoint (default: models/best_model.pth)",
    )

    args = parser.parse_args()

    # ── Validate image path ────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load inference engine ──────────────────────────────────────────────────
    print("\n[predict.py] Loading model...")
    t_load = time.time()

    kwargs = {"device": args.device}
    if args.model:
        kwargs["model_path"] = args.model

    engine = CoinInference(**kwargs)
    print(f"[predict.py] Model ready in {int((time.time() - t_load) * 1000)} ms")

    # ── Run prediction ─────────────────────────────────────────────────────────
    if args.tta:
        print("[predict.py] Running 5-pass TTA inference...")
    else:
        print("[predict.py] Running single-pass inference...")

    result = engine.predict(image_path, tta=args.tta)

    # ── Print result ───────────────────────────────────────────────────────────
    print_result(result)


if __name__ == "__main__":
    main()
