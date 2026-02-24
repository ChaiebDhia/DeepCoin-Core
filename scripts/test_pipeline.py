"""
Quick end-to-end test of the full Layer 3 pipeline.
Usage: python scripts/test_pipeline.py
"""
import sys
import warnings
import io
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
# Fix Windows console encoding for Greek/Unicode text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.agents.gatekeeper import Gatekeeper

TEST_IMAGE = "data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg"

print("Loading Gatekeeper (CNN + KB + all agents)...")
gk = Gatekeeper(save_pdf=True)
print("Gatekeeper ready.\n")

print(f"Analyzing: {TEST_IMAGE}")
result = gk.analyze(TEST_IMAGE, tta=False)

print("\n" + "=" * 60)
print(result["report"])
print("=" * 60)

state = result["state"]
print(f"\nRoute taken  : {state.get('route_taken')}")
print(f"CNN label    : {state['cnn_prediction']['label']}")
print(f"CNN conf     : {state['cnn_prediction']['confidence']:.1%}")
print(f"PDF saved    : {result['pdf_path']}")
