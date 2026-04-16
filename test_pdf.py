import sys
from src.agents.gatekeeper import Gatekeeper

gk = Gatekeeper()
try:
    res = gk.analyze("data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg", tta=True, language="fr")
    print(res.get("pdf_path"))
except Exception as e:
    import traceback
    traceback.print_exc()
