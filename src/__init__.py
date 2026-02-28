"""
DeepCoin-Core
=============
End-to-end archaeological coin classification and historical analysis system.

Fine-tuned EfficientNet-B3 CNN + LangGraph multi-agent RAG pipeline.

Layers:
    0  CNN Training      (scripts/train.py)
    1  Inference Engine  (src/core/inference.py)
    2  RAG Knowledge Base(src/core/rag_engine.py)
    3  Agent System      (src/agents/)
    4  FastAPI Backend   (src/api/)
    5  Next.js Frontend  (frontend/)  — pending
    6  Docker + Infra    (docker-compose.yml) — pending
    7  Tests + CI/CD     (tests/, .github/workflows/) — pending

Academic context:
    PFE — ESPRIT School of Engineering x YEBNI, Tunisia
    Student: Dhia Chaieb <dhia.chaieb@esprit.tn>
    Period:  February – July 2026
"""

__version__ = "0.4.0"
__author__  = "Dhia Chaieb"
__email__   = "dhia.chaieb@esprit.tn"
