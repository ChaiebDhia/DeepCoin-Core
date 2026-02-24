# 🪙 DeepCoin-Core

> **An Agentic Multi-Modal System for Archaeological Numismatics & Historical Synthesis**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15+-000000.svg?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-F7DF1E.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/features/actions)

---

## Overview

**DeepCoin-Core** is an end-to-end industrial AI system that classifies degraded archaeological coins and synthesizes verified historical reports. It combines a fine-tuned **EfficientNet-B3 CNN** for visual classification with a **5-agent LangGraph state machine** for historical reasoning, delivering sub-second inference and professional PDF reports.

**Context**: PFE Final Year Engineering Internship — ESPRIT School of Engineering × YEBNI, Tunisia (Feb – Jul 2026)

---

## The Problem

Archaeological coins present a unique classification challenge:

- **Physical degradation** — worn by centuries of circulation and corrosion
- **Fragmentation** — broken or incomplete specimens
- **Fine-grained similarity** — subtle visual differences between hundreds of types
- **Data scarcity** — severe long-tail distribution (many types have <10 images)
- **Domain gap** — standard ImageNet pre-trained models fail on ancient coinage

A numismatic expert would spend 1–2 hours identifying a single corroded specimen. A museum with 500 unidentified coins faces weeks of manual work.

**DeepCoin reduces this to under 2 seconds per coin, always returning a useful output.**

---

## Solution: Two-Stage Hybrid AI Pipeline

### Stage 1 — Deep Learning (Physical Analysis)

```
Raw coin photo
    ↓
CLAHE Enhancement (LAB color space, L-channel only)
    → Reveals worn surface details without color distortion
    ↓
Aspect-preserving resize → 299×299 with zero-padding
    → Preserves coin geometry (no distortion)
    ↓
EfficientNet-B3 (12M params, ImageNet pretrained, fine-tuned)
    → 1536-dimensional feature extraction
    → Softmax probabilities across 438 coin classes
    → Output: top-1 class + confidence score + top-5 predictions
```

### Stage 2 — Agentic AI (Historical Reasoning)

The CNN confidence score routes the analysis to the appropriate specialist agent:

```
confidence > 0.85  →  Historian Agent
                      ChromaDB semantic search → Nomisma SPARQL → LLM synthesis
                      Returns: emperor, period, mint, significance, sources

0.40 ≤ conf ≤ 0.85 →  Forensic Validator Agent
                      OpenCV color histogram analysis (metal type detection)
                      Historical consistency checks
                      → If anomaly: Human Review Queue
                      → If clean: synthesis

confidence < 0.40  →  Visual Investigator Agent
                      Vision-Language Model (Gemini 2.5 Flash via GitHub Models)
                      Zero-shot attribute extraction: metal, portrait, inscription, symbols
                      Ensures no empty response for unknown coin types

All paths → Editor-in-Chief (Synthesis Agent)
          → Structured Markdown report → PDF
          → FastAPI response → Next.js renders report
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER (Browser)                                 │
│         Next.js 15 + TypeScript + Tailwind CSS + shadcn/ui          │
│    Upload → Live Agent Progress (WebSocket) → PDF Report Viewer     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Nginx (Port 80) — Reverse Proxy                  │
│          /api/* → FastAPI (port 8000)  |  /* → Next.js (port 3000) │
└──────────┬──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                  FastAPI + Uvicorn (Port 8000)                      │
│  POST /api/classify  |  GET /api/health  |  GET /api/history        │
│  GET  /api/history/{id}  |  WS /ws/classify/{session_id}           │
└──────────┬──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│             LangGraph State Machine — Gatekeeper Orchestrator       │
│                                                                     │
│  [preprocess] → [vision_cnn] → route_by_confidence()               │
│                                                                     │
│  conf < 0.40  → [investigator] ──────────────────────┐             │
│  0.40–0.85   → [validator] → human_review? ──────────┤             │
│  conf > 0.85  → [historian] ─────────────────────────┤             │
│                                                       ▼             │
│                                               [synthesis]           │
│                                         Markdown → PDF report       │
└──────────┬────────────────────────────────────────────┬────────────┘
           │                                            │
┌──────────▼──────────┐              ┌──────────────────▼────────────┐
│  ChromaDB           │              │  LLM Fallback Chain            │
│  CN metadata        │              │  1. GitHub Models (Gemini 2.5) │
│  indexed vectors    │              │     → free, Copilot Pro        │
│  semantic search    │              │  2. Google AI Studio (Gemini)  │
└─────────────────────┘              │     → free tier, 1,500 req/day │
                                     │  3. Nomisma.org SPARQL         │
┌──────────────────────────────────┐ │     → academic linked open data│
│  Data Persistence Layer          │ │  4. Wikipedia API (last resort)│
│  PostgreSQL — history, audit log │ │     → prose only, flagged      │
│  Redis       — result cache      │ └────────────────────────────────┘
│  LocalStack S3 — image storage   │
└──────────────────────────────────┘
```

---

## The 5-Agent Sovereign Squad

### 1. Gatekeeper (Orchestrator)
The LangGraph state machine brain. Routes every analysis based on CNN confidence to the appropriate specialist. Manages the `CoinState` context shared by all agents. Handles retries, failures, and human-in-the-loop breakpoints.

**Routing thresholds**: `< 0.40` → Investigator | `0.40–0.85` → Validator | `> 0.85` → Historian

### 2. Visual Investigator (Attribute Expert)
Handles low-confidence and out-of-distribution coins. Sends the coin image to a Vision-Language Model and extracts structured visual attributes — metal color, portrait direction, visible inscription characters, iconographic symbols. Guarantees a useful analytical output even when the CNN cannot classify.

### 3. Forensic Validator (Truth Seeker)
Applies OpenCV-based forensic analysis to mid-confidence predictions. Computes HSV color histograms to verify the detected metal type matches the expected metal for the predicted coin class. Flags inconsistencies and routes suspicious specimens to the human review queue.

### 4. Historian (RAG Specialist)
Retrieves verified historical context for high-confidence predictions. Queries ChromaDB (populated from CN dataset metadata) using semantic search, enriches with Nomisma.org structured data (emperors, reign dates, mint locations), then uses an LLM to synthesize a narrative from those verified sources — never from the LLM's own memory alone.

### 5. Editor-in-Chief (Synthesis Agent)
Compiles all agent outputs into a single structured Markdown report, then converts it to PDF. The report includes: coin image, classification confidence, historical narrative, forensic assessment, visual description (when Investigator was used), related coin types, complete source citations, and expert review status.

---

## LangGraph State Contract

All agents communicate exclusively through this shared state:

```python
class CoinState(TypedDict):
    image_path: str
    preprocessed_image: bytes
    cnn_prediction: dict        # {"class_id": int, "label": str, "confidence": float, "top5": list}
    visual_description: str     # Populated by Investigator; empty string if not used
    validation_result: dict     # Populated by Validator; empty dict if not used
    historical_context: str     # Populated by Historian; empty string if not used
    final_report: str           # Populated by Editor-in-Chief (Markdown)
    human_review_required: bool
    human_approved: bool
    route_taken: Literal["investigator", "validator", "historian"]
```

---

## Technology Stack

### Deep Learning

| Technology | Version | Purpose |
|---|---|---|
| PyTorch | 2.6.0+cu124 | Neural network training and inference |
| torchvision | 0.21+ | EfficientNet-B3 pretrained weights |
| EfficientNet-B3 | ImageNet pretrained | 12M param CNN, 1536-dim features, 438-class head |
| OpenCV | 4.10+ | CLAHE preprocessing, HSV histogram forensics |
| Albumentations | 1.4+ | Training augmentation pipeline |
| NumPy | 2.x | Numerical operations |

### Agentic AI

| Technology | Version | Purpose |
|---|---|---|
| LangGraph | 0.3+ | State machine orchestration with cycles and conditional routing |
| LangChain | 0.3+ | Agent tooling and prompt management |
| Gemini 2.5 Flash | via GitHub Models | VLM for visual description (free, Copilot Pro) |
| Gemini 2.5 Flash | via Google AI Studio | LLM fallback (free tier, 1,500 req/day) |
| ChromaDB | 0.6+ | Local vector database for RAG |
| sentence-transformers | 3.3+ | Text embeddings for semantic search |

### Backend

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.115+ | Async Python web framework with auto-docs |
| Uvicorn | 0.40+ | ASGI server |
| Pydantic | 2.x | Request/response validation schemas |
| SQLAlchemy | 2.x | PostgreSQL ORM (async) |
| Alembic | Latest | Database migration versioning |

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 15 | React framework with Server Components |
| TypeScript | 5 | Type-safe JavaScript |
| Tailwind CSS | 4 | Utility-first styling |
| shadcn/ui | Latest | Accessible component library (Radix UI) |
| TanStack Query | 5 | Server state management and caching |
| Zustand | 4 | Lightweight client state |

### Infrastructure

| Technology | Version | Purpose |
|---|---|---|
| Docker Compose | 2.x | Multi-container local orchestration |
| PostgreSQL | 17 | Relational database (ACID, JSONB) |
| Redis | 7 | Result caching and session management |
| Nginx | 1.27 | Reverse proxy and load balancing |
| LocalStack | 3.x | AWS S3 + Lambda local simulation |
| GitHub Actions | — | CI/CD: test, lint, format on every push |
| pytest | 8.x | Python unit and integration testing |
| Jest + Playwright | 30 / 1.50+ | Frontend unit and end-to-end testing |

---

## Performance

| Metric | Value | Notes |
|---|---|---|
| CNN Test Accuracy | 79.08% | Single-pass, 438 classes, 1,152 test images |
| CNN Accuracy (TTA) | **80.03%** | 5-pass Test-Time Augmentation (+0.95%) |
| Mean F1 Score | 0.7763 | Macro-averaged across 438 classes |
| Top Confusion Pair | 3314 → 3987 | 10× misclassification frequency |
| Target Accuracy | >85% | Gap: ~5pp, addressable with ensemble/larger model |
| Training Duration | ~103 min | RTX 3050 Ti (4.3GB VRAM), CUDA 12.4 |
| Best Epoch | 52 / 100 | Val accuracy 79.25%, early stopping patience=10 |
| Inference Target | <500ms | Per image end-to-end |
| API Response Target | <2s | Upload → full report |

---

## Dataset

### Corpus Nummorum (CN) v1

| Property | Value |
|---|---|
| Source | [corpus-nummorum.eu](https://www.corpus-nummorum.eu/) |
| Total images | 115,160 ancient coin photographs |
| Original classes | 9,716 unique coin types |
| Distribution | Severe long-tail — majority of types have <10 images |

### Filtered Dataset (Training Ready)

| Property | Value |
|---|---|
| Classes | 438 (filtered: ≥10 images per class) |
| Total images | 7,677 preprocessed images |
| Average per class | 17.5 images |
| Image size | 299×299 RGB JPEG |
| Preprocessing | CLAHE (LAB, clipLimit=2.0, tile=8×8) + aspect-preserving resize |
| Split | 70% train / 15% validation / 15% test (stratified, seed=42) |

**Filtering rationale**: CNNs cannot reliably learn to classify from 1–3 examples. Applying a hard threshold of ≥10 images per class sacrifices breadth (438 vs 9,716 types) in exchange for classification reliability. Transfer learning from ImageNet reduces the minimum data requirement from thousands of examples to tens.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- NVIDIA GPU with CUDA 12.x (recommended for inference speed)

### Setup

```bash
# Clone
git clone https://github.com/ChaiebDhia/DeepCoin-Core.git
cd DeepCoin-Core

# Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (never commit this file):

```env
LLM_PROVIDER=github_models
GITHUB_TOKEN=ghp_your_token_here        # GitHub PAT with models:read scope
GOOGLE_AI_API_KEY=your_key_here         # Google AI Studio key (fallback)
LLM_MODEL=Gemini-2.5-Flash

CHROMA_DB_PATH=./data/chromadb
POSTGRES_URL=postgresql://localhost:5432/deepcoin
REDIS_URL=redis://localhost:6379
LOCALSTACK_ENDPOINT=http://localhost:4566
```

### Data Preprocessing

```bash
# Place raw CN dataset in data/raw/CN_dataset_v1/
python src/data_pipeline/prep_engine.py
# Output: 7,677 processed images in data/processed/
```

### CNN Training (already completed — model in models/)

```bash
python scripts/train.py
# Trains EfficientNet-B3, saves best checkpoint to models/best_model.pth
# Best result: 79.08% test accuracy (80.03% with TTA)
```

### Model Evaluation

```bash
# Standard evaluation
python scripts/audit.py
# Output: confusion matrix, per-class F1, top-K accuracy

# Test-Time Augmentation evaluation
python scripts/evaluate_tta.py
# Result: +0.95% accuracy improvement
```

### Full Stack (Docker)

```bash
# Coming in Layer 6 — Docker Compose setup
docker compose up
# Services: FastAPI + Next.js + ChromaDB + PostgreSQL + Redis + Nginx + LocalStack
```

---

## Project Structure

```
DeepCoin-Core/
│
├── src/
│   ├── data_pipeline/
│   │   └── prep_engine.py          # CLAHE + aspect-preserving resize pipeline
│   ├── core/
│   │   ├── model_factory.py        # EfficientNet-B3 definition (Dropout=0.4)
│   │   ├── dataset.py              # DeepCoinDataset + Albumentations transforms
│   │   ├── inference.py            # CoinInference class [Layer 1 — NEXT]
│   │   └── knowledge_base.py       # ChromaDB client wrapper [Layer 2]
│   ├── agents/
│   │   ├── gatekeeper.py           # LangGraph orchestrator + routing logic
│   │   ├── historian.py            # RAG specialist (ChromaDB + Nomisma + LLM)
│   │   ├── investigator.py         # VLM agent (Gemini 2.5 Flash via GitHub Models)
│   │   ├── validator.py            # Forensic analysis (OpenCV histograms)
│   │   └── synthesis.py            # Report generation (Markdown → PDF)
│   └── api/
│       ├── main.py                 # FastAPI application entry point
│       ├── routes/
│       │   ├── classify.py         # POST /api/classify [Layer 4]
│       │   └── history.py          # GET /api/history [Layer 4]
│       └── schemas.py              # Pydantic request/response models [Layer 4]
│
├── scripts/
│   ├── train.py                    # CNN training loop (V3, 729 lines) ✅
│   ├── audit.py                    # Model audit — F1, confusion matrix ✅
│   ├── evaluate_tta.py             # TTA evaluation ✅
│   ├── predict.py                  # CLI inference tool [Layer 1 — NEXT]
│   └── build_knowledge_base.py     # CSV + Nomisma → ChromaDB [Layer 2]
│
├── models/
│   ├── best_model.pth              # V3 weights (epoch 52, val 79.25%, test 79.08%, TTA 80.03%) ✅
│   ├── best_model_v1_80pct.pth     # Early checkpoint (epoch 3, val 21.33%) — misleading name, NOT the 80% model
│   └── class_mapping.pth           # {class_to_idx, idx_to_class, n=438} ✅
│
├── data/
│   ├── processed/                  # 7,677 images × 438 classes ✅
│   ├── metadata/                   # CN dataset CSVs → Layer 2
│   └── raw/                        # Original dataset (gitignored)
│
├── tests/
│   ├── unit/                       # Per-module unit tests → Layer 7
│   └── integration/                # End-to-end pipeline tests → Layer 7
│
├── frontend/                       # Next.js 15 app → Layer 5
├── notebooks/                      # Exploration and analysis
│
├── docker-compose.yml              # 7-service orchestration → Layer 6
├── .github/workflows/ci.yml        # pytest + flake8 + black → Layer 7
├── requirements.txt                # All Python dependencies ✅
├── .env                            # Secrets (gitignored — NEVER COMMIT)
└── .gitignore
```

---

## Build Layers — Current Progress

The system is built in strict dependency order. Each layer is completed and tested before the next begins.

| Layer | Name | Status | Description |
|---|---|---|---|
| 0 | Foundation | ✅ Complete | Trained CNN model at 80.03% accuracy |
| 1 | Inference Engine | 🔲 Next | `inference.py` + `predict.py` — load model, classify new images |
| 2 | Knowledge Base | 🔲 Pending | CN CSVs → ChromaDB vector DB for RAG |
| 3 | Agents | 🔲 Pending | All 5 agents fully implemented and tested |
| 4 | FastAPI Routes | 🔲 Pending | `/api/classify`, `/api/history`, WebSocket |
| 5 | Next.js Frontend | 🔲 Pending | Upload UI, live progress, PDF viewer |
| 6 | Docker + Infra | 🔲 Pending | Full Docker Compose stack, 7 services |
| 7 | Tests + CI/CD | 🔲 Ongoing | pytest, GitHub Actions, black, flake8 |

---

## CNN Training Configuration (V3)

```python
model         = EfficientNet-B3 (ImageNet pretrained, Dropout=0.4)
optimizer     = AdamW(lr=1e-4, weight_decay=0.01)
scheduler     = CosineAnnealingLR(T_max=100, eta_min=1e-6)
loss          = CrossEntropyLoss(label_smoothing=0.1)
augmentation  = Albumentations pipeline (rotation, brightness, elastic)
mixup         = alpha=0.2 (Beta distribution blending)
amp           = torch.amp.GradScaler('cuda') + autocast
gradient_clip = max_norm=1.0
batch_size    = 16  (GPU memory constraint: RTX 3050 Ti, 4.3GB VRAM)
early_stop    = patience=10 epochs on validation accuracy
pin_memory    = True
non_blocking  = True (async GPU transfer)
seed          = 42
```

---

## Key Engineering Decisions

| Decision | Choice | Rationale |
|---|---|---|
| CNN architecture | EfficientNet-B3 | Optimal accuracy/parameter ratio; B7 exceeds VRAM budget |
| Preprocessing | CLAHE in LAB space | Enhances contrast without distorting metal color values |
| Resize strategy | Aspect-preserving + zero-padding | Preserves coin geometry for accurate feature extraction |
| Agent framework | LangGraph | Conditional routing, cycles, human-in-loop — impossible in CrewAI |
| VLM provider | GitHub Models (Gemini 2.5 Flash) | Free with Copilot Pro student, vision-capable, OpenAI-compatible |
| VLM fallback | Google AI Studio | Free tier 1,500 req/day; identical model; 1-line config switch |
| Primary data source | CN dataset metadata CSVs | On-disk, validated by Berlin-Brandenburg Academy of Sciences |
| External data | Nomisma.org SPARQL | Academic numismatic linked open data — structured, authoritative |
| Wikipedia | Last resort only | Unverifiable for facts; prose only; always flagged in output |
| Vector DB | ChromaDB | Local, embeddable, zero-config for development |
| Backend | FastAPI (async) | Auto-docs, Pydantic validation, async request handling |
| Cloud simulation | LocalStack | Demonstrate S3/Lambda skills without AWS account costs |

---

## Data Sources for Historical Context

The system uses a verified fallback chain — always using the most authoritative available source:

1. **CN Dataset Metadata** (primary) — Structured CSV data from Corpus Nummorum, validated by the Berlin-Brandenburg Academy of Sciences and the German Research Foundation (DFG)
2. **Nomisma.org SPARQL** (secondary) — Academic linked open data for numismatics; emperor names, reign periods, mint locations as structured RDF
3. **LLM Synthesis** (tertiary) — Gemini 2.5 Flash generates narrative prose from the structured data retrieved in steps 1–2; the LLM writes, it does not invent
4. **Wikipedia API** (last resort) — Used only for emperor biography narrative when no structured source covers the subject; always marked `"Source: Wikipedia (unverified)"`

---

## API Reference (Planned)

```
POST   /api/classify
       Body: multipart/form-data { image: File }
       Returns: ClassificationResult (JSON)

GET    /api/health
       Returns: { api, ml_model, agents, database, chromadb }

GET    /api/history
       Returns: List[ClassificationSummary]

GET    /api/history/{id}
       Returns: ClassificationResult (full)

WS     /ws/classify/{session_id}
       Streams: AgentProgressEvent (live agent status)
```

---

## Contributing

Contributions are welcome. Please follow the workflow:

```bash
git checkout -b feature/your-feature-name
# make changes
git commit -m "feat: description of change"
git push origin feature/your-feature-name
# open Pull Request
```

All PRs must pass: `pytest` + `flake8` + `black --check` via GitHub Actions CI.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **[Corpus Nummorum](https://www.corpus-nummorum.eu/)** — Dataset, funded by the German Research Foundation (DFG) and Berlin-Brandenburg Academy of Sciences
- **[Nomisma.org](https://nomisma.org/)** — Numismatic linked open data standards
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[LangChain / LangGraph](https://langchain-ai.github.io/langgraph/)** — Agent orchestration
- **[FastAPI](https://fastapi.tiangolo.com/)** — Backend framework

---

## Contact

**Dhia Chaieb** — ESPRIT School of Engineering, Tunisia  
dhia.chaieb@esprit.tn | [@ChaiebDhia](https://github.com/ChaiebDhia)  
Internship partner: [YEBNI — Information & Communication](https://yebni.com)
