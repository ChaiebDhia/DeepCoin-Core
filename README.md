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
The LangGraph state machine brain. Routes every analysis based on CNN confidence to the appropriate specialist. Manages the `CoinState` shared by all agents. Features: structured `logging` with per-node timing (cnn / historian / validator / investigator / synthesis), retry with exponential backoff (2×, 1.5s→3.0s) on LLM 429/503 errors, and graceful per-node degradation — one failing agent stores `{"_error": ...}` in its result and lets the pipeline reach synthesis instead of crashing.

**Routing thresholds**: `< 0.40` → Investigator | `0.40–0.85` → Validator + Historian | `> 0.85` → Historian

### 2. Visual Investigator (Attribute Expert)
Handles low-confidence and out-of-distribution coins. Sends the coin image to a Vision-Language Model (`qwen3-vl:4b` via Ollama) and extracts structured visual attributes — metal color, portrait direction, visible inscription characters, iconographic symbols. When no vision model is available, falls back to a local OpenCV analysis: HSV histogram across 3 crop sizes for metal detection, Sobel edge density for condition estimate. Cross-references the full 9,541-type KB — not just the 438 CNN training types. Always returns a useful analytical output, even if the CNN cannot classify.

### 3. Forensic Validator (Truth Seeker)
Applies multi-scale OpenCV HSV forensic analysis to mid-confidence predictions. Runs independently on three center-crop sizes (40% / 60% / 80%) and takes a majority vote on the detected metal (gold / bronze / silver). Returns `detection_confidence` (float 0–1, mean pixel coverage) and `uncertainty` (low/medium/high based on scale agreement). Compares detected metal against the KB-stored expected material for the predicted type. Flags material mismatches in the report.

### 4. Historian (RAG Specialist)
Retrieves verified historical context for high-confidence predictions. Queries the hybrid RAG engine (BM25 + ChromaDB vector + RRF merge over 47,705 chunks). Injects retrieved chunks as labeled `[CONTEXT 1–5]` blocks into the LLM prompt with a strict instruction: *"Using ONLY the contexts above (cite [CONTEXT N]), write a 3-paragraph professional analysis. Do not add any fact not present in the context."* Supports 4 LLM providers: GitHub Models → Google AI Studio → Ollama (gemma3:4b) → fallback (KB field concatenation, no hallucination).

### 5. Editor-in-Chief (Synthesis Agent)
Compiles all agent outputs into a single structured plain-text report and converts it to a professional PDF. PDF is rendered entirely with direct fpdf2 draw primitives — no Markdown parsing, no external font loading. Features: navy header band, bordered tables with alternating row shading, blue section rule lines, Greek-to-Latin transliteration (`_GREEK_MAP`, 48 chars) so ancient legends render correctly in Latin-1 encoded fonts.

---

## LangGraph State Contract

All agents communicate exclusively through this shared state:

```python
class CoinState(TypedDict, total=False):
    # inputs
    image_path          : str
    use_tta             : bool
    # after cnn_node
    cnn_prediction      : dict    # {"class_id": int, "label": str, "confidence": float, "top5": list, "tta_used": bool}
    route_taken         : Literal["historian", "validator", "investigator"]
    # agent outputs
    historian_result    : dict    # narrative, mint, date, material, llm_used, _error (if any)
    validator_result    : dict    # status, detection_confidence, uncertainty, warning, _error (if any)
    investigator_result : dict    # visual_description, detected_features, kb_matches, llm_used, _error (if any)
    # per-node timing (seconds, set progressively by each node)
    node_timings        : dict    # {"cnn": 0.54, "historian": 14.37, "synthesis": 0.47}
    # final outputs
    report              : str     # plain-text summary
    pdf_path            : Optional[str]
```

**Key design rule**: the `label` field (folder name = CN type ID, e.g. `"1015"`) must be used for all KB lookups, NOT `class_id` (which is the 0–437 softmax tensor index).

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
| CNN Accuracy (TTA) | **80.03%** | 8-pass Test-Time Augmentation (+0.78%) |
| Mean F1 Score | 0.7763 | Macro-averaged across 438 classes |
| Top Confusion Pair | 3314 → 3987 | 10× misclassification frequency |
| Target Accuracy | >85% | Gap: ~5pp |
| Training Duration | ~103 min | RTX 3050 Ti (4.3 GB VRAM), CUDA 12.4 |
| Best Epoch | 52 / 100 | Val accuracy 79.25%, early stopping patience=10 |
| Pipeline (Historian) | ~15 s | CNN + RAG lookup + Ollama gemma3:4b + PDF |
| Pipeline (Validator) | ~10 s | CNN + multi-scale HSV + Historian + PDF |
| Pipeline (Investigator) | ~3 s | CNN + OpenCV fallback + KB search + PDF |
| KB search latency | < 1 ms | ChromaDB + BM25, 47,705 vectors |
| PDF generation | ~0.4 s | Direct fpdf2 draw, Greek transliteration |

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
# LLM providers (any one is enough — system tries in priority order)
GITHUB_TOKEN=ghp_your_token_here        # GitHub PAT with models:read scope (Priority 1)
GOOGLE_API_KEY=your_key_here            # Google AI Studio API key (Priority 2)

# Local Ollama (Priority 3 — runs fully offline)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_VISION_MODEL=qwen3-vl:4b        # For Investigator; fallback to OpenCV if not pulled

# Database (Layers 4-6)
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

### Run the Full Pipeline (Layer 3 — Production Ready)

```bash
# Run all 3 routing paths (historian / validator / investigator)
# Logs: per-node timing, confidence, route, PDF path
python scripts/test_pipeline.py 2>$null

# Example single image prediction (CLI)
python scripts/predict.py data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg --tta
# Output: type=1015  confidence=91.1%  label=1015  tta=True

# Rebuild ChromaDB (only needed if metadata changes)
python scripts/rebuild_chroma.py
# Duration: ~9 min  |  Output: 47,705 vectors in data/metadata/chroma_db_rag/
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
| 0 | CNN Training | ✅ Complete | EfficientNet-B3 trained at 80.03% TTA accuracy (438 classes, 7,677 images) |
| 1 | Inference Engine | ✅ Complete | `inference.py` + `predict.py` — TTA, device auto-resolve, structured prediction dict |
| 2 | Knowledge Base | ✅ Complete | 9,541 types scraped × 5 semantic chunks = 47,705 ChromaDB vectors + BM25 index |
| 3 | Agent System | ✅ Complete | All 5 agents enterprise-grade; logging, retry, graceful degradation; 3/3 routes tested |
| 4 | FastAPI Routes | 🔲 Next | `POST /api/classify`, `GET /api/history`, WebSocket live progress |
| 5 | Next.js Frontend | 🔲 Pending | Upload UI, real-time agent progress, PDF inline viewer |
| 6 | Docker + Infra | 🔲 Pending | Full Docker Compose stack (7 services), Redis cache, LocalStack S3 |
| 7 | Tests + CI/CD | 🔲 Pending | pytest, Jest, Playwright, GitHub Actions |

**Layer 3 end-to-end results (February 27, 2026):**
```
Route 1 — HISTORIAN    : type=1015   conf=91.1%   time=15.4s   PDF ✓   [PASS]
Route 2 — VALIDATOR    : type=21027  conf=42.9%   material=consistent  det_conf=0.73  time=9.8s    PDF ✓   [PASS]
Route 3 — INVESTIGATOR : type=544    conf=21.3%   KB_matches=3  llm=False (OpenCV fallback)  time=3.1s  PDF ✓   [PASS]
```

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
| LLM Provider Chain | Priority order | 1. GitHub Models (Gemini 2.5 Flash, free) → 2. Google AI Studio (free tier) → 3. Ollama gemma3:4b (local) → 4. Structured fallback (no LLM, no crash) |
| Vision LLM | qwen3-vl:4b via Ollama | For Investigator; OpenCV fallback if not downloaded |
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
