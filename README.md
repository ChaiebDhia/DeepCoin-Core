# DeepCoin-Core

> **From a degraded photograph to a verified numismatic intelligence report  in under 20 seconds.**
>
> DeepCoin-Core fuses a fine-tuned **Deep Learning** vision model with a production-grade **Retrieval-Augmented Generation (RAG)** pipeline and a five-agent AI system to classify, validate, and narrate archaeological coins at museum quality.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What is Deep Learning? What is RAG?](#what-is-deep-learning-what-is-rag)
3. [System Architecture](#system-architecture)
4. [The Five Agents](#the-five-agents)
5. [Deep Learning Model](#deep-learning-model)
6. [How We Built the Knowledge Base](#how-we-built-the-knowledge-base)
7. [RAG Engine](#rag-engine)
8. [Technology Stack](#technology-stack)
9. [Performance](#performance)
10. [Quick Start](#quick-start)
11. [API Reference](#api-reference)
12. [Project Structure](#project-structure)
13. [Build Layers](#build-layers)
14. [Engineering Decisions](#engineering-decisions)
15. [Academic Context](#academic-context)

---

## The Problem

Archaeological coin collections contain hundreds of thousands of degraded, worn, and corroded specimens. Identifying a single ancient coin by hand requires a trained numismatist consulting multiple reference catalogues  a process that can take hours. Museums, auction houses, and research institutions need a way to:

- **Classify** a coin photograph to a known historical type instantly
- **Validate** the result against known physical properties (metal, weight, mint)
- **Narrate** the coin's historical context without hallucinating wrong dates or dynasties
- **Handle gracefully** coins that have never been digitally catalogued

> This project is a **PFE (Final Year Engineering Internship)** at ESPRIT School of Engineering in partnership with YEBNI, Tunisia.
> Student: **Dhia Chaieb** | dhia.chaieb@esprit.tn

---

## What is Deep Learning? What is RAG?

These two technologies are the scientific core of DeepCoin-Core. If you are a jury member, museum curator, or encadrant reading this  here is what they mean and why we chose them.

### Deep Learning  Teaching a Machine to See

**Deep Learning** is a branch of artificial intelligence where a neural network learns to recognise patterns from examples, without being explicitly programmed with rules.

For DeepCoin-Core, we use a **Convolutional Neural Network (CNN)** called **EfficientNet-B3**. A CNN processes an image by passing it through dozens of mathematical filters that detect progressively more complex features: first edges, then shapes, then textures, then semantic concepts like "helmeted portrait" or "eagle reverse."

The key technique is **fine-tuning** (also called transfer learning):

1. EfficientNet-B3 was pre-trained by Google on 1.2 million photographs from ImageNet  teaching it universal visual concepts (textures, edges, object shapes).
2. We then continued training it on our own dataset of 7,677 ancient coin images across 438 coin types from the Corpus Nummorum catalogue.
3. The network learned to map a coin photograph to one of those 438 types, producing a **confidence score** (e.g. "91.1%  Type 1015, Maroneia, Thrace").

Why does this matter? We went from zero coin-recognition capability to **80.03% test accuracy** (using Test-Time Augmentation) in approximately 100 minutes of GPU training. A human expert would take years to develop equivalent breadth.

**Why EfficientNet-B3 specifically?**
EfficientNet was invented by Google Brain researchers who discovered that scaling a network's depth, width, and resolution *simultaneously* (compound scaling) is far more efficient than scaling them independently. B3 is the sweet spot for our hardware: 12 million parameters, 1536-dimensional feature output, fits entirely in 4.3 GB GPU VRAM.

---

### RAG  Retrieval-Augmented Generation

A Large Language Model (LLM) like Gemini is trained on billions of internet documents. It can write beautiful prose. But it can also **hallucinate**  confidently stating a wrong date, dynasty, or mint because it has seen plausible-sounding numismatic text during training.

**RAG** (Retrieval-Augmented Generation) solves this by separating **facts** from **writing**:

```
Step 1  RETRIEVE:  Search the Knowledge Base for verified facts about the coin
Step 2  AUGMENT:   Inject those facts as labeled context blocks into the LLM prompt
Step 3  GENERATE:  The LLM writes prose quality  but ONLY from the provided context
```

In DeepCoin-Core, the LLM prompt looks like this:

```
[CONTEXT 1  Identity]   type: 1015  |  denomination: drachm  |  region: Thrace  |  date: c.365330 BC
[CONTEXT 2  Obverse]    prancing horse right  |  legend: MAR
[CONTEXT 3  Reverse]    bunch of grapes on vine branch  |  legend: EPI ZINONOS
[CONTEXT 4  Material]   silver  |  weight: 2.44 g  |  mint: Maroneia
[CONTEXT 5  Context]    persons: Magistrate Zenon

INSTRUCTION: Using ONLY the contexts above (cite [CONTEXT N]),
             write a 3-paragraph professional numismatic analysis.
             Do not add any fact not present in the context.
```

The result: **zero hallucination on structured facts**. The LLM contributes only prose quality  it cannot invent a wrong date or wrong mint because those fields come directly from the Corpus Nummorum Knowledge Base.

The three-word summary: **RAG makes the LLM cite its sources.**

---

## System Architecture

```
  +---------------------------+
  |   RAW COIN PHOTOGRAPH     |
  +-------------+-------------+
                |
  +-------------v-------------+
  |    CLAHE Enhancement      |
  |   LAB colour space        |
  |   L-channel, clip=2.0     |
  +-------------+-------------+
                |
  +-------------v-------------+
  |  Aspect-preserving resize |
  |  299x299 + zero-padding   |
  |  (no stretch, no distort) |
  +-------------+-------------+
                |
  +-------------v-------------+
  |      EfficientNet-B3      |
  |  fine-tuned on 438 types  |
  |  => class + confidence    |
  +---+----------+------------+
      |          |            |
  conf > 85%  40-85%     conf < 40%
      |          |            |
  +---v-----+ +-v-------+ +---v---------+
  |Historian| |Validator| | Investigator|
  |RAG + LLM| |OpenCV   | | VLM + CV    |
  |narrative| |HSV check| | fallback    |
  +---+-----+ +-+-------+ +---+---------+
      |          |             |
      +----------+-------------+
                 |
  +--------------v------------+
  |      Synthesis Agent      |
  |  Plain-text + PDF Report  |
  +--------------+------------+
                 |
  +--------------v------------+
  |    FastAPI REST Backend   |
  |  POST /api/classify       |
  |  GET  /api/history        |
  |  GET  /api/reports/{file} |
  +---------------------------+


  RAG ENGINE  (Historian + Investigator)
  +----------------------------------------------+
  |  Query                                       |
  |    +-- BM25 keyword index  (rank-bm25)       |
  |    +-- ChromaDB vector search  (cosine sim.) |
  |                   |                          |
  |    RRF merge:  score = SUM 1 / (60 + rank_r) |
  |                   |                          |
  |    Top-k results => [CONTEXT 1-5] blocks     |
  |    Source: Corpus Nummorum (DFG-funded)      |
  |    Coverage: 9,541 types  /  47,705 vectors  |
  +----------------------------------------------+
```

---

## The Five Agents

| Agent | File | Role | Technology |
|-------|------|------|-----------|
| **Gatekeeper** | `gatekeeper.py` | LangGraph orchestrator  routes by confidence, manages state, retries on 429/503 | LangGraph state machine |
| **Historian** | `historian.py` | Fetches 5 semantic context chunks from KB, calls LLM with grounded prompt, writes narrative | RAG + Gemini / Ollama |
| **Investigator** | `investigator.py` | For unknown coins  extracts visual attributes via VLM or OpenCV, cross-references all 9,541 KB types | Vision LLM + OpenCV HSV/Sobel |
| **Validator** | `validator.py` | Multi-scale HSV metal detection: compares detected metal (gold/silver/bronze) to KB-expected metal at 3 crop sizes | OpenCV + ChromaDB |
| **Synthesis** | `synthesis.py` | Assembles all agent outputs into a professional PDF report with navy header, bordered tables, and page numbers | fpdf2 direct-draw |

### The Three Routes

| Route | Trigger | What happens |
|-------|---------|-------------|
| **Historian** | confidence > 85% | CNN is confident. Fetch KB record for exact type  RAG context chunks  LLM narrative  PDF |
| **Validator** | 40%  conf  85% | CNN is uncertain. Run OpenCV forensic metal check before committing to the classification  PDF |
| **Investigator** | confidence < 40% | CNN cannot classify. Describe coin visually  search all 9,541 KB types for closest match  PDF |

### Graceful Degradation  The Core Philosophy

Traditional AI systems fail silently: a wrong high-confidence answer is the worst possible outcome for a museum curator.

DeepCoin-Core applies three levels of degradation:

```
Level 1 (CNN known):     CNN classifies  Historian writes grounded narrative
Level 2 (CNN uncertain): CNN hesitates  Validator forensic check + best-guess narrative
Level 3 (Truly unknown): CNN fails  Investigator detects attributes  KB finds closest 3 neighbours
                         Report says: "No exact match. Closest types: [list]"
```

The system **never returns an empty answer**. It always returns the maximum possible information.

---

## Deep Learning Model

### EfficientNet-B3  Architecture

| Component | Detail |
|-----------|--------|
| Architecture | EfficientNet-B3 (compound-scaled CNN) |
| Pre-training | ImageNet (1.2M images, 1,000 classes  Google Brain) |
| Fine-tuning | CN dataset, 438 coin types, 7,677 images |
| Input shape | 299  299  3 (RGB) |
| Feature vector | 1,536-dimensional (before classification head) |
| Output head | `Dropout(0.4)  Linear(1536, 438)` |
| Parameters | ~12M total |

### Training Configuration

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| Optimizer | AdamW, lr=1e-4, wd=0.01 | Weight decay prevents memorising rare classes |
| Scheduler | CosineAnnealingLR (T_max=100, η_min=1e-6) | Smooth decay avoids sharp lr steps that cause instability |
| Loss | CrossEntropyLoss (label_smoothing=0.1) | Smoothing penalises over-confident predictions |
| Mixup | α = 0.2 (Beta distribution blending) | λA + (1-λ)B fuses two training images; prevents decision boundary memorisation |
| Batch size | 16 | RTX 3050 Ti 4.3 GB VRAM constraint |
| AMP | `torch.amp.GradScaler` + `autocast` | Float16 halves VRAM; GradScaler prevents float16 underflow |
| Gradient clip | max_norm = 1.0 | Catches explosion in first epochs on new head |
| Sampler | `WeightedRandomSampler` (weight = 1/class_count) | Fixes 40:1 class imbalance (204 vs 5 images/class) |
| Early stopping | patience = 10 on val accuracy | Stops at epoch 62  saves best at epoch 52 |
| Random seed | 42 | Full reproducibility |

### Preprocessing Pipeline

```
Raw photo  CLAHE (LAB L-channel, clipLimit=2.0, tile=88)
           Aspect-preserving resize (longest edge = 299, pad shorter edge with zeros)
           Albumentations augmentation (train only):
              Rotate 15°  |  BrightnessContrast 20%  |  GaussNoise
              ElasticTransform  |  HorizontalFlip  |  HV Flip
           Normalise: mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]
```

**Why CLAHE in LAB?** Ancient coins have patina  green/brown metal oxidation that proves archaeological authenticity. Standard histogram equalisation destroys patina colour by operating on all three RGB channels simultaneously. LAB colour space separates luminance (L) from colour (A, B). Applying CLAHE only to L enhances inscription detail and surface relief while preserving the diagnostic metal colours.

**Why aspect-preserving resize?** Coins are round. Stretching a 2:1 original image to 1:1 deforms the geometry. EfficientNet-B3 learns shape proportions  a stretched coin looks like a different coin.

### Results

| Metric | Value |
|--------|-------|
| Best epoch | 52 / 100 |
| Validation accuracy | 79.25% |
| Test accuracy (single pass) | 79.08% |
| **Test accuracy (TTA 8)** | **80.03%** |
| Macro F1 (438 classes) | 0.7763 |
| Training time | ~103 min on RTX 3050 Ti |

**TTA (Test-Time Augmentation):** 8 forward passes per coin (original + 7 flips/crops), averaged softmax probabilities  +0.78% accuracy gain.

---

## How We Built the Knowledge Base

The Knowledge Base is the factual backbone of the entire RAG pipeline. Here is exactly how it was constructed.

### Source: Corpus Nummorum

[Corpus Nummorum](https://www.corpus-nummorum.eu/) is a DFG-funded digital catalogue maintained by the Berlin-Brandenburg Academy of Sciences. It contains structured records for over 9,000 ancient coin types  denomination, authority, region, mint, material, weight, obverse/reverse descriptions, legends, and literature references.

### Scraping  9,541 Types in 2h 41min

```
Target:     9,716 type IDs (entire CN database)
Scraped:    9,541 types successfully
Failed:     175 types returned HTTP errors (missing or removed records)
Rate limit: 1 request/second (polite scraping)
Duration:   ~2 hours 41 minutes
Resumable:  --resume flag skips already-fetched IDs (crash-safe)
Output:     data/metadata/cn_types_metadata_full.json (~3.2 MB)
```

Each type record stores 15 structured fields: type_id, denomination, authority, region, date_range, obverse_description, obverse_legend, reverse_description, reverse_legend, material, weight, diameter, mint, persons, references.

### Chunking  5 Semantic Vectors Per Coin

A single coin record contains very different types of information. Packing all 15 fields into one 200-word text blob and asking ChromaDB to embed it produces a single averaged vector  "silver horse eagle Maroneia" blurs into a generic ancient-coin vector.

Instead, we split each type into **5 focused semantic chunks**:

| Chunk | Fields | Why it works |
|-------|--------|-------------|
| `identity` | type_id, denomination, authority, region, date_range | "silver drachm Thrace 365 BC"  matches classification queries |
| `obverse` | obverse_description, obverse_legend | "prancing horse MAR"  matches portrait/icon queries |
| `reverse` | reverse_description, reverse_legend | "bunch of grapes EPI ZINONOS"  matches reverse queries |
| `material` | material, weight, diameter, mint | "silver 2.44g Maroneia"  matches forensic validation queries |
| `context` | persons, references, notes | "Magistrate Zenon"  matches provenance queries |

```
9,541 types  5 chunks = 47,705 vectors stored in ChromaDB
Embedding model:  all-MiniLM-L6-v2  (384-dimensional, 22 MB, CPU)
Build time:       9.0 minutes
On-disk size:     ~180 MB
```

The `in_training_set: bool` tag on every record marks whether the type is one of the 438 CNN-trained classes or a KB-only extended entry.

---

## RAG Engine

File: `src/core/rag_engine.py`

### Hybrid Search: BM25 + Vector + RRF

Pure vector search has a weakness: exact keyword matches. If a curator searches for "silver tetradrachm Philip II", vector similarity might surface semantically related types that contain none of those exact words. BM25 (the algorithm behind search engines) catches exact matches that embedding space misses.

We run both in parallel and merge them using **Reciprocal Rank Fusion**:

```
score(doc) = Σ  1 / (60 + rank_r(doc))
              r  {BM25, ChromaDB vector}
```

Where `rank_r(doc)` is the position of the document in each ranked list (1 = top result). The constant 60 prevents outlier top-1 results from dominating.

**Why RRF and not a cross-encoder reranker?** A cross-encoder (BERT-based) would add ~65 MB model + 200ms latency per query. RRF achieves ~95% of reranker accuracy at zero latency overhead for a corpus of 9,541 records. Engineering pragmatism.

### Grounded LLM Prompting

```
[CONTEXT 1  Identity]  denomination: drachm | region: Thrace | date: c.365330 BC
[CONTEXT 2  Obverse]   prancing horse right | legend: MAR
[CONTEXT 3  Reverse]   bunch of grapes | legend: EPI ZINONOS
[CONTEXT 4  Material]  silver | weight: 2.44 g | mint: Maroneia
[CONTEXT 5  Context]   persons: Magistrate Zenon

INSTRUCTION: You are an expert numismatist.
             Using ONLY the contexts above (cite [CONTEXT N]),
             write a 3-paragraph professional analysis.
             Do not add any fact not present in the context.
```

The LLM cannot invent a wrong emperor, wrong dynasty, or wrong mint  every structured fact is injected from the KB. The model contributes only narrative prose quality.

### Per-Agent Search Scope

| Agent | Search scope | Filter |
|-------|-------------|--------|
| Historian | Exact lookup for CNN-predicted type | `type_id = predicted` |
| Validator | Material and identity chunks | `chunk_type in [material, identity]` |
| Investigator | ALL 9,541 types  widest possible net | No filter  maximum coverage |

---

## Technology Stack

### Deep Learning

| Component | Version | Role |
|-----------|---------|------|
| PyTorch | 2.6.0+cu124 | Neural network framework |
| torchvision | 0.21+ | EfficientNet-B3 pretrained weights |
| EfficientNet-B3 | ImageNet pretrained | 12M params, 1536-dim features |
| OpenCV | 4.13.0 | CLAHE preprocessing, HSV forensic analysis |
| Albumentations | 1.4+ | Training augmentation pipeline |
| CUDA | 12.4 | GPU acceleration (RTX 3050 Ti) |

### RAG & Knowledge Base

| Component | Version | Role |
|-----------|---------|------|
| ChromaDB | 0.6+ | Persistent local vector database (47,705 vectors) |
| sentence-transformers | 3.3+ | `all-MiniLM-L6-v2` text embeddings (384-dim) |
| rank-bm25 | latest | BM25Okapi keyword index |
| LangGraph | 0.3+ | State machine agent orchestration |
| LangChain | 0.3+ | Prompt templates, agent tooling |

### LLM Provider Chain

```
Priority 1: GITHUB_TOKEN   GitHub Models API (Gemini 2.5 Flash, free with Copilot Pro)
Priority 2: GOOGLE_API_KEY  Google AI Studio  (Gemini 2.5 Flash, 1,500 req/day free)
Priority 3: OLLAMA_HOST     Local Ollama       (gemma3:4b text  /  qwen3-vl:4b vision)
Priority 4: No key set      Structured fallback (KB fields concatenated, no crash, no hallucination)
```

### Backend

| Component | Version | Role |
|-----------|---------|------|
| FastAPI | 0.115+ | Async REST API, auto-docs |
| Uvicorn | 0.40+ | ASGI server |
| Pydantic v2 | 2.x | Request/response schema validation |

### Frontend (Layer 5  upcoming)

| Component | Version | Role |
|-----------|---------|------|
| Next.js | 15 (App Router) | React framework with Server Components |
| TypeScript | 5 | Type-safe frontend |
| Tailwind CSS | 4 | Utility-first styling |
| shadcn/ui | latest | Radix UI component library |
| TanStack Query | 5 | Server state management |

### Infrastructure (Layer 6  upcoming)

| Component | Version | Role |
|-----------|---------|------|
| Docker Compose | 2.x | Multi-container orchestration |
| PostgreSQL | 17 | Persistent analysis history |
| Redis | 7 | Result caching |
| Nginx | 1.27 | Reverse proxy |
| GitHub Actions |  | CI: pytest + flake8 + black |

---

## Performance

| Metric | Target | Current |
|--------|--------|---------|
| CNN test accuracy (TTA 8) | > 80% | **80.03%**  |
| CNN macro F1 (438 classes) | > 0.75 | **0.7763**  |
| Historian route latency | < 25s | ~1520s (Ollama gemma3:4b) |
| Validator route latency | < 15s | ~9.8s  |
| Investigator route (CV only) | < 5s | ~3.1s  |
| PDF generation | < 1s | ~0.40.5s  |
| KB hybrid search | < 50ms | < 1ms  |
| Knowledge base coverage | 9,716 types | 9,541 types (98.2%) |

---

## Quick Start

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA 12.4 (CPU inference also supported, slower)
- 8 GB disk for models + processed dataset

### Installation

```bash
git clone https://github.com/ChaiebDhia/DeepCoin-Core.git
cd DeepCoin-Core
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### Set API Keys (optional  fallback works without them)

```bash
# Windows PowerShell
$env:GITHUB_TOKEN  = "your_github_token"    # GitHub Copilot Pro token
$env:GOOGLE_API_KEY = "your_google_key"     # Google AI Studio key
```

### Build the Knowledge Base

```bash
# Scrape all 9,716 CN types (one-time, ~2h 41min)
python scripts/build_knowledge_base.py --all-types
# Resume if interrupted
python scripts/build_knowledge_base.py --all-types --resume

# Build ChromaDB vector index from scraped data (~9 min)
python scripts/rebuild_chroma.py
```

### Run the Pipeline

```bash
# Single coin classification
python scripts/predict.py --image data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg

# Full end-to-end test (all 3 routes)
python scripts/test_pipeline.py

# Start FastAPI server
uvicorn src.api.main:app --port 8000 --log-level info
```

### Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# Classify a coin
curl -X POST http://localhost:8000/api/classify \
     -F "file=@data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg"

# View history
curl http://localhost:8000/api/history
```

---

## API Reference

### POST `/api/classify`

Upload a coin image for full pipeline analysis.

**Request:** `multipart/form-data`, field `file` (JPEG/PNG, max 10 MB)

**Response:**
```json
{
  "id": "uuid-string",
  "cnn_result": {
    "label": "1015",
    "confidence": 0.911,
    "top5": [
      {"rank": 1, "label": "1015", "confidence": 0.911},
      {"rank": 2, "label": "1017", "confidence": 0.034}
    ]
  },
  "route_taken": "historian",
  "report": "Expert Commentary text...",
  "pdf_path": "reports/uuid_coin.pdf",
  "node_timings": {"cnn": "0.54s", "historian": "14.2s", "synthesis": "0.47s"},
  "created_at": "2026-02-28T10:23:45"
}
```

**Status codes:** `200 OK` | `400 Bad Request` (invalid file) | `422 Unprocessable Entity` (validation) | `500 Internal Server Error`

---

### GET `/api/health`

Returns system status for all 5 components.

```json
{
  "status": "ok",
  "components": {
    "cnn_model": "ok",
    "rag_engine": "ok",
    "llm_provider": "ollama",
    "pdf_generator": "ok",
    "history_store": "ok"
  }
}
```

Returns `503 Service Unavailable` if any component is degraded.

---

### GET `/api/history`

Returns paginated analysis history.

**Query params:** `page` (default 1), `per_page` (default 20, max 100)

---

### GET `/api/history/{id}`

Returns the full analysis result for a specific classification.

---

### GET `/api/reports/{filename}`

Serves a generated PDF report. Path traversal protected.

---

## Project Structure

```
deepcoin/
 src/
    data_pipeline/
       prep_engine.py          # CLAHE + aspect-preserving resize
    core/
       model_factory.py        # EfficientNet-B3 definition
       dataset.py              # PyTorch Dataset + Albumentations transforms
       inference.py            # CoinInference: TTA, device resolution
       knowledge_base.py       # Legacy ChromaDB wrapper (434 types, fallback)
       rag_engine.py           # Hybrid BM25+vector+RRF search (47,705 vectors)
    agents/
       gatekeeper.py           # LangGraph orchestrator
       historian.py            # RAG + LLM narrative generator
       investigator.py         # VLM visual agent + OpenCV fallback
       validator.py            # Multi-scale HSV forensic validator
       synthesis.py            # Professional PDF generator (fpdf2)
    api/
        main.py                 # FastAPI app lifespan + CORS + health
        schemas.py              # Pydantic v2 request/response models
        _store.py               # Thread-safe JSON history store
        routes/
            classify.py         # POST /api/classify (5-layer security)
            history.py          # GET /api/history + GET /api/history/{id}
 scripts/
    train.py                    # CNN training V3 (729 lines  AMP, Mixup, TTA)
    evaluate_tta.py             # TTA evaluation (+0.78% = 80.03%)
    predict.py                  # CLI inference tool
    test_pipeline.py            # End-to-end test (all 3 routes)
    build_knowledge_base.py     # CN scraper + KB builder (--all-types, --resume)
    rebuild_chroma.py           # ChromaDB rebuild from JSON metadata
 models/
    best_model.pth              # EfficientNet-B3 V3  epoch 52, 80.03% TTA
    class_mapping.pth           # {class_to_idx, idx_to_class, n=438}
 data/
    processed/                  # 7,677 images  438 classes (299299 JPEG)
    metadata/
        cn_types_metadata_full.json  # 9,541 CN types (~3.2 MB)
        chroma_db/              # Legacy 434-vector index (fallback)
        chroma_db_rag/          # Production 47,705-vector index
 reports/                        # Generated PDF output
 requirements.txt
 docker-compose.yml              # 7-service skeleton (Layer 6)
 .github/
     copilot-instructions.md     # Persistent AI context file
```

---

## Build Layers

| # | Layer | Status | Key files |
|---|-------|--------|-----------|
| 0 | **CNN Training** |  Complete | `scripts/train.py`, `src/core/model_factory.py` |
| 1 | **Inference Engine** |  Complete | `src/core/inference.py`, `scripts/predict.py` |
| 2 | **Knowledge Base + RAG** |  Complete | `src/core/rag_engine.py`, `scripts/rebuild_chroma.py` |
| 3 | **Agent System** |  Complete | `src/agents/`  all 5 agents |
| 4 | **FastAPI Backend** |  Complete | `src/api/`  classify, history, health, PDF serving |
| 5 | **Next.js Frontend** |  Next | `frontend/`  Next.js 15, TypeScript, Tailwind, shadcn/ui |
| 6 | **Docker + Infrastructure** |  Pending | `docker-compose.yml`  7 services |
| 7 | **Tests + CI/CD** |  Pending | `tests/`, `.github/workflows/ci.yml` |

---

## Engineering Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CNN backbone | EfficientNet-B3 | Compound scaling (depth+width+resolution); B3 fits 4.3 GB VRAM; B7 does not |
| Preprocessing | CLAHE in LAB space | Enhances inscription contrast without destroying diagnostic metal patina colours |
| Resize strategy | Aspect-preserving + zero-pad to 299299 | Coins are round  stretching deforms geometry and misleads the CNN |
| Training technique | Transfer learning (ImageNet  438 coins) | 80% accuracy from 7,677 images; impossible from scratch |
| Class imbalance fix | WeightedRandomSampler (weight = 1/class_count) | 40:1 imbalance (204 vs 5 images/class); sampler equalises exposure |
| Generalization | Mixup α=0.2 + label smoothing 0.1 | Prevents overfit on small dataset; smoothed labels reduce over-confidence |
| GPU efficiency | AMP (float16) + GradScaler | Halves VRAM, ~2 epoch speed on RTX 3050 Ti |
| Agent framework | LangGraph (not CrewAI) | Conditional routing + cycles + explicit state; production-ready |
| KB scope | All 9,541 CN types | CNN and KB have independent constraints  KB is pure text, no image threshold applies |
| Chunking | 5 semantic chunks per coin | Clean, targeted embeddings; "silver coin" hits material chunk not blurred blob |
| Search | Hybrid BM25 + vector + RRF | BM25 catches exact keyword matches; vectors catch semantic similarity; RRF merges both |
| Reranking | RRF (not cross-encoder) | 9,541 records  formula beats 65 MB BERT model at zero latency overhead |
| LLM grounding | [CONTEXT N] citation blocks | LLM writes, KB provides facts  eliminates hallucination on structured fields |
| Fallback | 4-tier LLM chain + OpenCV | System works offline; investigator functions without any API key |
| Architecture style | Modular monolith | 1-person PFE team; microservices = premature; monolith with clean interfaces = correct |
| PDF engine | fpdf2 direct-draw | Zero Markdown parsing, zero font dependency, full layout control |

---

## Academic Context

| Field | Value |
|-------|-------|
| **Institution** | ESPRIT School of Engineering, Manouba, Tunisia |
| **Company** | YEBNI  Information & Communication, Tunisia |
| **Project type** | PFE (Projet de Fin d'Études)  5-month final year internship |
| **Period** | February  July 2026 |
| **Student** | Dhia Chaieb  dhia.chaieb@esprit.tn |
| **GitHub** | [ChaiebDhia/DeepCoin-Core](https://github.com/ChaiebDhia/DeepCoin-Core) |
| **Dataset** | Corpus Nummorum v1  115,160 images, 9,716 coin types, DFG-funded |
| **Scientific domain** | Fine-grained visual recognition + archaeological numismatics |
| **Key contribution** | Hybrid CNN + multi-agent RAG system with graceful degradation for out-of-distribution coins |

### Research Questions Addressed

1. Can transfer learning from ImageNet classify ancient coins reliably?  **Yes  80.03% TTA on 438 classes**
2. Does hybrid BM25+vector search outperform vector-only for numismatic KB?  **Yes  exact keyword recall improves on material/mint queries**
3. Can RAG grounding eliminate LLM hallucination on structured numismatic facts?  **Yes  [CONTEXT N] citation format produces zero invented dates/mints in testing**
4. Can graceful degradation replace "I don't know" with useful output?  **Yes  all 3 routing paths produce valid reports in end-to-end tests**

---

## License

MIT License  see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Corpus Nummorum**  Berlin-Brandenburg Academy of Sciences, DFG-funded digital numismatic catalogue
- **Google Brain**  EfficientNet architecture (Tan & Le, 2019)
- **Meta AI**  Albumentations augmentation library
- **LangChain AI**  LangGraph state machine framework
- **YEBNI**  Company internship supervisor and domain expertise
- **ESPRIT School of Engineering**  Academic supervision

---

*DeepCoin-Core  Archaeological numismatics meets enterprise AI.*
*Dhia Chaieb  ESPRIT  YEBNI  2026*
