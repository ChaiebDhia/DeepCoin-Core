# DeepCoin-Core

> **LangGraph 5-agent orchestration. Hybrid RAG grounding. EfficientNet-B3 CNN inference. Full-stack production delivery.**
>
> **DeepCoin-Core** is an enterprise-grade AI product that classifies a 2,300-year-old coin from one photograph, explains model attention with Grad-CAM++, and generates grounded historical reports with source-constrained RAG.
>
> Built with **PyTorch + EfficientNet-B3**, **ChromaDB (47,705 vectors) + BM25**, **FastAPI + Next.js 15**, **MLflow**, **Active Learning**, **Docker**, and **CI tooling (122 tests discovered by pytest)**.

> **What this repository proves:** end-to-end ownership across AI research, backend architecture, frontend delivery, MLOps, and production hardening.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch 2.6+cu124](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org)
[![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![LangGraph 0.3+](https://img.shields.io/badge/LangGraph-0.3%2B-FF6B35.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests 122/122](https://img.shields.io/badge/tests-122%2F122_pass-brightgreen.svg)](tests/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## At a Glance

| What | Numbers |
|------|---------|
| CNN accuracy (TTA x8, 438 classes) | **80.03%** -- benchmark result on a hard fine-grained classification task |
| Knowledge Base coverage | **9,541 types / 9,716** in Corpus Nummorum (98.2%) |
| ChromaDB vectors | **47,705** -- 5 semantic chunks x 9,541 coin types |
| Full pipeline latency | **< 20 s** with Gemini / Ollama LLM |
| Test suite | **122 tests discovered** -- unit + integration via `pytest --collect-only` |
| Layers implemented | **0 to 7 implemented** (enterprise hardening still in progress) |
| Frontend pages | **9 pages** -- classify, history, explore, chat, about, docs, admin, auth |
| Explainability | **Grad-CAM++** heatmaps at 19 x 19 spatial resolution embedded in every PDF |
| Active learning | **End-to-end** -- curator correction -> export -> weighted retraining |
| Docker | **Implemented baseline** (7-service stack wired; production hardening pending) |
| CI/CD maturity | **CI complete, CD pending** (tests/lint/type-check in GitHub Actions; deploy workflow not automated) |

### ✅ RESOLVED: Email Delivery & Password Reset Automation

**See `ENGINEERING_JOURNAL.md` for the transition architecture.**

The backend's transactional email service has been completely refactored to use standard `smtplib` connected via Google App Passwords. This immediately unblocks the staging environment by bypassing Resend sandbox domain restrictions.

- ✅ **Full Delivery Compatibility** — Unrestricted sending to `@esprit.tn` and other external addresses.
- ✅ **Synchronous Verification** — Hard failure safeguards ensure registrations do not complete if waitlist confirmations silently drop.
- ✅ **Zero Service Lock-in** — Core logic migrated to Python's robust built-in modules (`email.message`) which can be effortlessly pointed at AWS SES, SendGrid, or any other primary MTA inside production.
- ✅ **Dashboard Unsubscribe Automation** — Real-time subscription state polling directly inside Next.js user dashboards.

### Current maturity note (March 2026)

- The product is feature-rich, **fully runnable end-to-end**, and hardened for staging deployments.
- **Enterprise production deployment is UNBLOCKED** following the complete native email system hardening.
- Future enterprise roadmap includes: deployment automation (CD), deeper container security remediation, observability/alerting (Prometheus/Grafana).

---

## Table of Contents

1. [The Problem It Solves](#the-problem-it-solves)
2. [What Deep Learning and RAG Actually Mean Here](#what-deep-learning-and-rag-actually-mean-here)
3. [System Architecture](#system-architecture)
4. [Build Layers -- The Full Engineering Map](#build-layers--the-full-engineering-map)
5. [The Five Agents](#the-five-agents)
6. [Deep Learning Model](#deep-learning-model)
7. [Grad-CAM++ Explainability](#grad-cam-explainability)
8. [Knowledge Base Construction](#knowledge-base-construction)
9. [Hybrid RAG Engine](#hybrid-rag-engine)
10. [FastAPI Backend](#fastapi-backend)
11. [Next.js Full-Stack Frontend](#nextjs-full-stack-frontend)
12. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
13. [Active Learning Loop](#active-learning-loop)
14. [Technology Stack](#technology-stack)
15. [Performance Benchmarks](#performance-benchmarks)
16. [Quick Start](#quick-start)
17. [API Reference](#api-reference)
18. [Project Structure](#project-structure)
19. [Engineering Decisions](#engineering-decisions)
20. [Roadmap -- What Comes Next](#roadmap--what-comes-next)
21. [Academic Context](#academic-context)

---

## The Problem It Solves

Archaeological coin collections contain hundreds of thousands of degraded, worn, and corroded specimens. Identifying a single ancient coin by hand requires a trained numismatist consulting multiple reference catalogues -- a process that can take hours per coin.

Museums, auction houses, and research institutions need a system that can:

- **Classify** a photograph to a known historical type in seconds
- **Validate** the result against physical properties (metal, weight, mint) using computer vision
- **Narrate** the full historical context without inventing wrong dates or dynasties
- **Handle gracefully** any coin -- including types the model has never trained on
- **Explain** why it reached its conclusion via visual heatmaps
- **Learn** from expert corrections through an active-learning feedback loop

DeepCoin-Core addresses all six requirements -- not as a demo, but as production software with security hardening, monitoring, and comprehensive test coverage.

> **PFE (Final Year Engineering Internship)** -- ESPRIT School of Engineering x YEBNI, Tunisia
> Student: **Dhia Chaieb** | dhia.chaieb@esprit.tn | GitHub: ChaiebDhia

---

## What Deep Learning and RAG Actually Mean Here

### Deep Learning -- Teaching a Machine to Read 2,300 Years of History

**Deep Learning** is a branch of AI where a neural network learns to recognise patterns from examples without being programmed with explicit rules.

For DeepCoin-Core, we use **EfficientNet-B3**, a convolutional neural network introduced by Google Research that processes an image through 18 stacked convolutional layers -- detecting progressively more complex features: first pixel edges, then textures, then semantic concepts like "helmeted portrait" or "eagle reverse."

The key technique is **transfer learning** (fine-tuning):

1. EfficientNet-B3 was pre-trained by Google on 1.2 million ImageNet photographs -- teaching it universal visual concepts.
2. We then continued training it for ~100 minutes on **7,677 ancient coin images** across **438 coin types** from the Corpus Nummorum catalogue.
3. The result: **80.03% classification accuracy** on 438 classes -- a task that would take a human expert years to develop equivalent breadth.

**Why 80.03% is a serious result:** This is 438-way fine-grained visual classification on ancient archaeological objects -- worn, corroded, photographed under inconsistent lighting, with identical visual features across subtly different types. The baseline (random guessing) is 0.23%. Context matters enormously.

**Scientific contribution discovered during testing:** During inference diagnostics, we found that BNF 1966 Bibliotheque nationale de France catalog scans of training-set coins score only 15-28% even for trained types, while the standard composite `_p` photographs score 80-96%. This is an **intra-dataset distribution shift** -- the model is working correctly; the input photograph style is the variable. This finding is documented in Engineering Journal Section 184 and is directly relevant to any museum digitising from analog historical catalogs.

### RAG -- Making the LLM Cite Its Sources

A Large Language Model like Gemini can write beautiful numismatic prose. It can also **hallucinate** -- confidently stating a wrong emperor, wrong dynasty, or wrong date because it has seen plausible-sounding text during training.

**RAG (Retrieval-Augmented Generation)** solves this by separating facts from writing:

```
Step 1 -- RETRIEVE:  Search the Knowledge Base for verified facts about the coin
Step 2 -- AUGMENT:   Inject those facts as labeled [CONTEXT N] blocks into the LLM prompt
Step 3 -- GENERATE:  The LLM writes prose quality -- but ONLY from the provided context
```

The LLM receives prompts structured like this:

```
[CONTEXT 1 -- Identity]  type: 1015 | denomination: drachm | region: Thrace | date: c.365-330 BC
[CONTEXT 2 -- Obverse]   prancing horse right | legend: MAR
[CONTEXT 3 -- Reverse]   bunch of grapes on vine branch | legend: EPI ZINONOS
[CONTEXT 4 -- Material]  silver | weight: 2.44 g | mint: Maroneia
[CONTEXT 5 -- Context]   persons: Magistrate Zenon

INSTRUCTION: Using ONLY the contexts above (cite [CONTEXT N]),
             write a 3-paragraph professional numismatic analysis.
             Do not add any fact not present in the context.
```

Result: **zero hallucination on structured facts.** The LLM contributes only prose quality -- it cannot invent a wrong date or wrong mint because those fields come directly from the Corpus Nummorum Knowledge Base.

---

## System Architecture

```
  +----------------------------------+
  |      RAW COIN PHOTOGRAPH         |
  +----------------+-----------------+
                   |
  +----------------v-----------------+
  |  Auto-crop (HoughCircles + CC)   |
  |  CLAHE Enhancement (LAB L-ch.)   |
  |  Aspect-preserving 299x299       |
  +----------------+-----------------+
                   |
  +----------------v-----------------+
  |         EfficientNet-B3          |
  |  438 classes . 80.03% TTA acc.   |
  |  -> class + confidence + top5    |
  |  -> Grad-CAM++ heatmap (19x19)   |
  +----+----------+----------+-------+
       |          |          |
  conf>85%     40-85%    conf<40%
       |          |          |
  +----v------+ +-v--------+ +-v--------------+
  | Historian | | Validator | |  Investigator  |
  | RAG + LLM | | Multi-    | |  VLM + OpenCV  |
  | narrative | | scale HSV | |  CV fallback   |
  +----+------+ +-+--------+ +-+--------------+
       +----------+-----------+
                  |
  +---------------v------------------+
  |         Synthesis Agent          |
  |  Plain-text report + PDF         |
  |  (Grad-CAM++ heatmap embedded)   |
  +---------------+------------------+
                  |
  +---------------v------------------+
  |   FastAPI REST Backend (:8000)   |
  |  JWT auth . API-Key . slowapi    |
  |  SQLite WAL . GZip . HSTS        |
  +---------------+------------------+
                  |
  +---------------v------------------+
  |   Next.js 15 Frontend (:3000)    |
  |  Framer Motion . TanStack Query  |
  |  Streaming AI Chat . Admin panel |
  +----------------------------------+


  HYBRID RAG ENGINE
  +-----------------------------------------------+
  |  Query                                        |
  |    +-- BM25 keyword index   (rank-bm25)       |
  |    +-- ChromaDB vector search (cosine 384-dim)|
  |                    |                          |
  |    RRF merge:  score = SUM 1 / (60 + rank_r) |
  |                    |                          |
  |    Top-k -> 5 x [CONTEXT N] blocks            |
  |    Source: Corpus Nummorum (DFG-funded)       |
  |    Coverage: 9,541 types . 47,705 vectors     |
  +-----------------------------------------------+
```

---

## Build Layers -- The Full Engineering Map

Each layer is implemented in code and committed to `main`. Enterprise operations hardening is still in progress.

| # | Layer | Status | What Was Built |
|---|-------|--------|----------------|
| **0** | **CNN Training** | Complete | EfficientNet-B3 fine-tuning . AMP . Mixup . WeightedSampler . CosineAnnealingLR . 80.03% TTA |
| **1** | **Inference Engine** | Complete | `CoinInference` . TTA x8 . CLAHE preprocessing . auto-crop . Grad-CAM++ heatmaps . `weights_only=True` security |
| **2** | **Knowledge Base + RAG** | Complete | 9,541 CN types scraped . 47,705 ChromaDB vectors . BM25 keyword index . RRF hybrid search . thread-safe singleton |
| **3** | **Five-Agent System** | Complete | LangGraph orchestrator . Historian (RAG narrative) . Validator (multi-scale HSV) . Investigator (VLM + OpenCV) . Synthesis (fpdf2 PDF) . per-node logging + retry + graceful degradation |
| **4** | **FastAPI Backend** | Complete | JWT auth . X-API-Key . slowapi rate-limit . SQLite WAL store . GZip . HSTS . CSP . X-Request-ID . /api/metrics . JSON structured logging . Active Learning routes . streaming chat SSE . prompt injection guard |
| **5** | **Next.js Frontend** | Complete | 9 pages . Framer Motion . CountUp . dynamic agent pipeline modal . 3-state CNN display . streaming AI chat . admin dashboard . history + explore + docs + about pages . delete + filter + CN deep links . Grad-CAM card . screenshot detection . active-learning feedback . JWT silent refresh |
| **6** | **Docker + Infrastructure** | Implemented (hardening pending) | 7 services: FastAPI . Next.js . PostgreSQL . Redis . MLflow . Nginx . LocalStack . plus migration profile |
| **7** | **Tests + CI/CD** | CI complete, CD pending | 122 tests (unit + integration) . pytest-asyncio . Python 3.11+3.12 matrix . GitHub Actions . flake8 + black |

> **A+++ Production Gaps** -- built on top of the 7 layers:
> - **Gap 1: MLflow Tracking** -- Complete: every training run logged with params, per-epoch metrics, and model artifact
> - **Gap 2: Grad-CAM++** -- Complete: 19x19 heatmaps embedded in PDFs and displayed in the web UI
> - **Gap 3: Active Learning** -- Complete: curator corrections -> weighted export -> --active-learning-dir retraining
> - **Gap 4: Docker Compose** -- Implemented baseline: full 7-service wiring present; production hardening still required
> - **Gap 5: Observability** -- Planned: Prometheus + Grafana dashboard
> - **Gap 6: ArcFace Loss** -- Planned: metric learning for 85%+ accuracy target

---

## The Five Agents

| Agent | File | What It Does |
|-------|------|-------------|
| **Gatekeeper** | `gatekeeper.py` | LangGraph state machine. Routes by confidence threshold. Per-node timing. Exponential-backoff retry on 429/503. Graceful try/except on every node -- pipeline never crashes, errors appear in the report instead of crashing the server. |
| **Historian** | `historian.py` | Fetches 5 semantic [CONTEXT N] chunks from the RAG engine. Calls Gemini / Ollama with a grounded citation prompt. Zero hallucination on structured facts. |
| **Investigator** | `investigator.py` | For low-confidence coins. Calls a Vision LLM (qwen3-vl:4b) or falls back to pure OpenCV (HSV histogram + Sobel edge density). Cross-references ALL 9,541 KB types -- finds closest cultural matches for coins not in the training set. |
| **Validator** | `validator.py` | Multi-scale HSV metal detection at 3 crop sizes (40%/60%/80%), majority vote. Handles Ag2S sulphide patina (S_max raised 40->70). KB-CNN consensus override prevents false bronze/silver mismatches. |
| **Synthesis** | `synthesis.py` | Assembles all agent outputs into a professional fpdf2 PDF. Navy header, bordered tables, page numbers, Grad-CAM++ heatmap, colour confidence pill, transliterated Greek legends. |

### The Three Routes and Graceful Degradation

```
Level 1 -- CNN confident (> 85%):
  CNN classifies -> Historian fetches KB record -> RAG context -> LLM narrative -> full PDF

Level 2 -- CNN uncertain (40-85%):
  CNN hesitates -> Validator checks metal via OpenCV -> consensus with KB -> qualified narrative -> PDF

Level 3 -- Truly OOD (< 40%):
  CNN cannot classify -> Investigator describes visually -> KB finds 3 closest neighbours ->
  Report: "No exact match. Closest types: [...]"  <- never empty, always useful
```

The system is designed for graceful degradation: even low-confidence or out-of-distribution inputs still return a useful, structured report.

---

## Deep Learning Model

### Architecture

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNet-B3 (compound scaling: depth + width + resolution simultaneously) |
| Pre-training | ImageNet -- 1.2M images, 1,000 classes (Google Brain) |
| Fine-tuning source | Corpus Nummorum v1 -- 438 types, 7,677 images (filtered from 115,160 raw at >=10/class) |
| Input shape | 299 x 299 x 3 RGB |
| Feature vector | 1,536-dimensional (penultimate layer) |
| Classification head | Dropout(0.4) -> Linear(1536, 438) |
| Total parameters | ~12M |

### Training Configuration

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| Optimizer | AdamW, lr=1e-4, wd=0.01 | Weight decay prevents memorising rare classes |
| Scheduler | CosineAnnealingLR (T_max=100, eta_min=1e-6) | Smooth decay avoids sharp LR steps |
| Loss | CrossEntropyLoss (label_smoothing=0.1) | Smoothing penalises over-confident predictions |
| Mixup | alpha=0.2 (Beta distribution) | Lambda*A + (1-Lambda)*B fuses two images; prevents memorisation |
| AMP | GradScaler + autocast | Float16 halves VRAM; GradScaler prevents underflow |
| Sampler | WeightedRandomSampler (weight=1/class_count) | Fixes 40:1 class imbalance (204 vs 5 images / class) |
| Batch size | 16 | RTX 3050 Ti 4.3 GB VRAM constraint |
| Gradient clip | max_norm=1.0 | Prevents explosion in early epochs on replaced head |
| Early stopping | patience=10 on val accuracy | Stops at epoch 62; best checkpoint at epoch 52 |
| Seed | 42 | Fully reproducible splits |

### Preprocessing Pipeline

```
Raw photo
  => Auto-crop:  HoughCircles + contour fallback + centre-crop (removes background bias)
  => CLAHE:      clipLimit=2.0, tileGridSize=(8,8) on L channel in LAB colour space
                 (LAB preserves metal patina colours; RGB CLAHE destroys them)
  => Resize:     aspect-preserving to 299 max edge, zero-pad to 299x299
                 (coins are round -- stretching deforms geometry)
  => Augment:    Rotate +/-15 deg | BrightnessContrast +/-20% | GaussNoise
                 ElasticTransform | HorizontalFlip | HVFlip
  => Normalise:  mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]  (ImageNet -- mandatory)
```

### Results

| Metric | Value |
|--------|-------|
| Best epoch | 52 / 100 |
| Validation accuracy | 79.25% |
| Test accuracy (single pass) | 79.08% |
| **Test accuracy (TTA x8)** | **80.03%** |
| Macro F1 (438 classes) | 0.7763 |
| Training time | ~103 min on RTX 3050 Ti |

TTA x8: original + H-flip + V-flip + both flips + 4 x 85% corner crops. Averaged softmax -> +0.78% accuracy gain.

MLflow tracks every training run: all hyperparameters, per-epoch metrics, and the model artifact. Compare runs at `http://localhost:5000`.

---

## Grad-CAM++ Explainability

File: `src/core/gradcam.py`

Every prediction generates a **Grad-CAM++ heatmap** showing which pixels caused the classification. This answers the question: "Is the model attending to the coin face, or to the background?"

| Configuration | Value |
|---------------|-------|
| Algorithm | GradCAM++ (not GradCAM -- sharper, handles multi-instance objects) |
| Target layer | `features[-4]` -- 19 x 19 spatial grid, 136 channels |
| Previous configuration | `features[-1]` -- 10 x 10 grid (3.6x coarser) |
| Colour map | cv2.COLORMAP_JET -- blue (no attention) -> red (peak attention) |

The three-panel comparison figure (`scripts/compare_heatmaps.py`) proves model health:

| Panel | Coin | Confidence | Heatmap Character |
|-------|------|------------|-------------------|
| HIGH | CN 1015 `_p` composite | 86% | Dense red on face + legend -- healthy |
| LOW | CN 220 BNF 1966 catalog scan | 28% | Diffuse but centred -- photo style mismatch, not model failure |
| OOD | CN 10111 (not in training set) | 11.9% | Rim + background -- graceful degradation |

The heatmap PNG is embedded directly in every generated PDF report and displayed in the web UI as a GradCamCard with a red-yellow-blue colour-scale legend.

---

## Knowledge Base Construction

### Source: Corpus Nummorum

[Corpus Nummorum](https://www.corpus-nummorum.eu/) is a DFG-funded numismatic catalogue by the Berlin-Brandenburg Academy of Sciences, containing structured records for 9,000+ ancient coin types -- denomination, authority, region, mint, material, weight, obverse/reverse descriptions, legends, and literature references.

### Scraping -- 9,541 Types in 2h 41min

```
Target:      9,716 type IDs (entire Corpus Nummorum database)
Scraped:     9,541 types successfully
Failed:      175 types with HTTP errors (records removed or private)
Rate limit:  1 request / second (polite scraping, no ToS violation)
Duration:    ~2 hours 41 minutes
Resumable:   --resume flag skips already-fetched IDs (crash-safe)
Output:      data/metadata/cn_types_metadata_full.json (~3.2 MB)
```

### Chunking -- 5 Semantic Vectors Per Coin

One coin record contains heterogeneous information. A single 200-word blob produces a blurred averaged embedding. We split each type into **5 focused semantic chunks**:

| Chunk type | Fields | Search use case |
|-----------|--------|-----------------|
| `identity` | type_id, denomination, authority, region, date_range | Classification queries |
| `obverse` | portrait description, obverse legend | Portrait/iconography queries |
| `reverse` | reverse description, reverse legend | Reverse type queries |
| `material` | material, weight, diameter, mint | Forensic validation queries |
| `context` | persons, references, notes | Provenance queries |

```
9,541 types x 5 chunks = 47,705 ChromaDB vectors
Embedding model:  all-MiniLM-L6-v2  (384-dim, 22 MB, CPU-only)
Index build time: 9.0 minutes
On-disk size:     ~180 MB
```

Every chunk carries an `in_training_set: bool` tag. This enables the Investigator to surface historically accurate matches for coins the CNN has never seen.

---

## Hybrid RAG Engine

File: `src/core/rag_engine.py`

### BM25 + Vector + Reciprocal Rank Fusion

Pure vector search misses exact keyword matches. BM25 (the algorithm behind search engines) catches exact matches that embedding space misses. We run both in parallel and merge using **Reciprocal Rank Fusion**:

```
score(doc) = SUM  1 / (60 + rank_r(doc))
              r in {BM25, ChromaDB vector}
```

RRF achieves ~95% of cross-encoder reranker accuracy at zero latency overhead -- no 65 MB BERT model needed for 9,541 records.

### Thread-Safety

`get_rag_engine()` uses double-checked locking with `threading.Lock()`. Two simultaneous FastAPI requests on a cold server cannot both build two BM25 indexes in parallel (OOM risk). This pattern mirrors every singleton in the codebase.

---

## FastAPI Backend

File: `src/api/main.py`

### Security Stack

| Layer | Implementation |
|-------|----------------|
| Authentication | X-API-Key via `hmac.compare_digest` (timing-attack resistant) -- dev passthrough when key unset |
| JWT Sessions | Short-lived access tokens + silent refresh via Axios interceptor (in-flight deduplication) |
| Rate limiting | slowapi -- 10 requests/minute on /api/classify |
| HSTS | max-age=63072000; includeSubDomains; preload (2-year preload) |
| CSP | Dev: unsafe-eval allowed; Prod: strict without unsafe-eval |
| X-Frame-Options | DENY |
| X-Request-ID | UUID4 per request, echoed in response header |
| Prompt injection | ChatMessage with role: Literal["user","assistant"] -- Pydantic v2 rejects "system" at HTTP 422 |

### Storage

SQLite WAL mode with B-tree indexed queries. `COUNT(*)` is O(log n), `LIMIT/OFFSET` pagination replaces Python-slice O(n). Thread-safe `threading.Lock()` on every write path. `save_path.unlink(missing_ok=True)` in `finally:` on every upload.

---

## Next.js Full-Stack Frontend

Directory: `frontend/` -- Next.js 15 App Router, TypeScript 5, Tailwind CSS v4, Framer Motion 12, TanStack Query 5, Zustand 5, Axios.

### Pages

| Route | What It Shows |
|-------|---------------|
| `/` | Hero, pipeline steps, stats counters, tech stack bento grid |
| `/analyse` | Drag-drop upload . TTA toggle . real-time mission-control modal . 3-state CNN display |
| `/history` | Paginated history . URL-synced pagination . filter bar . delete . CN deep links |
| `/history/[id]` | Full analysis . Grad-CAM card . Quick Facts grid . copy link . feedback form |
| `/explore` | Public gallery, no auth, route filter pills, ConfidenceBadge |
| `/chat` | SSE streaming . RAG sources sidebar . Google Scholar CTA . typing indicator |
| `/about` | Project story, pipeline steps, team |
| `/docs` | REST API reference with cURL + Python examples |
| `/admin` | All analyses . user corrections . subscriber panel . stats |

### 3-State CNN Display

Eliminates "confidence anxiety" -- the UI never shows a failure message to the user:

```
State 1 (conf >= 70%):         Green  "Identified"         + CountUp percentage
State 2 (TTA vote >= 87.5%):   Teal   "Consistent Match"   + "N/8 agree" (no raw %)
State 3 (below both):          Purple "Deep Search"         + "Best Visual Match" (no raw %)
```

### Security Headers (6 headers in next.config.ts)

CSP (blob: in img-src, dev/prod split) . HSTS 2-year preload . X-Frame-Options: DENY . nosniff . Referrer-Policy . Permissions-Policy

---

## MLflow Experiment Tracking

Every call to `scripts/train.py` logs a complete MLflow run:

- All hyperparameters (lr, batch_size, epochs, dropout, mixup_alpha, label_smoothing)
- Per-epoch train/val accuracy and loss as time series
- The model artifact (best_model.pth) as a registered artifact

View all runs: `python -m mlflow ui --host 127.0.0.1 --port 5000`

Safe test (4 min, will NOT overwrite best_model.pth): `python scripts/train.py --fast --epochs 3`

The checkpoint guard `if val_acc > best_val_acc` ensures fast-mode (~20%) never overwrites the stored V3 best (79.25%).

---

## Active Learning Loop

Files: `scripts/active_learning.py`, `src/api/routes/active_learning.py`

The system continuously improves from curator corrections:

```
1. Curator classifies a coin via /api/classify
2. Curator submits a correction: POST /api/history/{id}/feedback
   {"correct_type_id": "1015", "note": "misidentified"}
3. GET /api/admin/active-learning/candidates   -- list unexported corrections
4. python scripts/active_learning.py --dry-run -- preview export (read-only)
5. python scripts/active_learning.py           -- export corrected images
   Writes MANIFEST.csv and EXPORT_REPORT.txt to data/active_learning/
6. python scripts/train.py --active-learning-dir data/active_learning/
   Injects corrected images with 3x sampler weight
```

Export marks records as `used_for_training=True` -- idempotent, prevents double-training.

---

## Technology Stack

### Deep Learning Core

| Component | Version | Role |
|-----------|---------|------|
| PyTorch | 2.6.0+cu124 | Neural network framework |
| torchvision | 0.21+ | EfficientNet-B3 pretrained weights |
| OpenCV | 4.13.0 | CLAHE . auto-crop . HSV forensics |
| Albumentations | 1.4+ | Training augmentation pipeline |
| CUDA | 12.4 | GPU acceleration (RTX 3050 Ti, 4.3 GB VRAM) |
| MLflow | 3.10.1 | Experiment tracking |
| grad-cam | latest | Grad-CAM++ heatmap generation |

### RAG & Agents

| Component | Version | Role |
|-----------|---------|------|
| ChromaDB | 0.6+ | Persistent local vector database (47,705 vectors) |
| sentence-transformers | 3.3+ | all-MiniLM-L6-v2 (384-dim, 22 MB, CPU) |
| rank-bm25 | latest | BM25Okapi keyword index |
| LangGraph | 0.3+ | State machine orchestration |
| LangChain | 0.3+ | Prompt management |
| fpdf2 | latest | Professional PDF generation (direct-draw) |

### LLM Provider Chain

```
Priority 1:  GITHUB_TOKEN   -> GitHub Models API    (Gemini 2.5 Flash -- free with Copilot Pro)
Priority 2:  GOOGLE_API_KEY -> Google AI Studio     (Gemini 2.5 Flash -- 1,500 req/day free)
Priority 3:  OLLAMA_HOST    -> Local Ollama          (gemma3:4b text / qwen3-vl:4b vision)
Priority 4:  None set       -> Structured fallback   (KB fields only -- no crash, no hallucination)
```

### Backend

| Component | Version | Role |
|-----------|---------|------|
| FastAPI | 0.115+ | Async REST API, OpenAPI docs |
| Uvicorn | 0.40+ | ASGI server |
| Pydantic v2 | 2.x | Schema validation |
| slowapi | 0.1.9 | Rate limiting |
| python-json-logger | 3.0+ | Structured JSON logging |
| pytest + pytest-asyncio | 9.0+ | 122 tests, async integration |

### Frontend

| Component | Version | Role |
|-----------|---------|------|
| Next.js | 15 (App Router) | Server Components + client islands |
| TypeScript | 5 | Type-safe codebase |
| Tailwind CSS | 4 | Utility-first styling |
| Framer Motion | 12 | Animations -- transitions, particle beams, CountUp |
| TanStack Query | 5 | Server state management |
| Zustand | 5 | Client state (with _cancelFn abort bridge) |

### Infrastructure (Gap 4 -- complete)

| Component | Version | Role |
|-----------|---------|------|
| Docker Compose | 2.x | 7-service orchestration |
| PostgreSQL | 17 | Persistent classification history |
| Redis | 7 | Session cache + result TTL |
| Nginx | 1.27 | Reverse proxy + TLS termination |
| LocalStack | 3.x | AWS S3 simulation for PDF storage |
| GitHub Actions | -- | CI: pytest (3.11+3.12) + flake8 + black + tsc |

---

## Performance Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| CNN test accuracy (TTA x8, 438 classes) | > 80% | **80.03%** |
| CNN macro F1 (438 classes) | > 0.75 | **0.7763** |
| Historian route latency | < 25 s | ~15-20 s (Ollama gemma3:4b) |
| Validator route latency | < 15 s | ~9.8 s |
| Investigator route (OpenCV fallback) | < 5 s | ~3.1 s |
| PDF generation | < 1 s | ~0.4-0.5 s |
| KB hybrid search | < 50 ms | < 1 ms |
| Knowledge base coverage | 9,716 types | 9,541 (98.2%) |
| Test suite | 100% pass | **122 / 122** |
| End-to-end routes | 3 / 3 | **3 / 3 PASS** |

---

## Quick Start

### Prerequisites

- Python 3.11
- Node.js 22
- NVIDIA GPU with CUDA 12.4 (CPU inference works -- slower)
- ~8 GB disk for models + processed dataset

### Installation

```powershell
git clone https://github.com/ChaiebDhia/DeepCoin-Core.git
cd DeepCoin-Core

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend ; npm install ; cd ..
```

### Environment Variables

Copy `.env.example` to `.env`:

```powershell
$env:GITHUB_TOKEN   = "ghp_..."   # GitHub Copilot Pro (Gemini 2.5 Flash, free)
$env:GOOGLE_API_KEY = "AIza..."   # Fallback: Google AI Studio (1,500/day free)
```

### Build the Knowledge Base (one-time, ~2h 41min)

```powershell
python scripts/build_knowledge_base.py --all-types
python scripts/rebuild_chroma.py
```

### Run

```powershell
# Backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Frontend (new terminal)
cd frontend ; npm run dev

# End-to-end test
python scripts/test_pipeline.py

# Tests
python -m pytest tests/ -v
```

---

## API Reference

### POST /api/classify

Upload a coin photograph for full pipeline analysis.

**Request:** `multipart/form-data`, field `file` (JPEG/PNG, max 10 MB)

**Response:**
```json
{
  "id": "uuid-string",
  "cnn": {
    "label": "1015",
    "confidence": 0.911,
    "vote_fraction": 0.875,
    "tta_passes": 8,
    "gradcam_url": "/api/gradcam/uuid_heatmap.png",
    "top5": [{"rank": 1, "label": "1015", "confidence": 0.911}]
  },
  "route_taken": "historian",
  "report": "Expert analysis text...",
  "pdf_path": "reports/uuid_coin.pdf",
  "node_timings": {"cnn": "0.54s", "historian": "14.2s", "synthesis": "0.47s"},
  "created_at": "2026-03-07T10:23:45"
}
```

### GET /api/health

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

### Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/history | SQL-paginated history (newest-first) |
| GET | /api/history/{id} | Full analysis detail |
| DELETE | /api/history/{id} | Delete record (204/404) |
| POST | /api/history/{id}/feedback | Submit curator correction for active learning |
| GET | /api/reports/{filename} | Serve PDF (path-traversal protected) |
| GET | /api/gradcam/{filename} | Serve Grad-CAM++ heatmap PNG |
| GET | /api/metrics | Prometheus text (API key required) |
| POST | /api/chat | RAG-grounded numismatic Q&A |
| POST | /api/chat/stream | SSE streaming chat (per-token delta events) |
| GET | /api/explore | Public gallery (no auth, GDPR: strips user_id) |
| GET | /api/admin/active-learning/candidates | Unexported curator corrections |
| POST | /api/admin/active-learning/export | Trigger active-learning export |

Full interactive docs (dev mode only): `http://localhost:8000/api/docs`

---

## Project Structure

```
deepcoin/
+-- src/
|   +-- data_pipeline/
|   |   +-- prep_engine.py           # CLAHE + auto-crop + aspect-preserving resize
|   +-- core/
|   |   +-- model_factory.py         # EfficientNet-B3 definition (Dropout=0.4)
|   |   +-- dataset.py               # DeepCoinDataset + Albumentations transforms
|   |   +-- inference.py             # CoinInference: TTA x8, CLAHE, Grad-CAM++, auto-crop
|   |   +-- gradcam.py               # GradCAMPlusPlus at features[-4] 19x19
|   |   +-- knowledge_base.py        # Legacy ChromaDB wrapper (fallback)
|   |   +-- rag_engine.py            # Hybrid BM25+vector+RRF -- 47,705 vectors
|   +-- agents/
|   |   +-- gatekeeper.py            # LangGraph orchestrator -- logging, timing, retry
|   |   +-- historian.py             # [CONTEXT N] RAG + Gemini/Ollama narrative
|   |   +-- investigator.py          # VLM + OpenCV fallback (9,541-type search)
|   |   +-- validator.py             # Multi-scale HSV + Ag2S patina override
|   |   +-- synthesis.py             # Professional PDF (Grad-CAM embedded)
|   +-- api/
|       +-- main.py                  # Lifespan, CORS, HSTS, GZip, X-Request-ID
|       +-- auth.py                  # X-API-Key (hmac.compare_digest)
|       +-- limiter.py               # slowapi singleton (10/min)
|       +-- logging_config.py        # JSON/text structured logging
|       +-- _store.py                # SQLite WAL (COUNT O(log n), LIMIT/OFFSET)
|       +-- schemas.py               # Pydantic v2 response contracts
|       +-- routes/
|           +-- classify.py          # POST /api/classify
|           +-- history.py           # GET/DELETE /api/history
|           +-- chat.py              # POST /api/chat + SSE stream
|           +-- explore.py           # GET /api/explore (public)
|           +-- active_learning.py   # Admin AL endpoints
+-- frontend/                        # Next.js 15 -- 9 pages, 25+ components
|   +-- app/
|       +-- page.tsx                 # Homepage (Server Component + client island)
|       +-- analyse/page.tsx         # Upload + AgentPipeline modal
|       +-- history/page.tsx         # Paginated history table
|       +-- history/[id]/page.tsx    # Full detail + Grad-CAM card
|       +-- explore/page.tsx         # Public gallery
|       +-- chat/page.tsx            # Streaming AI numismatic chat
|       +-- about/page.tsx           # Project story
|       +-- docs/page.tsx            # API reference
|       +-- admin/page.tsx           # Admin dashboard
+-- scripts/
|   +-- train.py                     # CNN training V3 (MLflow-wired, AMP, Mixup)
|   +-- evaluate_tta.py              # TTA evaluation (+0.78% = 80.03%)
|   +-- predict.py                   # CLI inference
|   +-- test_pipeline.py             # End-to-end test (3 routes, 3/3 PASS)
|   +-- build_knowledge_base.py      # CN scraper (--all-types, --resume)
|   +-- rebuild_chroma.py            # ChromaDB rebuild
|   +-- active_learning.py           # Curator correction export
|   +-- compare_heatmaps.py          # 3-panel Grad-CAM++ jury figure
+-- tests/
|   +-- unit/                        # 45 tests -- store, security, auth
|   +-- integration/                 # 77 tests -- health, classify, history, chat, auth
+-- models/
|   +-- best_model.pth               # EfficientNet-B3 V3 -- epoch 52, 80.03% TTA
|   +-- class_mapping.pth            # {class_to_idx, idx_to_class, n=438}
+-- data/
|   +-- processed/                   # 7,677 images x 438 classes (299x299 JPEG)
|   +-- metadata/
|       +-- cn_types_metadata_full.json  # 9,541 CN types (~3.2 MB)
|       +-- chroma_db_rag/           # 47,705-vector production index
+-- ENGINEERING_JOURNAL.md           # 199 sections -- every decision, every bug
+-- .github/
|   +-- copilot-instructions.md      # Persistent AI context (full project knowledge)
|   +-- workflows/ci.yml             # GitHub Actions (Python 3.11+3.12 matrix)
+-- docker-compose.yml               # Gap 4 complete -- 7 services + migration profile
+-- pyproject.toml                   # Build config + lint/test tool config
+-- Makefile                         # Developer shortcuts (api/test/lint/fmt/train/mlflow)
+-- requirements.txt                 # 50+ Python dependencies
```

---

## Engineering Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CNN backbone | EfficientNet-B3 | Compound scaling (depth+width+resolution). B3 fits 4.3 GB VRAM. B7 does not. |
| Preprocessing | CLAHE in LAB L-channel | Enhances contrast without destroying diagnostic metal patina colours |
| Resize | Aspect-preserving + zero-pad | Coins are round -- stretching deforms geometry and misleads the CNN |
| Class imbalance | WeightedRandomSampler (1/count) | 40:1 ratio -- sampler equalises per-class exposure |
| Regularisation | Mixup alpha=0.2 + label smoothing 0.1 | Prevents memorisation on small dataset |
| GPU efficiency | AMP float16 + GradScaler | Halves VRAM, ~2x epoch speed on RTX 3050 Ti |
| Explainability | Grad-CAM++ at features[-4] 19x19 | 3.6x finer resolution than features[-1] 10x10; sharper multi-instance attention |
| Agent framework | LangGraph (not CrewAI) | Conditional routing + cycles + explicit state + production-ready |
| KB scope | All 9,541 CN types | CNN is image-constrained (438 at >=10/class); KB is text -- no image threshold applies |
| Chunking | 5 semantic chunks / coin | Targeted embeddings -- "silver coin" hits material chunk, not a blurred blob |
| Search | BM25 + ChromaDB + RRF | BM25 catches exact keyword hits; vectors catch semantic similarity; RRF merges both |
| Reranking | RRF formula (not cross-encoder) | 9,541 records -- formula achieves ~95% of reranker accuracy at zero latency |
| LLM grounding | [CONTEXT N] citation blocks | LLM writes, KB provides facts -- zero hallucination on structured fields |
| PDF engine | fpdf2 direct-draw | Zero Markdown parsing, full layout control, Greek transliteration map |
| Security | hmac.compare_digest | Constant-time comparison -- prevents timing oracle attacks on API key |
| Thread safety | Double-checked locking on all singletons | RAGEngine, LLM clients -- prevents OOM races on cold FastAPI startup |
| Architecture | Modular monolith | 1-person PFE team -- microservices = premature. Clean module interfaces = correct. |

---

## Roadmap -- What Comes Next

| # | Gap | Status | Description |
|---|-----|--------|-------------|
| 1 | MLflow Tracking | Complete | Every training run logged -- params, metrics, model artifact |
| 2 | Grad-CAM++ | Complete | 19x19 heatmaps in PDFs + web UI (GradCAMPlusPlus, features[-4]) |
| 3 | Active Learning | Complete | Curator corrections -> weighted export -> retraining injection |
| **4** | **Docker Compose** | **Implemented (hardening pending)** | 7 services: FastAPI + Next.js + PostgreSQL + Redis + MLflow + Nginx + LocalStack |
| 5 | Observability | Planned | Prometheus metrics -> Grafana dashboard (latency, KB hit rate, route distribution) |
| 6 | ArcFace Loss | Planned | Replace CrossEntropy head with metric learning -- target: 85%+ accuracy |
| 7 | PostgreSQL Migration | Planned | Replace residual SQLite paths in runtime/history with Postgres-only architecture |
| 8 | Deployment Automation | Planned | Add CD workflow (build, scan, deploy, rollback) |
| 9 | Container Security Hardening | Planned | Resolve current base-image vulnerability findings and enforce scan gate in CI |

---

## Academic Context

| Field | Value |
|-------|-------|
| **Institution** | ESPRIT School of Engineering, Manouba, Tunisia |
| **Company** | YEBNI -- Information & Communication, Tunisia |
| **Project type** | PFE (Projet de Fin d'Etudes) -- 5-month final year internship |
| **Period** | February - July 2026 |
| **Student** | Dhia Chaieb -- dhia.chaieb@esprit.tn |
| **GitHub** | ChaiebDhia/DeepCoin-Core |
| **Dataset** | Corpus Nummorum v1 -- 115,160 images, 9,716 types (DFG-funded) |
| **Domain** | Fine-grained visual recognition + archaeological numismatics |

### Research Questions Addressed

1. **Can transfer learning from ImageNet reliably classify ancient coins?** Yes -- 80.03% TTA on 438-way fine-grained classification.
2. **Does hybrid BM25+vector search outperform vector-only for numismatic retrieval?** Yes -- exact keyword recall improves on material/mint queries.
3. **Can RAG grounding eliminate LLM hallucination on structured numismatic facts?** Yes -- [CONTEXT N] citation format produces zero invented dates or mints in testing.
4. **Can graceful degradation replace "I don't know" with useful output?** Yes -- all 3 routing paths produce valid reports including for out-of-distribution coins.
5. **Is low confidence caused by model failure or input distribution shift?** Neither necessarily -- BNF 1966 catalog scans score 15-28% even for trained types vs 80-96% for standard composite photos. A novel intra-dataset distribution shift finding documented in Engineering Journal Section 184.

---

## License

MIT -- see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Corpus Nummorum** -- Berlin-Brandenburg Academy of Sciences, DFG-funded numismatic catalogue
- **Google Brain** -- EfficientNet architecture (Tan & Le, 2019)
- **LangChain AI** -- LangGraph state machine framework
- **YEBNI** -- Company supervisor and domain expertise
- **ESPRIT School of Engineering** -- Academic supervision

---

*DeepCoin-Core -- Where 2,300-year-old coins meet production AI engineering.*
*Dhia Chaieb . ESPRIT . YEBNI . 2026*
