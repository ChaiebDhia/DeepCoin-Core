# DeepCoin-Core — Copilot Persistent Context
# ============================================
# This file is automatically injected into every GitHub Copilot Chat session.
# It gives Copilot full knowledge of the project state, decisions, and rules.
# NEVER delete this file. Update it after every major milestone.
# Last updated: February 27, 2026

---

## 0. IRON RULES — READ THESE FIRST, NEVER VIOLATE THEM

1. **"Never go to the next layer unless all is engineered as experts will do — enterprise-grade and production-ready."**
2. **"Don't add any code unless we discuss it first."** Always present the plan, wait for "go" approval.
3. **"Explain everything like teaching — WHAT it does, WHY it's designed this way, HOW it fits."**
4. Every function must have detailed docstrings.
5. This is a PFE (Final Year Engineering Internship) — ESPRIT School of Engineering × YEBNI, Tunisia.
6. Student: **Dhia Chaieb** | GitHub: `ChaiebDhia` | Email: `dhia.chaieb@esprit.tn`
7. GitHub repo: `https://github.com/ChaiebDhia/DeepCoin-Core` | Branch: `main`
8. Always assume the Python venv at `C:\Users\Administrator\deepcoin\venv\` is active.
9. OS: Windows 11 | Shell: PowerShell 5.1 | Use `;` not `&&` to chain commands.
10. GPU: NVIDIA RTX 3050 Ti, 4.3 GB VRAM | CUDA 12.4 | PyTorch 2.6.0+cu124

---

## 1. PROJECT MISSION

Build an end-to-end industrial AI system that:
- Classifies degraded archaeological ancient coins from a photograph
- Routes the analysis through specialist AI agents based on confidence
- Returns a professional PDF report with historical narrative, forensic validation, and visual attributes
- Handles unknown coins gracefully (never returns "I don't know" — always returns useful output)
- Covers the full Corpus Nummorum domain (9,716 coin types in KB, 438 in CNN)

**The core philosophy:** "Failing gracefully is better than failing confidently."

---

## 2. ARCHITECTURE — TWO-STAGE HYBRID PIPELINE

### Stage 1 — Deep Learning (Visual Classification)
```
Raw coin photo
  → CLAHE Enhancement (LAB color space, L-channel only, clipLimit=2.0, tile=8×8)
  → Aspect-preserving resize to 299×299 with zero-padding (no stretching)
  → EfficientNet-B3 (12M params, ImageNet pretrained, fine-tuned on 438 coin classes)
  → 1536-dimensional feature vector
  → Softmax → top-1 class + confidence score + top-5 predictions
```

### Stage 2 — Agentic System (Historical Reasoning)
```
confidence > 0.85   →  Historian Agent (high confidence — RAG + LLM narrative)
0.40 ≤ conf ≤ 0.85  →  Validator Agent + Historian Agent (verify material first)
confidence < 0.40   →  Investigator Agent (VLM + local CV fallback — unknown coin)

All paths → Synthesis Agent → PDF report
```

### Agent Communication
All agents share a single LangGraph `CoinState` TypedDict:
```python
class CoinState(TypedDict, total=False):
    image_path        : str
    use_tta           : bool
    cnn_prediction    : dict   # {class_id, label, confidence, top5}
    route_taken       : Literal["historian", "validator", "investigator"]
    historian_result  : dict
    validator_result  : dict
    investigator_result: dict
    report            : str    # final Markdown
    pdf_path          : Optional[str]
```

---

## 3. COMPLETE TECHNOLOGY STACK

### Deep Learning
| Component | Version | Detail |
|-----------|---------|--------|
| PyTorch | 2.6.0+cu124 | Neural network framework |
| torchvision | 0.21+ | EfficientNet-B3 pretrained weights |
| EfficientNet-B3 | ImageNet pretrained | 12M params, 1536-dim features, 438-class output head |
| OpenCV | 4.13.0 | CLAHE preprocessing, HSV material detection |
| Albumentations | 1.4+ | Training augmentation pipeline |
| NumPy | 2.x | Numerical ops |
| scikit-learn | latest | Stratified splits, WeightedRandomSampler support |

### Agentic AI
| Component | Version | Detail |
|-----------|---------|--------|
| LangGraph | 0.3+ | State machine orchestration — conditional routing, cycles |
| LangChain | 0.3+ | Agent tooling, prompt management |
| openai SDK | latest | Used for BOTH GitHub Models AND Google AI Studio (both OpenAI-compatible) |
| ChromaDB | 0.6+ | Local vector database, persisted to disk |
| sentence-transformers | 3.3+ | `all-MiniLM-L6-v2` embedding model (384-dim, 22MB, CPU) |
| fpdf2 | latest | PDF generation — all direct draw primitives, NO Markdown parsing |
| rank-bm25 | latest | BM25Okapi keyword search (to be added in enterprise RAG upgrade) |

### LLM Provider Chain (priority order)
```
1. GITHUB_TOKEN env var  → GitHub Models API (Gemini 2.5 Flash)
   base_url: https://models.inference.ai.azure.com
   model: "gemini-2.5-flash"
   Free with GitHub Copilot Pro student

2. GOOGLE_API_KEY env var → Google AI Studio
   base_url: https://generativelanguage.googleapis.com/v1beta/openai/
   model: "gemini-2.5-flash"
   Free tier: 1,500 req/day

3. OLLAMA_HOST env var → Local Ollama (gemma3:4b or llama3.2:3b)
   Hook written, Ollama NOT currently installed
   gemma3:4b fits in 4.3 GB VRAM

4. None set → structured fallback (KB fields concatenated, no hallucination, no crash)
```

### Backend (Layer 4 — pending)
- FastAPI 0.115+ (async, auto-docs, Pydantic v2 validation)
- Uvicorn 0.40+
- SQLAlchemy 2.x async + Alembic migrations
- PostgreSQL 17

### Frontend (Layer 5 — pending)
- Next.js 15 (App Router, Server Components)
- TypeScript 5
- Tailwind CSS 4
- shadcn/ui (Radix UI)
- TanStack Query 5
- Zustand 4

### Infrastructure (Layer 6 — pending)
- Docker Compose 2.x (7 services)
- Redis 7 (result cache)
- Nginx 1.27 (reverse proxy)
- LocalStack 3.x (AWS S3 + Lambda simulation)
- GitHub Actions (CI: pytest + flake8 + black)

---

## 4. CNN MODEL — FULL DETAILS

### Architecture
- **Model**: EfficientNet-B3 (compound scaling: balanced depth/width/resolution)
- **Why B3 not B7**: B7 exceeds 4.3 GB VRAM budget; B3 is optimal param/accuracy ratio
- **Input**: 299×299 RGB
- **Feature extractor output**: 1536-dim vector
- **Output head**: `nn.Linear(1536, 438)` + Dropout(0.4) — replaced from original 1000-class head
- **Pretrained on**: ImageNet (1.2M images, 1000 classes)
- **Fine-tuned on**: 438 CN coin types, 7,677 images

### Training Configuration (V3 — `scripts/train.py`, 729 lines)
```python
optimizer     = AdamW(lr=1e-4, weight_decay=0.01)
scheduler     = CosineAnnealingLR(T_max=100, eta_min=1e-6)
loss          = CrossEntropyLoss(label_smoothing=0.1)
augmentation  = Albumentations (rotate ±15°, brightness ±20%, elastic, GaussNoise)
mixup         = alpha=0.2   # blends 2 images: λ×imgA + (1-λ)×imgB — prevents memorization
amp           = torch.amp.GradScaler('cuda') + autocast  # halves VRAM, ~2× faster
gradient_clip = max_norm=1.0
batch_size    = 16  # GPU memory constraint (4.3 GB VRAM)
early_stop    = patience=10 on val accuracy
pin_memory    = True
non_blocking  = True
seed          = 42
```

### Data Pipeline
```
115,160 raw images (9,716 unique coin types from Corpus Nummorum)
    ↓ filter: ≥10 images per class
438 viable classes, 7,677 images
    ↓ CLAHE in LAB color space
    ↓ Aspect-preserving resize to 299×299
    ↓ Stratified 70/15/15 split (seed=42)
    ↓ WeightedRandomSampler (fixes 40:1 class imbalance)
Training: 5,374 images | Validation: 1,151 | Test: 1,152
```

### Results
| Metric | Value |
|--------|-------|
| Best epoch | 52 / 100 |
| Val accuracy (epoch 52) | 79.25% |
| Test accuracy (single pass) | 79.08% |
| **Test accuracy (TTA ×8)** | **80.03%** |
| Mean F1 (macro, 438 classes) | 0.7763 |
| Top confusion pair | 3314 → 3987 (10× misclassification) |
| Training duration | ~103 min on RTX 3050 Ti |

### TTA (Test-Time Augmentation)
- 8 passes: original + horizontal flip + 2×vertical variants + 4×crops
- Predictions averaged → +0.78% accuracy gain over single-pass
- Implemented in `src/core/inference.py` → `CoinInference.predict(tta=True)`

### Saved Artefacts
```
models/best_model.pth           # V3 weights (epoch 52) — the real model
models/best_model_v1_80pct.pth  # MISLEADING NAME — actually epoch 3, val 21.33%, NOT the 80% model
models/class_mapping.pth        # {class_to_idx: {"1015": 0, ...}, idx_to_class: {0: "1015", ...}, n: 438}
```

---

## 5. LAYER-BY-LAYER STATUS

### Layer 0 — CNN Training ✅ COMPLETE
File: `scripts/train.py` (729 lines)
Status: EfficientNet-B3 trained, 80.03% TTA accuracy achieved.

### Layer 1 — Inference Engine ✅ COMPLETE
Files: `src/core/inference.py`, `scripts/predict.py`
- `CoinInference`: loads model once, runs TTA, returns structured prediction dict
- Device resolution: `"auto"` resolved to `"cuda"` or `"cpu"` before PyTorch sees it
- Bug fixed: original code passed `"auto"` directly to `.to(device)` → RuntimeError

### Layer 2 — Knowledge Base ✅ COMPLETE (but needs expansion)
Files: `src/core/knowledge_base.py`, `scripts/build_knowledge_base.py`, `data/metadata/cn_types_metadata.json`

**Current state** (needs upgrade):
- ChromaDB collection: `cn_coin_types`, 434 documents
- Embedding: `all-MiniLM-L6-v2` (384-dim, cosine similarity)
- One document per coin = one 200-word text blob per type (BAD — to be fixed)
- Only 438 types in KB (BAD — should be 9,716)
- `search(query, n, where)` → vector-only search (no BM25)
- `search_by_id(type_id)` → exact ID lookup
- `build_from_metadata(path)` → builds ChromaDB from JSON

**Known critical gaps**:
1. Only 438 types — should be ALL 9,716 from Corpus Nummorum
2. One blob per coin — should be 5 semantic chunks per type
3. Vector-only search — no BM25, no hybrid, no RRF
4. `in_training_set` tag missing (needed to distinguish CNN scope from KB scope)

### Layer 3 — Agent System ✅ WORKING → 🔧 ENTERPRISE UPGRADE IN PROGRESS
All 5 agents written, end-to-end test passing (type 1015, 91.1%, historian route, PDF generated).

**Latest commit**: `113514b` — Greek transliteration fix + footer band removal

#### Agent Files and Current State:

**`src/agents/gatekeeper.py`** (245 lines) — LangGraph orchestrator
- `CoinState` TypedDict: full shared pipeline state
- `Gatekeeper.__init__()`: loads ALL agents once, resolves `"auto"` device
- Routing thresholds: `HIGH_CONF=0.85`, `LOW_CONF=0.40` (class constants)
- Routes: historian / validator+historian / investigator
- **Pending upgrades**: structured logging, retry (up to 2× on 429/503), graceful degradation per node, per-node timing

**`src/agents/historian.py`** (212 lines) — RAG + LLM narrative
- `_get_llm()`: GitHub Models / Google AI Studio lazy singleton
- `research(cnn_prediction)→dict`: calls `search_by_id()` → passes raw document string to Gemini
- `_generate_narrative(record, confidence)`: single-turn Gemini call
- `_fallback_narrative(record)`: field concatenation when no LLM key
- **Pending upgrades**: true RAG (hybrid search → 5-chunk injection → grounded generation), multi-query retrieval, citation refs, "Related Types" section from full 9,716 KB

**`src/agents/investigator.py`** — VLM visual agent
- Base64-encodes image → Gemini Vision 6-point structured prompt
- KB cross-reference: uses Gemini description as semantic search query
- `_parse_features(description)`: naive regex extraction
- **Pending upgrades**: local CV fallback (HSV histogram + Sobel edges + ORB keypoints when no API key), search full 9,716 KB (not just 438), better feature parsing

**`src/agents/validator.py`** — OpenCV forensic material validator
- Crops centre 60% of coin, HSV mask analysis
- Gold threshold: H 15-35, S 80-255 | Bronze: H 5-25, S 50-180 | Silver: S < 40
- 15% pixel fraction threshold (hardcoded)
- `_materials_match()`: simplistic string comparison
- **Pending upgrades**: multi-scale (40%/60%/80% crops), confidence score 0-100%, uncertainty flag (low/medium/high), per-channel std analysis, cross-reference KB on mismatch

**`src/agents/synthesis.py`** — Professional PDF generator ✅ COMPLETE, NO CHANGES NEEDED
- `synthesize(state)→str`: clean plain-text summary
- `to_pdf(state, output_path)`: ALL direct fpdf2 draw — NO Markdown parsing
- Navy header band, bordered tables with alternating shading, blue section rule lines
- `_GREEK_MAP`: dict-based Greek→Latin transliteration (Κ→K, Ε→E, Ρ→R, etc.)
- Bug fixed: Greek `???` chars replaced via transliteration map
- Bug fixed: duplicate footer band removed (header already carries branding)
- Signature change from `to_pdf(markdown_str, path)` → `to_pdf(state_dict, path)`

### Layer 4 — FastAPI Backend 🔲 PENDING
Files to create: `src/api/main.py`, `src/api/routes/classify.py`, `src/api/routes/history.py`, `src/api/schemas.py`
Endpoints planned: `POST /api/classify`, `GET /api/health`, `GET /api/history`, `GET /api/history/{id}`, `WS /ws/classify/{session_id}`

### Layer 5 — Next.js Frontend 🔲 PENDING
Directory: `frontend/`
Stack: Next.js 15 App Router, TypeScript 5, Tailwind CSS 4, shadcn/ui, TanStack Query 5, Zustand 4

### Layer 6 — Docker + Infrastructure 🔲 PENDING
File: `docker-compose.yml` (skeleton exists)
7 services: FastAPI + Next.js + ChromaDB + PostgreSQL + Redis + Nginx + LocalStack

### Layer 7 — Tests + CI/CD 🔲 PENDING
Directories: `tests/unit/`, `tests/integration/`
Stack: pytest 8.x, Jest, Playwright, GitHub Actions (`.github/workflows/ci.yml`)

---

## 6. THE ENTERPRISE UPGRADE PLAN (CURRENT ACTIVE WORK)

This is the work happening NOW before moving to Layer 4.

### The Problem Statement
Current state covers only 4.5% of the CN numismatic domain (438 / 9,716 types). This is the core gap to fix.

### Full 9,716-Type KB Strategy (APPROVED)
- CNN training was limited to 438 types (image threshold ≥10 per class)
- KB is pure text — has NO image constraint — should cover all 9,716 types
- `in_training_set: bool` tag distinguishes CNN-known from KB-only types
- Impact: Investigator transforms from "fallback agent" into "numismatic detective"
- Scrape cost: ~2.7 hours at 1 req/sec (one-time, resumable with `--resume`)

### 5 Semantic Chunks Per Coin
Each coin record split into 5 ChromaDB documents with tagged `chunk_type`:
```
chunk_type="identity"  → type_id, denomination, authority, region, date_range
chunk_type="obverse"   → obverse description + legend
chunk_type="reverse"   → reverse description + legend
chunk_type="material"  → material, weight, diameter, mint
chunk_type="context"   → persons, references, notes
```
Result: 9,716 × 5 = 48,580 vectors (~180 MB ChromaDB on disk)
Why: Each chunk embeds cleanly; "silver coin" search hits material chunks, "eagle reverse" hits reverse chunks.

### Hybrid Search Architecture
```
Query → BM25 keyword search (rank-bm25) → ranked list A
      → ChromaDB vector search            → ranked list B
      → RRF merge: score(d) = Σ 1/(60 + rank_r(d))
      → final re-ranked list
```
No cross-encoder model (overkill for 9,716 records; RRF gives ~95% of accuracy at 0ms overhead).

### Per-Agent Search Scope
```python
historian()    → hybrid_search(query, where={"type_id": known_id})   # exact type + neighbors
validator()    → hybrid_search(query, where={"chunk_type": "material"})  # material-scoped
investigator() → hybrid_search(query)  # FULL CORPUS — no filter, maximum coverage
```

### Grounded LLM Prompt Pattern
```
[CONTEXT 1 — Identity] denomination: denarius | authority: Augustus | date: 27 BC–14 AD
[CONTEXT 2 — Obverse]  laureate head right | legend: CAESAR AVGVSTVS
[CONTEXT 3 — Reverse]  Caius and Lucius standing | legend: PRINCIP IVVENTVTIS
[CONTEXT 4 — Material] silver | weight: 3.9g | mint: Lugdunum
[CONTEXT 5 — Context]  persons: Augustus, Caius Caesar, Lucius Caesar

INSTRUCTION: You are an expert numismatist. Using ONLY the context above (cite [CONTEXT N]),
write a 3-paragraph professional analysis. Do not add facts not present in the context.
```
This pattern = zero hallucination on structured facts, LLM only adds interpretation.

### Build Order (strict dependency sequence)
```
STEP 0: Expand build_knowledge_base.py → --all-types flag (scrape 9,716)
STEP 1: Build src/core/rag_engine.py (NEW FILE — hybrid search foundation)
STEP 2: Rebuild ChromaDB index (5 chunks × 9,716 types = 48,580 vectors)
STEP 3: Upgrade historian.py (true RAG + "Related Types" section)
STEP 4: Upgrade investigator.py (full KB search + local CV fallback)
STEP 5: Upgrade validator.py (confidence scoring + multi-scale HSV)
STEP 6: Upgrade gatekeeper.py (logging + retry + graceful degradation)
STEP 7: End-to-end test all 3 routes
STEP 8: Commit and push
```

---

## 7. KEY ENGINEERING DECISIONS (with rationale)

| Decision | Choice | Why |
|----------|--------|-----|
| CNN architecture | EfficientNet-B3 | Compound scaling; B7 exceeds 4.3 GB VRAM |
| Preprocessing | CLAHE in LAB space | Enhances contrast without destroying metal patina colors |
| Resize strategy | Aspect-preserving + zero-padding | Preserves coin geometry |
| Agent framework | LangGraph (not CrewAI) | Conditional routing + cycles + human-in-loop |
| LLM provider | GitHub Models primary | Free with Copilot Pro student |
| Vector DB | ChromaDB | Local, embeddable, zero network dependency |
| Reranking | RRF score-based (not cross-encoder) | 9,716 records — math > extra 65MB model |
| Chunking | 5 semantic chunks per coin | Better embedding precision than 1 blob |
| Architecture style | Modular Monolith | 1-person PFE team; microservices = premature |
| KB scope | All 9,716 types | CNN and KB have independent constraints |
| Ollama | Hook ready, skip install for now | Progressive enhancement |
| Transfer learning norm | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] | ImageNet stats — mandatory for pretrained weights |
| Augmentation | Albumentations pipeline | 6× synthetic expansion from 7,677 images |
| Class imbalance | WeightedRandomSampler (1/class_count) | Fixes 40:1 imbalance between most/least common types |

---

## 8. FILE STRUCTURE (complete)

```
C:\Users\Administrator\deepcoin\
│
├── .github/
│   ├── copilot-instructions.md   ← THIS FILE — persistent context
│   └── workflows/
│       └── ci.yml                ← Layer 7 (pending)
│
├── src/
│   ├── data_pipeline/
│   │   └── prep_engine.py        ✅ CLAHE + aspect-preserving resize
│   ├── core/
│   │   ├── model_factory.py      ✅ EfficientNet-B3 definition (Dropout=0.4)
│   │   ├── dataset.py            ✅ DeepCoinDataset + Albumentations transforms
│   │   ├── inference.py          ✅ CoinInference (TTA, device auto-resolve)
│   │   ├── knowledge_base.py     ✅ ChromaDB wrapper — NEEDS UPGRADE (438→9716, chunking)
│   │   └── rag_engine.py         🔲 NEW — hybrid BM25+vector+RRF search engine
│   ├── agents/
│   │   ├── gatekeeper.py         ✅ LangGraph orchestrator — NEEDS logging+retry
│   │   ├── historian.py          ✅ LLM narrative — NEEDS true RAG upgrade
│   │   ├── investigator.py       ✅ VLM agent — NEEDS local CV fallback + full KB
│   │   ├── validator.py          ✅ OpenCV forensics — NEEDS confidence score
│   │   └── synthesis.py          ✅ PDF generator — COMPLETE, no changes needed
│   └── api/
│       ├── main.py               🔲 FastAPI entry point (Layer 4)
│       ├── routes/
│       │   ├── classify.py       🔲 POST /api/classify (Layer 4)
│       │   └── history.py        🔲 GET /api/history (Layer 4)
│       └── schemas.py            🔲 Pydantic models (Layer 4)
│
├── scripts/
│   ├── train.py                  ✅ CNN training V3 (729 lines, AMP+Mixup)
│   ├── audit.py                  ✅ F1 + confusion matrix evaluation
│   ├── evaluate_tta.py           ✅ TTA evaluation (+0.78% = 80.03%)
│   ├── predict.py                ✅ CLI inference tool
│   ├── test_pipeline.py          ✅ End-to-end test (type 1015, all 3 routes)
│   ├── test_dataset.py           ✅ Dataset validation
│   └── build_knowledge_base.py   ✅ Web scraper + ChromaDB builder — NEEDS --all-types flag
│
├── models/
│   ├── best_model.pth            ✅ V3 weights — epoch 52, val 79.25%, test 79.08%, TTA 80.03%
│   ├── best_model_v1_80pct.pth   ⚠️  MISLEADING NAME — epoch 3, val 21.33%, NOT 80%
│   └── class_mapping.pth         ✅ {class_to_idx, idx_to_class, n=438}
│
├── data/
│   ├── processed/                ✅ 7,677 images × 438 classes (299×299 JPEG)
│   ├── metadata/
│   │   ├── cn_types_metadata.json ✅ 515 KB — 438 types (needs expansion to 9,716)
│   │   └── chroma_db/            ✅ ChromaDB persisted — 434 vectors (needs rebuild)
│   └── raw/                      ⚠️  Original 115k images — gitignored, may be on disk
│
├── tests/
│   ├── unit/                     🔲 Layer 7
│   └── integration/              🔲 Layer 7
│
├── frontend/                     🔲 Next.js 15 (Layer 5)
├── notebooks/                    exploration
├── reports/                      PDF output directory
│
├── requirements.txt              ✅ All Python dependencies (50+ packages)
├── docker-compose.yml            🔲 7-service skeleton (Layer 6)
├── .env                          ⚠️  Secrets file — gitignored, NEVER commit
│                                    Contains: GITHUB_TOKEN, GOOGLE_API_KEY
└── .gitignore                    ✅ Excludes: data/, models/, venv/, .env, notes.md

```

---

## 9. ENVIRONMENT AND PATHS

```powershell
# Activate venv (always do this first)
& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1

# Working directory
C:\Users\Administrator\deepcoin\

# Python 3.11 in venv
C:\Users\Administrator\deepcoin\venv\Scripts\python.exe

# Key installed packages (selected)
torch==2.6.0+cu124
torchvision==0.21.0+cu124
efficientnet-pytorch (via torchvision models)
opencv-python==4.13.0
albumentations==1.4+
chromadb==0.6+
sentence-transformers==3.3+
langgraph==0.3+
langchain==0.3+
openai (latest)
fpdf2 (latest)
scikit-learn (latest)
tqdm
rank-bm25  ← to be installed during RAG upgrade
```

---

## 10. COMMIT HISTORY (significant milestones)

| Commit | Description |
|--------|-------------|
| Initial commits | Phase 0: project setup, venv, gitignore, README |
| — | Phase 1: CLAHE preprocessing pipeline, 7,677 images |
| — | Phase 3 (Dataset): DeepCoinDataset + Albumentations |
| — | Phase 4 (Training V3): AMP + Mixup + WeightedSampler |
| — | Phase 2 (KB): ChromaDB build, 434 docs |
| — | Layer 3 agents: all 5 written |
| — | Bug fixes: IndentationError historian, device 'auto' gatekeeper, multi_cell synthesis |
| — | PDF redesign: direct fpdf2 draw (navy header, bordered tables, no Markdown parsing) |
| `113514b` | Greek transliteration fix + duplicate footer band removal ← LATEST |

---

## 11. KNOWN BUGS AND RESOLVED BUGS

### Resolved ✅
- `IndentationError` in `historian.py` — leftover TODO stub in method
- `RuntimeError: device 'auto'` — `"auto"` was passed directly to `model.to(device)` instead of being resolved to `"cuda"` or `"cpu"` first
- `multi_cell` horizontal space error in `synthesis.py` — needed `set_x()` before every `multi_cell` call
- Greek `???` characters in PDF — Greek Unicode (ΚΕΡ) was not supported by fpdf2 default font; fixed with `_GREEK_MAP` dict-based transliteration
- Branding footer band appearing on extra page — `_draw_footer_band()` call removed (header band already carries branding)
- `to_pdf()` signature mismatch — changed from `(markdown_str, path)` to `(state_dict, path)` and updated gatekeeper call accordingly

### Known (to fix in enterprise upgrade)
- `knowledge_base.py`: 1 blob per coin instead of 5 semantic chunks
- `knowledge_base.py`: only 438 types instead of 9,716
- `historian.py`: raw document blob passed to LLM — not true RAG
- `investigator.py`: 100% dependent on Gemini Vision — no local CV fallback
- `validator.py`: binary match/mismatch — no confidence score
- `gatekeeper.py`: `print()` statements instead of `logging` module

---

## 12. DATA SOURCES AND FALLBACK CHAIN

```
Priority 1: CN Dataset metadata (primary)
  → Structured fields scraped from corpus-nummorum.eu
  → Validated by Berlin-Brandenburg Academy of Sciences (DFG-funded)
  → Stored in ChromaDB, searched via hybrid BM25+vector

Priority 2: Nomisma.org SPARQL (secondary)
  → Academic linked open data — emperor names, reign periods, mint locations
  → RDF structured data, authoritative for numismatic domain

Priority 3: LLM synthesis (tertiary)
  → Gemini 2.5 Flash generates prose from injected context chunks
  → LLM WRITES, it does not INVENT — all facts come from [CONTEXT N] blocks

Priority 4: Wikipedia API (last resort)
  → Only for emperor biography narrative when no structured source covers it
  → Always flagged in output: "Source: Wikipedia (unverified)"
```

---

## 13. PERFORMANCE TARGETS

| Metric | Target | Current |
|--------|--------|---------|
| CNN Top-1 accuracy | >85% | 80.03% (TTA) — gap ~5pp |
| CNN Top-5 accuracy | >95% | Not measured yet |
| Per-class recall (rare) | >50% | Unknown |
| Full pipeline latency | <2s | Not measured (agents pending upgrade) |
| PDF generation | <500ms | Approximately met |
| KB search latency | <50ms | Sub-ms (ChromaDB) |

---

## 14. ACADEMIC CONTEXT

- **Institution**: ESPRIT School of Engineering, Manouba, Tunisia
- **Company**: YEBNI — Information & Communication, Tunisia (yebni.com)
- **Type**: PFE (Projet de Fin d'Études) — 5-month final year internship
- **Period**: February – July 2026
- **Dataset**: Corpus Nummorum v1 — 115,160 images, 9,716 types, DFG-funded
- **Problem domain**: Fine-grained archaeological numismatics with long-tail distribution
- **Key contribution**: Hybrid CNN + multi-agent RAG system with graceful degradation for OOD inputs

---

## 15. HOW TO RESUME IN ANY NEW CHAT

1. The file you're reading is automatically injected — Copilot already knows everything.
2. Say: **"Continue the enterprise upgrade of Layer 3 — we're at STEP [N] of the build order"**
3. Or say: **"What is the current status and what should we do next?"**
4. Always activate venv first: `& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1`
5. The rule is still: **discuss plan first, wait for "go", then build.**
