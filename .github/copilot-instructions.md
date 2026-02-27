# DeepCoin-Core — Copilot Persistent Context
# ============================================
# This file is automatically injected into every GitHub Copilot Chat session.
# It gives Copilot full knowledge of the project state, decisions, and rules.
# NEVER delete this file. Update it after every major milestone.
# Last updated: February 27, 2026 — COMPLETE verified rewrite, all 6 agent files cross-checked, full history included

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

## 2. COMPLETE PROJECT HISTORY — FROM RAW DATA TO NOW

This is the full chronological record. Every phase, every problem, every fix.

---

### PHASE 0 — Environment Setup (early February 2026) ✅

**What we did:**
- Created `C:\Users\Administrator\deepcoin\` directory structure
- Initialized Python 3.11 virtual environment at `venv\`
- Set up Git repo: `https://github.com/ChaiebDhia/DeepCoin-Core`
- Created `.gitignore` (excludes: `data/`, `models/`, `venv/`, `.env`, `notes.md`, `The Project.md`)
- Created `requirements.txt` (50+ deps), professional `README.md`, `.gitkeep` files

**Problems:** None. Clean setup.

---

### PHASE 1 — Dataset Auditing (mid February 2026) ✅

**Tool:** `src/data_pipeline/auditor.py`

**Discovery — Long-tail distribution problem:**
```
Raw dataset: 115,160 images across 9,716 coin types (folders in data/raw/)
Most types have only 1–3 images → neural network cannot learn from that
Decision: apply ≥10 images per class threshold
Result: 9,716 types → 438 viable classes, 7,677 images retained
```

Why ≥10 is the right cutoff: Transfer learning (ImageNet pretrained) reduces minimum data need from ~1,000 to ~10 images/class. Below 10, the model memorises rather than generalises.

---

### PHASE 1b — Preprocessing Engine (mid February 2026) ✅

**File:** `src/data_pipeline/prep_engine.py`

**Step 1 — CLAHE in LAB colour space:**
- Convert BGR → LAB (separates luminance L from colour channels A, B)
- Apply CLAHE to L channel only: `clipLimit=2.0, tileGridSize=(8,8)`
- Convert back to BGR
- Why LAB not RGB: RGB CLAHE distorts metal patina colours (the green/brown oxidation proving archaeological authenticity). LAB preserves colours while enhancing contrast on the luminance channel.

**Step 2 — Aspect-preserving resize to 299×299:**
- Scale so longest edge = 299; use `INTER_AREA` (downscale) or `INTER_CUBIC` (upscale)
- Pad shorter edge with black zeros to reach 299×299
- Why not simple resize: stretch deforms coin geometry. The model must learn coins are round.

**Output:** `data/processed/[class_id]/[files]` — 7,677 images, 438 class folders.

---

### PHASE 2 — Dataset Class (February 20, 2026) ✅

**File:** `src/core/dataset.py` (248 lines)

`DeepCoinDataset(Dataset)` — PyTorch bridge between disk and training loop.
- Lazy loading: stores `(path, label)` tuples — NOT pixel arrays. Loading 7,677 images raw = 2.6 GB RAM. Lazy loading = one batch at a time = feasible.
- `class_to_idx`: maps folder name to integer (`"1015" → 0`). Neural networks only understand numbers.
- `get_train_transforms()`: 6 Albumentations augmentations + ImageNet normalisation
- `get_val_transforms()`: normalise only (honest evaluation — no augmentation)

**Augmentations:**
```python
A.Rotate(limit=15, p=0.5)                        # tilted photos
A.RandomBrightnessContrast(0.2, 0.2, p=0.5)      # lighting variation
A.GaussNoise(p=0.3)                               # low-quality cameras
A.ElasticTransform(p=0.3)                         # worn/bent coins
A.HorizontalFlip(p=0.5)                           # either orientation
A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # ImageNet stats — MANDATORY
```
ImageNet normalisation is MANDATORY. EfficientNet-B3 was pretrained with these exact stats. Wrong values → pretrained features activate incorrectly → ~15-20% accuracy loss.

**Critical discovery from `scripts/test_dataset.py`:**
```
Min images per class:  5  (class 5181)
Max images per class: 204 (class 3987)
Imbalance ratio:      40:1  ← must be corrected during training
```

---

### PHASE 3 — Model Architecture (February 2026) ✅

**File:** `src/core/model_factory.py`

`get_deepcoin_model(num_classes=438, dropout=0.4)`:
- Base: `torchvision.models.efficientnet_b3(pretrained=True)` — ImageNet weights
- Replace head: `nn.Linear(1536, 1000)` → `nn.Sequential(nn.Dropout(0.4), nn.Linear(1536, 438))`
- Dropout 0.4: 40% of neurons zeroed per forward pass → cannot rely on any single neuron → less memorisation

Why EfficientNet-B3: compound scaling (depth + width + resolution simultaneously). B3 = best accuracy/parameter ratio for 4.3 GB VRAM. B7 would need ~8 GB.

The 1536-dim vector before the head = coin's "fingerprint" — 18 convolution layers encoding all visual features.

---

### PHASE 4 — Training V3 (February 2026) ✅

**File:** `scripts/train.py` (729 lines)

```python
optimizer     = AdamW(lr=1e-4, weight_decay=0.01)
scheduler     = CosineAnnealingLR(T_max=100, eta_min=1e-6)
loss          = CrossEntropyLoss(label_smoothing=0.1)
augmentation  = Albumentations (6 transforms)
mixup_alpha   = 0.2        # Beta(0.2,0.2) blending
amp           = GradScaler('cuda') + autocast('cuda')   # halves VRAM
gradient_clip = max_norm=1.0
batch_size    = 16         # 4.3 GB VRAM constraint
early_stop    = patience=10 on val accuracy
seed          = 42
```

Mixup: `mixed = λ×A + (1-λ)×B` with `λ ~ Beta(0.2,0.2)`. Smooth decision boundaries. Reduces train/val gap by ~3-4% on small datasets.

AMP: float16 gradients → halves VRAM, ~2× faster/epoch. GradScaler prevents underflow that would corrupt float16.

WeightedRandomSampler: weight_i = 1/count(class_i) → each class seen approximately equally → fixes 40:1 imbalance.

**Data split (stratified, seed=42):**
```
Train:      5,374  (70%)  — sampler applied
Validation: 1,151  (15%)  — no augmentation
Test:       1,152  (15%)  — run ONCE at end
```

**Results:**
```
Best epoch:         52 / 100
Val accuracy:       79.25%
Test accuracy:      79.08%  (single-pass)
TTA accuracy (×8):  80.03%  ← official result
Macro F1:           0.7763  (438 classes)
Top confusion:      3314 → 3987  (10× misclassification)
Training time:      ~103 min on RTX 3050 Ti
Early stop:         epoch 62 (10 epochs no improvement)
```

---

### PHASE 4b — TTA Evaluation (February 2026) ✅

**File:** `scripts/evaluate_tta.py`

TTA (Test-Time Augmentation): 8 forward passes per coin, averaged softmax:
```
Pass 1: original
Pass 2: horizontal flip
Pass 3: vertical flip
Pass 4: both flips
Pass 5-8: four 85% corner crops
```
Same coin, 8 orientations → averaged prediction reduces noise → +0.78% gain.

**Saved artefacts:**
```
models/best_model.pth          ← V3, epoch 52 — THE REAL MODEL
models/best_model_v1_80pct.pth ← MISLEADING NAME. Epoch 3, val 21.33%. NOT the 80% model. Ignore.
models/class_mapping.pth       ← {class_to_idx, idx_to_class, n:438}
```

---

### PHASE 5 — Inference Engine (February 2026) ✅

**Files:** `src/core/inference.py`, `scripts/predict.py`

`CoinInference` — production wrapper:
- `predict(image_path, tta=False)` → `{class_id, label, confidence, top5, tta_used}`

**Bug found and fixed (Bug #2 — see Section 12):**
```
"auto" string passed directly to model.to("auto") → RuntimeError
Fix: resolve before passing: device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

### PHASE 6 — Knowledge Base (February 2026) ✅ (needs upgrade)

**Files:** `src/core/knowledge_base.py` (343 lines), `scripts/build_knowledge_base.py` (296 lines)

**Scraper:** Fetches `https://www.corpus-nummorum.eu/types/{id}` at 1 req/sec. Parses `<dl>` blocks → 15 structured fields. Saves every 50 types (crash-safe). SSL verification disabled (lab env).

**KB state:**
- ChromaDB `PersistentClient` at `data/metadata/chroma_db/`
- Collection `cn_coin_types`, embedding `all-MiniLM-L6-v2` (384-dim, cosine, CPU, 22 MB)
- 434 documents (4 types returned HTTP errors → filtered)
- Document format: one 200-word text blob per coin type

**Scraper bugs found and fixed (see Section 11 for details):**
- SSL certificate errors → disabled cert verification
- Emoji/navigation chars in scraped HTML → regex cleanup
- Mint field contained "Region:" suffix → regex split
- 4/438 types returned HTTP errors → error records filtered in `build_from_metadata()`

**API:**
```python
kb.search(query, n, where)   # cosine similarity: 1.0 - distance
kb.search_by_id(type_id)     # exact ID lookup via ChromaDB .get()
kb.build_from_metadata(path) # batch upsert (50/batch)
get_knowledge_base()         # module-level singleton
```

**BEFORE the upgrade — what the KB is today:**
- 438 coin types only (the CNN training subset — 4.5% of the full CN domain)
- Each coin stored as ONE 200-word text blob: all fields concatenated into a paragraph
- ChromaDB encodes that blob into one 384-dim vector
- When the Historian needs facts it calls `search_by_id("1015")` → gets the blob → sends the ENTIRE blob to Gemini
- Gemini sees an unstructured paragraph and must guess which field is which
- If the CNN predicted a coin type that's outside the 438 (or a truly unknown coin), the KB returns nothing

**AFTER the upgrade — what the KB will become:**
- All 9,716 CN types (one-time scrape, ~2.7 hours) — the KB now covers the FULL domain
- Each coin split into 5 focused chunks: `identity`, `obverse`, `reverse`, `material`, `context`
- 9,716 × 5 = 48,580 vectors in ChromaDB (~180 MB on disk)
- Hybrid search: BM25 keyword search + vector semantic search, merged with RRF formula
- Historian injects each chunk as a labeled `[CONTEXT N]` block → Gemini can only state facts from the context → zero hallucination
- Investigator searches ALL 9,716 types (no filter) → unknown coins now surface real matches
- `in_training_set: bool` tag on every record → easy to see if a match is CNN-known or KB-only

**Known gaps (to fix in enterprise upgrade):**
1. Only 438 types → should be ALL 9,716
2. One blob per coin → should be 5 semantic chunks
3. Vector-only search → no BM25, no hybrid, no RRF
4. `in_training_set` tag MISSING from `build_metadata_dict()`

---

### PHASE 7 — All 5 Agents (February 2026) ✅ WORKING

End-to-end test passing: type 1015, 91.1% confidence, historian route, PDF generated.

**The 5 agents and what each one does:**

| File | Role | Input | Output |
|------|------|-------|--------|
| `gatekeeper.py` | Orchestrator — runs the LangGraph state machine, routes by confidence | image path | final state dict |
| `historian.py` | Pulls KB facts + calls Gemini to write historical narrative | CNN prediction dict | narrative, mint, date, material... |
| `investigator.py` | For unknown coins — sends image to Gemini Vision, extracts visual attributes, cross-refs KB | image path | visual description, detected features, KB matches |
| `validator.py` | OpenCV forensic check — detects gold/silver/bronze from HSV pixel analysis, compares to expected material | image path + CNN prediction | match/mismatch, warning |
| `synthesis.py` | Assembles ALL agent outputs into one structured plain-text summary and a professional PDF | full CoinState dict | PDF file + text report |

See **Section 6 (Layer-by-Layer)** for exact per-agent code details.

---

### PHASE 8 — Bug Fixing Marathon (February 2026) ✅

All bugs fully documented in **Section 11 (Known Bugs)**.

---

### PHASE 9 — End-to-End Test (February 2026) ✅

**File:** `scripts/test_pipeline.py`

```
Input:    data/processed/1015/any_coin.jpg
CNN:      type 1015, 91.1% confidence
Route:    historian
KB:       found — Maroneia, Thrace, c.365-330 BC, silver drachm
LLM:      narrative generated (GITHUB_TOKEN) or fallback (no key)
PDF:      written to reports/
Exit:     0
```

Latest clean commit: `113514b` — Greek transliteration + footer band fix.
Persistent context file committed: `ca96c10`.

---

### CURRENT STATUS — Enterprise Layer 3 Upgrade (active)

All 6 key files fully audited. Plan approved. NO code changed yet.
See **Section 7 (Enterprise Upgrade Plan)** for the 8-step build order.

---

## 3. ARCHITECTURE — TWO-STAGE HYBRID PIPELINE

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
    image_path         : str
    use_tta            : bool
    cnn_prediction     : dict   # {class_id, label, confidence, top5}
    route_taken        : Literal["historian", "validator", "investigator"]
    historian_result   : dict
    validator_result   : dict
    investigator_result: dict
    report             : str    # final Markdown
    pdf_path           : Optional[str]
```

---

### Why CNN AND KB — They Cannot Replace Each Other

This question will come from the encadrant: *"If you scrape 9,716 types from the KB, why did you train a CNN? Why not just use the scraper?"*

**Answer: The CNN and KB solve completely different problems. Neither can do the other's job.**

| | CNN | Knowledge Base (KB) |
|---|---|---|
| **Input** | Raw pixel photograph | Text query or coin type_id |
| **Output** | "This looks like type 1015" (visual identity) | "Type 1015 is a silver drachm from Maroneia, 365–330 BC" (factual knowledge) |
| **What it learns** | Visual patterns — portrait style, iconography, metal texture, patina, geometric proportions | Nothing — it is a lookup table with semantic search on top |
| **Can it analyse a photo?** | Yes — that is its entire purpose | No — it has no vision, only text |
| **Can it explain history?** | No — it outputs a class index (e.g. `438`) | Yes — it stores the full structured record |
| **Generalises to unseen coins?** | Yes — extracts 1536-dim features, returns most visually similar known type | No — if type_id is not in the KB, it returns nothing |

**The scraping is data collection. The CNN is pattern recognition. The KB is the encyclopedia. RAG is the retrieval engine.**

A library full of books does not replace a librarian who can look at an artefact and say "this belongs on shelf 7." A librarian who knows which shelf it is on cannot write the book's contents from scratch.

---

### What Happens With Unknown Coins — 3 Cases

#### Case A — CNN trained on it, KB has it (438 CNN classes)
```
CNN: "type 1015, 91% confidence"
Route: Historian
KB: returns type 1015 record (mint, date, material, obverse, reverse, persons...)
RAG: retrieves 5 focused chunks → injects as [CONTEXT N] blocks → Gemini writes grounded narrative
Report: full professional PDF with historical analysis, forensic check, and visual attributes
```

#### Case B — CNN never trained on it, but KB has it (types 439–9,716 after upgrade)
```
CNN: misidentifies it as the closest visual match, but confidence is low (< 40%)
Route: Investigator (low confidence triggers VLM path)
Gemini Vision: analyses the photo → "silver coin, helmeted portrait right, legend ΑΝΤΙΟΧΟΥ, eagle reverse"
KB search (full 9,716 corpus): finds CN type 7432 — Seleucid tetradrachm of Antiochos I
Report says: "CNN could not classify this coin (not in training set).
             Visual analysis matched CN type 7432 from knowledge base.
             Confidence: KB match only — not CNN-verified."
```
This case transforms from a failure into a success specifically because the KB covers all 9,716 types.

#### Case C — Not in CNN, not in KB (completely unknown coin)
```
CNN: low confidence, Investigator route
Gemini Vision: still describes the coin — metal, portrait type, legend fragments, symbols
KB search: returns the 3 closest cultural neighbours (similar dynasty, region, period)
Report says: "No exact match in Corpus Nummorum. Closest neighbours: [3 types listed].
             Visual attributes detected: silver, laureate portrait, eagle reverse, possible Greek legend."
```
The system never returns "I don't know." It always returns maximum useful information. This is the *graceful degradation* principle built into the architecture.

---

### What RAG Does — The Three-Word Summary: "Makes Gemini Cite Its Sources"

**Without RAG (today):**
```
KB returns one 200-word blob → pasted into Gemini prompt → Gemini writes a paragraph
Problem: Gemini can misread fields, mix up obverse/reverse, or invent plausible-sounding facts
         because it sees unstructured text with no enforcement
```

**With RAG (after upgrade):**
```
KB returns 5 focused chunks (identity, obverse, reverse, material, context)
→ Each chunk injected as a labeled block:
    [CONTEXT 1 — Identity]  type: 1015 | denom: drachm | region: Thrace | date: 365-330 BC
    [CONTEXT 2 — Obverse]   prancing horse right | legend: MAR
    [CONTEXT 3 — Reverse]   bunch of grapes | legend: EPI ZINONOS
    [CONTEXT 4 — Material]  silver | weight: 2.44 g | mint: Maroneia
    [CONTEXT 5 — Context]   persons: Magistrate Zenon
→ Strict prompt instruction: "Using ONLY the contexts above (cite [CONTEXT N]),
   write a 3-paragraph analysis. Do not add any fact not present in the context."
→ Gemini writes a grounded, citable narrative
```

RAG = **R**etrieve the right chunks → **A**ugment the prompt with them → **G**enerate from those facts only.
The LLM is used for natural language writing quality, not for inventing historical knowledge.

---

## 4. COMPLETE TECHNOLOGY STACK

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

## 5. CNN MODEL — FULL DETAILS

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

## 6. LAYER-BY-LAYER STATUS

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

## 7. THE ENTERPRISE UPGRADE PLAN (CURRENT ACTIVE WORK)

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

## 8. KEY ENGINEERING DECISIONS (with rationale)

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

## 9. FILE STRUCTURE (complete)

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

## 10. ENVIRONMENT AND PATHS

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

## 11. COMMIT HISTORY (significant milestones)

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

## 12. KNOWN BUGS AND RESOLVED BUGS

---

### FULLY RESOLVED BUGS ✅

#### Bug 1 — `IndentationError` in `historian.py`
- **When:** First test run of historian agent
- **Symptom:** `IndentationError: unexpected indent` at startup
- **Root cause:** A leftover `pass` / TODO stub inside a method body was deleted, leaving orphaned indentation on the next line
- **Fix:** Cleaned the method body — removed the stub, completed the method properly

---

#### Bug 2 — `RuntimeError: Invalid device string 'auto'`
- **File:** `src/agents/gatekeeper.py` → propagated from device config
- **When:** First time running the full pipeline with `device="auto"`
- **Symptom:** `RuntimeError: Invalid device string: 'auto'` from PyTorch
- **Root cause:** `"auto"` was passed directly as a device string to `CoinInference(device="auto")` → PyTorch only accepts `"cuda"`, `"cpu"`, `"cuda:0"` etc.
- **Fix:** Added device resolution before instantiation:
```python
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

#### Bug 3 — `multi_cell` horizontal position drift in `synthesis.py`
- **When:** Rendering bordered tables in the PDF
- **Symptom:** Table cells overflowed page margins; text ran off the right edge
- **Root cause:** fpdf2's `multi_cell()` does NOT preserve the X cursor. After each cell, the cursor drifted right. Subsequent `multi_cell()` calls started at the wrong X position.
- **Fix:** Added `pdf.set_x(col_x)` immediately before every `multi_cell()` call to restore correct column position.

---

#### Bug 4 — Greek characters rendered as `???` in PDF
- **File:** `src/agents/synthesis.py`
- **When:** Rendering coins with Greek legends (e.g., `ΚΕΡ`, `ΜΑΡ`, `ΣΑΤ`)
- **Symptom:** All Greek Unicode characters replaced by `?` in the PDF output
- **Root cause:** fpdf2's built-in fonts (Helvetica/Arial) use Latin-1 encoding. Python's `str.encode("latin-1")` replaces any character outside the Latin-1 range (U+0100+) with `?`. Greek alphabet is U+0370–U+03FF — entirely outside Latin-1.
- **Fix:** Added `_GREEK_MAP` dict (48 characters — full uppercase + lowercase Greek → Latin) and `_s(text)` wrapper function. **Every** text string passed to fpdf2 goes through `_s()` first:
```python
_GREEK_MAP = {"Α":"A","Β":"B","Γ":"G","Δ":"D","Ε":"E","Ζ":"Z","Η":"E",
              "Θ":"TH","Ι":"I","Κ":"K","Λ":"L","Μ":"M","Ν":"N",
              "Ξ":"X","Ο":"O","Π":"P","Ρ":"R","Σ":"S","Τ":"T",
              "Υ":"Y","Φ":"PH","Χ":"CH","Ψ":"PS","Ω":"O", ...}

def _s(text: str) -> str:
    """Transliterate Greek, then encode to latin-1 safely."""
    for gr, lat in _GREEK_MAP.items():
        text = text.replace(gr, lat)
    return text.encode("latin-1", "replace").decode("latin-1")
```

---

#### Bug 5 — Extra blank page with branding footer
- **File:** `src/agents/synthesis.py`
- **When:** Any coin analysis that fills almost a full PDF page
- **Symptom:** PDF had an extra blank page at the end with only the navy branding band
- **Root cause:** `_draw_footer_band()` was called unconditionally at the end of `to_pdf()`. If the content had already filled the previous page to capacity, fpdf2 automatically opened a new page before rendering the footer band.
- **Fix:** Removed `_draw_footer_band()` call entirely (the navy header band already carries branding). Footer was purely cosmetic and caused layout corruption.

---

#### Bug 6 — `to_pdf()` signature mismatch between Synthesis and Gatekeeper
- **Files:** `src/agents/synthesis.py` (changed), `src/agents/gatekeeper.py` (also needed update)
- **When:** PDF redesign refactor (replacing Markdown parsing with direct fpdf2 draw)
- **Symptom:** `TypeError: to_pdf() takes 2 positional arguments but 3 were given`
- **Root cause:** `synthesis.py` was refactored:
  - Old: `to_pdf(markdown_str: str, path: str)` — took the text report as input
  - New: `to_pdf(state: dict, path: str)` — takes the full CoinState dict directly
  But `gatekeeper.py` was still calling the old signature: `synthesis.to_pdf(state["report"], pdf_path)`
- **Fix:** Updated `synthesis_node` inside `gatekeeper.py`:
```python
# Old (broken):
synthesis.to_pdf(state.get("report", ""), pdf_path)
# New (correct):
synthesis.to_pdf(state, pdf_path)
```

---

#### Bugs 7-10 — Scraper bugs in `build_knowledge_base.py`

**Bug 7 — SSL certificate error:**
- **Symptom:** `ssl.SSLCertVerificationError` when fetching corpus-nummorum.eu in lab environment
- **Root cause:** Corporate/lab network intercepts HTTPS — certificate chain validation fails
- **Fix:** `ssl.create_default_context()` with `check_hostname=False, verify_mode=ssl.CERT_NONE`

**Bug 8 — Emoji/navigation garbage in scraped text:**
- **Symptom:** Metadata fields contained chars like `🔍❐✤` from website navigation icons
- **Root cause:** BeautifulSoup extracts ALL text from `<dl>` elements including icon characters
- **Fix:** `re.sub(r"[^\x00-\x7F\u00C0-\u024F\u0370-\u03FF]", "", s)` in `_clean()` function — strips non-Latin/non-Greek Unicode from all scraped text

**Bug 9 — Mint field "Region:" contamination:**
- **Symptom:** `mint = "Maroneia  Region: Thrace  Typology: Type Group X"`
- **Root cause:** HTML `<dl>` for Mint sometimes contained the Region and Typology sub-labels inline with the value
- **Fix:**
```python
mint_parts = re.split(r"\s+Region:", raw_mint)
mint = mint_parts[0].strip()
region = re.sub(r"\s+Typology.*", "", mint_parts[1]).strip() if len(mint_parts) > 1 else ""
```

**Bug 10 — 4 types returned HTTP errors:**
- **Symptom:** After scraping 438 types, only 434 documents appeared in ChromaDB
- **Root cause:** 4 type IDs in `class_mapping.pth` returned 404/500 from corpus-nummorum.eu (likely types removed from the database since the dataset was published)
- **Fix:** `build_from_metadata()` filters error records:
```python
records = [r for r in metadata if "error" not in r]
```

---

### KNOWN ISSUES (scheduled for enterprise upgrade)

| Component | Issue | Planned Fix |
|-----------|-------|-------------|
| `knowledge_base.py` | 1 blob per coin instead of 5 semantic chunks | STEP 2 — rebuild with chunking |
| `knowledge_base.py` | Only 438 types (4.5% of CN) | STEP 0+2 — scrape 9,716 + rebuild |
| `knowledge_base.py` | No `in_training_set` tag | STEP 2 — add to metadata dict |
| `historian.py` | Raw document blob to LLM — not true RAG | STEP 3 — hybrid search + [CONTEXT N] injection |
| `historian.py` | No "Related Types" section | STEP 3 — second search over full KB |
| `investigator.py` | 100% dependent on Gemini Vision | STEP 4 — local CV fallback (HSV+Sobel+ORB) |
| `investigator.py` | Only searches 434-record KB | STEP 4 — full 9,716 corpus |
| `validator.py` | Binary match/mismatch only | STEP 5 — confidence score 0-100% |
| `validator.py` | Single scale (60% crop only) | STEP 5 — multi-scale 40/60/80% |
| `gatekeeper.py` | `print()` statements only | STEP 6 — `logging.getLogger(__name__)` |
| `gatekeeper.py` | No retry on 429/503 | STEP 6 — 2× retry with backoff |
| `gatekeeper.py` | No per-node timing | STEP 6 — `time.perf_counter()` per node |

---

## 13. DATA SOURCES AND FALLBACK CHAIN

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

## 14. PERFORMANCE TARGETS

| Metric | Target | Current |
|--------|--------|---------|
| CNN Top-1 accuracy | >85% | 80.03% (TTA) — gap ~5pp |
| CNN Top-5 accuracy | >95% | Not measured yet |
| Per-class recall (rare) | >50% | Unknown |
| Full pipeline latency | <2s | Not measured (agents pending upgrade) |
| PDF generation | <500ms | Approximately met |
| KB search latency | <50ms | Sub-ms (ChromaDB) |

---

## 15. ACADEMIC CONTEXT

- **Institution**: ESPRIT School of Engineering, Manouba, Tunisia
- **Company**: YEBNI — Information & Communication, Tunisia (yebni.com)
- **Type**: PFE (Projet de Fin d'Études) — 5-month final year internship
- **Period**: February – July 2026
- **Dataset**: Corpus Nummorum v1 — 115,160 images, 9,716 types, DFG-funded
- **Problem domain**: Fine-grained archaeological numismatics with long-tail distribution
- **Key contribution**: Hybrid CNN + multi-agent RAG system with graceful degradation for OOD inputs

---

## 16. HOW TO RESUME IN ANY NEW CHAT

1. **This file is already injected.** Copilot knows everything — no re-explaining needed.
2. Say: **"Continue the enterprise upgrade — we're at STEP [N] of the build order in Section 7."**
3. Or say: **"What is the current status and what should we do next?"**
4. Always activate venv first: `& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1`
5. Iron rule still applies: **discuss plan first → wait for "go" → then build.**
6. Current next action: verify `data/raw/` folder exists on disk (needed for STEP 0 of the upgrade plan), then begin building the `--all-types` flag in `scripts/build_knowledge_base.py`.

```powershell
# Quick health check on resume
& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1
Get-ChildItem "C:\Users\Administrator\deepcoin\data\raw" -ErrorAction SilentlyContinue | Measure-Object
python scripts/test_pipeline.py
```

**Build order reminder (Section 7):**
```
STEP 0 — build_knowledge_base.py --all-types flag   (check data/raw/ first)
STEP 1 — src/core/rag_engine.py  (NEW FILE — hybrid BM25+vector+RRF)
STEP 2 — rebuild ChromaDB        (5 chunks × 9,716 = 48,580 vectors)
STEP 3 — historian.py upgrade    (true RAG + [CONTEXT N] injection)
STEP 4 — investigator.py upgrade (local CV fallback + full KB search)
STEP 5 — validator.py upgrade    (confidence score + multi-scale HSV)
STEP 6 — gatekeeper.py upgrade   (logging + retry + graceful degradation)
STEP 7 — end-to-end test         (all 3 routes)
STEP 8 — commit + push + update this file
```
