# DeepCoin-Core — Copilot Persistent Context
# ============================================
# This file is automatically injected into every GitHub Copilot Chat session.
# It gives Copilot full knowledge of the project state, decisions, and rules.
# NEVER delete this file. Update it after every major milestone.
# Last updated: March 6, 2026 — A+++ roadmap implemented: MLflow tracking (train.py), Grad-CAM explainability (src/core/gradcam.py + inference.py + synthesis PDF). Engineering Journal sections 160-167 added. Next: Layer 6 (Docker) OR ArcFace accuracy improvement.

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

### PHASE 10 — Enterprise RAG Upgrade (February 27, 2026) ✅ COMPLETE

This phase transformed the system from a 438-type demo into a production-grade pipeline covering 97.7% more of the CN numismatic domain.

**STEP 0 — Expand the scraper to all 9,716 types**

File: `scripts/build_knowledge_base.py` — added `--all-types` flag.

The original scraper only fetched the 438 CNN training types. The KB is pure text — it has NO image constraint — so there is no reason to limit it to the CNN training set.

Scrape stats:
```
9,716 type IDs targeted
9,541 successfully scraped (175 returned HTTP errors during the run)
Output: data/metadata/cn_types_metadata_full.json  (~3.2 MB)
Speed: 1 req/sec (rate-limited to respect corpus-nummorum.eu)
Duration: ~2h 41min
Resumable: --resume flag skips already-fetched IDs
```

Bug found and fixed during this step:
- ETA formula displayed "~161h 56min" instead of "~2h 41min" (see Bug 11)

Commit: `0abf192`

---

**STEP 1 — Build `src/core/rag_engine.py`**

New file: `src/core/rag_engine.py` (674 lines)

**WHY a new file instead of extending knowledge_base.py:**
The old KB was a thin ChromaDB wrapper. The RAG engine is a different beast — it needs BM25 index management, RRF score merging, per-chunk metadata, and a `get_context_blocks()` method that returns 5 structured blocks. Mixing these concerns would make knowledge_base.py unmaintainable. The old KB is kept as a fallback reference.

**Architecture:**
```python
class RAGEngine:
    # WHAT: BM25 keyword index + ChromaDB vector index + RRF merger
    # WHY BM25: "silver" matches all silver coins exactly — vector search alone
    #           can miss exact keyword hits when the embedding moves words around
    # WHY RRF: No cross-encoder model needed for 9,716 records;
    #           score(d) = sum(1 / (60 + rank_r(d))) gives ~95% of reranker accuracy
    #           at zero latency overhead

    def search(query, n, where=None)       # hybrid BM25+vector+RRF
    def get_by_id(type_id)                 # exact type lookup
    def get_context_blocks(type_id)        # returns 5 labeled [CONTEXT N] strings
    def populate_chroma()                  # one-time build (called by rebuild_chroma.py)
    def is_chroma_built()                  # check before rebuild
    def corpus_size()                      # returns record count
```

**5 Semantic Chunks per coin type:**
```
chunk_type="identity"  → type_id, denomination, authority, region, date_range
chunk_type="obverse"   → obverse description + legend
chunk_type="reverse"   → reverse description + legend
chunk_type="material"  → material, weight, diameter, mint
chunk_type="context"   → persons, references, notes
```

Smoke test result: `9,541 records loaded, 47,705 chunks prepared, BM25 working`

Commit: `514d674`

---

**STEP 2 — Rebuild ChromaDB index**

New script: `scripts/rebuild_chroma.py`

Old DB: `data/metadata/chroma_db/` — 434 vectors (1 blob per type, 438 types)
New DB: `data/metadata/chroma_db_rag/` — 47,705 vectors (5 chunks per type, 9,541 types)

```
Vectors: 47,705 / 47,705 (100%)
Duration: 9.0 minutes
Batch size: 500 (ChromaDB upsert limit)
Speed: 11.3 ms/chunk
On-disk size: ~180 MB
```

The old DB is preserved at `chroma_db/` for fallback. The new DB at `chroma_db_rag/` is the production index.

Commit: `0ef040c` (same as STEP 3)

---

**STEP 3 — Upgrade `historian.py` to true RAG**

File: `src/agents/historian.py`

Before STEP 3, the historian did:
```
get_by_id("1015") → ONE 200-word blob → pasted into Gemini prompt → Gemini guesses field structure
```

After STEP 3, it does:
```
get_by_id("1015") → RAGEngine.get_context_blocks("1015") → 5 labeled blocks → grounded prompt:
  [CONTEXT 1 — Identity]   denomination: drachm | region: Thrace | date: c.365–330 BC
  [CONTEXT 2 — Obverse]    bunch of grapes on vine branch | legend MAR
  [CONTEXT 3 — Reverse]    legend EPI ZINONOS
  [CONTEXT 4 — Material]   silver | weight: 2.44g | mint: Maroneia
  [CONTEXT 5 — Context]    persons: Magistrate Zenon

  INSTRUCTION: Using ONLY the contexts above (cite [CONTEXT N]),
               write a 3-paragraph professional analysis.
               Do not add any fact not present in the context.
```

Result: zero hallucination on structured facts, LLM only contributes prose quality.

**Critical bug found and fixed during STEP 3 (see Bug 12):**
`class_id` is the raw softmax output index (0 to 437), NOT the CN type ID. Using `class_id` directly to call `get_by_id()` would look up year index 0 instead of type 1015. Must use `label_str` (the folder name, e.g. `"1015"`).

Before fix: researcher for coin 1015 returned type 5045 (wrong dynasty, wrong region).
After fix: returns correct Maroneia drachm.

Commit: `0ef040c`

---

**STEP 4 — Upgrade `investigator.py`**

File: `src/agents/investigator.py`

Two changes:
1. **KB cross-reference scope**: switched from `self._kb.search()` (434 types) to `self._rag.search()` (9,541 types). A low-confidence coin may match a type outside the CNN training set.
2. **OpenCV fallback when no vision LLM key is available:**

```python
def _opencv_fallback(self, image_path: str) -> tuple[str, dict]:
    """
    Pure local analysis when no vision API key is set or the model is not downloaded.

    WHAT: Runs two independent OpenCV analyses:
      1. HSV histogram on 3 crop sizes (40%/60%/80%) with majority vote
         → gold: H 15-35, S 80-255 | bronze: H 5-25, S 50-180 | silver: S < 40
      2. Sobel edge density (gradient magnitude > 30 threshold)
         → higher density = better preserved / more detail visible

    WHY: The system must NEVER return an empty analysis. If qwen3-vl:4b is not
    downloaded, a pure-Python OpenCV fallback still extracts useful attributes
    (metal estimate + condition estimate) that the RAG search can use as a query.
    """
```

Test result: `"silver/gold coin... well-preserved (Sobel 84.2)"`

Commit: `0cfe540`

---

**STEP 5 — Upgrade `validator.py`**

File: `src/agents/validator.py`

Three changes:
1. **`label_str` fix** — same issue as historian. Was using `class_id` (0-437) for KB lookup. Fixed to use `label_str` (CN type ID string).
2. **Multi-scale HSV**: 3 crop sizes (40 % / 60 % / 80 % of coin center) run independently; majority vote determines the winning metal; single-scale was unreliable on coins with worn edges.
3. **`detection_confidence` + `uncertainty`**:
   - `detection_confidence` (float 0.0–1.0): mean pixel coverage of the winning metal mask across all scales that agree
   - `uncertainty`: `"low"` (3/3 scales agree) | `"medium"` (2/3) | `"high"` (1/3 — effectively unknown)

Test for route 2 (conf=42.9%): `status=consistent  det_conf=0.73  uncertainty=low`

Commit: `3a82ba2`

---

**STEP 6 — Upgrade `gatekeeper.py`**

File: `src/agents/gatekeeper.py` (grew from 245 to 330 lines)

Four engineering improvements:

**1. Structured logging** (`logging.getLogger(__name__)` replaces all bare `print()`):
- Every node emits key metrics at INFO level: label, confidence, route decision, elapsed time, result summary
- `logging.basicConfig()` in `__init__` — no-op if caller already configured logging (FastAPI will)
- PDF errors now logged with `exc_info=True` — full stack trace captured, not lost to stdout

**2. Per-node timing** (`time.perf_counter()`):
- Each node writes its elapsed seconds into `state["node_timings"][node_name]`
- `analyze()` logs a summary: `total=20.86s  timings={'cnn': '0.54s', 'historian': '19.85s', 'synthesis': '0.47s'}`
- `node_timings: dict` added to `CoinState` TypedDict
- Now we know exactly which node is slow (historian LLM call = 14–20s)

**3. Retry with exponential backoff** (`_retry_call(fn, retries=2, backoff=1.5)`):
- Wraps historian and investigator LLM calls
- Retries on 429 (rate limit) and 503 (service unavailable)
- Detects via `exc.status_code` (openai SDK) OR string matching on the error message
- Backoff: 1.5s → 3.0s between retries
- Rationale: >95% of transient 429 errors resolve within 5 seconds

**4. Graceful per-node degradation** (`try/except Exception`):
- Each node catches all exceptions, writes `{"_error": str(exc)}` into the result dict
- The pipeline continues to synthesis — which includes the error in the report instead of crashing
- CNN node: NOT wrapped (a CNN failure means no prediction at all — surfacing the error is correct)
- All other nodes: fully protected

Bug fixed during this step (see Bug 13): PDF error was printed with bare `print()`. Now `logger.error(exc_info=True)`.

Commit: `3bc9d05`

---

**STEP 7 — End-to-end test all 3 routes**

File: `scripts/test_pipeline.py` (completely rewritten from single-route to 3-route test)

Test images discovered by scanning 40 random classes:
- Route 1 (historian, > 85%): `data/processed/1015/CN_type_1015_cn_coin_5943_p.jpg` — always type 1015, ~91%
- Route 2 (validator, 40-85%): `data/processed/21027/CN_type_21027_cn_coin_6169_p.jpg` — conf 42.9%
- Route 3 (investigator, < 40%): `data/processed/544/CN_type_544_cn_coin_2324_p.jpg` — conf 21.3%

Results:
```
Route 1 — HISTORIAN   : type=1015  conf=91.1%  time=15.4s   PDF saved   [PASS]
Route 2 — VALIDATOR   : label=12884 conf=42.9%  material=consistent  det_conf=0.73  uncertainty=low  time=9.8s   PDF saved   [PASS]
Route 3 — INVESTIGATOR: label=532   conf=21.3%  KB matches=3  llm_used=False (OpenCV fallback, qwen3-vl:4b not downloaded)  time=3.1s   PDF saved   [PASS]

RESULTS: 3/3 passed — all routes OK    EXIT: 0
```

Test exit code: 0 (clean). `sys.exit(1)` fires only if any assertion fails.

Commit: `9622f66`

---

**STEP 8 — Commit, push, update persistent context**

All changes pushed to `ChaiebDhia/DeepCoin-Core` branch `main`. Persistent context file updated.

Latest commit: `5a12ed1` — copilot-instructions.md update

---

### CURRENT STATUS — Enterprise Layer 3 Upgrade ✅ COMPLETE (all 8 steps done)

**All 8 steps done. Layer 3 is enterprise-grade and production-ready.**
All 3 routing paths tested: historian (91.1%), validator (42.9%), investigator (21.3%) — 3/3 PASS.
Next: Layer 4 — FastAPI backend.
See **Section 7 (Enterprise Upgrade Plan)** for the full build order record.

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
- Best epoch: 52 / 100 | Val: 79.25% | Test: 79.08% | TTA (×5): **80.03%**
- TTA count corrected: README previously stated "8 passes" — fixed to "5 passes" to match `_TTA_TRANSFORMS` list in `inference.py`
- Augmentation pipeline: Rotate ±15°, BrightnessContrast ±20%, GaussNoise, ElasticTransform, HorizontalFlip + ImageNet normalisation
- Class imbalance (40:1) handled via `WeightedRandomSampler(weight_i = 1/count_i)`
- AMP (`GradScaler` + `autocast`) halves VRAM, ~2× faster/epoch on RTX 3050 Ti
- **Audit fix (Layer 0-3 audit, commit 8354450):** `weights_only=True` added to both `torch.load()` calls in `train.py` (`--resume` checkpoint path + final best-model test eval) — prevents arbitrary code execution via malicious pickle; matches the same fix already applied to `inference.py` in Layer 4 audit
- **Audit fix (Layer 0-3 audit, commit 8354450):** `None` guard added after `cv2.imread()` in `dataset.py` `__getitem__` — raises `ValueError` with the full file path on corrupt/empty/unsupported JPEG instead of propagating a cryptic `TypeError` mid-batch that kills the training job
- **MLflow tracking (A+++ Gap 1):** `mlflow.set_experiment()`, `mlflow.log_params()`, `mlflow.log_metrics(step=epoch)`, `mlflow.pytorch.log_model()` wired throughout `train.py`. Every run logged to `mlruns/`. View with `make mlflow` → `http://localhost:5000`.

### Layer 1 — Inference Engine ✅ COMPLETE (CLAHE fix + Grad-CAM)
Files: `src/core/inference.py`, `src/core/gradcam.py`, `scripts/predict.py`
- `CoinInference`: loads model once, runs TTA, returns structured prediction dict
- Device resolution: `"auto"` resolved to `"cuda"` or `"cpu"` before PyTorch sees it (Bug #2 fix)
- **Security patch (Layer 4 audit, commit 1b210ef):** both `torch.load()` calls now use `weights_only=True`
  - Prevents arbitrary code execution via malicious pickle in `.pth` files
  - Compatible with standard `torch.save(model.state_dict(), path)` outputs
- TTA: 5 forward passes (original + HFlip + Rotate +10° + Rotate -10° + BrightnessShift), averaged softmax → +0.78% accuracy
- **CLAHE fix (commit bc99423):** `_load_image()` now applies CLAHE before BGR→RGB conversion, exactly matching the `prep_engine.py` training pipeline
  - Skipping CLAHE caused train/inference distribution mismatch → 5–15% confidence on raw photos even for known coin types
  - Parameters: `clipLimit=2.0, tileGridSize=(8,8)` on L channel in LAB colourspace (identical to training)
- **Grad-CAM (A+++ Gap 2):** `predict(gradcam=True)` generates a heatmap PNG via `src/core/gradcam.py::generate_gradcam()`. `gatekeeper.py` passes `gradcam=True` on every call. PNG path stored in `cnn_prediction["gradcam_path"]` and embedded in the PDF by `synthesis.to_pdf()`.

### Layer 2 — Knowledge Base ✅ UPGRADED TO FULL CORPUS
Files: `src/core/knowledge_base.py` (legacy fallback), `src/core/rag_engine.py` (production), `scripts/build_knowledge_base.py`, `scripts/rebuild_chroma.py`

**Final state:**
- `knowledge_base.py`: original 434-vector DB kept at `data/metadata/chroma_db/` — used as fallback only
- `rag_engine.py`: 47,705-vector DB at `data/metadata/chroma_db_rag/` — 9,541 types × 5 semantic chunks
- Hybrid search: BM25Okapi keyword index + ChromaDB vector similarity + RRF (k=60) merge
- `get_context_blocks(type_id)` → returns 5 labeled `[CONTEXT N]` strings ready to inject into LLM prompt
- `in_training_set: bool` tag on every chunk record
- Rebuild script: `scripts/rebuild_chroma.py` (wipe-safe, 9.0 min, 11.3 ms/chunk)
- **Audit fix (Layer 0-3 audit, commit 8354450):** `get_rag_engine()` singleton now thread-safe via double-checked locking (`threading.Lock()`) — without the lock, two simultaneous FastAPI requests on a cold server could both enter `if _engine_instance is None` and build two BM25 indexes in parallel, causing OOM; pattern mirrors `get_gatekeeper()` in `gatekeeper.py`

### Layer 3 — Agent System ✅ ENTERPRISE UPGRADE COMPLETE
All 5 agents fully upgraded. All 3 routes tested and passing.

**Latest commit**: `9622f66` — STEP 7 test_pipeline.py, 3/3 routes PASS

#### Agent Files and Current State:

**`src/agents/gatekeeper.py`** (330 lines) — LangGraph orchestrator ✅ UPGRADED
- `CoinState` TypedDict: 11 fields (added `node_timings: dict`)
- `Gatekeeper.__init__()`: `logging.basicConfig()` call + `logger.info()` on init and ready events
- `Gatekeeper.analyze()`: logs entry + pipeline-complete summary with per-node timing dict
- `_build_graph()`: exposes `_retry_call(fn, retries=2, backoff=1.5)` — 1.5s/3.0s backoff on 429/503
- Each node: `time.perf_counter()` start/stop, `logger.info()` with key metrics, `try/except` graceful degradation
- `synthesis_node`: PDF error logged with `exc_info=True` instead of bare `print()`
- Routing thresholds: > 0.85 → historian | 0.40–0.85 → validator | < 0.40 → investigator

**`src/agents/historian.py`** — RAG + LLM narrative ✅ UPGRADED
- `_get_llm(capability)`: separate `_text_client`/`_vision_client` caches — 4-provider chain (GitHub/Google/Ollama/fallback)
- `research(cnn_prediction)→dict`: `label_str` lookup (NOT raw class_id), `get_by_id()` → hybrid RAG search, `get_context_blocks()` for [CONTEXT 1-5] injection
- `_generate_narrative()`: grounded prompt — Gemini cites [CONTEXT N], `max_tokens=800`
- `_fallback_narrative()`: field concatenation when no LLM key
- **Audit fix (8354450):** `_llm_lock = threading.Lock()` added as module global; `_get_llm()` uses double-checked locking so concurrent FastAPI requests cannot race on `_text_client` / `_vision_client` assignment

**`src/agents/investigator.py`** — VLM visual agent ✅ UPGRADED
- KB cross-reference via `self._rag.search()` — all 9,541 types (not just 438)
- `_opencv_fallback()`: HSV histogram (3 crop sizes) → metal detection; Sobel edge density → condition estimate; used when no vision LLM available
- `qwen3-vl:4b` not downloaded yet → fallback always active; pull later: `ollama pull qwen3-vl:4b`

**`src/agents/validator.py`** — OpenCV forensic material validator ✅ UPGRADED + BUG 18 FIXED
- Multi-scale HSV: 40%/60%/80% crop sizes, majority vote on gold/bronze/silver detection
- `detection_confidence` (float 0-1): mean pixel coverage of winning metal across agreeing scales
- `uncertainty` flag: low (3/3 agree) / medium (2/3) / high (1/3)
- `label_str` lookup fix (same as historian — NOT raw class_id)
- **Bug 18 fix:** `silver_mask` S_max raised 40→70 — captures ancient Ag₂S patina (S=55-80) without overlapping true bronze (S>70)
- **Bug 18 fix:** consensus override — `detected=="bronze" AND expected=="silver" AND cnn_conf>=0.40` → `status="uncertain"` + patina-ambiguity warning; CNN+KB evidence overrides HSV alone
- **Audit fix (8354450):** `from collections import Counter` moved from inside `_detect_material()` hot-path to module top-level imports — avoids Python re-importing the stdlib module on every validator call

**`src/agents/synthesis.py`** — Professional PDF generator ✅ COMPLETE
- `synthesize(state)→str`: clean plain-text summary
- `to_pdf(state, output_path)`: ALL direct fpdf2 draw — NO Markdown parsing
- Navy header band, bordered tables with alternating shading, blue section rule lines
- `_GREEK_MAP`: dict-based Greek→Latin transliteration (Κ→K, Ε→E, Ρ→R, etc.)
- Bug fixed: Greek `???` chars replaced via transliteration map
- Bug fixed: duplicate footer band removed (header already carries branding)
- Signature change from `to_pdf(markdown_str, path)` → `to_pdf(state_dict, path)`
- **Audit fix (8354450):** `import re as _re` removed from inside `_enrich_label()` and `_basename()` — both now use module-level `re`; eliminates a redundant stdlib import executed on every PDF render call

### Layer 4 — FastAPI Backend ✅ ENTERPRISE-HARDENED (latest: 6dad389)
- `src/api/main.py`: lifespan, real CORS (`ALLOWED_ORIGINS` env), real health endpoint (5 component checks → 503 if degraded), `_cleanup_old_files(max_age_hours=24)` at startup, `/api/metrics` (auth-gated, Prometheus text), GZipMiddleware(500B), X-Request-ID ASGI middleware, ENV-gated docs (`docs_url=None` in production)
- `src/api/logging_config.py` (NEW): `configure_logging()` — `LOG_FORMAT=json|text`, `LOG_LEVEL`, silences httpx/chromadb/sentence_transformers; `python-json-logger>=3.0.0` dependency
- `src/api/schemas.py`: Pydantic v2 — `ClassifyResponse`, `CnnResult`, `Top5Item`, `HistoryListResponse`, `HistorySummary`
- `src/api/_store.py`: **SQLite** (WAL mode, B-tree indexed) — `count()` O(log n) SQL COUNT, `load_page(skip,limit)` SQL LIMIT/OFFSET, replaces O(n) Python-slice pagination
- `src/api/auth.py`: `require_api_key` dependency — `hmac.compare_digest`, dev-mode passthrough when `DEEPCOIN_API_KEY` unset
- `src/api/limiter.py`: slowapi singleton, `10/minute` on `/api/classify`
- `src/api/routes/classify.py`: `Depends(require_api_key)` + `@limiter.limit("10/minute")` + `asyncio.Semaphore(1)` GPU guard + `save_path.unlink(missing_ok=True)` in `finally:` + sync `history_append`
- `src/api/routes/history.py`: `GET /api/history` (SQL paginated, newest-first) + `GET /api/history/{id}`
- `GET /api/reports/{filename}`: PDF serving with path traversal protection
- `src/__init__.py`: `__version__ = "0.4.0"` — single version source of truth
- Tests: **36/36 unit tests passing**
- Server start: `uvicorn src.api.main:app --port 8000 --log-level info`

### Layer 5 — Next.js Frontend ✅ COMPLETE v3 + Phase 3+4 UX (latest: e92c1ba)
Directory: `frontend/` (25+ files, 0 TypeScript errors)
Stack: Next.js 15 App Router, TypeScript 5, Tailwind CSS v4, CVA, TanStack Query 5, Zustand 5, Axios, Framer Motion 12, react-countup 6
Pages: `/` (classify + hero), `/history` (paginated table, URL-synced), `/history/[id]` (full detail)
Components: CoinUploader (drag-drop + TTA toggle + AbortController + Cancel button + canvas downsize), AgentPipeline (fullscreen modal, mission control, X button), AnalysisPanel (animated bars + CountUp + route colours + 3-state CNN display), HistoryTable (filter bar + delete), HealthDot
Animations: Framer Motion AnimatePresence transitions, particle-beam connectors in AgentPipeline, CountUp confidence number, CSS cubic-bezier bar growth
Error boundaries: `app/error.tsx`, `app/history/error.tsx`, `app/history/[id]/error.tsx`
URL pagination: `history/page.tsx` uses `useSearchParams` + `useRouter` wrapped in `<Suspense>`
Security: 6 HTTP headers (CSP dev/prod split, `blob:` in img-src, HSTS 2yr preload, X-Frame-Options:DENY, nosniff, Referrer-Policy, Permissions-Policy), AbortController cancellation, blob URL lifecycle management (`useMemo` + cleanup `useEffect`), reactive Zustand selectors
Design: dark navy brand palette matching PDF report; shadcn-style CVA component system

**3-way CNN display states (702e3eb):**
- `DISPLAY_CONF_THRESHOLD = 0.70`, `TTA_VOTE_THRESHOLD = 0.75` constants in AnalysisPanel
- State 1 (conf ≥ 0.70): green CountUp % — Identified
- State 2 (vote_fraction ≥ 0.75): teal "TTA Consensus" badge + "N/8 agree" — no raw %
- State 3 (below both): purple "Deep Search" badge + "Best visual match" — no raw %, no failure language
- Header badge follows same 3-state colour (green/teal/purple)
- `types/api.ts`: added `vote_fraction: number | null`, `tta_passes: number`, `temperature: number` to `CnnResult`

**Confidence anxiety elimination (451f3f2):**
- State 3 reframed: no "could not classify" language, no raw %, "Deep Search" / "Best Visual Match"
- Removed duplicate raw confidence block (technical debt that defeated State-3 suppression)
- Investigator banner: "Deep Investigation Mode" positive framing
- TTA label fixed: hardcoded "5 passes" → reads `cnn.tta_passes` (correct: 8)
- History detail page full rewrite: Quick Facts grid, action bar, `getConfidenceTier()`, metadata strip

**Phase 3 UX — CN links, delete, filter bar (0455d45):**
- Backend: `delete_by_id(record_id) -> bool` in `_store.py` (threading.Lock, DELETE SQL)
- Backend: `DELETE /api/history/{id}` endpoint (204/404 REST semantics)
- Frontend: `deleteHistoryItem(id)` in `lib/api.ts`; `useMutation` + `invalidateQueries` in history page
- HistoryTable full rewrite: filter bar (search input + route pills, client-side `useMemo`)
- Delete button: HTML5-compliant sibling of `<Link>` (not nested inside `<a>`), `Trash2` icon, `window.confirm()` guard
- Top-5 table: CN label → `<a href=corpus-nummorum.eu/types/{id} target=_blank rel=noopener noreferrer>` with `↗`

**Phase 4 UX — CTA banners, linked badges, stats, copy link (e92c1ba):**
- CTA banner in CnnSection below top-5: gradient border, ExternalLink hover micro-animation (+2px diagonal)
- Header badge (all 3 states): wrapped in `<a style="display:contents">` — zero layout impact, badge = CN link
- CN Type rows in HistorianSection + ValidatorSection: blue `<a>` + ExternalLink icon
- History detail `<h1>` CN label: direct external link to CN record
- History stats strip: SQL total count (global) + per-route breakdown + avg confidence (page window)
- Copy link button: `navigator.clipboard.writeText(window.location.href)`, 2s `Check` / "Copied!" feedback

**Cancel / Abort architecture:**
- `CoinUploader`: red Cancel button replaces Analyse during loading; `handleCancel()` calls `abortRef.current.abort()` + `reset()`
- `AgentPipeline`: X button in modal header; hover state via `useState(xHovered)` (not DOM mutation — Framer Motion conflict)
- `store.ts`: `_cancelFn: (() => void) | null` + `setCancelFn` action — sibling-communication bridge between CoinUploader and AgentPipeline

**Client-side image downsize (P16):**
- `downsizeImage(file, maxPx=1024)` — canvas JPEG 0.85, returns original if already ≤ 1024px
- Called before every `classifyCoin()` — eliminates large DSLR uploads hitting the API

**Runtime proxy fixes:**
- IPv6: `DEEPCOIN_API_URL=http://127.0.0.1:8000` in `.env.local` (Node.js `localhost` → `::1` bug)
- Turbopack timeout: `classifyApiClient` (direct to FastAPI, 180s) — bypasses ~30s proxy timeout
- CSP `connect-src http://127.0.0.1:8000` added to allow direct browser→FastAPI calls

Prod build verified: `next build` clean, 5 routes compiled (4 static + 1 dynamic), tsc: 0 errors

### Layer 6 — Docker + Infrastructure 🔲 PENDING
File: `docker-compose.yml` (skeleton exists)
7 services: FastAPI + Next.js + ChromaDB + PostgreSQL + Redis + Nginx + LocalStack

### Layer 7 — Tests + CI/CD ✅ COMPLETE (122 tests + GitHub Actions)
Files: `tests/unit/` (45 tests), `tests/integration/` (69+ tests), `.github/workflows/ci.yml`

**Test count breakdown:**
```
Preprocessing:              2 (CLAHE + resize)
Unit: security/mime:       16
Unit: audit logging:       11
Unit: API key auth:         8
Unit: SQLite store:        10
Integration: health:       11
Integration: classify:     17
Integration: history:       9
Integration: chat:         17
Integration: auth flow:    15
TOTAL:                    122 / 122 PASS
```

**conftest.py fixtures (tests/integration/conftest.py — 384 lines):**
- Module-level env setup (DATABASE_URL sqlite+aiosqlite, SECRET_KEY, ENV=test)
- `_reset_rate_limiter` (autouse): clears slowapi MemoryStorage before every test
- `_patch_gatekeeper_globally` (session, autouse): patches `src.agents.gatekeeper.Gatekeeper`
- `override_db` (function): overrides `get_db` with AsyncMock session (session.delete=AsyncMock!)
- `override_auth` / `override_guest`: mock user injection via `app.dependency_overrides`
- `client` / `auth_client`: `AsyncClient(ASGITransport(app))` + `app.state.gk=_MockGatekeeper()` set directly

**Three bugs found and fixed during Layer 7:**
1. pytest-asyncio not installed → install `pytest-asyncio>=0.24.0`
2. `app.state.gk` never set (lifespan doesn't fire reliably via ASGITransport) → set directly in fixture
3. `await db.delete()` on MagicMock → change to AsyncMock

**`.github/workflows/ci.yml`:**
- Trigger: push to main + PRs to main
- Concurrency: cancel-in-progress on same branch
- Job 1 (python-ci): Python 3.11 + 3.12 matrix | pip cache | torch CPU wheel first | flake8 + black + pytest unit + pytest integration
- Job 2 (frontend-ci): Node 22 | npm ci | `tsc --noEmit` + next lint
- Env: DATABASE_URL=sqlite+aiosqlite, ENV=test, no LLM keys

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
✅ STEP 0: Expand build_knowledge_base.py → --all-types flag (scrape 9,716)
         Code complete + smoke test passed. Full scrape running (~2h 42min).
         Output: data/metadata/cn_types_metadata_full.json
         Bug fixed: ETA formula (divided by 60 twice — now divides by 3600 for hours)
✅ STEP 1: Build src/core/rag_engine.py (NEW FILE — hybrid search foundation)
         Code complete + smoke test passed. 6,876 records, 34,380 chunks, BM25 working.
         Commit: 514d674
🔲 STEP 2: Rebuild ChromaDB index (5 chunks × 9,716 types = 48,580 vectors)
🔲 STEP 3: Upgrade historian.py (true RAG + "Related Types" section)
🔲 STEP 4: Upgrade investigator.py (full KB search + local CV fallback)
🔲 STEP 5: Upgrade validator.py (confidence scoring + multi-scale HSV)
🔲 STEP 6: Upgrade gatekeeper.py (logging + retry + graceful degradation)
🔲 STEP 7: End-to-end test all 3 routes
🔲 STEP 8: Commit and push
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
│   │   ├── knowledge_base.py     ✅ ChromaDB wrapper (438 types, kept for fallback)
│   │   └── rag_engine.py         ✅ hybrid BM25+vector+RRF search engine — 47,705 vectors
│   ├── agents/
│   │   ├── gatekeeper.py         ✅ LangGraph orchestrator — logging, timing, retry, degradation
│   │   ├── historian.py          ✅ true RAG + [CONTEXT N] citation + Ollama provider
│   │   ├── investigator.py       ✅ RAG 9,541 types + OpenCV fallback
│   │   ├── validator.py          ✅ multi-scale HSV, detection_confidence, uncertainty
│   │   └── synthesis.py          ✅ PDF generator — COMPLETE, no changes needed
│   └── api/
│       ├── main.py               ✅ FastAPI — health, metrics, cleanup, version
│       ├── auth.py               ✅ X-API-Key auth (hmac.compare_digest)
│       ├── limiter.py            ✅ slowapi singleton (10/min on classify)
│       ├── _store.py             ✅ SQLite store (WAL, B-tree, Repository Pattern)
│       ├── routes/
│       │   ├── classify.py       ✅ POST /api/classify — auth + rate-limit wired
│       │   └── history.py        ✅ GET /api/history + GET /api/history/{id}
│       └── schemas.py            ✅ Pydantic v2 — ClassifyResponse, HistoryListResponse
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
│   ├── __init__.py               ✅
│   ├── unit/
│   │   ├── __init__.py           ✅
│   │   ├── test_store.py         ✅ 10 tests — SQLite store (append, upsert, ordering)
│   │   ├── test_api_security.py  ✅ 16 tests — _sanitise_filename, _detect_mime
│   │   └── test_auth.py          ✅ 8 tests — require_api_key, hmac timing resistance
│   └── integration/              🔲 Layer 7
│
├── frontend/                     🔲 Next.js 15 (Layer 5)
├── notebooks/                    exploration
├── reports/                      PDF output directory
│
├── requirements.txt              ✅ All Python dependencies (50+ packages)
├── pyproject.toml                ✅ Build system, tool.pytest, tool.black, tool.flake8
├── Makefile                      ✅ api / test / lint / fmt / train / pipeline targets
├── .env.example                  ✅ Documented template for all environment variables
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
rank-bm25           # installed (STEP 1 — RAG engine BM25 index)
ollama (0.17.4)     # for local LLM inference (gemma3:4b downloaded, deepseek-r1:8b downloading)
slowapi (0.1.9)     # rate limiting for FastAPI
pytest (9.0.2)      # unit testing (34 tests across 3 files)
# qwen3-vl:4b      # NOT yet downloaded; pull when needed: ollama pull qwen3-vl:4b
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
| `113514b` | Greek transliteration fix + duplicate footer band removal |
| `0abf192` | STEP 0: build_knowledge_base.py --all-types, 9,541 scraped, resume bug fix |
| `514d674` | STEP 1: src/core/rag_engine.py — BM25+vector+RRF, 47,705 chunks |
| `0ef040c` | STEP 2+3: ChromaDB rebuilt 47,705 vectors; historian.py true RAG + label_str fix |
| `0cfe540` | STEP 4: investigator.py — RAG search 9,541 types + OpenCV fallback |
| `3a82ba2` | STEP 5: validator.py — multi-scale HSV, detection_confidence, uncertainty |
| `3bc9d05` | STEP 6: gatekeeper.py — logging, per-node timing, retry, graceful degradation |
| `9622f66` | STEP 7+8: test_pipeline.py 3/3 routes PASS + git push |
| `e1b3756` | Ollama-first LLM priority (historian + investigator) |
| `083937f` | _TYPO_MAP curly quote normalisation in synthesis |
| `ce417c7` | historian prompt + _clean_narrative() helper |
| `509834f` | 4 synthesis PDF fixes (CONTEXT markers, Markdown, table layout, staircase) |
| `29162b3` | 5 PDF data fixes (NLP artifact, legend prefix, UUID header, VLM Markdown, inscription scope) |
| `08b2622` | Enterprise PDF upgrade: _safe, _conf_color, _PDF class, colored pill, rrf score normalised |
| `9fd433a` | fix: metal detection priority + KB rrf_score key in investigator |
| `7e04b94` | feat: _enrich_label() — user-friendly coin names in all PDF tables |
| `a731bcd` | fix: 8 PDF quality fixes (em-dash, bad denominations, v.Chr.→BC, pipe legend, CN Reference label, Unclassified Specimen, section title) |
| `68a3c21` | fix: strip Wait-loop reasoning artifact + date differentiation in top-5 |
| `d7a0459` | fix: 3 PDF layout bugs (detected table page split, compound denom, top-5 overflow) |
| `55e1946` | fix: 3 KB data quality bugs (metal rescue, denom parens, date period suffix) |
| `0f31fbd` | fix: paragraph page-break + author attribution (header + footer) |
| `c03158b` | fix: trim header attribution to 'Prepared by: Dhia Chaieb' only |
| `16e7835` | docs: enterprise README overhaul — RAG/DL explainers, scraping story, remove Wikipedia/Nomisma, Layer 4 ✅ |
| `1b210ef` | feat: auth (X-API-Key), rate-limiting (slowapi), SQLite store (WAL), /api/metrics, 34 unit tests, pyproject.toml, Makefile, .env.example |
| `4be8e56` | docs: Engineering Journal sections 27-30 + copilot-instructions Layer 0-1 updates |
| `8354450` | fix: Layer 0-3 enterprise audit — 6 security & hardening fixes (weights_only×2, None guard, thread-safe RAG singleton, historian lock, Counter import, re import) |
| `f61113f` | feat: Layer 5 v1 — Next.js 15 frontend, 22 files, 0 TS errors |
| `c016996` | docs: Engineering Journal Section 32 |
| `b0fa6da` | feat: Layer 5 v2 — AgentPipeline mission control, Framer Motion, CountUp, error boundaries, URL pagination |
| `91613c2` | docs: Engineering Journal Section 33 — Layer 5 v2 |
| `8d6962a` | fix: Layer 5 security audit — CSP headers, AbortController, blob URL cleanup, getState() anti-pattern |
| `f2c24ec` | docs: document proxy fixes (IPv6 + Turbopack timeout) in copilot-instructions |
| `2f6c3f7` | fix: CSP connect-src for direct classify client + devIndicators:false + history 500 guard |
| `cf3be7f` | fix: health dot "healthy" status + AgentPipeline fullscreen modal + asChild warning + synthesis cycling |
| `a2e8e50` | fix: synthesis log idx guard (bail-out replaces Math.min cap) |
| `d732767` | fix: synthesis stage messages — user-friendly wording |
| `bc99423` | fix: CLAHE in inference._load_image() + investigator UX (purple badge, context banners) |
| `47d3ef9` | fix: anchor lib/ gitignore to root; track frontend/lib/*.ts (api.ts, store.ts, utils.ts) |
| `1ab77e6` | feat: Cancel button (AbortController) + CLAHE singleton in CoinInference.__init__() |
| `9ddad23` | feat: X button on AgentPipeline modal + _cancelFn bridge in Zustand store |
| `c7ef23d` | feat: P2-P9 audit — SQL COUNT, GPU semaphore, docs gate, SQL pagination, upload delete, GZip, metrics auth, noopener |
| `6dad389` | feat: P10-P16 audit — HSTS, JSON logging, RAG BM25 warning, CSP prod, sync history, X-Request-ID, canvas downsize |
| `b3e7030` | fix: strip non-ASCII chars from upload filenames (re.ASCII flag) + open(rb)+frombuffer in cv imread calls |
| `07f8ca6` | fix: probe Ollama model availability before use — prevents indefinite hang; bump classify timeout to 10 min |
| `cadfac0` | feat: auto-crop coin region before CLAHE — HoughCircles + contour fallback + centre-crop; fixes low confidence on screenshots |
| `9e09438` | feat: CNN station shows preprocessing steps (auto-crop + CLAHE) in mission control log; friendlier subtitles |
| `c8b74a7` | feat: preprocessor stage in mission control + glow fix + 8-pass TTA |
| `29098e6` | fix: temperature scaling + vote-fraction routing override for screenshot uploads |
| `702e3eb` | feat: 3-way CNN display — Identified / TTA Consensus / Deep Search; DISPLAY_CONF_THRESHOLD=0.70, TTA_VOTE_THRESHOLD=0.75 |
| `451f3f2` | ux: eliminate confidence anxiety + enrich history detail page (Quick Facts grid, getConfidenceTier, TTA label fix) |
| `0455d45` | feat: CN links in top-5, delete button + filter bar (Phase 3 UX) + DELETE /api/history/{id} backend |
| `e92c1ba` | feat: CN CTAs, linked type rows, stats strip, copy link (Phase 4 UX) |
| `ca16ead` | docs: engineering journal sections 41-45 — Phase 3+4 UX, HSV patina false-mismatch analysis |
| `44b208b` | fix: TTA threshold 0.75→0.875; State 2 label → "Consistent Match"; "CNN confidence:"; honest tooltip |
| `f76d274` | fix: auto-crop skip `min(h,w)<200` → `max(h,w)<400` — restored 97.5% confidence on processed images |
| `9befeb3` | feat(frontend): screenshot warning banner (3 heuristics) + coin-flip header + mascot speech bubble + typing dots |
| `9f8ce0d` | feat: mark-as-wrong feedback — add_feedback() store + POST /api/history/{id}/feedback + inline form |
| `349e636` | docs: Engineering Journal sections 54-56 — TTA/auto-crop fix, screenshot warning, animation, feedback ← prev |
| `80c682e` | feat: enterprise homepage redesign — 11 new components, Server Component page shell, client island pattern |
| `20b7813` | fix: ClientFetchError (SessionSync + module-level cache), /analyse page, /admin dashboard, TechStack v1, nav links |
| `64f6991` | fix: next.config.ts fallback rewrite — stops /api/auth/session from routing to FastAPI |
| `ebc3050` | fix: dev auto-activate (register+login), auth.config.ts 403 propagation, LoginForm error map, TechStack bento redesign |
| `8a820b4` | fix: RegisterForm server-driven success message + CoinUploader reset-on-mount (frozen /analyse fix) |
| `47245da` | feat: auth-guard Analyse CTA (HeroSection), NavLinks Server Component refactor (public-only links), health dot moved left |
| `391e62e` | feat: POST/GET /api/subscribers (thread-safe, idempotent), data/subscribers.json gitignored |
| `932a67f` | feat: /about + /docs pages (Server Components), /explore gallery (Client Component), admin subscriber panel, Next.js route handler proxy for X-API-Key |
| `06116a5` | feat: enterprise chat redesign v2, AI Chat CTA in AnalysisPanel, TutorialModal, admin access guide |
| `584fe2c` | fix: prompt injection guard (ChatMessage Literal roles), chat SSE streaming (POST /api/chat/stream + chatQueryStream + streaming cursor), explore date_range fix, stale comment cleanup. Layer 7 (Tests + CI/CD) is next. |
| `40118e5` | feat: JWT silent refresh (proxy route + Axios interceptor + SessionSync update() bridge + NextAuth expiry tracking), confirm-subscription UX cleanup (remove broken confirm link), Docker CVE fix (Python 3.12-slim + Node 22-alpine) |
| `b3bd803` | fix: 8 bugs — admin stats (stale backend + isError card), chat stream 404 (backend restart), /api/docs (docs_url=/api/docs), pagination always-visible (HistoryTable + admin Pagination), StatsBar useInView margin 0px, /confirm-subscription restored (Server Component SSR confirm), /contact page (mailto flow), Contact nav link |
| `19721b9` | docs: Engineering Journal sections 46-62 |
| `8eb9b3c` | feat: dashboard created_at auth chain, chat memory (non-numismatic guard bypass for history), google/scholar userQuery restore, POST /auth/resend-verification, forgot-password + reset-password + verify-email pages, LoginForm resend + forgot-pwd link |
| `3752283` | docs: Engineering Journal sections 63-67 — auth chain, chat memory, google/scholar, complete auth flow |
| `6c6a7cf` | docs: update persistent context — commits 19721b9, 8eb9b3c, 3752283 added |
| `ce6c2f9` | feat: MLflow tracking (train.py) + Grad-CAM explainability (gradcam.py, inference.py, synthesis.py, gatekeeper.py) + Engineering Journal sections 160-167 ← LATEST |

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

#### Bug 11 — ETA printed as "~161h 56min" instead of "~2h 41min"
- **File:** `scripts/build_knowledge_base.py` → `main()` ETA block
- **When:** First full `--all-types` run (9,716 types). ETA line read "~161h 56min at 1 req/sec".
- **Root cause:** The formula divided by 60 once, treating the result as hours:
  ```python
  eta_min = len(class_ids) // 60   # 9716 // 60 = 161 ← this is MINUTES, not hours
  eta_sec = len(class_ids) % 60
  print(f"~{eta_min}h {eta_sec:02d}min")  # printed 161h 56min ← WRONG
  ```
  At 1 req/sec, 9,716 requests = 9,716 **seconds** total. Correct conversion needs `// 3600` for hours.
- **Fix:**
  ```python
  _total_s  = len(class_ids)           # seconds at 1 req/sec
  eta_hours = _total_s // 3600         # 9716 // 3600 = 2
  eta_min   = (_total_s % 3600) // 60  # (9716 % 3600) // 60 = 41
  print(f"~{eta_hours}h {eta_min:02d}min at 1 req/sec")  # ~2h 41min ← CORRECT
  ```

---

#### Bug 12 — `class_id` is NOT the CN type ID
- **Files:** `src/agents/historian.py`, `src/agents/validator.py`
- **When:** STEP 3 — first run of historian with RAG lookup
- **Symptom:** For coin image from class `1015/`, historian returned historical data for type 5045 (a completely different dynasty, region, and period). The coin was Maroneia Thrace but the narrative described a different mint entirely.
- **Root cause:** `cnn_prediction["class_id"]` is the **softmax tensor index** (integer 0–437), assigned by `enumerate()` over the alphabetically sorted class folder names. It is NOT the CN type number. The folder `1015/` happens to be at index 0 in the sorted list, so `class_id=0` maps to type 1015. Using that raw integer `0` to call `get_by_id(0)` looked up a completely different type.
  ```python
  # WRONG — class_id is 0, 1, 2 ... 437 (sort order position)
  cn_type_id = cnn_prediction["class_id"]       # e.g. 0
  kb_record  = rag.get_by_id(cn_type_id)         # looks up type "0" — doesn't exist

  # CORRECT — label is the original folder name = CN type ID
  label_str  = cnn_prediction["label"]           # e.g. "1015"
  cn_type_id = int(label_str) if label_str.isdigit() else label_str
  kb_record  = rag.get_by_id(cn_type_id)         # looks up type 1015 ✔
  ```
- **Fix:** Every agent that needs the CN type ID for KB lookup must use `label_str`, not `class_id`. Applied to `historian.py` (STEP 3) and `validator.py` (STEP 5).

---

#### Bug 13 — PDF error silently lost to bare `print()`
- **File:** `src/agents/gatekeeper.py` — `synthesis_node`
- **When:** Present from the initial agent implementation; discovered and fixed in STEP 6
- **Symptom:** If PDF rendering raised an exception, the error was printed to stdout with `print(f"[Gatekeeper] PDF error: {_pdf_err}")` + `traceback.print_exc()`. In a production setting (FastAPI server, Docker), stdout may be redirected or suppressed. The error would be silently lost and the caller would only see `pdf_path: null` with no explanation.
- **Root cause:** Early implementation used `print()` as a placeholder during development. Never upgraded to the logging system.
- **Fix:**
  ```python
  # Old (broken):
  except Exception as _pdf_err:
      print(f"[Gatekeeper] PDF error: {_pdf_err}")
      import traceback; traceback.print_exc()
      pdf_path = None

  # New (correct):
  except Exception as pdf_err:
      logger.error("synthesis_node PDF error: %s", pdf_err, exc_info=True)
      pdf_path = None
  ```
  `exc_info=True` captures the full stack trace in the log record, regardless of how the process output is redirected.

---

#### Bug 14 — Metal detection priority: "silver" matched before "bronze"
- **File:** `src/agents/investigator.py` — `_parse_features()`
- **When:** Route 3 (investigator) with a bronze coin; discovered during post-enterprise-upgrade PDF review
- **Symptom:** PDF showed "Metal Color: silver" when VLM description clearly stated "The coin is bronze... rather than silver or gold"
- **Root cause:** `_parse_features()` scanned with this loop order: `("silver", "bronze", "gold", ...)`. The word "silver" appeared in the VLM text as a negation ("rather than **silver**"), but the loop matched it first and broke. Bronze was never reached.
- **Fix:** Reordered loop to `("bronze", "gold", "electrum", "billon", "copper", "silver")` — specific, less-ambiguous metals first; "silver" last as a fallback.
  ```python
  # Old (wrong order):
  for m in ("silver", "bronze", "gold", "copper", "billon", "electrum"):
  # New (specific metals first):
  for m in ("bronze", "gold", "electrum", "billon", "copper", "silver"):
  ```

---

#### Bug 15 — KB Similarity always 0% (`rrf_score` key mismatch)
- **File:** `src/agents/investigator.py` — `investigate()`, line ~116
- **When:** All Route 3 (investigator) PDF runs; discovered during post-enterprise-upgrade PDF review
- **Symptom:** Every KB match showed "0%" similarity in the PDF KNOWLEDGE BASE MATCHES table, even after the normalisation fix in `08b2622`
- **Root cause:** `rag_engine.search()` returns records with key `rrf_score` (the RRF merged score). The investigator called `hit.get("score", 0.0)` — wrong key name, always returned `0.0`. The normalisation code (`max_score = scores[0]; result / max_score`) then computed `0/0` → all zeros.
- **Fix:**
  ```python
  # Old (wrong key):
  "score": hit.get("score", 0.0),
  # New (correct key with legacy fallback):
  "score": hit.get("rrf_score", hit.get("score", 0.0)),
  ```

---

#### Bug 16 — Train/inference CLAHE mismatch → low confidence on raw photos
- **File:** `src/core/inference.py` — `_load_image()`
- **When:** Any real-world photo submitted through the frontend (not preprocessed images from `data/processed/`)
- **Symptom:** Confidence always 5–15% even for coin types with >10 training images; correctly classified coins go to investigator route instead of historian
- **Root cause:** `prep_engine.py` saves training images after applying CLAHE (clipLimit=2.0, tileGridSize=(8,8)) on the L channel in LAB colourspace. `_load_image()` loaded raw BGR → converted to RGB with NO CLAHE preprocessing. The model's convolutional filters were calibrated to CLAHE-enhanced contrast levels; raw photos have lower effective contrast → weaker activations at every layer → softmax probability mass spreads flat → top-1 confidence collapses.
- **Fix:** Insert the same CLAHE pipeline between `cv2.imread()` and `cv2.cvtColor(BGR2RGB)` in `_load_image()`:
  ```python
  lab        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
  l, a, b    = cv2.split(lab)
  clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l_eq       = clahe.apply(l)
  lab_eq     = cv2.merge((l_eq, a, b))
  img_bgr    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
  ```
- **Commit:** `bc99423`

---

#### Bug 17 — `lib/` gitignore rule silently excluded `frontend/lib/`
- **File:** `.gitignore` line 12
- **When:** Attempting to stage `frontend/lib/utils.ts` after the CLAHE fix commit
- **Symptom:** `git add frontend/lib/utils.ts` silently skipped the file; `git commit` reported "2 files changed" instead of 3; the investigator badge colour fix never reached the remote
- **Root cause:** The Python stdlib rule `lib/` (no leading slash) matches **any** directory named `lib/` anywhere in the repo tree — including `frontend/lib/`. Git's gitignore spec: patterns without a leading `/` match in all subdirectories.
- **Fix:** Changed `lib/` → `/lib/` (and `lib64/` → `/lib64/`) to anchor the pattern to the repo root only. Then staged and committed `frontend/lib/utils.ts`, `api.ts`, and `store.ts` (all three had been silently invisible to git).
- **Commit:** `47d3ef9`

---

#### Bug 18 — HSV patina/silver false mismatch → `"bronze detected / silver expected"` at 94% confidence
- **File:** `src/agents/validator.py`
- **When:** Every validator run on a patinated ancient silver coin (Route 2: conf 40–85%)
- **Symptom:** Validator returned `status="mismatch"`, `uncertainty="low"`, `det_confidence≈0.94` with warning "image appears bronze but type is recorded as silver" for genuine silver coins that simply had ancient Ag₂S sulphide patina.
- **Root cause:** Ancient silver sulphide patina (Ag₂S) turns the surface dark brownish-grey; in HSV it reads as `H≈15–25, S≈55–80`. The OLD silver mask used `S_max=40` which missed all patinated silver entirely (`S=55–80 > 40`). The bronze window covers `H:5–25, S:50–180` — patina falls squarely inside it. All 3 crop scales voted "bronze" → `vote_count=3 → uncertainty="low" → det_confidence≈0.94`. The system was confidently wrong.
- **Fix 1 — raise silver saturation ceiling `S_max 40 → 70`:**
  `silver_mask = cv2.inRange(hsv, np.array([0,0,80]), np.array([179,70,255]))` — captures lightly toned silver (S=40–70). True bronze has vivid reddish warmth (S>70) so the range does not overlap.
- **Fix 2 — consensus override for Ag₂S ambiguity:**
  If `detected=="bronze"` AND `expected=="silver"` AND `cnn_conf >= 0.40` AND `uncertainty in ("low","medium")` → set `status="uncertain"`, emit a specific patina-ambiguity warning message instead of a false mismatch. The CNN and KB together are two independent lines of evidence for silver; HSV alone should not override both.
- **Note:** consensus threshold is `cnn_conf >= 0.40` (not 0.85) because Route 2 is exactly the 40–85% band. At the lower end (40%) the CNN is uncertain too, but KB still says silver.
- **Commit:** see latest

---

### KNOWN ISSUES (all resolved)

All Layer 3 enterprise upgrade items are COMPLETE.
All PDF quality issues resolved through commits 509834f → 68a3c21.
All Layer 5 live-testing UX issues resolved through d732767 → 47d3ef9.
Phase 3+4 UX complete through e92c1ba.
**Bug 18 (HSV patina/silver) fixed — see above.**
No open engineering issues remain.
See Section 7 Build Order for what was fixed and in which commit.

---

## 14. PERFORMANCE TARGETS

| Metric | Target | Current |
|--------|--------|---------|
| CNN Top-1 accuracy | >85% | 80.03% (TTA) — gap ~5pp |
| CNN Top-5 accuracy | >95% | Not measured yet |
| Per-class recall (rare) | >50% | Unknown |
| Full pipeline latency | <20s (LLM) / <3s (no LLM) | Historian: ~15s (Ollama gemma3:4b) / Validator: ~10s / Investigator: ~3s (OpenCV only) |
| PDF generation | <500ms | ~0.40–0.47s measured |
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
2. Say: **"Start Layer 6 — Docker."** or **"What is the current status and what should we do next?"**
3. Always activate venv first: `& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1`
4. Iron rule still applies: **discuss plan first → wait for "go" → then build.**
5. Layers 0–5 are complete and enterprise-grade. Layer 6 (Docker) is next.

```powershell
# Quick health check on resume
& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1
# Verify pipeline still passes
& "C:\Users\Administrator\deepcoin\venv\Scripts\python.exe" scripts/test_pipeline.py 2>$null
# Show exit code
Write-Host "EXIT: $LASTEXITCODE"
```

**Enterprise RAG upgrade: ALL 8 STEPS COMPLETE ✅**
```
✅ STEP 0 — build_knowledge_base.py --all-types   9,541/9,716 scraped  0abf192
✅ STEP 1 — src/core/rag_engine.py                47,705 chunks        514d674
✅ STEP 2 — ChromaDB rebuilt                      47,705 vectors       0ef040c
✅ STEP 3 — historian.py true RAG                 [CONTEXT N]          0ef040c
✅ STEP 4 — investigator.py upgrade               OpenCV fallback      0cfe540
✅ STEP 5 — validator.py upgrade                  multi-scale HSV      3a82ba2
✅ STEP 6 — gatekeeper.py upgrade                 logging+retry        3bc9d05
✅ STEP 7 — end-to-end test                       3/3 PASS             9622f66
✅ STEP 8 — commit + push                         pushed to GitHub     9622f66
```

**Layer 4 FastAPI backend: COMPLETE ✅**
```
✅ src/api/main.py         lifespan, CORS, real health              7055768
✅ src/api/schemas.py      Pydantic v2 response contracts           7055768
✅ src/api/_store.py       thread-safe JSON history store           7055768
✅ src/api/routes/         classify.py + history.py                 7055768
✅ ENGINEERING_JOURNAL.md  Section 23                               4bb9878
```
**PDF quality fixes: ALL COMPLETE ✅**
```
✅ [CONTEXT N] markers stripped from PDFs                509834f
✅ Markdown symbols removed from PDF text               509834f
✅ Table staircase layout fixed                         509834f
✅ Accented/special chars (?) fixed                     083937f
✅ Ollama-first LLM priority                            e1b3756
✅ NLP artifact ("go to the NLP result") removed         29162b3
✅ "Legend Legend" double prefix fixed                   29162b3
✅ UUID removed from PDF header                         29162b3
✅ VLM description Markdown cleaned                     29162b3
✅ Inscription field scoped to INSCRIPTIONS section      29162b3
✅ Enterprise PDF: color pill, route label, page numbers 08b2622
✅ KB Similarity normalised 0-100%                      08b2622
✅ Metal detection priority fixed (bronze before silver) 9fd433a
✅ KB Similarity 0% bug fixed (rrf_score key)           9fd433a
✅ Raw CN IDs replaced with human-readable coin names    7e04b94
✅ em-dash latin-1 crash fixed                          a731bcd
✅ Bad denomination artifacts filtered (_BAD_DENOMS)     a731bcd
✅ v.Chr./n.Chr. → BC/AD German date fix                a731bcd
✅ | Legend: → / Legend: format fix                     a731bcd
✅ Stripe shows 'Corpus Nummorum · CN XXXX'             a731bcd
✅ <40% conf shows 'Unclassified Specimen'              a731bcd
✅ CN Reference moved to last row in record table       a731bcd
✅ Section title stripped of route suffix               a731bcd
✅ Wait-loop reasoning stripped (_strip_wait_loops)      68a3c21
✅ Section body capped at 350 chars (_cap_sections)      68a3c21
✅ Top-5 dates shown to differentiate same-name variants 68a3c21
✅ Detected Attributes table page-overflow guard         d7a0459
✅ Compound denom first-word filter                      d7a0459
✅ Top-5 description truncation at 48 chars              d7a0459
✅ Metal rescue from empty material via denom scan       55e1946
✅ Denom parenthetical strip (e.g. "Large (Bronze)")     55e1946
✅ Date period suffix stripped (_clean_kb_date)          55e1946
✅ Paragraph page-break on \n\n boundary                 0f31fbd
✅ Author attribution in PDF header                      0f31fbd
✅ Header trimmed to "Prepared by: Dhia Chaieb"          c03158b
✅ README enterprise overhaul (RAG/DL, no Wikipedia)     16e7835
```

**Layer 4 audit fixes complete (1b210ef). Layer 4 is now enterprise-grade.**

```
✅ weights_only=True          torch.load() security (no arbitrary pickle)
✅ X-API-Key auth             hmac.compare_digest, dev-mode passthrough
✅ Rate limiting              slowapi, 10 req/min on /api/classify
✅ SQLite store               WAL mode, B-tree, O(log n) writes
✅ File cleanup               _cleanup_old_files(24h) at startup
✅ /api/metrics               Prometheus text — 5 metrics
✅ __version__                single source in src/__init__.py
✅ pyproject.toml             build config + lint/test tool config
✅ Makefile                   developer shortcuts
✅ .env.example               self-documenting template
✅ Unit tests                 34/34 pass in 1.31s at commit 1b210ef
```

**Layer 0-3 audit fixes complete (8354450). All layers 0-4 are now enterprise-grade.**

```
✅ weights_only=True (train.py ×2)       torch.load() security in training — matches inference.py fix
✅ None guard (dataset.py)               cv2.imread() failure → ValueError with path, not TypeError
✅ Thread-safe RAGEngine singleton       get_rag_engine() double-checked locking — prevents OOM race
✅ Thread-safe LLM clients (historian)   _llm_lock guards _text_client/_vision_client assignment
✅ Counter import moved (validator)      from collections import Counter at module top, not per-call
✅ re import removed (synthesis)         module-level re reused, no per-render local import
✅ Unit tests                           36/36 pass at commit 8354450
```

**Layer 5 Next.js frontend: COMPLETE v2 (b0fa6da).**

```
✅ Next.js 15 App Router           25 files, 0 TS errors (v2)
✅ Framer Motion 12                AnimatePresence hero/result transitions
✅ AgentPipeline.tsx               Mission Control — 4 stations, particle beams, chat log
✅ CountUp confidence              counts 0 → 91.1% on result reveal
✅ Animated confidence bars        0 → real value, 700ms cubic-bezier
✅ Per-route colour coding         CNN=blue, historian=emerald, validator=amber, investigator=purple
✅ CSS particle-flow keyframes     radial-gradient dot flows along connector rails
✅ app/error.tsx                   Root error boundary
✅ app/history/error.tsx           History list error boundary
✅ app/history/[id]/error.tsx      Detail page error boundary
✅ history/page.tsx URL pagination useSearchParams + Suspense wrapper
✅ Prod build verified             16.1s, 5 routes (4 static + 1 dynamic)
✅ ENGINEERING_JOURNAL.md          Section 33 added
```

**Layer 5 security audit: COMPLETE (8d6962a).**

```
✅ next.config.ts HTTP headers     CSP (blob: allowed), X-Frame-Options:DENY, nosniff, Referrer-Policy, Permissions-Policy
✅ AbortController                 classifyCoin() accepts AbortSignal; reset + unmount cancel in-flight request
✅ blob URL lifecycle              URL.createObjectURL() in useMemo + cleanup useEffect (no leak)
✅ errorMessage selector           reactive Zustand selector replaces getState() anti-pattern
tsc: 0 errors | build: clean (5 routes)
```

**Layer 5 runtime proxy fixes: COMPLETE (live testing session).**

```
✅ IPv6 fix (.env.local)           DEEPCOIN_API_URL=http://127.0.0.1:8000 (was localhost → ::1)
✅ Turbopack timeout fix           classifyApiClient direct to FastAPI (NEXT_PUBLIC_CLASSIFY_URL)
   lib/api.ts: two clients — apiClient (proxied, fast calls) + classifyApiClient (direct, 180s)
   classify POST now: browser → http://127.0.0.1:8000/api/classify (CORS allowed)
   health/history: still proxied via /api/* rewrite (unchanged)
```

**Layer 5 live-testing UX fixes: COMPLETE (bc99423 + d732767 + a2e8e50 + cf3be7f + 2f6c3f7).**

```
✅ Health dot stuck "Connecting…"   API returns "healthy"; code checked === "ok" → fixed type + logic
✅ AgentPipeline inline → modal     Fixed fullscreen overlay with backdrop blur
✅ Synthesis log cycling            idx >= length early-return guard (Math.min still re-emitted last msg)
✅ Synthesis messages internal text User-friendly: Compiling / Assembling / Generating PDF
✅ asChild DOM prop warning         button.tsx Radix Slot fix
✅ History 500 on classify          try/except around history_append in classify route
✅ CSP connect-src                  Added http://127.0.0.1:8000 (was blocking direct classify calls)
✅ devIndicators: false             Removed Next.js dev overlay icons
```

**Layer 1 CLAHE fix: COMPLETE (bc99423).**

```
✅ inference._load_image()          CLAHE(clip=2.0, tile=8x8) on L channel in LAB before BGR→RGB
   Root cause: raw photos lacked contrast enhancement applied during training
   Symptom: 5–15% confidence on raw photos even for well-known coin types
   Fix: identical CLAHE pipeline as prep_engine.py, inserted before cvtColor(BGR2RGB)
```

**Layer 5 investigator UX: COMPLETE (bc99423).**

```
✅ Route badge                      investigator: red → purple ("Visual Investigation")
✅ CnnSection low-conf callout      conf<40%: explains 438/9716 constraint, sets expectations
✅ InvestigatorSection banner       Opens with context: "low conf is expected, KB covers 9541 types"
✅ frontend/lib/*.ts now tracked    .gitignore lib/→/lib/ anchors Python rule; api.ts+store.ts+utils.ts committed
```

**Cancel / X button abort architecture: COMPLETE (1ab77e6 + 9ddad23).**

```
✅ Cancel button (CoinUploader)     AbortController.abort() + setCancelFn(null) + reset() during isLoading
✅ X button (AgentPipeline modal)   useState(xHovered) hover state; onCancel prop wired from store._cancelFn
✅ Zustand _cancelFn bridge         sibling communication: CoinUploader → store → AgentPipeline
✅ CLAHE singleton                  cv2.createCLAHE() in __init__, reused across all TTA passes
```

**P2–P9 backend audit: COMPLETE (c7ef23d) — 36/36 tests passing.**

```
✅ P2: history count O(log n)       SELECT COUNT(*) replaces len(load_all())
✅ P3: GPU semaphore                asyncio.Semaphore(1) prevents concurrent CUDA OOM
✅ P4: docs gated by ENV            docs_url=None when ENV=production
✅ P5: SQL pagination               LIMIT/OFFSET replaces Python-slice O(n) memory
✅ P6: upload file cleanup          save_path.unlink(missing_ok=True) in finally:
✅ P7: GZip middleware              minimum_size=500 — 60–70% compression on classify responses
✅ P8: metrics auth                 Depends(require_api_key) on /api/metrics
✅ P9: noopener noreferrer          PDF link target=_blank security fix
```

**P10–P16 deep hardening: COMPLETE (6dad389) — 36/36 tests, 0 TS errors.**

```
✅ P10: HSTS header                 max-age=63072000; includeSubDomains; preload
✅ P11: JSON structured logging     src/api/logging_config.py; LOG_FORMAT=json|text; silences noisy libs
✅ P12: RAG BM25 warning            logger.warning() when ChromaDB returns nothing — no silent fallback
✅ P13: CSP unsafe-eval prod        removed from production CSP; dev-only via isDev flag
✅ P14: sync history append         removed asyncio.to_thread — SQLite WAL write < 1ms, no benefit
✅ P15: X-Request-ID middleware     every request gets UUID4; echoed in response header
✅ P16: canvas downsize             downsizeImage(file, maxPx=1024) before upload; JPEG 0.85
```

**Layer 5 Phase 3+4 UX: COMPLETE (e92c1ba) — 0 TS errors.**

```
✅ 3-way CNN display               Identified (green) / TTA Consensus (teal) / Deep Search (purple)
✅ DISPLAY_CONF_THRESHOLD=0.70     raw % hidden below 0.70 — replaces "Not Identified" framing
✅ TTA_VOTE_THRESHOLD=0.75         6/8 TTA agreement → State 2 (TTA Consensus)
✅ vote_fraction / tta_passes      new fields in types/api.ts CnnResult
✅ Confidence anxiety fix          no failure language in State 3; duplicate block removed
✅ TTA label fix                   hardcoded "5 passes" → reads cnn.tta_passes (correct: 8)
✅ History detail rewrite          Quick Facts grid, action bar, getConfidenceTier(), metadata strip
✅ delete_by_id() + DELETE endpoint  204/404 REST semantics, threading.Lock
✅ useMutation + invalidateQueries  TanStack Query idiomatic delete wiring
✅ HistoryTable filter bar          search + route pills, client-side useMemo
✅ Delete button HTML5-compliant    sibling of <Link> (not nested), Trash2, window.confirm
✅ CN links in top-5               blue <a> with ↗ and noopener noreferrer
✅ CTA banner                      below top-5, ExternalLink hover micro-animation
✅ Header badge as CN link          display:contents wrapper — zero layout impact
✅ CN Type rows linked             ExternalLink icon in Historian + Validator sections
✅ History <h1> linked              direct CN record link
✅ Stats strip                     SQL total (global) + route breakdown + avg conf (page)
✅ Copy link button                navigator.clipboard + 2s Check feedback
```

**Bug 18 — HSV patina/silver false mismatch: ✅ FIXED.**
**TTA UX overhaul: ✅ COMPLETE (44b208b).**
**Auto-crop inference bug: ✅ FIXED (f76d274).**
**Screenshot warning + mascot animation: ✅ COMPLETE (9befeb3).**
**Mark-as-wrong feedback: ✅ COMPLETE (9f8ce0d).**
**Homepage redesign + 4 post-Layer-6 bug fixes: ✅ COMPLETE (80c682e → 8a820b4).**
**Navigation overhaul + subscriber endpoint + new public pages + AI Chat + admin panels + 3 bug fixes: ✅ COMPLETE (47245da → d1a6783).**
**Enterprise chat redesign v2, AI Chat CTA, TutorialModal, admin access guide: ✅ COMPLETE (06116a5).**
**Prompt injection guard + Chat SSE streaming + explore date_range fix + stale comment cleanup: ✅ COMPLETE (pending commit).**

```
Fix:  Prompt injection — ChatMessage(BaseModel) with role: Literal["user","assistant"]
      replaces list[dict[str,str]] in ChatRequest.conversation_history;
      Pydantic v2 rejects "system" role at HTTP 422 before any LLM call.
Feat: POST /api/chat/stream — SSE endpoint; daemon-thread + asyncio.Queue pattern;
      "sources" event first, then per-token "delta" events, then "done".
Feat: chatQueryStream() in lib/api.ts — native fetch + ReadableStream; ChatStreamCallbacks.
Feat: chat/page.tsx handleSubmit rewritten for streaming; placeholder AI message + blinking
      cursor; onDelta appends tokens in-place; onDone clears cursor; abort on unmount.
Fix:  _build_item() in kb.py: record.get("date_range") or record.get("date","") — explore
      page coin cards now show correct date_range instead of empty string.
Fix:  classify.py comment: removed stale "history_append writes to SQLite" reference.
```

```
Feat: enterprise chat page redesign (424 lines) — EmptyState, MessageBubble, SourceChip, GoogleSearchCTA,
      CopyButton, TypingIndicator, animated starter questions, provider badge, stats pills, glow input
Fix:  chatQuery uses classifyApiClient (direct FastAPI, 180s) -- fixes chat proxy timeout root cause
Feat: ?q= URL param via useSearchParams + Suspense -- context injected from AnalysisPanel CTA
Feat: AnalysisPanel purple CTA -- "Continue research in DeepCoin AI" when confidence < 0.70
      and no TTA consensus -- injects CN label into /chat?q= for context pre-load
Feat: TutorialModal -- floating gold ? button (bottom-right), 6-step guided tour,
      Framer Motion AnimatePresence, progress dots, colored icons -- global in layout.tsx
Feat: admin/page.tsx Access Restricted early-return for analyst role -- psql promotion guide
```

```
Fix: TTA_VOTE_THRESHOLD 0.75 → 0.875 (7/8 passes required for "Consistent Match")
Fix: "similarity score:" → "CNN confidence:"; tooltip no longer blames image quality
Fix: auto-crop skip condition min(h,w)<200 → max(h,w)<400 (restores 97.5% on processed images)
Fix: silver S_max raised 40 → 70 in validator + Ag₂S consensus override
Feat: detectScreenshot() in CoinUploader — 3 heuristics — orange warning banner
Feat: 🪙 coin-flip Framer Motion header in AgentPipeline
Feat: mascot speech bubble (active agent + latest message + bouncing typing dots)
Feat: add_feedback() in _store.py + POST /api/history/{id}/feedback + inline form in AnalysisPanel
Feat: homepage redesign — Server Component shell, 11 new components, client island AnalyseSection
Fix: ClientFetchError — SessionSync component + module-level _authToken cache (Bug 19)
Fix: /api/auth/session → FastAPI — next.config.ts fallback rewrite (Bug 20)
Fix: login after register — dev auto-activate (register+login), 403 propagation, LoginForm error map (Bug 21)
Fix: /analyse page frozen — Zustand singleton reset on CoinUploader mount (Bug 22)
Fix: RegisterForm success message from API response, not hardcoded
Feat: /analyse dedicated page (Server Component), /admin dashboard (Client Component)
Feat: TechStack bento grid redesign — hero tile + 4 pillar cards + dataset credit banner
```

**NEXT: Layer 6 — Docker Compose** or **A+++ Gap 3 (Active Learning)** — see Section 17 below.

---

## 17. A+++ PRODUCTION ROADMAP — TODO CHECKLIST

This section tracks improvements beyond Layers 0–7. Update the checkbox and add journal section when each is completed.

| # | Gap | Files | Status |
|---|-----|-------|--------|
| ✅ 1 | MLflow experiment tracking | `scripts/train.py`, `requirements.txt`, `Makefile` | DONE — Section 166 |
| ✅ 2 | Grad-CAM explainability in PDF | `src/core/gradcam.py`, `src/core/inference.py`, `src/agents/synthesis.py`, `src/agents/gatekeeper.py` | DONE — Section 167 |
| 🔲 3 | Active Learning loop | `scripts/active_learning.py`, add `used_for_training` col to feedback table | TODO |
| 🔲 4 | Docker full wiring | `docker-compose.yml`, PostgreSQL migration, Redis cache, LocalStack PDF | TODO |
| 🔲 5 | Observability dashboard | `prometheus.yml`, Grafana service, `docker-compose.yml` | TODO |
| 🔲 6 | ArcFace loss → 85%+ accuracy | `src/core/arcface_head.py`, `scripts/train.py` | TODO |

**How to continue**: pick any 🔲 item, say "implement Gap N" and the session will have full context from Section 165–167 of the Engineering Journal.

---

```
feat: JWT silent refresh — /api/auth/refresh-access-token Route Handler (proxy to FastAPI)
      Axios response interceptor: 401 → _attemptRefresh() → retry with new token
      In-flight deduplication: _refreshQueue prevents parallel refresh race condition
      SessionSync: now exposes update() via setSessionUpdateFn() for session cookie sync
      auth.config.ts: expires_in extracted from login response, access_expires_at stored in JWT callback
      types/next-auth.d.ts: User.expires_in, JWT.access_expires_at, Session.user.access_expires_at
feat: confirm-subscription UX cleanup — EmailCapture simplified success state (removed broken confirm link)
      No double opt-in required for waitlist; single "You're on the list!" message
      /confirm-subscription/page.tsx retained as admin utility for future SMTP integration
feat: Docker CVE fix — python:3.11-slim-bookworm (1 HIGH CVE) → python:3.12-slim (clean)
                        node:20-alpine (9 HIGH CVEs) → node:22-alpine (LTS, clean)
      WHY Python 3.12: importlib.resources path traversal CVE fixed; fully compatible with PyTorch 2.6, OpenCV 4.x
      WHY Node 22: updated embedded OpenSSL 3.3.x removes all 9 HIGH CVEs; Next.js 15 supports Node 22
```

```
Fix: auth-guard Analyse CTA (HeroSection) — authed→/analyse, anon→/login?callbackUrl=/analyse
Fix: health dot moved left of auth controls in header
Fix: POST /api/subscribers — thread-safe, idempotent, JSON file, data/subscribers.json gitignored
Feat: /about page — Server Component, pipeline steps, metrics, team (932a67f)
Feat: /explore page — Client gallery, route filter pills, ConfidenceBadge, CN links, AI Chat CTA
Feat: /docs page — Server Component, 8 REST endpoints documented, cURL + Python examples
Feat: admin subscriber panel — email table, count badge, CSV export, Next.js route handler proxy
Bug 23: explore empty for anon → /api/explore public endpoint + explorePublic() → FIXED
Bug 24: PDF download returns JSON → reports TTL raised 24h→720h (30 days) → FIXED
Bug 25: feedback corrections invisible → GET /api/admin/feedback endpoint → FIXED
Feat: /api/explore (public, no auth, GDPR: strips user_id + pdf_path)
Feat: /api/admin/feedback (admin/curator, joinedload, 20/page, pages field)
Feat: /api/admin/analyses (admin/curator, route+search filter, pdf_url, user_email)
Feat: /api/chat (public, RAG search → LLM → ChatSource[], provider badge, asyncio.to_thread)
Feat: /chat page — MessageBubble, SourceChip, starter questions, typing indicator, no auth
Feat: admin AllAnalyses table — paginated, search, route filter, PDF links, user email
Feat: admin UserCorrections panel — feedback table, red badge, CN links, active learning framing
Feat: NavLinks AI Chat link — between Explore and About
Feat: 8 new TypeScript interfaces (ExploreItem, FeedbackItem, AdminAnalysisItem, ChatResponse…)
Feat: 4 new API functions (explorePublic, getAdminFeedback, getAdminAnalyses, chatQuery)
Docs: Engineering Journal sections 78-83 (+6 sections)
```

---

### PHASE 11 — Post-Layer-6 Frontend Polish (March 3, 2026) ✅ COMPLETE

After completing the Docker stack (Layer 6), 5 commits addressed homepage quality and auth flow bugs.

**Commits:** `80c682e` → `20b7813` → `64f6991` → `ebc3050` → `8a820b4`

#### Enterprise Homepage Redesign (80c682e)
Rewrote `app/page.tsx` as a pure Server Component with 11 new components:
- `HeroSection` — full-viewport landing, floating coins, shimmer headline, BADGES strip, 2 CTAs
- `StatsBar` — 5 count-up counters (80.03%, 9,716, 47,705, <20s, 46 tests) with Framer Motion `useMotionValue`
- `PipelineSteps` — 4-step explainer: Upload → CNN → Agents → PDF with tech chips and animated connectors
- `ValueCards` — 3 expert-objection-answering cards (forensic, RAG, graceful degradation)
- `TechStack` — v1 pill design (later replaced); v2 bento grid in ebc3050
- `AnalyseSection` — **client island** pattern: isolates Zustand + CoinUploader from the Server Component shell
- `EmailCapture`, `ForWhoCards`, `footer.tsx` — support sections
- `globals.css` additions: `animate-shimmer-text` keyframe, floating coin keyframes, CSS custom property brand palette

**WHY Server Component shell**: homepage components above the fold ship zero interactive JS. Only `AnalyseSection` (Zustand, CoinUploader, AgentPipeline) is a client bundle. Total above-fold JS reduced ~73%.

#### Bug 19 — ClientFetchError (20b7813)
- **Root cause:** `getSession()` (standalone next-auth function) was called inside Axios interceptors. `getSession()` fires a network request on every call. During SSR, the call fails and NextAuth internally calls `console.error` before our `try/catch` can suppress it.
- **Fix:** Module-level `_authToken` cache + `setAuthToken()` in `lib/api.ts`. New `SessionSync.tsx` component watches `useSession()` (reads from React context, zero network) and writes to the cache. Interceptors read synchronously.

#### Bug 20 — /api/auth/session forwarded to FastAPI (64f6991)
- **Root cause:** Plain-array `rewrites()` return = `afterFiles` in Next.js 15 Turbopack. Turbopack ordering bug: afterFiles fired BEFORE App Router handlers, so `/api/auth/session` was proxied to FastAPI (which returned `{"detail": "Not Found"}`).
- **Fix:** Structured `{ beforeFiles: [], afterFiles: [], fallback: [...] }`. Fallback is last in the resolution order — after all Next.js route handlers. NextAuth handles `/api/auth/**` at step 5; FastAPI handles all other `/api/**` at step 8.

#### Bug 21 — Login fails after register (ebc3050)
- **Root cause chain:** Users created as `status=pending` → no SMTP in dev → pending 403 → auth.config.ts returned `null` for all non-200 → `CredentialsSignin` error → wrong UI message
- **Fix chain:** `router.py register()` and `login()` auto-activate in dev (`ENV != "production"`). `auth.config.ts` throws `Error` for 403 → `CallbackRouteError`. `LoginForm.tsx` maps both error types to correct messages. TechStack redesigned to bento grid in the same commit.

#### Bug 22 — /analyse page frozen (8a820b4)
- **Root cause:** Zustand is a module-level singleton. `phase: "processing"` from an abandoned homepage analysis persisted across Client-Side Navigation to `/analyse`. `AgentPipeline` (fixed inset-0 z-50) rendered immediately, blocking the entire page.
- **Fix:** `reset()` called in `CoinUploader` mount effect. Also fixed `RegisterForm` to display server's `message` field (dev: "sign in immediately"; prod: "check your email").
Say: "Start Layer 6 — Docker."

**New pages as of d1a6783:**
```
app/
├── page.tsx              ← Server Component homepage (stable)
├── analyse/page.tsx      ← Dedicated analyser (stable)
├── admin/page.tsx        ← All-analyses + feedback + subscriber panels
├── history/page.tsx      ← Paginated user history (stable)
├── history/[id]/page.tsx ← Full analysis detail (stable)
├── explore/page.tsx      ← Public gallery (fixed — /api/explore, no auth)
├── chat/page.tsx         ← NEW: AI numismatic Q&A over 47,705 KB chunks
├── about/page.tsx        ← NEW: Project story (Server Component, SEO)
└── docs/page.tsx         ← NEW: REST API reference (Server Component, SEO)
```
