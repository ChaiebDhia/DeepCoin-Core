# CLAUDE.md — DeepCoin-Core Persistent Project Memory

Last updated: 2026-05-13
Workspace root: `C:\Users\Administrator\deepcoin`
Primary repo: `ChaiebDhia/DeepCoin-Core` (branch: `main`)

---

## 0) Purpose of this file

This file is the **single-session bootstrap memory** for any AI assistant or contributor.
It captures:
- System architecture
- Project structure
- Delivery progress by layer
- Key engineering decisions
- Bug/fix ledger (critical issues only)
- Runbook commands
- What is done vs what is pending

For **full line-by-line history**, use:
- `ENGINEERING_JOURNAL.md` (master history; 199+ sections)
- `.github/copilot-instructions.md` (persistent operational context)
- `README.md` (public architecture + setup + status)

---

## 1) Mission and product definition

DeepCoin-Core is an end-to-end AI system that:
1. Classifies an ancient coin from one image (CNN).
2. Routes analysis through specialized agents by confidence.
3. Grounds historical output with RAG (hybrid BM25 + vector + RRF).
4. Generates a professional PDF report with explainability.
5. Handles unknown/out-of-distribution inputs gracefully (never empty output).

Academic context:
- PFE (Final Year Engineering Internship)
- ESPRIT × YEBNI
- Student: Dhia Chaieb

---

## 2) Current status snapshot (high-level)

### Delivery status
- Core build (Layers 0–5): **implemented**
- Layer 6 (Docker/infra): **implemented baseline, enterprise hardening pending**
- Layer 7 (CI/CD): **CI implemented, CD/deployment automation pending**
- A+++ quality upgrades: **5/6 complete** (MLflow, Grad-CAM++, Active Learning, Docker wiring, Observability dashboard complete; ArcFace track remains roadmap item)
- Advanced Features: **Implemented i18n (French/English toggle), Admin Coin Inventory Workflow, Prometheus & Alertmanager Observability Stack**

### Core metrics (latest documented)
- CNN accuracy: **80.03%** (TTA)
- Classes in CNN: **438**
- KB coverage: **9,541 / 9,716** types
- Chroma vectors: **47,705** (5 chunks per type)
- Tests: **122 discovered via pytest collect-only**
- Grad-CAM++: **19×19 target map** integration in reports/UI

### Product characteristics
- Full-stack architecture implemented and runnable locally
- Security hardening present (auth, rate limits, headers, safe loading)
- Graceful degradation across routes
- Active learning feedback loop implemented
- Enterprise operations maturity still in progress (deployment, observability, image hardening)

---

## 3) Architecture (authoritative summary)

## Stage A — Vision inference
1. Input image
2. Preprocessing: auto-crop + CLAHE (LAB L-channel) + aspect-preserving resize (299x299)
3. EfficientNet-B3 inference
4. Output: top label + confidence + top-5 + optional Grad-CAM++

## Stage B — Agentic orchestration (LangGraph)
Routing thresholds:
- `confidence > 0.85` → Historian path
- `0.40 <= confidence <= 0.85` → Validator + historian path
- `confidence < 0.40` → Investigator path

Agents:
- `gatekeeper.py`: orchestration, timing, retry, graceful degradation
- `historian.py`: grounded historical synthesis via RAG contexts
- `validator.py`: OpenCV material consistency analysis
- `investigator.py`: low-confidence visual analysis + broad KB lookup
- `synthesis.py`: normalized final narrative + PDF generation

## Stage C — Reporting and delivery
- Structured response API + persisted analysis
- PDF report in `reports/`
- Frontend visualization (including Grad-CAM card)

---

## 4) RAG architecture (production shape)

Engine:
- `src/core/rag_engine.py`

Data model:
- Corpus records split into 5 semantic chunks:
  - `identity`
  - `obverse`
  - `reverse`
  - `material`
  - `context`

Indexing:
- Vector index: ChromaDB (all-MiniLM-L6-v2 embeddings)
- Keyword index: BM25
- Merge strategy: Reciprocal Rank Fusion (RRF)

Retrieval behavior:
- Historian: usually type-constrained retrieval + context blocks
- Validator: material-focused signals
- Investigator: broad corpus retrieval for unknowns

Guarantee:
- LLM narrative is grounded using context blocks; fallback modes exist when model/provider unavailable.

---

## 5) Deep Learning pipeline summary

Model:
- EfficientNet-B3 (transfer learning)
- Custom classification head for 438 classes

Training characteristics (documented setup):
- AdamW + cosine annealing
- label smoothing
- Albumentations augmentation
- mixup
- AMP/scaler
- weighted sampling for class imbalance
- early stopping

Data facts:
- Raw: 115,160 images / 9,716 types
- Filtered training subset (>=10 images/class): 7,677 images / 438 classes

Inference quality:
- TTA enabled path
- preprocessing parity fixes applied (CLAHE alignment)
- Grad-CAM upgraded to Grad-CAM++ and finer target layer resolution

---

## 6) Backend summary (FastAPI)

Key files:
- `src/api/main.py`
- `src/api/routes/classify.py`
- `src/api/routes/history.py`
- `src/api/routes/active_learning.py`
- `src/api/auth.py`
- `src/api/limiter.py`
- `src/api/_store.py`

Capabilities:
- classify endpoint with routing + report generation
- history retrieval/detail/delete/feedback flows
- active learning candidate/mark/export endpoints
- auth controls (JWT / API key depending on route)
- rate limiting and request safety controls
- health and metrics endpoints

Hardening themes:
- safe model loading (`weights_only=True` where relevant)
- controlled file serving and cleanup
- security headers, CORS discipline, request IDs, logging

---

## 7) Frontend summary (Next.js 15)

Location:
- `frontend/`

Major user surfaces:
- Analyze flow
- History list/detail
- Explore/gallery
- Chat
- Docs/About
- Admin dashboards

Notable UX/engineering features:
- mission-control pipeline visualization
- 3-state confidence presentation
- cancel/abort flow for long analyses
- Grad-CAM display card
- feedback/active-learning hooks
- auth/session refresh improvements

---

## 8) Infrastructure + CI/CD

Containerization:
- Docker + Docker Compose setup (multi-service stack)
- Nginx reverse proxy
- PostgreSQL, Redis, MLflow, LocalStack integration in stack design

Runtime stack (compose):
- `postgres` (PostgreSQL 17)
- `redis` (Redis 7)
- `api` (FastAPI)
- `web` (Next.js 15)
- `nginx` (reverse proxy)
- `mlflow` (tracking server)
- `localstack` (S3 simulation)

Quality gates:
- Unit + integration tests
- GitHub Actions CI
- Lint/format/test workflows

Observed testing baseline:
- 122 tests passing (documented)

---

## 9) Project structure (practical map)

- `src/data_pipeline/` — preprocessing components
- `src/core/` — model factory, inference, KB/RAG core
- `src/agents/` — gatekeeper + 4 specialist agents + synthesis
- `src/api/` — FastAPI app, routes, auth, store
- `scripts/` — training, eval, KB build, pipeline tests, active learning
- `tests/` — unit/integration coverage
- `frontend/` — Next.js app
- `data/` — processed/raw/metadata stores
- `models/` — trained weights + class mapping
- `reports/` — generated PDFs and explainability artifacts
- `.github/workflows/` — CI pipelines

---

## 10) Bug and fix ledger (critical condensed memory)

This is a condensed list of major fixes repeatedly referenced during development.
For full bug chronology and root-cause narratives, see `ENGINEERING_JOURNAL.md` and `.github/copilot-instructions.md`.

1. Device string bug (`"auto"`) in inference path → explicit cuda/cpu resolution.
2. Historian/validator type-id mismatch (`class_id` vs true CN label) → switched to label string for KB lookup.
3. PDF rendering defects:
   - multi-cell x drift
   - duplicate footer page artifacts
   - Greek transliteration handling
   - signature mismatch between gatekeeper/synthesis
4. Scraper/KB ingestion issues:
   - SSL/cert handling
   - noisy scraped characters cleanup
   - mint/region parsing contamination
   - filtered error records
5. ETA reporting bug in full scrape command corrected.
6. Investigator metal parsing priority fixed (bronze/silver phrase ambiguity).
7. Investigator RRF score key mismatch fixed (`rrf_score` vs `score`).
8. Train/inference distribution mismatch fixed by adding CLAHE parity in inference.
9. `.gitignore` path anchoring bug fixed to avoid excluding `frontend/lib/`.
10. Validator patina false mismatch fixed:
    - silver saturation threshold widened
    - uncertainty + consensus override logic
11. Security hardening:
    - `weights_only=True` load safeguards
    - singleton race protection locks
    - import cleanup in hot paths
12. Frontend/runtime reliability fixes:
    - proxy/IPv6 timeout routing
    - stale state, modal flow, cancellation wiring
    - CSP and resource loading issues
13. Chat/security fixes:
    - role validation for prompt injection guard
    - SSE streaming architecture and stability
14. Auth/session reliability:
    - JWT refresh flow and retry queueing
    - route/proxy ordering corrections
15. Multiple admin/history/report path and UX consistency defects resolved.

---

## 11) Key engineering decisions (why the system looks like this)

- EfficientNet-B3 chosen for performance/VRAM balance.
- CLAHE in LAB (L-channel only) to preserve patina-relevant color semantics.
- Aspect-preserving resize/pad to avoid geometric distortion of coins.
- LangGraph selected for deterministic routing and stateful orchestration.
- Hybrid retrieval (BM25 + vector + RRF) chosen over pure vector for better lexical recall.
- 5-chunk semantic document design improves retrieval specificity and prompt grounding.
- Graceful degradation is mandatory: never return “nothing useful.”

---

## 12) Environment assumptions and commands

Environment assumptions:
- OS: Windows
- Shell: PowerShell 5.1
- Python venv: `C:\Users\Administrator\deepcoin\venv\`

Typical start:
```powershell
& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1
```

Common run targets:
```powershell
# Pipeline test
& C:\Users\Administrator\deepcoin\venv\Scripts\python.exe scripts/test_pipeline.py

# API
uvicorn src.api.main:app --port 8000 --log-level info

# Frontend (inside frontend/)
npm run dev
```

---

## 13) Where we are now / next priorities

### Shipped baseline
- End-to-end system is production-shaped across core product capabilities.
- Documentation corpus is extensive (README + Engineering Journal + persistent context).
- Authentication flows include register, verify, refresh, forgot password, and reset password endpoints.
⚠️  **CRITICAL DISCOVERY**: Password reset emails do NOT work in production.
  See `ENTERPRISE_AUDIT.md` section 5 for full analysis.
  Root cause: No error handling when RESEND_API_KEY is missing.
  Impact: Registration and password reset completely broken in production.
### Ongoing priorities (BLOCKING PRODUCTION)

**P0 — CRITICAL (BLOCKING DEPLOYMENT):**
1. **Email delivery system hardening** (see ENTERPRISE_AUDIT.md § 6)
   - Add error detection when RESEND_API_KEY is missing
   - Add email_log table for audit trail
   - Fail registration if email send fails in production
   - Implement retry logic with exponential backoff
   - Add Resend webhook handler for delivery confirmation

**P1 — HIGH (Do within 1 week):**
2. Observability dashboard implementation (Prometheus/Grafana + alert rules)
3. CI → CD promotion (build/push/deploy pipeline, rollback strategy)
4. Container hardening (resolve current Node image vulnerability findings)
5. Email rate limiting (prevent reset spam)
6. Admin email logs dashboard

**P2 — MEDIUM (Do within 2 weeks):**
7. End-to-end regression tests for auth/report/chat paths
8. Email template versioning (move from hard-coded to DB)
9. SendGrid + AWS SES fallback providers
10. Optional model improvements (ArcFace experiment track)

---

## 14) How to resume safely in a new session

1. Read this file (`CLAUDE.md`) first.
2. **CRITICAL**: Read `ENTERPRISE_AUDIT.md` section 5 (email/password reset gaps).
3. Read `README.md` for architecture and run instructions.
4. Read `ENGINEERING_JOURNAL.md` relevant sections for deep history.
5. Check `.github/copilot-instructions.md` for persistent constraints/context.
6. Validate runtime state quickly:
   - env activation
   - API health endpoint
   - one pipeline smoke test
   - **NEW**: Verify RESEND_API_KEY is set in .env (critical for auth)

---

## 15) Truth hierarchy (anti-drift)

If any conflict appears between files, trust in this order:
1. Code in `src/` + tests in `tests/`
2. `ENGINEERING_JOURNAL.md` (full chronology)
3. `.github/copilot-instructions.md` (persistent operating memory)
4. `README.md` (public-facing summary)
5. `CLAUDE.md` (this operational quick-memory file)

---

## 16) Change log for this file

- 2026-03-20: Initial creation of `CLAUDE.md` as persistent architecture/progress/bug memory summary, aligned to current project corpus.
- 2026-03-20: Enterprise hardening pass — added layer matrix, timeline, endpoint map, artifact inventory, risk register, bootstrap prompt, maintenance protocol, and reconciliation notes.
- 2026-03-20: Reality audit pass — downgraded overstated completion claims, clarified CI vs CD, added enterprise backlog and risk items for deployment/observability/security gaps.

---

## 17) Layer completion matrix (operational truth)

| Layer | Status | Key outputs | Verification signal |
|---|---|---|---|
| 0 — CNN training | Implemented | EfficientNet-B3 model + class mapping | 80.03% TTA documented |
| 1 — Inference | Implemented | TTA inference + CLAHE parity + Grad-CAM++ | End-to-end classify + heatmap |
| 2 — KB/RAG | Implemented | 9,541 types, 47,705 vectors, hybrid retrieval | Context-grounded historian output |
| 3 — Agent graph | Implemented | Gatekeeper + 4 specialist agents + synthesis | 3-route behavior implemented |
| 4 — FastAPI | Implemented | Auth, limits, metrics, history, active learning, chat stream | 122 tests discovered by pytest |
| 5 — Frontend | Implemented | Analyze/history/explore/chat/docs/about/admin/auth pages | TS/build checks present in CI |
| 6 — Docker infra | Implemented (hardening pending) | 7-service compose stack + nginx routing | Security scanning currently flags Node image vulns |
| 7 — CI/CD + tests | CI complete, CD pending | GitHub Actions CI + test matrix | CI workflow present; deploy automation not yet wired |

---

## 18) Milestone timeline (condensed)

Use this when someone asks “how did we get here?” without reading all journal sections.

1. Dataset audit and filtering (9,716 → 438 trainable classes)
2. CLAHE + resize preprocessing pipeline stabilized
3. EfficientNet-B3 training stack finalized (AMP, mixup, weighted sampling)
4. 80.03% TTA benchmark achieved on 438 classes
5. Legacy KB built (small scope), then enterprise RAG redesign started
6. Full corpus scrape expanded to 9,541 types
7. RAG engine created with BM25 + vector + RRF hybrid ranking
8. Chroma index rebuilt to 47,705 vectors (5 semantic chunks per type)
9. Historian/validator/investigator upgraded to true RAG and robust fallbacks
10. Gatekeeper hardened (logging, timing, retry, graceful degradation)
11. FastAPI production hardening (auth, rate limiting, metrics, security headers)
12. Next.js enterprise frontend (mission control UX, history/admin/chat/docs pages)
13. Grad-CAM++ upgraded to 19×19 resolution and embedded in reports/UI
14. Active learning feedback loop implemented end-to-end
15. Docker 7-service stack wired; CI stabilized at 122 tests discovered
16. Current phase: enterprise operations hardening (CD, observability, vulnerability remediation)

---

## 19) Endpoint map (high-value API surface)

Primary groups:
- Classification/reporting: classify + report serving + gradcam serving
- History: list/detail/delete/feedback workflows
- Health/ops: health + metrics
- Active learning: candidate retrieval, correction marking, export/retrain support
- Auth/session: login/register/refresh/authenticated user flows
- Chat/research: synchronous chat + SSE streaming chat

Design note:
- Keep auth-protected and public endpoints clearly separated; avoid accidental exposure through proxy rewrites.

---

## 20) Model and data artifacts inventory

Core artifacts:
- `models/best_model.pth` — primary CNN weights
- `models/class_mapping.pth` — label/index map
- `data/metadata/chroma_db_rag/` — production vector index
- `data/metadata/cn_types_metadata_full.json` (or equivalent full metadata dump)
- `reports/` — generated PDFs and Grad-CAM images

Operational constraints:
- Model and metadata are intentionally large and generally excluded from git.
- Runtime requires local availability of model + metadata mounts for full capability.

---

## 21) Risk register (what can still break)

1. **Data distribution shift risk**
    - Catalog-photo style mismatch can depress confidence despite in-training labels.
    - Mitigation: warning UX + preprocessing parity + confidence-aware messaging.

2. **LLM provider volatility risk**
    - API quota/rate/availability can vary.
    - Mitigation: provider fallback chain + local CV fallback + retry logic.

3. **Infra drift risk**
    - Docs may lag behind repo state.
    - Mitigation: truth hierarchy + periodic context refresh protocol.

4. **Operational secret hygiene risk**
    - Wrongly managed `.env` can leak tokens.
    - Mitigation: `.env.example`, gitignore discipline, runtime env injection.

5. **Performance regression risk**
    - Feature additions can degrade latency.
    - Mitigation: keep timing logs, watch route-specific runtime, preserve test gates.

6. **Container security drift risk**
    - Base images may accumulate high vulnerabilities over time.
    - Mitigation: pin patched tags/digests, add image scanning gate in CI, rebuild frequently.

7. **CI/CD scope gap risk**
    - CI validates code but no automatic deployment path exists.
    - Mitigation: add staging/prod CD workflow with migration + health-check gates.

---

## 22) Session bootstrap prompt (copy/paste)

Use this in a new AI session to restore context quickly:

"Read `CLAUDE.md`, `README.md`, `ENGINEERING_JOURNAL.md`, and `.github/copilot-instructions.md`. Summarize current architecture, shipped layers, unresolved risks, latest bug fixes, and immediate next priorities. Then propose a minimal safe plan for the requested task and update `CLAUDE.md` change log after implementation."

---

## 23) Context maintenance protocol

After every major change set:
1. Update project code/tests first.
2. Update `README.md` if external-facing behavior changed.
3. Append relevant section(s) in `ENGINEERING_JOURNAL.md` for full chronology.
4. Refresh `.github/copilot-instructions.md` persistent context.
5. Update `CLAUDE.md` snapshot sections:
    - status, metrics, timeline, risks, next priorities
6. Add one line in `CLAUDE.md` change log with date + scope.

This prevents "memory drift" between implementation and narrative documents.

---

## 24) Reconciliation notes (important)

- Some historical documents may still contain earlier "pending" statements for layers that are now complete.
- When conflicts appear, do not delete history; instead preserve chronology and rely on the truth hierarchy in Section 15.
- `CLAUDE.md` is a snapshot, not a replacement for the journal.

---

## 25) Operational smoke checklist (quick confidence run)

Run this sequence after major merges or environment changes:

1. Activate environment
    - `& C:\Users\Administrator\deepcoin\venv\Scripts\Activate.ps1`

2. API health
    - `uvicorn src.api.main:app --port 8000 --log-level info`
    - Check `GET /api/health`

3. Frontend boot
    - In `frontend/`: `npm run dev`
    - Validate upload flow and one history detail page

4. Pipeline sanity
    - `& C:\Users\Administrator\deepcoin\venv\Scripts\python.exe scripts/test_pipeline.py`

5. Tests baseline
    - `pytest tests/unit -q`
    - `pytest tests/integration -q`

6. If any mismatch appears
    - Update `ENGINEERING_JOURNAL.md` first (chronology)
    - Then refresh `.github/copilot-instructions.md` and `CLAUDE.md`

---

## 26) Enterprise backlog (authoritative next actions)

Priority P0:
- Resolve documentation drift: keep `README.md`, `.github/copilot-instructions.md`, and `CLAUDE.md` aligned after each milestone.

Priority P1 (must-have for enterprise readiness):
- Add CD pipeline (build image, scan, push, deploy, rollback hooks).
- Add observability stack wiring (Prometheus scraping + Grafana dashboards + alert thresholds).
- Fix container vulnerability findings in frontend image and enforce scan gate in CI.
- Add backup/restore runbook for PostgreSQL and report artifacts.

Priority P2 (quality/compliance):
- Add full E2E test suite covering auth (forgot/reset password), classify, report download, and chat stream.
- Add SAST/secret scanning in CI (`pip-audit`, `npm audit`, `trivy`, secret scanner).
- Add release checklist with semantic version tagging and changelog automation.

- **Latest Capabilities**: i18n localization (FR/EN) on frontend, full observability stack (Alertmanager + Grafana), and complete UML architecture diagrams.
