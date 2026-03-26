import re

with open('ENGINEERING_JOURNAL.md', 'rb') as f:
    btext = f.read()

text = btext.decode('utf-8', errors='ignore')

part9_start = re.search(r'## PART 9[\s\-]+Recap:\s*How Every File Connects to Dhia\'s Work', text)
next_h2 = re.search(r'\n##\s', text[part9_start.end():])

# Calculate byte positions
prefix = btext[:len(text[:part9_start.start()].encode('utf-8', errors='ignore'))]
if next_h2:
    suffix_byte_idx = len(text[:part9_start.end() + next_h2.start()].encode('utf-8', errors='ignore'))
    suffix = btext[suffix_byte_idx:]
else:
    suffix = b''

new_part9 = b'''## PART 9  Recap: How Every File Connects to Dhia's Work (The Full Enterprise Blueprint)

This section maps out every critical file in the DeepCoin-Core architecture, detailing *what* it is, *why* it exists, *when* it is called, and *how* it connects within our ecosystem. 

### 1. CORE MACHINE LEARNING & PIPELINES (The Vision Layer)
- **src/data_pipeline/prep_engine.py**: The Preprocessor. Converts raw Corpus Nummorum images into 299x299 training inputs using CLAHE over the LAB L-channel. *Why:* CNNs fail on uncropped, low-contrast photos. *When:* Data ingestion phase.
- **src/core/dataset.py**: The PyTorch Bridge. Bridges 7,677 JPEGs on disk to the GPU by supplying batches of tensor images. Implement lazy loading. *Why:* Loading into RAM causes OOM errors. *When:* During model training.
- **src/core/model_factory.py**: The Architecture Definition. Defines the EfficientNet-B3 backbone linked to a custom 438-class linear head. *Why:* standardizes weights loading for training and inference alike.
- **scripts/train.py**: The Training Loop (729 lines). Handles mixed-precision (AMP), Mixup data augmentation, early stopping, and class imbalance via WeightedRandomSampler. Also integrated with MLflow. *When:* Explicit training cycles.
- **src/core/inference.py**: The Production Prediction Engine. Wraps the model in a unified singleton. Takes uploaded user bytes, applies identical CLAHE logic, runs 5 TTA (Test-Time Augmentation) crops, and averages the logits. *Why:* Hardened production inference without training overhang.
- **src/core/gradcam.py**: The Explainability Extractor (GradCAM++). Hooks into eatures[-4] (19x19) of the model to compute heatmaps of where the CNN is "looking". *Why:* Overcomes Black Box AI limitations so numismatists trust the result.

### 2. THE RAG KNOWLEDGE BASE (The Memory Layer)
- **src/core/rag_engine.py**: The Semantic Brain. Hybrid Search Engine leveraging BM25 (keyword exact matches) + ChromaDB (vector semantics) merged via RRF. Holds 47,705 chunks mapping 9,541 coin types. *Why:* Softmax hallucinates unknown coins; Vector embeddings don't.
- **src/core/knowledge_base.py**: Legacy Vector wrapper. Kept solely as architectural reference.
- **scripts/rebuild_chroma.py**: The DB Builder. Computes embeddings using ll-MiniLM-L6-v2 in batches of 500. *When:* Upon system initialization or external DB wipe.

### 3. THE MULTI-AGENT STATE MACHINE (The Reasoning Layer)
- **src/agents/gatekeeper.py**: The LangGraph Orchestrator. Evaluates the cnn_prediction confidence. >0.85 -> Historian; 0.40 to 0.85 -> Validator; <0.40 -> Investigator. *Why:* Traffic controller preventing illogical LLM calls.
- **src/agents/historian.py**: The High-Confidence Grounder. If CNN is sure, it calls RAG for 5 contexts, then forces Gemini to write a structured narrative citing [CONTEXT N].
- **src/agents/validator.py**: The Skeptic. Mid-confidence trigger. Ingests raw HSV masks trying to find a mismatch against expected metals. Rejects bad CNN reads.
- **src/agents/investigator.py**: The Detective. Unconfident states. Passes the image to Gemini Vision to describe raw features, then searches the entire 9,541 RAG corpus to find the best match strictly through semantics.
- **src/agents/synthesis.py**: The Reporter. Combines all state logic from Gatekeeper and outputs a heavily branded, strictly-transliterated fpdf2 PDF report.

### 4. THE FASTAPI BACKEND (The Infrastructure Layer)
- **src/api/main.py**: The Uvicorn Entrypoint. Mounts the FastAPI application. Owns strict CORS, health checks, ASGI ID Middlewares, GZip byte compression, and docs_url environment gating.
- **src/api/_store.py**: Thread-safe WAL-enabled SQLite database wrapper. Exposes isolated ppend, load_page, delete_by_id functionality with explicit thread locking to prevent concurrency crashes. 
- **src/api/auth.py & src/api/auth/email.py**: The Security Vault. Uses hmac.compare_digest to mitigate timing attacks on API calls. Integrates vanilla smtplib configurations for Password Recovery and Registration flows to prevent Silent Resend failures.
- **src/api/schemas.py**: The Contract Enforcer (Pydantic v2). Enforces extreme validation typing. 
- **src/api/routes/active_learning.py**: The Feedback Handler. Accepts user corrections on misclassified coins and dumps them for the next PyTorch Datasets ine_tune() cycle.

### 5. THE NEXT.JS V15 FRONTEND (The Client Layer)
- **rontend/app/page.tsx**: The SSR Shell. Ships zero interactive JS above the fold. Radically drops initial DOM payload weight. Uses Client-Islands for interactive areas.
- **rontend/lib/store.ts**: Zustand Memory Matrix. Allows the CoinUploader cross-communication with AgentPipeline modal without ugly prop-drilling or Context-API renders. Holds abort controllers.
- **rontend/Dockerfile**: Alpine Node.js configuration to deploy extreme-optimized Standalone Next Builds inside ghcr.io reducing the Image from 600MB to ~150MB removing all NPM CVEs.

### 6. THE DEVOPS PLATFORM (The Reliability Layer)
- **docker-compose.yml**: Defines the 7 interconnected nodes (pi, web, edis, postgres, 
ginx, localstack, mlflow). Connects everything to shared Docker bridged networks.
- **.github/workflows/cd.yml**: The CI/CD Watchdog. Rebuilds images on GitHub PRs, runs severe Trivy security scans blocking vulnerabilities from Production, handles deployments automatically to GHCR.
'''

ultimate_sections = b'''

---

## 212. Zero-to-Hero: Hardware Prerequisite Playbook

Before a beginner attempts to reproduce this masterpiece, it is critical they configure the hardware environment explicitly:
- **Operating System**: Windows 11 strictly, running PowerShell 5.1 (requires chaining via ; instead of Linux &&).
- **Python Runtime**: DeepCoin utilizes a strict Multi-Architecture limit. The core logic executes best on local Python 3.11.8 environments (to prevent Starlette decoding issues), whereas Docker uses linux 3.12-slim strictly for mitigating Node/OpenSSL CVEs. 
- **GPU Requisites**: An Nvidia RTX 3050 Ti with precisely 4.3 GB VRAM budget. This forces the PyTorch atch_size=16 maximum. Attempts to train EfficientNet-B7 will dynamically Out-Of-Memory (CUDA OOM), hence why B3 is the engineering cap ceiling.
- **Dependencies**: The required driver is cu124 matching 	orch==2.6.0+cu124.

---

## 213. Zero-to-Hero: Day-in-the-Life Debugging Guide (Top 3 Killers)

The majority of software engineering isn't writing perfect code - it's debugging broken logic perfectly.

**1. The Silent SMTP Blackhole**
- *Symptom:* You register an account, get a "Pending Email" modal, but the database saves the user and you receive zero emails.
- *Root Cause:* Early third-party abstractions like Resend will return HTTP 200 successes even if SDK keys are invalid, resulting in dangling broken registrations.
- *Fix:* Stripped third-party abstractions. Used pure python smtplib. Throws a hard Exception when unverified, executing an instant database transaction rollback.

**2. The FastAPI "Response stringified" Crash**
- *Symptom:* API returns AssertionError: Status code 204 must not have a response body. 
- *Root Cause:* Placing rom __future__ import annotations inside route files explicitly stringified Python object payloads, confusing FastAPI 0.115's JSON execution parser on DELETE requests.
- *Fix:* Scrubbed global future annotations on REST deletion routes, enforcing explicit Response(status_code=204).

**3. The Uvicorn VRAM Starvation Death**
- *Symptom:* "Pytorch CUDA Out of Memory" when launching FastAPI via --workers 4.
- *Root Cause:* The CNN .pth weight dictionary is roughly 150MB. Instantiating CoinInference locks VRAM natively. Spawning 4 Uvicorn asynchronous workers attempts to instantiate 4 concurrent CNN graphs across the memory cache resulting in catastrophic VRAM depletion.
- *Fix:* Limited execution to --workers 1 protected by a strict syncio.Semaphore(1) in the API routes. Scaling requires external queueing (e.g. Celery / Triton) rather than native threaded replication holding GPU context.

---

## 214. Architect's Horizon: The System Topography

*(Note: As markdown natively cannot dynamically draw complex structures perfectly, here is the mental topology.)*

- **USER:** Navigates standard URL (Intercepted by NGINX Proxy) -> Proxied into Next.js.
- **CLIENT NODE:** Standalone Front-end (React/Zustand) triggers HTTP POST to /api/classify -> Intercepted again by NGINX.
- **BACKEND FASTAPI:** Uvicorn single-worker accepts bytes.
  - -> _store.py registers audit to PostgreSQL.
  - -> inference.py pulls ImageNet graph across GPU blocks -> Returns Top-5 logits and GradCAM heat.
  - -> gatekeeper.py executes conditional DAG routing.
     - *IF Confident:* Calls ChromaDB RAG Vector Store + Gemini Text LLM via HTTP.
     - *IF Weak:* Triggers CV2 multi-scaled HSV algorithms natively.
  - -> synthesis.py encodes results as a stateless pdf2 document -> Dumped into LocalStack.
- **FINAL ROUTE:** User's NextJS store updates State Machine React variables. PDF is fetched effortlessly via cached CDN streams dynamically.
'''

with open('ENGINEERING_JOURNAL.md', 'wb') as f:
    f.write(prefix + new_part9 + suffix + ultimate_sections)

print("Updated successfully")
