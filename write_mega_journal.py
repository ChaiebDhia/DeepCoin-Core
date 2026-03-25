import re

with open('ENGINEERING_JOURNAL.md', 'r', encoding='utf-8') as f:
    text = f.read()

MEGA_SECTION = """
---

## 209. The Grand Architect's Masterclass: Complete Retrospective from Section 186 to 208 (0-to-Hero)
**Date:** March 25, 2026

**Why This Section Exists**
As a Senior AI Engineer, I am taking a moment to stop writing pure code and turn around to teach you, the junior developer, exactly **what** we built, **why** we built it, and **how** it connects. If you were handed this project tomorrow, you need to understand every moving part from MLflow, to Docker, to Python environment mismatches, to Zero Trust authentication. We are pulling all the fragmented commits and fixes from Section 186 all the way to 208 into one cohesive, deeply explained masterclass.

Let's break down the entire architecture.

### Part 1: Explainability & Active Learning (Sections 186-188, 197-198)
**The Concept:** When a neural network makes a prediction, it is usually a "black box." A user uploads an ancient coin, the AI says "Roman Denarius," but the user asks, "Why?" 
To solve this, we implemented **Grad-CAM++**. 
- **What is Grad-CAM++?** It stands for Gradient-weighted Class Activation Mapping. It looks at the final convolutional layers of our PyTorch EfficientNet-B3 model and tracks where the gradients (the learning signals) flow. It generates a heatmap (red for high attention, blue for low) overlaid on the coin. It proves to the archeologist *what part of the coin the AI was looking at*.
- **The Active Learning Loop:** If the AI is wrong, the frontend has a "Mark as Wrong" button. This triggers the Active Learning pipeline. The misclassified coin is saved to a specific `data/active_learning/candidates/` directory via FastAPI. When someone triggers retraining (`scripts/active_learning.py`), the system loads these new images, merges them into the `_InMemoryDataset`, and fine-tunes the PyTorch weights. 

**MLflow Integration:** How do we track if the new training run is better than the old one? **MLflow**. It is an open-source platform for the machine learning lifecycle. Inside `scripts/train.py`, we injected `mlflow.set_experiment()`, `mlflow.log_params()`, and `mlflow.log_metrics()`. Every time the model trains, it logs the Epochs, the Loss, and the Validation Accuracy directly to a local tracking server running on port 5000. 

### Part 2: Containerization & DevOps (Sections 189-196, 199)
**What is Docker?** Imagine trying to run this application on your laptop. You need PyTorch, Python 3.11, Next.js, Node.js 22, PostgreSQL, Redis, and Nginx. Installing all of that manually is a nightmare. Docker fixes this. 
Docker creates "Containers" — lightweight, isolated, mini-computers. Each container runs exactly one thing.
- **The 7 Services (docker-compose.yml):**
  1. `api`: The FastAPI backend.
  2. `web`: The Next.js frontend.
  3. `postgres`: The relational database for auth and history.
  4. `redis`: An in-memory cache for fast lookups.
  5. `nginx`: The proxy server. It acts as a traffic cop. If you go to port 80, it routes `/api` to FastAPI, and everything else to Next.js.
  6. `mlflow`: The machine learning tracker.
  7. `localstack`: A local clone of Amazon AWS S3 for saving PDF reports reliably.

**How They Communicate:** We use a custom Docker network. Instead of hardcoding IP addresses, Docker allows containers to reach each other via their names. The Next.js app literally sends a request to `http://api:8000`.

### Part 3: Fixing P0 Bugs: The Auth & Python Environment Crisis (Sections 202-206)
**The Python 3.11 vs 3.12 Dilemma:**
We hit a paradox. Our PyTorch AI models require extreme stability with CUDA (your GPU), which works flawlessly on Python 3.11 locally. However, Python 3.11 Docker images had massive security vulnerabilities (CVEs). 
- **The Solution:** We instantiated a **Multi-Architecture Boundary**. The AI runs on Python 3.11 in the local `venv`, but the production Docker containers pull `python:3.12-slim`. We get local AI stability + production security.

**The FastAPI 204 Bug & Future Annotations:**
When you ask an API to delete data (like deleting a user or search history), the standard HTTP response is `204 No Content`. But FastAPI 0.115 crashed completely, throwing a `FastAPI 204 Assertions Error`. Why?
Because in Python, developers often import `from __future__ import annotations` at the top of a file so they can use modern type hints. Unfortunately, FastAPI's JSON serializer was breaking when it saw stringified annotations mixed with a 204 (which forbids any response body). 
- **The Fix:** We stripped `from __future__ import annotations` globally and manually forced `response_class=Response` into the FastAPI `@router.delete()` decorators, explicitly commanding FastAPI: "Do not attempt to serialize a JSON response body. Return nothing but the strict HTTP header."

**The Silent SMTP Email Bug:**
Our password reset logic failed silently in production because it relied on `Resend` (a modern email API), but if the API key was missing, it simply returned `True` (assuming success) and failed to email the user. 
- **The Fix:** We ripped it out. We migrated to a raw, robust `smtplib` connection and added strict error handling. If the email doesn't send, the database strictly rolls back the transaction. 

### Part 4: Zero Trust Observability (Section 200, Gap 5)
If we want to know *how much RAM our AI uses at 3 AM*, we need **Prometheus** (a database that pulls system metrics every 5 seconds) and **Grafana** (a dashboard to visualize those metrics in beautiful charts).

**The Zero Trust Problem:** 
Prometheus needed to scrape `/api/metrics` from our FastAPI server. But we had locked that URL behind an `X-API-Key` check (`require_api_key`), meaning Prometheus was getting a `401 Unauthorized`. 
A bad engineer would just remove the password. We are Senior Engineers. We practice **Zero Trust** — meaning nobody, not even our own internal tools, gets a free pass.
- **The Fix:** We updated `api_key.py` to gracefully accept `Authorization: Bearer <token>` alongside `X-API-Key`. We then injected that token securely into Prometheus using a mounted Docker file (`bearer_token_file`). Prometheus now proves its identity on every single scrape. Security is maintained.

### Summary
If you rebuild this project, remember the hierarchy:
1. **Frontend (Next.js)** captures intent.
2. **Backend (FastAPI)** routes the logic securely.
3. **Database (PostgreSQL + SQLite)** stores state and vector mappings.
4. **AI Core (EfficientNet-B3 + RAG)** makes the intelligent deductions.
5. **Observability (Grafana + MLflow)** proves it works under load.
"""

new_toc_entry = "\n177. [Section 209 — The Grand Architect's Masterclass: Complete Retrospective from Section 186 to 208 (0-to-Hero)](#section-209--the-grand-architects-masterclass-complete-retrospective-from-section-186-to-208-0-to-hero)"

parts = text.split('## Complete Table of Contents\n', 1)
if len(parts) > 1:
    toc_and_body = parts[1]
    toc_end_index = toc_and_body.find('\n---')
    toc_block = toc_and_body[:toc_end_index]
    
    if 'Section 209' not in toc_block:
        new_toc = toc_block + new_toc_entry
        text = text.replace(toc_block, new_toc)

# Check if section 209 already exists at the bottom to avoid duplication
if '209. The Grand Architect' not in text:
    text = text + MEGA_SECTION

with open('ENGINEERING_JOURNAL.md', 'w', encoding='utf-8') as f:
    f.write(text)
