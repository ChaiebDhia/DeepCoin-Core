# Makefile — DeepCoin Core
# ========================
# Common developer tasks.  Works with GNU Make on Windows (Git Bash / Make for Windows).
# On pure PowerShell, use the equivalent venv\Scripts\<cmd> commands directly.
#
# Install Make on Windows: winget install GnuWin32.Make
#                      or: choco install make
#
# Usage:
#   make api         → start FastAPI dev server on :8000
#   make test        → run all pytest tests
#   make lint        → flake8 + black --check
#   make fmt         → black auto-format (writes files)
#   make train       → launch CNN training (requires GPU + data/processed/)
#   make predict     → run a single inference on a sample coin (demo)
#   make clean       → remove __pycache__, .pytest_cache, *.pyc
#   make pipeline    → run end-to-end 3-route smoke test

PYTHON   := venv/Scripts/python.exe
UVICORN  := venv/Scripts/uvicorn.exe
PYTEST   := venv/Scripts/pytest.exe
BLACK    := venv/Scripts/black.exe
FLAKE8   := venv/Scripts/flake8.exe

# Fallback: if the venv exe doesn't exist, try system python
ifeq ($(wildcard $(PYTHON)),)
  PYTHON  := python
  UVICORN := uvicorn
  PYTEST  := pytest
  BLACK   := black
  FLAKE8  := flake8
endif

.PHONY: api test lint fmt train predict pipeline clean help

# ── Development server ────────────────────────────────────────────────────────
api:
	@echo "Starting DeepCoin API on http://localhost:8000 ..."
	@echo "Docs: http://localhost:8000/docs"
	$(UVICORN) src.api.main:app \
	    --port 8000 \
	    --log-level info \
	    --timeout-keep-alive 600 \
	    --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	@echo "Running test suite..."
	$(PYTEST) tests/ -v --tb=short

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

# ── Linting and formatting ────────────────────────────────────────────────────
lint:
	@echo "Running flake8..."
	$(FLAKE8) src/ --max-line-length 110 --extend-ignore E203,W503
	@echo "Checking black formatting..."
	$(BLACK) src/ tests/ --check --line-length 110

fmt:
	@echo "Auto-formatting with black..."
	$(BLACK) src/ tests/ --line-length 110

# ── Model training ────────────────────────────────────────────────────────────
train:
	@echo "Starting CNN training (EfficientNet-B3, 438 classes)..."
	@echo "Expected time: ~103 min on RTX 3050 Ti"
	$(PYTHON) scripts/train.py

# ── MLflow experiment dashboard ───────────────────────────────────────────────
mlflow:
	@echo "Opening MLflow UI at http://localhost:5000 ..."
	@echo "Run 'make train' first to generate run data."
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Single-image inference demo ───────────────────────────────────────────────
predict:
	$(PYTHON) scripts/predict.py \
	    --image data/processed/1015/$(firstword $(wildcard data/processed/1015/*.jpg)) \
	    --tta

# ── End-to-end pipeline test (all 3 routes) ───────────────────────────────────
pipeline:
	@echo "Running end-to-end pipeline test (3 routes)..."
	$(PYTHON) scripts/test_pipeline.py

# ── Knowledge base rebuild ────────────────────────────────────────────────────
rebuild-kb:
	@echo "Rebuilding ChromaDB index (47,705 vectors, ~9 min)..."
	$(PYTHON) scripts/rebuild_chroma.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	@echo "Clean complete."

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "DeepCoin Core — Available make targets:"
	@echo "  make api          Start FastAPI dev server (:8000)"
	@echo "  make test         Run full pytest suite"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make lint         flake8 + black --check"
	@echo "  make fmt          Auto-format with black"
	@echo "  make train        Launch CNN training (~103 min)"
	@echo "  make predict      Single-image inference demo"
	@echo "  make pipeline     End-to-end 3-route smoke test"
	@echo "  make rebuild-kb   Rebuild ChromaDB from metadata JSON"
	@echo "  make clean        Remove __pycache__, .pyc, .pytest_cache"
	@echo ""
