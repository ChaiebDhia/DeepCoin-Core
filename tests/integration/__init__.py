# tests/integration/__init__.py
# Integration test package marker.
# These tests exercise the FastAPI application layer end-to-end using
# httpx AsyncClient + ASGITransport (no real network, no real GPU, no
# real PostgreSQL — all external dependencies are mocked at the boundary).
