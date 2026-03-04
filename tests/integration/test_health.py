"""
tests/integration/test_health.py
=================================
Integration tests for the system info and health endpoints.

Endpoints covered:
    GET /             — root info endpoint
    GET /api/health   — readiness probe used by Docker HEALTHCHECK
    GET /api/metrics  — Prometheus-format metrics (auth-gated)

WHY test health endpoints:
    These are the first thing a load balancer or orchestrator calls.
    A broken health endpoint means:
      - Zero pods receive traffic (Kubernetes marks them NotReady)
      - Docker Compose restart loops indefinitely
    If the API changes the response shape, this test catches it immediately.

WHY test metrics auth:
    The /api/metrics endpoint is protected by X-API-Key in production.
    Without a test, a refactor could accidentally remove the Depends() call
    and expose internal metrics to any anonymous caller.
"""
from __future__ import annotations

import pytest


# ── GET / ─────────────────────────────────────────────────────────────────────

class TestRoot:
    """
    GET / returns the service identity document.

    WHY test the root:
        Nginx upstream health checks, monitoring probes, and SDK auto-discovery
        all call the root. It must always return 200 with consistent shape.
    """

    async def test_root_returns_200(self, client):
        """The root endpoint must respond with HTTP 200 OK."""
        response = await client.get("/")
        assert response.status_code == 200

    async def test_root_contains_service_name(self, client):
        """
        The root JSON must include the service identifier.

        WHY: The load balancer's health-check script parses this field to
        distinguish the DeepCoin API from other services on the same IP.
        """
        response = await client.get("/")
        body = response.json()
        assert body.get("service") == "DeepCoin API"

    async def test_root_contains_version(self, client):
        """
        Every response must include the API version string.

        WHY: Frontend and SDK callers use this to detect breaking changes
        and can display a banner if the backend version is lower than expected.
        """
        response = await client.get("/")
        body = response.json()
        assert "version" in body
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0


# ── GET /api/health ───────────────────────────────────────────────────────────

class TestHealth:
    """
    GET /api/health — readiness probe.

    The endpoint performs real checks (model file, ChromaDB directory,
    Gatekeeper reference in app.state). In the test environment:
      - Model files DO NOT exist on disk → model_file component = "MISSING"
      - ChromaDB directory may not exist → chroma_db component = "MISSING"
      - app.state.gk IS set by the mocked lifespan → gatekeeper = "ok"
    This results in status="degraded" rather than "healthy" — which is
    correct and expected in CI where model artefacts are not present.
    """

    async def test_health_returns_200_or_503(self, client):
        """
        Health must return HTTP 200 (healthy) or 503 (degraded).

        WHY 503 allowed:
            In CI, model files and ChromaDB do not exist. The API is running
            but degraded — which is a truthful, correct status for that
            environment. 503 signals "not ready for traffic" cleanly.
        """
        response = await client.get("/api/health")
        assert response.status_code in (200, 503)

    async def test_health_has_status_and_components(self, client):
        """
        The health response body must always contain 'status' and 'components'.

        WHY: The load balancer parses 'status' to decide routing. Monitoring
        dashboards iterate 'components' to show per-service indicators. Both
        fields are part of the public contract — removing either is a
        breaking change.
        """
        response = await client.get("/api/health")
        body = response.json()
        assert "status" in body,     "health response must contain 'status'"
        assert "components" in body, "health response must contain 'components'"

    async def test_health_status_value_is_valid(self, client):
        """
        'status' must be exactly 'healthy' or 'degraded' — never a random string.

        WHY: The Next.js HealthDot component switches colour based on this value.
        An unexpected value (e.g. 'ok' or 'running') would leave the dot stuck
        in the loading state on every page.
        """
        response = await client.get("/api/health")
        assert response.json()["status"] in ("healthy", "degraded")

    async def test_health_components_have_expected_keys(self, client):
        """
        All five subsystem keys must be present in 'components'.

        WHY: Each key maps to a monitored resource. If a key disappears
        (e.g. refactor renames 'gatekeeper' to 'model') monitoring dashboards
        and alerting rules break silently.
        """
        response = await client.get("/api/health")
        components = response.json()["components"]
        expected = {"model_file", "mapping_file", "chroma_db", "gatekeeper", "llm_provider"}
        assert expected == set(components.keys())

    async def test_health_version_matches_package_version(self, client):
        """
        The 'version' field in health must match src.__version__.

        WHY: Ops engineers use this to verify which deployment is running.
        A mismatch means the Docker image was built from a stale tag.
        """
        from src import __version__
        response = await client.get("/api/health")
        assert response.json().get("version") == __version__


# ── GET /api/metrics ─────────────────────────────────────────────────────────

class TestMetrics:
    """
    GET /api/metrics — Prometheus text format, protected by X-API-Key.

    WHY unit-test the metrics endpoint:
        The endpoint is protected by `Depends(require_api_key)`. A refactor
        that accidentally removes that dependency would expose internal metrics
        (uptime, history count, upload volume) to any unauthenticated caller.
        This test verifies the authentication gate is always present.
    """

    async def test_metrics_requires_api_key_in_production(self, client):
        """
        Without an X-API-Key header and with DEEPCOIN_API_KEY set in the env,
        /api/metrics must return 403 Forbidden or 401 Unauthorized.

        WHY: Anonymous access to Prometheus metrics leaks operational data
        (uptime, request volume, model load state) that could aid attackers
        in timing attacks or infrastructure mapping.
        """
        import os
        original = os.environ.get("DEEPCOIN_API_KEY")
        os.environ["DEEPCOIN_API_KEY"] = "production-secret-key"
        try:
            response = await client.get("/api/metrics")
            # Should be denied — either 401 or 403
            assert response.status_code in (401, 403), (
                f"Expected 401 or 403 but got {response.status_code}. "
                "The metrics endpoint must be auth-gated."
            )
        finally:
            if original is None:
                os.environ.pop("DEEPCOIN_API_KEY", None)
            else:
                os.environ["DEEPCOIN_API_KEY"] = original

    async def test_metrics_accessible_in_dev_mode(self, client):
        """
        In dev mode (no DEEPCOIN_API_KEY env var), the metrics endpoint must
        be accessible so developers can curl it locally without setup.

        WHY: Requiring API key setup for local development introduces friction
        that causes developers to skip metrics monitoring entirely. Dev-mode
        passthrough is the correct balance between security and DX.
        """
        import os
        os.environ.pop("DEEPCOIN_API_KEY", None)   # ensure dev mode
        response = await client.get("/api/metrics")
        assert response.status_code == 200

    async def test_metrics_returns_prometheus_format(self, client):
        """
        The response must be Prometheus text exposition format, not JSON.

        WHY: Prometheus scraping requires the specific text format. If someone
        changes the response to JSON, the Prometheus scrape job silently stops
        collecting metrics — no alerts fire until the dashboard is checked.
        """
        import os
        os.environ.pop("DEEPCOIN_API_KEY", None)
        response = await client.get("/api/metrics")
        assert response.status_code == 200
        text = response.text
        # Every Prometheus metric block starts with "# HELP"
        assert "# HELP" in text, "Prometheus text format requires '# HELP' comment lines"
        assert "# TYPE" in text, "Prometheus text format requires '# TYPE' comment lines"
        assert "deepcoin_uptime_seconds" in text
