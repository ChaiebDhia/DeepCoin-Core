"""
tests/integration/test_admin_coins.py
====================================
Integration tests for admin coin inventory routes.

WHAT this suite covers:
1. Auth boundary (401 for anonymous, 403 for non-privileged users)
2. Prefill request validation for empty payloads
3. Delete not-found behavior
4. Basic stats response contract for privileged users
"""

from __future__ import annotations

import uuid

from src.api.db.models import UserRole


class TestAdminCoinAuth:
    async def test_list_requires_auth(self, client):
        response = await client.get("/api/admin/coins")
        assert response.status_code == 401

    async def test_list_forbidden_for_analyst(self, auth_client):
        # auth_client fixture injects an analyst role by default.
        response = await auth_client.get("/api/admin/coins")
        assert response.status_code == 403


class TestAdminCoinValidation:
    async def test_prefill_requires_type_or_query(self, auth_client, mock_current_user):
        mock_current_user.role = UserRole.admin
        response = await auth_client.post("/api/admin/coins/prefill", json={})

        assert response.status_code == 422
        body = response.json()
        assert "type_id or query" in str(body.get("detail", ""))


class TestAdminCoinDelete:
    async def test_delete_missing_coin_returns_404(self, auth_client, mock_current_user):
        mock_current_user.role = UserRole.admin
        fake_id = str(uuid.uuid4())

        response = await auth_client.delete(f"/api/admin/coins/{fake_id}")
        assert response.status_code == 404


class TestAdminCoinStats:
    async def test_stats_shape_for_privileged_user(self, auth_client, mock_current_user, override_db):
        mock_current_user.role = UserRole.admin

        # The admin_coins stats route performs multiple execute() calls.
        # Reuse the same result object and force deterministic scalar/all values.
        override_db.execute.return_value.scalar_one.return_value = 0
        override_db.execute.return_value.all.return_value = []

        response = await auth_client.get("/api/admin/coins/stats")
        assert response.status_code == 200

        body = response.json()
        for key in (
            "total",
            "manual_count",
            "ai_prefilled",
            "in_training_set",
            "with_gallery",
            "by_source_type",
            "by_region",
            "by_mint",
            "map_points",
        ):
            assert key in body
