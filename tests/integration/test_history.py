"""
tests/integration/test_history.py
===================================
Integration tests for the history endpoints.

Endpoints covered:
    GET    /api/history           — paginated list of user's analyses
    GET    /api/history/{id}      — single classification detail
    DELETE /api/history/{id}      — remove a classification
    POST   /api/history/{id}/feedback — mark as wrong correction

WHAT these tests verify:
  1. Authentication is enforced — unauthenticated requests get 401
  2. Empty history returns a valid response (not 500)
  3. Non-existent IDs return 404 (not 500 or a wrong-ID result)
  4. Delete returns correct HTTP status codes
  5. Feedback submission schema is validated

WHY these tests matter:
    The history feature is what transforms DeepCoin from a "one-off classify"
    tool into a research assistant. Museum curators need to review past
    analyses; researchers need to export history for study. A bug that breaks
    history pagination silently delivers corrupted records to users who may
    not notice until they try to use old data.

Test strategy:
    Uses the `auth_client` fixture (mock DB + mock authenticated user).
    DB return values are configured per-test via the `override_db` fixture.
    No real PostgreSQL — all SQL is intercepted by AsyncMock.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock


# ── GET /api/history — unauthenticated ───────────────────────────────────────

class TestHistoryAuthentication:
    """
    Every history endpoint requires authentication.

    WHY strict auth on history:
        Analysis history is personal data. Without authentication, a URL
        enumeration attack would expose every user's coin analyses:
            GET /api/history?skip=0&limit=100 → full database dump
        Users would not expect their museum research to be publicly accessible.
    """

    async def test_get_history_requires_auth(self, client):
        """
        GET /api/history without auth token must return 401 Unauthorized.

        WHY 401 not 403:
            401 means "authentication required" — the client can retry with
            credentials. 403 means "you are authenticated but not permitted."
            Since the user is not authenticated at all, 401 is correct.
        """
        response = await client.get("/api/history")
        assert response.status_code == 401, (
            f"Expected 401 but got {response.status_code}. "
            "GET /api/history must require authentication."
        )

    async def test_get_history_detail_requires_auth(self, client):
        """GET /api/history/{id} without auth must return 401."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/api/history/{fake_id}")
        assert response.status_code == 401

    async def test_delete_history_requires_auth(self, client):
        """DELETE /api/history/{id} without auth must return 401."""
        fake_id = str(uuid.uuid4())
        response = await client.delete(f"/api/history/{fake_id}")
        assert response.status_code == 401


# ── GET /api/history — authenticated ─────────────────────────────────────────

class TestHistoryList:
    """
    GET /api/history returns a paginated list of the authenticated user's
    classification records.

    WHY test empty list:
        A new user with no analyses must get a 200 with an empty list,
        not a 404 or 500. Many frontend list components crash on null
        instead of an empty array.
    """

    async def test_empty_history_returns_200(self, auth_client, override_db):
        """
        When the user has no analyses, GET /api/history must return HTTP 200
        with an empty items list.

        WHY not 404:
            404 means "this resource does not exist." A user's history
            resource exists from the moment of account creation — it is just
            empty. 404 would confuse the frontend into showing an error page.
        """
        # Default mock_db configuration returns [] for scalars().all()
        response = await auth_client.get("/api/history")
        assert response.status_code == 200

    async def test_history_response_is_list_shape(self, auth_client, override_db):
        """
        The response body must contain an 'items' array and pagination metadata.

        WHY test structure and not just status:
            A 200 with the wrong shape (e.g. returning a plain list instead of
            a {items, total, page} envelope) would cause silent frontend bugs
            where the history table renders empty while the API appears healthy.
        """
        response = await auth_client.get("/api/history")
        body = response.json()
        assert "items" in body, (
            "GET /api/history must return {items: [...], ...} envelope, "
            f"got: {list(body.keys())}"
        )
        assert isinstance(body["items"], list)

    async def test_history_accepts_pagination_params(self, auth_client, override_db):
        """
        skip and limit query parameters must be accepted without error.

        WHY: Pagination is the primary interface for the history table.
        If the endpoint rejects valid pagination params (e.g. raising 422),
        every page load beyond the first page fails.
        """
        response = await auth_client.get("/api/history?skip=0&limit=5")
        assert response.status_code == 200

    async def test_history_invalid_limit_returns_422(self, auth_client, override_db):
        """
        A limit above the allowed maximum must return 422 Unprocessable Entity.

        WHY: Without a max limit, a single request could return the entire
        database: GET /api/history?limit=999999. Pydantic validation on the
        Query parameter enforces the cap.
        """
        # history.py uses limit: int = Query(20, le=100) — over 100 should fail
        response = await auth_client.get("/api/history?limit=9999")
        assert response.status_code == 422


# ── GET /api/history/{id} — authenticated ────────────────────────────────────

class TestHistoryDetail:
    """
    GET /api/history/{id} returns the full classification record for one analysis.
    """

    async def test_detail_not_found_returns_404(self, auth_client, override_db):
        """
        When the requested ID does not exist in the database,
        the endpoint must return 404 Not Found.

        WHY 404 not 500:
            A database query that returns no rows is a normal, expected
            condition. Raising a 500 would signal a system error to the client
            when really the record just doesn't exist.
        """
        # Configure the mock to return None for the ID lookup
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        result_mock.scalars.return_value.first.return_value = None
        override_db.execute.return_value = result_mock

        fake_id = str(uuid.uuid4())
        response = await auth_client.get(f"/api/history/{fake_id}")
        assert response.status_code == 404, (
            f"Expected 404 for non-existent ID but got {response.status_code}"
        )


# ── DELETE /api/history/{id} ─────────────────────────────────────────────────

class TestHistoryDelete:
    """
    DELETE /api/history/{id} removes a classification record.

    Access control:
        - The owner (current_user.id == classification.user_id) can delete
        - Admins/curators can delete any record
        - Other users get 403 Forbidden
    """

    async def test_delete_nonexistent_returns_404(self, auth_client, override_db):
        """
        DELETE on a non-existent ID must return 404.

        WHY: A frontend that retries a delete (network glitch) must not get
        a 500 on the second attempt. 404 tells it the record is gone.
        """
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        result_mock.scalars.return_value.first.return_value = None
        override_db.execute.return_value = result_mock

        fake_id = str(uuid.uuid4())
        response = await auth_client.delete(f"/api/history/{fake_id}")
        assert response.status_code == 404

    async def test_delete_accessible_to_owner(self, auth_client, override_db, mock_current_user):
        """
        A user deleting their OWN record must receive a success response (204).

        WHY test ownership:
            Without the user_id check, any authenticated user can delete
            anyone else's classification: DELETE /api/history/<victim-uuid>
            This is a horizontal privilege escalation vulnerability.
        """
        from unittest.mock import MagicMock as _MM

        # Create a mock Classification row owned by the mock_current_user
        mock_row = _MM()
        mock_row.id         = uuid.uuid4()
        mock_row.user_id    = mock_current_user.id   # owns this record
        mock_row.pdf_path   = None

        result_mock = _MM()
        result_mock.scalar_one_or_none.return_value = mock_row
        result_mock.scalars.return_value.first.return_value = mock_row
        override_db.execute.return_value = result_mock

        response = await auth_client.delete(f"/api/history/{mock_row.id}")
        # 204 No Content is the correct REST response for successful DELETE
        assert response.status_code in (204, 200), (
            f"Expected 204 or 200 for owner delete but got {response.status_code}"
        )


# ── POST /api/history/{id}/feedback ──────────────────────────────────────────

class TestHistoryFeedback:
    """
    POST /api/history/{id}/feedback — "mark as wrong" active learning.

    WHY test feedback submission:
        Feedback is the active-learning pipeline input. Every incorrect
        prediction that a museum curator marks gives data for the next
        model retraining cycle. If feedback submission is broken, that
        pipeline silently starves.

    Schema validation:
        FeedbackRequest requires correct_type_id: str (CN type number).
        Missing or wrong-type fields must return 422.
    """

    async def test_feedback_missing_body_returns_422(self, auth_client, override_db):
        """
        POST /api/history/{id}/feedback with no body must return 422.
        'correct_type_id' is required and Pydantic must enforce it.
        """
        fake_id = str(uuid.uuid4())
        response = await auth_client.post(f"/api/history/{fake_id}/feedback", json={})
        assert response.status_code == 422

    async def test_feedback_missing_type_id_returns_422(self, auth_client, override_db):
        """
        FeedbackRequest.correct_type_id is a required field.
        Omitting it must trigger Pydantic validation error (422).
        """
        fake_id = str(uuid.uuid4())
        response = await auth_client.post(
            f"/api/history/{fake_id}/feedback",
            json={"note": "this is a wrong prediction"},  # missing correct_type_id
        )
        assert response.status_code == 422
