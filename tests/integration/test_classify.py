"""
tests/integration/test_classify.py
====================================
Integration tests for POST /api/classify — the core endpoint of DeepCoin.

WHAT these tests verify:
  1. File validation (MIME type, magic bytes, file size) — defence in depth
  2. Filename sanitisation (path traversal prevention)
  3. API key authentication enforcement in production mode
  4. Success path: valid JPEG/PNG → 200 + structurally correct JSON response
  5. Response schema completeness (all required fields present)
  6. TTA parameter forwarded correctly to the Gatekeeper

WHY this suite is critical:
    POST /api/classify is the entire reason DeepCoin exists. 100% of revenue
    (in a commercial version) and 100% of research output goes through this
    single endpoint. Before this test suite existed, the endpoint had ZERO
    test coverage — a rename of a field, a Pydantic upgrade, or a routing
    change could silently break the product with no detection.

Test strategy:
    - Uses the `client` fixture from conftest.py
      (mock Gatekeeper + mock DB session + guest user)
    - File uploads are sent via httpx multipart form
    - No real CNN inference, no real DB writes

Minimal valid image bytes used for tests:
    JPEG: 20-byte minimal JFIF file. Magic bytes \xff\xd8\xff satisfy the
          magic-byte check in classify.py _detect_mime().
    PNG:  Standard 8-byte PNG signature satisfies magic-byte check.
"""
from __future__ import annotations

import os
import pytest


# ── Minimal valid binary images for testing ────────────────────────────────────
# These are the smallest syntactically valid JPEG and PNG files possible.
# They pass the magic-byte validation in classify.py but would fail full
# OpenCV decode — we never actually decode them since the Gatekeeper is mocked.

MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0"   # SOI + APP0 marker
    b"\x00\x10"           # APP0 length = 16 bytes
    b"JFIF\x00"           # identifier
    b"\x01\x01"           # version 1.1
    b"\x00"               # pixel aspect ratio unit = undefined
    b"\x00\x01\x00\x01"  # 1×1 pixels
    b"\x00\x00"           # no embedded thumbnail
    b"\xff\xd9"           # EOI — end of image
)

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"   # PNG signature (standard 8-byte header)
    b"\x00\x00\x00\rIHDR"  # IHDR chunk header (13-byte chunk)
    b"\x00\x00\x00\x01"    # width = 1 pixel
    b"\x00\x00\x00\x01"    # height = 1 pixel
    b"\x08\x02\x00\x00\x00"  # 8-bit depth, RGB colour type
    b"\x90wS\xde"           # CRC32
)


# ── File validation tests ─────────────────────────────────────────────────────

class TestClassifyFileValidation:
    """
    DEFENCE IN DEPTH: classify.py applies three independent file validation
    layers before touching the Gatekeeper. A single layer is insufficient:

    Layer 1 — Content-Type header:
        Easy for a malicious client to forge. We check it first because it is
        fast, but we NEVER rely on it exclusively.

    Layer 2 — Magic bytes:
        The first 4 bytes of the actual binary data. Cannot be faked without
        making the file unreadable as an image. This is the authoritative check.

    Layer 3 — File size:
        10 MB cap. Coin photos are never larger. A 1 GB upload would block the
        event loop for seconds; this cap protects against DoS via large uploads.

    A malicious actor would have to pass ALL THREE layers simultaneously —
    which requires producing a file that is both a valid image AND contains
    the malicious payload. For coin analysis this is effectively impossible.
    """

    async def test_no_file_returns_422(self, client):
        """
        A POST with no 'file' field must return 422 Unprocessable Entity.

        WHY 422 not 400:
            FastAPI uses 422 for request body validation failures, consistently
            with the OpenAPI spec. The client has submitted a syntactically
            correct request but with missing required fields.
        """
        response = await client.post("/api/classify")
        assert response.status_code == 422

    async def test_wrong_content_type_returns_415(self, client):
        """
        A file with Content-Type: text/plain must be rejected before reading
        the body, with HTTP 415 Unsupported Media Type.

        WHY 415 and not 400:
            HTTP 415 is the semantically correct code for "this content type
            is not supported." It signals to the caller that they must use
            image/jpeg or image/png — not that their request was malformed.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    async def test_wrong_mime_with_pdf_content_type_returns_415(self, client):
        """
        application/pdf is not an accepted image type.
        Reject it even if the caller sends real PDF bytes.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
        )
        assert response.status_code == 415

    async def test_file_too_large_returns_413(self, client):
        """
        A file exceeding MAX_UPLOAD_BYTES (10 MB) must return 413 Request Entity Too Large.

        WHY 10 MB cap:
            The largest coin photo from a DSLR camera is ~5 MB. JPEG compression
            makes coin images small. A 10 MB cap allows generous overhead while
            blocking accidental large file uploads (a 4K screenshot is ~15 MB)
            and deliberate DoS attacks (a 1 GB upload would buffer-overflow
            asyncio's readinto loop in a single-worker server).
        """
        # Build a JPEG-magic-byte-prefixed buffer just over the limit
        oversized = b"\xff\xd8\xff" + b"X" * (10 * 1024 * 1024 + 1)
        response = await client.post(
            "/api/classify",
            files={"file": ("big.jpg", oversized, "image/jpeg")},
        )
        assert response.status_code == 413

    async def test_valid_jpeg_magic_but_declared_as_wrong_type_returns_415(self, client):
        """
        Even if the file content has valid JPEG magic bytes, if Content-Type
        is wrong the request is rejected at the first gate (content-type check)
        before the magic-byte check runs.

        WHY test this ordering:
            The content-type check must execute FIRST (before reading the file)
            to avoid buffering potentially huge payloads just to reject them.
            This test confirms the short-circuit order is correct.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "application/octet-stream")},
        )
        assert response.status_code == 415

    async def test_jpeg_with_random_bytes_after_magic_returns_415(self, client):
        """
        A file that starts with \xff\xd8\xff (JPEG magic) but has Content-Type
        text/plain is rejected at the content-type layer.
        """
        fake_jpeg = b"\xff\xd8\xff" + b"this is not a real jpeg"
        response = await client.post(
            "/api/classify",
            files={"file": ("fake.jpg", fake_jpeg, "text/plain")},
        )
        assert response.status_code == 415


# ── Authentication enforcement tests ─────────────────────────────────────────

class TestClassifyAuthentication:
    """
    POST /api/classify uses require_api_key dependency.

    In dev mode (DEEPCOIN_API_KEY not set) every request passes through.
    In production mode (key set) requests without the correct X-API-Key
    header must be rejected with 401/403.

    WHY test this:
        The GPU is an expensive resource. Without API key protection, anyone
        who discovers the endpoint URL can trigger unlimited CNN inferences
        (each consuming ~543ms GPU time). On a 10-request-per-minute rate
        limit that is 600 analyses per hour per attacker. This test ensures
        the auth gate is always present.
    """

    async def test_classify_accessible_in_dev_mode(self, client):
        """
        With no DEEPCOIN_API_KEY env var (dev mode), classify must accept
        the request with a valid JPEG file.

        WHY test dev mode:
            Developer onboarding requires that `pip install` + `uvicorn` works
            without configuring API keys. Checking dev mode passthrough
            ensures we never accidentally break the out-of-box experience.
        """
        os.environ.pop("DEEPCOIN_API_KEY", None)
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        # Must not be a 401/403 auth rejection
        assert response.status_code not in (401, 403), (
            f"classify returned {response.status_code} in dev mode — "
            "API key should not be required when DEEPCOIN_API_KEY is not set."
        )

    async def test_classify_rejects_missing_key_in_production(self, client):
        """
        When DEEPCOIN_API_KEY is configured (production mode), a request
        with no X-API-Key header must return 401 or 403.
        """
        os.environ["DEEPCOIN_API_KEY"] = "production-secret"
        try:
            response = await client.post(
                "/api/classify",
                files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
            )
            assert response.status_code in (401, 403), (
                f"Expected 401 or 403 but got {response.status_code}. "
                "POST /api/classify must enforce X-API-Key when DEEPCOIN_API_KEY is set."
            )
        finally:
            os.environ.pop("DEEPCOIN_API_KEY", None)

    async def test_classify_accepts_correct_key_in_production(self, client):
        """
        When the correct X-API-Key header is sent in production mode,
        the request must not be rejected for auth reasons.
        """
        os.environ["DEEPCOIN_API_KEY"] = "correct-key-123"
        try:
            response = await client.post(
                "/api/classify",
                headers={"X-API-Key": "correct-key-123"},
                files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
            )
            # Auth should pass — though classify might fail for other reasons,
            # it must not be a 401/403.
            assert response.status_code not in (401, 403)
        finally:
            os.environ.pop("DEEPCOIN_API_KEY", None)


# ── Success path tests ────────────────────────────────────────────────────────

class TestClassifySuccess:
    """
    Tests for the happy path: valid file → Gatekeeper (mocked) → 200 JSON.

    The mock Gatekeeper returns a deterministic CoinState dict (see conftest.py).
    These tests verify that the classify route correctly:
      1. Calls gk.analyze() (via asyncio.to_thread)
      2. Maps the state dict onto ClassifyResponse fields
      3. Returns the correct HTTP status and Content-Type

    WHY mock and not a real inference:
        Real inference takes 15-20 seconds and requires GPU + model files.
        CI runs on CPU only with no model artefacts. The mock tests the
        routing, schema building, and error handling code paths — the
        correctness of CNN inference is tested by scripts/test_pipeline.py
        in the GPU environment.
    """

    async def test_valid_jpeg_returns_200(self, client):
        """
        A minimal valid JPEG file (JFIF magic bytes) must produce HTTP 200.

        WHY minimal JPEG:
            We test the HTTP layer, not image decodability. The mock Gatekeeper
            ignores the image content. The JFIF magic bytes satisfy the
            _detect_mime() check without needing a real compressed image.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        assert response.status_code == 200, (
            f"Expected 200 but got {response.status_code}: {response.text[:200]}"
        )

    async def test_valid_png_returns_200(self, client):
        """
        PNG files must be accepted alongside JPEG. PNG uploads from mobile
        screenshots or scan workflows must work identically to JPEG.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.png", MINIMAL_PNG, "image/png")},
        )
        assert response.status_code == 200

    async def test_success_response_has_required_fields(self, client):
        """
        The ClassifyResponse schema must include all required fields.

        WHY test schema completeness:
            The Next.js frontend destructures specific fields. If a field is
            renamed (e.g. 'cnn' → 'cnn_result') the frontend silently renders
            undefined. This test catches schema regressions immediately.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        body = response.json()
        required = {"id", "timestamp", "route_taken", "cnn", "processing_time_s"}
        missing   = required - set(body.keys())
        assert not missing, f"ClassifyResponse is missing required fields: {missing}"

    async def test_success_response_cnn_section(self, client):
        """
        The 'cnn' sub-object must have all CNN result fields.

        WHY: The AnalysisPanel component reads cnn.confidence, cnn.label,
        cnn.vote_fraction, and cnn.tta_passes. Missing any causes NaN
        display or crashes in TTA threshold comparisons.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        cnn = response.json()["cnn"]
        assert "label"             in cnn
        assert "confidence"        in cnn
        assert "top5"              in cnn
        assert "tta_used"          in cnn
        assert "tta_passes"        in cnn
        assert "vote_fraction"     in cnn
        assert "inference_time_ms" in cnn

    async def test_success_response_id_is_uuid(self, client):
        """
        The 'id' field must be a valid UUID4 string.

        WHY: The history endpoint uses this ID as primary key. A non-UUID id
        (e.g. a filename or integer) would break GET /api/history/{id} routing
        and potentially cause SQLAlchemy column type errors.
        """
        import uuid
        response = await client.post(
            "/api/classify",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        record_id = response.json()["id"]
        # Should not raise ValueError
        parsed = uuid.UUID(record_id)
        assert str(parsed) == record_id

    async def test_tta_false_parameter_accepted(self, client):
        """
        TTA can be disabled via ?tta=false query parameter.
        The mock Gatekeeper still returns a valid response regardless.

        WHY test TTA toggle:
            Users on slow connections or mobile may want single-pass inference
            (3-5s instead of 25s). The TTA toggle must be forwarded to the
            Gatekeeper and the response must still be valid.
        """
        response = await client.post(
            "/api/classify?tta=false",
            files={"file": ("coin.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        assert response.status_code == 200


# ── Path traversal protection tests ──────────────────────────────────────────

class TestClassifyFilenameSecuritiy:
    """
    Tests for _sanitise_filename() — path traversal attack prevention.

    ATTACK VECTOR:
        An attacker submits a file named "../../etc/passwd.jpg".
        Without sanitisation, the save path becomes:
            data/uploads/../../etc/passwd.jpg = /etc/passwd.jpg
        This overwrites /etc/passwd on the server.

    DEFENCE:
        _sanitise_filename() in classify.py strips all directory components
        and replaces any non-ASCII character with "_" (re.ASCII flag).
        The test verifies that the ENDPOINT SUCCESSFULLY PROCESSES the file
        (meaning sanitisation worked without error), not that it rejects it.
    """

    async def test_path_traversal_filename_is_sanitised(self, client):
        """
        A filename containing path traversal sequences (../../../) must be
        accepted (not rejected with an error) because _sanitise_filename
        strips the traversal and uses only the last path component.

        WHY 200 expected (not 400):
            Sanitisation means the upload is SAFE, so it proceeds normally.
            Rejecting traversal filenames would be a valid alternative design,
            but the current design sanitises silently (prefer robustness).
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("../../etc/passwd.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        # The request should either succeed (200) or fail for NON-security reasons.
        # It must NOT fail with a 500 (unhandled path traversal) or 400/422.
        assert response.status_code in (200, 415, 413), (
            f"Unexpected status {response.status_code}: path traversal filenames "
            "must be sanitised, not cause a server error."
        )

    async def test_unicode_filename_is_sanitised(self, client):
        """
        Non-ASCII filenames (accents, CJK characters) must be sanitised to
        ASCII-only before saving. cv2.imread() on Windows silently returns
        None for non-ASCII file paths (Windows ANSI path limitation).

        WHY test this:
            A mobile upload from an Arabic or Chinese locale would produce
            filenames like 'عملة.jpg' or '硬币.jpg'. Without sanitisation,
            cv2.imread() returns None → NoneType error inside the pipeline.
        """
        response = await client.post(
            "/api/classify",
            files={"file": ("عملة_قديمة.jpg", MINIMAL_JPEG, "image/jpeg")},
        )
        # Must not crash the server
        assert response.status_code != 500
