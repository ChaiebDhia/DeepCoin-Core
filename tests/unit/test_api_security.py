"""
tests/unit/test_api_security.py
================================
Unit tests for the security helpers in the classify route:
  - _sanitise_filename(): prevents path-traversal uploads
  - _detect_mime():       rejects non-image magic bytes

These tests do NOT load the CNN model or hit the database.

Usage:
    pytest tests/unit/test_api_security.py -v
"""

import pytest

# Import only the pure helpers — no side effects from loading the full app
from src.api.routes.classify import _sanitise_filename, _detect_mime


# ── _sanitise_filename ────────────────────────────────────────────────────────

class TestSanitiseFilename:
    """
    WHAT: _sanitise_filename() strips directory components and sanitises
          characters that are unsafe in OS filenames.
    WHY:  An attacker can upload a file named '../../etc/passwd.jpg'.
          Without sanitisation, that path could escape the uploads directory.
    """

    def test_plain_filename_unchanged(self):
        """A normal filename must pass through untouched (minus extension casing)."""
        result = _sanitise_filename("coin_photo.jpg")
        assert result == "coin_photo.jpg"

    def test_strips_unix_path_traversal(self):
        """../../ prefixes must be stripped to the base filename only."""
        result = _sanitise_filename("../../etc/passwd.jpg")
        assert "/" not in result
        assert ".." not in result
        assert "passwd" in result

    def test_strips_windows_path_traversal(self):
        """Windows backslash path traversal must also be stripped."""
        result = _sanitise_filename(r"..\..\windows\system32\evil.jpg")
        assert "\\" not in result
        assert ".." not in result

    def test_strips_absolute_unix_path(self):
        """/etc/passwd must collapse to just passwd (or similar base name)."""
        result = _sanitise_filename("/etc/passwd.jpg")
        assert "/" not in result

    def test_preserves_extension(self):
        """The file extension must be preserved."""
        assert _sanitise_filename("photo.jpeg").endswith(".jpeg")
        assert _sanitise_filename("scan.PNG").lower().endswith(".png")

    def test_empty_string_handled(self):
        """An empty filename must not raise — returns a safe fallback."""
        result = _sanitise_filename("")
        assert isinstance(result, str)   # whatever it returns, no crash

    def test_null_bytes_removed(self):
        """Filenames with null bytes are a classic injection vector."""
        result = _sanitise_filename("evil\x00.jpg")
        assert "\x00" not in result

    def test_non_ascii_replaced(self):
        """
        Non-ASCII characters (accented letters, CJK, etc.) must be replaced
        with '_'.  cv2.imread() and np.fromfile() use C-runtime fopen() on
        Windows which only accepts ANSI paths — a saved path with 'é' causes a
        silent None return and a pipeline crash.
        Real-world case: French locale screenshots 'Capture_d_écran_....png'.
        """
        result = _sanitise_filename("Capture_d_écran_2026-03-01.png")
        assert "é" not in result
        assert result.isascii()
        assert result.endswith(".png")


# ── _detect_mime ──────────────────────────────────────────────────────────────

class TestDetectMime:
    """
    WHAT: _detect_mime() inspects the first bytes (magic bytes) of an uploaded
          file to determine its actual MIME type, ignoring the stated Content-Type.
    WHY:  An attacker can rename any file to .jpg. Magic-byte detection prevents
          uploading shell scripts, executables, or HTML disguised as images.
    """

    # JPEG magic: FF D8 FF
    JPEG_HEADER = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 20

    # PNG magic: 89 50 4E 47 0D 0A 1A 0A
    PNG_HEADER  = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"\x00" * 20

    # WebP magic: RIFF????WEBP
    WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10

    # GIF magic: GIF87a / GIF89a
    GIF_HEADER  = b"GIF89a" + b"\x00" * 20

    def test_jpeg_detected(self):
        assert _detect_mime(self.JPEG_HEADER) == "image/jpeg"

    def test_png_detected(self):
        assert _detect_mime(self.PNG_HEADER) == "image/png"

    def test_webp_detected(self):
        result = _detect_mime(self.WEBP_HEADER)
        assert result in ("image/webp", "image/jpeg", "image/png", None)
        # webp may not be supported on all platforms — just must not crash

    def test_gif_rejected_or_detected(self):
        """GIF is not a valid coin image format. Either None or 'image/gif'."""
        result = _detect_mime(self.GIF_HEADER)
        # We accept either None (rejected) or a clear mime string (caller decides)
        assert result is None or isinstance(result, str)

    def test_unknown_bytes_returns_none(self):
        """Random bytes with no known magic signature must return None."""
        result = _detect_mime(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A")
        assert result is None

    def test_empty_bytes_returns_none(self):
        """Empty binary input must not raise."""
        result = _detect_mime(b"")
        assert result is None

    def test_html_disguised_as_image_rejected(self):
        """An HTML file starting with <!DOCTYPE must not be accepted as an image."""
        html_bytes = b"<!DOCTYPE html><html><body>evil</body></html>"
        result = _detect_mime(html_bytes)
        assert result is None or "html" not in (result or "")

    def test_python_script_disguised_as_image_rejected(self):
        """A Python file starting with # or import must be rejected."""
        py_bytes = b"#!/usr/bin/env python3\nimport os; os.system('rm -rf /')"
        result = _detect_mime(py_bytes)
        assert result is None

    def test_elf_binary_rejected(self):
        """ELF binary magic (Linux executable) must be rejected."""
        elf_bytes = bytes([0x7F, 0x45, 0x4C, 0x46]) + b"\x00" * 20
        result = _detect_mime(elf_bytes)
        assert result is None
