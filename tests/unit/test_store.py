"""
tests/unit/test_store.py
========================
Unit tests for src/api/_store.py (SQLite history store).

These tests use a temporary directory for the database so they never
touch the production data/history.db file.

Usage:
    pytest tests/unit/test_store.py -v
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Point the store at a temp directory during tests
_TMP = tempfile.mkdtemp()
os.environ.setdefault("_TEST_DB_OVERRIDE", "1")


# ── patch _DB_PATH before importing the module ────────────────────────────────

import src.api._store as store_module
_original_db_path = store_module._DB_PATH
store_module._DB_PATH = Path(_TMP) / "test_history.db"


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    """
    Ensure a clean database for every test.

    Why autouse=True:
        Every test gets an isolated state. Tests that write records don't
        pollute the assertion counts of tests that count records.
    """
    db_path = store_module._DB_PATH
    if db_path.exists():
        db_path.unlink()
    store_module.ensure_store()
    yield
    # Cleanup after test
    if db_path.exists():
        db_path.unlink()


def _make_record(record_id: str = "test-uuid-001", label: str = "1015",
                 confidence: float = 0.91, route: str = "historian") -> dict:
    """Helper — build a minimal ClassifyResponse-shaped dict."""
    return {
        "id":              record_id,
        "timestamp":       "2026-02-28T10:00:00+00:00",
        "image_filename":  "coin.jpg",
        "route_taken":     route,
        "cnn": {
            "class_id":          0,
            "label":             label,
            "confidence":        confidence,
            "top5":              [],
            "inference_time_ms": 142,
            "tta_used":          False,
        },
        "narrative":            "Test narrative.",
        "mint":                 "Maroneia",
        "region":               "Thrace",
        "date_range":           "c.365-330 BC",
        "material":             "silver",
        "denomination":         "drachm",
        "material_status":      None,
        "material_confidence":  None,
        "visual_description":   None,
        "kb_match_count":       None,
        "pdf_url":              "/api/reports/test.pdf",
        "processing_time_s":    15.4,
    }


# ── tests ─────────────────────────────────────────────────────────────────────

class TestEnsureStore:
    def test_creates_db_file(self):
        """ensure_store() must create the database file."""
        assert store_module._DB_PATH.exists(), "DB file should exist after ensure_store()"

    def test_idempotent(self):
        """Calling ensure_store() twice must not raise or corrupt the DB."""
        store_module.ensure_store()   # second call
        assert store_module._DB_PATH.exists()


class TestAppend:
    def test_append_single_record(self):
        """append() must persist one record retrievable by get_by_id."""
        rec = _make_record()
        store_module.append(rec)

        result = store_module.get_by_id("test-uuid-001")
        assert result is not None, "Record must be retrievable after append"
        assert result["id"] == "test-uuid-001"

    def test_append_preserves_all_fields(self):
        """The full payload including nested cnn dict must survive round-trip."""
        rec = _make_record(label="3987", confidence=0.55)
        store_module.append(rec)

        result = store_module.get_by_id(rec["id"])
        assert result["cnn"]["label"] == "3987"
        assert abs(result["cnn"]["confidence"] - 0.55) < 1e-6
        assert result["mint"] == "Maroneia"

    def test_append_multiple_records(self):
        """load_all() must return all appended records."""
        for i in range(5):
            store_module.append(_make_record(record_id=f"uuid-{i:03d}", label=str(1000 + i)))

        all_records = store_module.load_all()
        assert len(all_records) == 5, f"Expected 5 records, got {len(all_records)}"

    def test_append_upsert_on_duplicate_id(self):
        """Inserting a second record with the same id must overwrite (not duplicate)."""
        rec = _make_record(route="historian")
        store_module.append(rec)

        rec_updated = {**rec, "route_taken": "investigator"}
        store_module.append(rec_updated)

        all_records = store_module.load_all()
        assert len(all_records) == 1, "Duplicate id must upsert, not insert twice"
        assert all_records[0]["route_taken"] == "investigator"


class TestLoadAll:
    def test_empty_store_returns_empty_list(self):
        """load_all() on a fresh store must return []."""
        records = store_module.load_all()
        assert records == []

    def test_newest_first_ordering(self):
        """Records must be returned newest-first (by timestamp)."""
        store_module.append(_make_record("id-old", label="1015") |
                            {"timestamp": "2026-01-01T00:00:00+00:00"})
        store_module.append(_make_record("id-new", label="3987") |
                            {"timestamp": "2026-02-28T00:00:00+00:00"})

        records = store_module.load_all()
        assert records[0]["id"] == "id-new", "Newest record must come first"
        assert records[1]["id"] == "id-old"


class TestGetById:
    def test_returns_none_for_missing_id(self):
        """get_by_id() must return None when the id does not exist."""
        result = store_module.get_by_id("does-not-exist")
        assert result is None

    def test_returns_correct_record(self):
        """get_by_id() must return exactly the record with the given id."""
        r1 = _make_record(record_id="id-alpha", label="1015")
        r2 = _make_record(record_id="id-beta",  label="3987")
        store_module.append(r1)
        store_module.append(r2)

        result = store_module.get_by_id("id-beta")
        assert result is not None
        assert result["cnn"]["label"] == "3987"
        assert result["id"] == "id-beta"
