"""
src/api/_store.py
==================
.. deprecated:: 0.5.0
    This module (SQLite-backed history store) is superseded by the
    `classifications` and `feedback` PostgreSQL tables in
    `src/api/db/models.py` (A4 migration, March 2026).

    The routes POST /api/classify, GET /api/history, DELETE /api/history/{id},
    and POST /api/history/{id}/feedback now write/read directly from PostgreSQL
    via SQLAlchemy async sessions.

    This module is retained for:
      - Unit tests that verify the SQLite store in isolation
      - Any legacy data migration scripts (read old SQLite → write to PostgreSQL)

    DO NOT add new callers to this module.

Classification history store backed by SQLite (stdlib sqlite3).

WHY SQLite OVER the previous JSON FILE store
--------------------------------------------
The JSON store read and rewrote the entire file on every append — O(n) writes.
With 10,000 records that's 10,000 × full-file rewrites.  SQLite:

  - INSERT is O(log n) with a B-tree index on the `id` column
  - SELECT by id is O(log n) via primary key lookup
  - SELECT all is a sequential scan — still fast at any realistic scale
  - Crash-safe: SQLite uses WAL (Write-Ahead Logging) to prevent corruption
  - Zero extra dependencies: sqlite3 is in the Python standard library
  - Migration path: swap for PostgreSQL by changing only this module

WHY NOT SQLAlchemy (yet):
    SQLAlchemy async requires Python 3.10+ async generators and adds ~80 MB
    of dependencies.  For a single-worker, single-user service, raw sqlite3
    with a threading.Lock is simpler, faster to understand, and produces
    identical output.  When we move to PostgreSQL (Layer 6), we switch to
    SQLAlchemy async — but that is a one-module change thanks to the
    Repository Pattern used here.

WHY threading.Lock:
    FastAPI workers share the same process (workers=1 in our config).
    Multiple requests can call append() / load_all() concurrently.
    SQLite's built-in thread safety mode (check_same_thread=False) allows
    multiple threads to use the same connection, but still requires external
    locking for read-then-write sequences to be atomic.

SCHEMA
------
    Table: classifications
        id          TEXT    PRIMARY KEY   — UUID, matches ClassifyResponse.id
        timestamp   TEXT    NOT NULL      — ISO 8601 UTC
        label       TEXT    NOT NULL      — CNN predicted CN type ID
        confidence  REAL    NOT NULL      — CNN softmax confidence (0-1)
        route_taken TEXT    NOT NULL      — historian | validator | investigator
        payload     TEXT    NOT NULL      — full JSON blob (ClassifyResponse.model_dump())

    The payload column stores the complete response dict as JSON so we
    don't need a migration every time ClassifyResponse gains a new field.
    The indexed columns (id, timestamp, label) cover every query pattern.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _ROOT / "data" / "history.db"
_lock    = threading.Lock()

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS classifications (
    id          TEXT    PRIMARY KEY,
    timestamp   TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    route_taken TEXT    NOT NULL,
    payload     TEXT    NOT NULL
);
"""

_CREATE_IDX_TS = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON classifications(timestamp DESC);
"""


# ── connection helper ─────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL mode enabled.

    WHY WAL mode:
        Default SQLite journal mode (DELETE) locks the database file for the
        full duration of a write.  WAL (Write-Ahead Logging) allows concurrent
        reads during a write — important for the history route (reads) running
        alongside the classify route (writes).

    WHY check_same_thread=False:
        We use a threading.Lock to serialise writes.  Python's sqlite3 module
        with check_same_thread=False allows multiple threads to share the
        connection safely when combined with explicit locking.
    """
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── public API ─────────────────────────────────────────────────────────────────

def ensure_store() -> None:
    """
    Create the database file and classifications table if they do not exist.

    Called once at API startup before any request is processed.
    Idempotent — safe to call multiple times.
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_IDX_TS)
            conn.commit()
        finally:
            conn.close()
    logger.info("History store ready: %s", _DB_PATH)


def append(record: dict) -> None:
    """
    Persist one classification result to the database.

    WHAT: Inserts a row into the classifications table.
          The full record dict is stored as a JSON blob in the payload column.
          Key fields are promoted to indexed columns for fast queries.

    Args:
        record: Full classification result dict (ClassifyResponse.model_dump())

    Thread-safe: protected by _lock.
    """
    cnn    = record.get("cnn", {})
    label  = str(cnn.get("label", record.get("label", "")))
    conf   = float(cnn.get("confidence", record.get("confidence", 0.0)))

    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO classifications
                    (id, timestamp, label, confidence, route_taken, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record.get("timestamp", ""),
                    label,
                    conf,
                    record.get("route_taken", ""),
                    json.dumps(record, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def load_all() -> list[dict]:
    """
    Return all records ordered newest-first.

    Returns:
        List of full record dicts (deserialized from JSON payload column).
        Empty list on any database error — API never crashes on a corrupt store.

    Thread-safe: read-only SELECT, still protected by _lock for consistency.
    """
    with _lock:
        try:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT payload FROM classifications ORDER BY timestamp DESC"
            ).fetchall()
            conn.close()
            return [json.loads(row["payload"]) for row in rows]
        except Exception as exc:
            logger.error("History store read error, returning empty: %s", exc)
            return []


def get_by_id(record_id: str) -> Optional[dict]:
    """
    Return one record by its UUID, or None if not found.

    SELECT by primary key — O(log n), constant time regardless of history size.
    """
    with _lock:
        try:
            conn = _get_conn()
            row = conn.execute(
                "SELECT payload FROM classifications WHERE id = ?",
                (record_id,),
            ).fetchone()
            conn.close()
            return json.loads(row["payload"]) if row else None
        except Exception as exc:
            logger.error("History store get_by_id error for %s: %s", record_id, exc)
            return None


def count() -> int:
    """
    Return the total number of classification records via SELECT COUNT(*).

    WHY a dedicated function:
        The /api/metrics and /api/history endpoints both need the total count.
        Using len(load_all()) would deserialise every JSON payload row just
        to count them — O(n) deserialization work when O(1) SQL count suffices.
        SELECT COUNT(*) reads only the index, not the payload column.
    """
    with _lock:
        try:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT COUNT(*) FROM classifications"
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception as exc:
            logger.error("History store count error: %s", exc)
            return 0


def delete_by_id(record_id: str) -> bool:
    """
    Delete one classification record from the database by its UUID.

    WHAT: Executes a DELETE statement by primary key.

    WHY return bool:
        The caller (DELETE /api/history/{id} endpoint) needs to distinguish
        between "deleted successfully" and "id not found" to return the
        correct HTTP status code (204 vs 404).  rowcount == 0 means no row
        matched the id.

    Args:
        record_id: UUID string matching the `id` column.

    Returns:
        True if exactly one row was deleted.  False if id not found or on error.

    Thread-safe: protected by _lock.
    """
    with _lock:
        try:
            conn    = _get_conn()
            cursor  = conn.execute(
                "DELETE FROM classifications WHERE id = ?",
                (record_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted
        except Exception as exc:
            logger.error("History store delete_by_id error for %s: %s", record_id, exc)
            return False


def add_feedback(record_id: str, correct_type_id: str, note: str) -> bool:
    """
    Attach user feedback ("mark as wrong" correction) to an existing record.

    WHAT:
        Reads the stored JSON payload, adds a `feedback` sub-dict, and writes
        it back via SQL UPDATE.  The rest of the row (indexed columns) is
        unchanged — we only patch the payload blob.

    WHY store feedback inside the payload:
        Adding a dedicated `feedback` column would require an ALTER TABLE
        migration.  Embedding it in the JSON payload is zero-migration and
        consistent with how every other optional field (mint, narrative, etc.)
        is already stored.

    Args:
        record_id:       UUID of the classification to correct.
        correct_type_id: The true CN type ID the user says it should be.
        note:            Optional free-text explanation from the user.

    Returns:
        True on success, False if the record was not found or on DB error.

    Thread-safe: protected by _lock.
    """
    with _lock:
        try:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT payload FROM classifications WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                conn.close()
                return False

            payload = json.loads(row["payload"])
            payload["feedback"] = {
                "correct_type_id": correct_type_id,
                "note":            note,
                "submitted_at":    datetime.now(timezone.utc).isoformat(),
            }
            conn.execute(
                "UPDATE classifications SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), record_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.error("History store add_feedback error for %s: %s", record_id, exc)
            return False


def load_page(skip: int = 0, limit: int = 20) -> list[dict]:
    """
    Return a paginated slice of records ordered newest-first.

    WHY SQL LIMIT/OFFSET instead of Python slice:
        The previous implementation called load_all() and sliced in Python.
        For 10,000 records that means deserialising 10,000 JSON payloads to
        return 20.  LIMIT/OFFSET reads only the requested rows from the
        B-tree, so memory and CPU usage are O(limit) not O(total).

    Args:
        skip:  Number of records to skip (page offset).
        limit: Maximum records to return.

    Returns:
        List of record dicts (already newest-first — no reversal needed).
    """
    with _lock:
        try:
            conn = _get_conn()
            rows = conn.execute(
                """
                SELECT payload FROM classifications
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, skip),
            ).fetchall()
            conn.close()
            return [json.loads(row["payload"]) for row in rows]
        except Exception as exc:
            logger.error("History store load_page error: %s", exc)
            return []


# ── Active Learning API ──────────────────────────────────────────────────────

def get_feedback_candidates() -> list[dict]:
    """
    Return all classification records that have user feedback attached
    and have NOT yet been exported for active-learning retraining.

    WHAT:
        Loads every row from `classifications`, deserialises the JSON payload,
        and returns only those where:
          1. `payload["feedback"]` is present (user clicked "mark as wrong")
          2. `payload["feedback"]["used_for_training"]` is False or absent

    WHY two conditions:
        Condition 1 filters the >95% of records that have no correction.
        Condition 2 prevents re-exporting the same sample in every run.
        Once a sample is exported and the trainer uses it, it must not appear
        in the next batch — otherwise the training set would accumulate
        duplicates and over-fit to the same corrections.

    Returns:
        List of full record dicts ready for export. Each dict contains:
          - id, label, confidence, route_taken, timestamp
          - feedback.correct_type_id  — the curator's correction
          - feedback.note             — optional curator comment
          - cnn.gradcam_path          — heatmap path if available
          - pdf_path                  — path to the report PDF

    PERFORMANCE NOTE:
        This does a full table scan.  For a PFE deployment (<10,000 records),
        this is fine.  For production scale, add:
          CREATE INDEX ix_classifications_has_feedback
          ON classifications (json_extract(payload, '$.feedback')) WHERE ...
        SQLite 3.38+ supports partial indexes — available in Python 3.11's
        bundled SQLite.
    """
    candidates = []
    with _lock:
        try:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT payload FROM classifications ORDER BY timestamp DESC"
            ).fetchall()
            conn.close()
            for row in rows:
                record   = json.loads(row["payload"])
                feedback = record.get("feedback")
                if feedback and not feedback.get("used_for_training", False):
                    candidates.append(record)
        except Exception as exc:
            logger.error("get_feedback_candidates error: %s", exc)
    return candidates


def mark_used_for_training(record_ids: list[str]) -> int:
    """
    Mark a batch of classification records as exported for active-learning.

    WHAT:
        For each record_id in the list:
          1. Loads the payload from the database
          2. Sets `payload["feedback"]["used_for_training"] = True`
          3. Records the export timestamp in `payload["feedback"]["exported_at"]`
          4. Writes the modified payload back via SQL UPDATE

    WHY timestamp the export:
        `exported_at` tells the trainer WHEN this sample entered the
        active-learning dataset.  If two export batches are run, the trainer
        can filter by date to understand which batch each sample came from.
        This is standard practice in Active Learning MLOps pipelines
        (see Google's WIT — What-If Tool).

    Args:
        record_ids: List of UUIDs to mark.  Typically the ids returned by
                    get_feedback_candidates() after the caller has successfully
                    written the export files to disk.

    Returns:
        Number of records actually updated (0 if any are already marked or
        not found — idempotent).

    WHY update IDs one-at-a-time not with IN (?,...):
        SQLite's parameter binding for IN clauses requires dynamic generation
        of placeholder strings ((",?)*n)[1:]).  For the small batch sizes
        expected here (<500 corrections), per-row updates are cleaner and
        safer.  If batch size ever exceeds 1,000 you'd want the IN version.

    Thread-safe: protected by _lock; all updates in one transaction.
    """
    updated = 0
    exported_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        try:
            conn = _get_conn()
            for record_id in record_ids:
                row = conn.execute(
                    "SELECT payload FROM classifications WHERE id = ?",
                    (record_id,),
                ).fetchone()
                if row is None:
                    continue
                payload  = json.loads(row["payload"])
                feedback = payload.get("feedback", {})
                if feedback.get("used_for_training", False):
                    continue   # already marked — idempotent
                feedback["used_for_training"] = True
                feedback["exported_at"]       = exported_at
                payload["feedback"] = feedback
                conn.execute(
                    "UPDATE classifications SET payload = ? WHERE id = ?",
                    (json.dumps(payload, ensure_ascii=False), record_id),
                )
                updated += 1
            conn.commit()
            conn.close()
            logger.info("mark_used_for_training: %d records marked", updated)
        except Exception as exc:
            logger.error("mark_used_for_training error: %s", exc)
    return updated
