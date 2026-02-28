"""
src/api/_store.py
==================
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
