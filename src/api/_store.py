"""
src/api/_store.py
==================
Thread-safe JSON file store for classification history.

WHY a file store and not a database (yet):
    PostgreSQL is Layer 6. Until then, a JSON file gives us persistence
    without any infrastructure dependency. When we migrate to PostgreSQL,
    only this module changes — the routes stay identical.
    This is the Repository Pattern: the data layer is isolated behind a
    simple interface (append / load_all / get_by_id). Swapping the backend
    means rewriting one 30-line module, not hunting through routes.

WHY threading.Lock:
    FastAPI workers share the same process. Multiple requests could
    call append() simultaneously — one would read the file, the other
    would also read the same file before the first has written, and the
    first write would be silently overwritten.
    The lock ensures only one thread reads-then-writes at a time.
    This is NOT needed for reads because Python dict iteration is GIL-safe,
    but we lock reads too for consistency.

NOTE: This is single-process safe. For multi-worker (workers > 1) deployments,
    replace with PostgreSQL (see Layer 6). Our workers=1 constraint in main.py
    means this is safe for production until Layer 6.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT         = Path(__file__).resolve().parent.parent.parent
_HISTORY_FILE = _ROOT / "data" / "history.json"
_lock         = threading.Lock()


def ensure_store() -> None:
    """Create the history file if it does not exist. Called at startup."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _HISTORY_FILE.exists():
        _HISTORY_FILE.write_text("[]", encoding="utf-8")
        logger.info("History store created: %s", _HISTORY_FILE)


def append(record: dict) -> None:
    """
    Append one record to history.

    WHAT: Reads the full JSON file, appends the record, writes it back.
    WHY not append-only: JSON arrays cannot be appended incrementally.
    WHY not a DB yet: See module docstring. Repository Pattern isolates this.
    Thread-safe: protected by _lock.
    """
    with _lock:
        records = _load_raw()
        records.append(record)
        _HISTORY_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_all() -> list[dict]:
    """Return all records. Thread-safe read."""
    with _lock:
        return _load_raw()


def get_by_id(record_id: str) -> Optional[dict]:
    """Return one record by UUID, or None if not found."""
    with _lock:
        for r in _load_raw():
            if r.get("id") == record_id:
                return r
    return None


def _load_raw() -> list[dict]:
    """
    Read and parse history.json.
    Returns [] on any read/parse error so the API never crashes on a corrupt file.
    Caller must hold _lock.
    """
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("History file corrupt, returning empty: %s", exc)
        return []
