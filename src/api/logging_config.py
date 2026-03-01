"""
src/api/logging_config.py
==========================
Structured JSON logging configuration for the FastAPI backend.

WHY JSON logs (P11):
    Plain-text logs like "2026-03-01 19:03:21  INFO  classify: id=..."
    are fine to read manually but break every log aggregation tool:
    - Datadog, Grafana Loki, AWS CloudWatch — all expect JSON
    - Structured fields (id, route, confidence) are queryable in dashboards
    - Regex-parsing of plain text is fragile and maintenance-heavy

JSON log line example:
    {
      "timestamp": "2026-03-01T19:03:24.123Z",
      "level":     "INFO",
      "name":      "src.agents.gatekeeper",
      "message":   "Pipeline complete",
      "route":     "investigator",
      "total_s":   59.12
    }

WHY pythonjsonlogger (not manual json.dumps in every logger call):
    - Inherits Python's standard logging hierarchy (getLogger(__name__) still works)
    - Existing logger.info("msg", key=val) calls are unchanged
    - Format is configurable without touching every module

WHY call at startup only (not module import):
    Libraries that import this module before configure_logging() is called
    should not have their logging silently modified. Calling configure_logging()
    once in the FastAPI lifespan startup guarantees it happens exactly once
    and only when running as a server (not during pytest imports).
"""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """
    Configure the root logger to emit structured JSON lines.

    In development (LOG_FORMAT=text or default), uses a human-readable
    format to keep the terminal readable during local testing.

    In production (LOG_FORMAT=json), switches to pythonjsonlogger's
    JsonFormatter so every log line is a parseable JSON object.

    Controlled by:
        LOG_FORMAT=json   → JSON lines (production / Docker)
        LOG_FORMAT=text   → coloured text (default, dev)
        LOG_LEVEL=DEBUG   → debug verbosity (default: INFO)
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_level  = os.getenv("LOG_LEVEL",  "INFO").upper()

    level = getattr(logging, log_level, logging.INFO)

    if log_format == "json":
        try:
            from pythonjsonlogger.json import JsonFormatter  # type: ignore[import]
        except ImportError:
            from pythonjsonlogger import jsonlogger          # type: ignore[import]
            JsonFormatter = jsonlogger.JsonFormatter         # type: ignore[assignment]

        formatter = JsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
        )
    else:
        # Human-readable text for local development
        formatter = logging.Formatter(
            fmt="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Only configure if no handlers are set (avoids double-logging in tests)
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(level)
    else:
        # Update existing handlers with the new formatter
        for h in root.handlers:
            h.setFormatter(formatter)
        root.setLevel(level)

    # Silence noisy third-party loggers that spam at INFO
    for noisy in ("httpx", "httpcore", "chromadb", "hpack", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
