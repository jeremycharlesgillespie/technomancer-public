"""
Structured JSON Logging — JSON-per-line rotating file handler.

Attaches a :class:`logging.handlers.RotatingFileHandler` to the root logger
that writes one JSON object per line to ``logs/executor.jsonl`` so the
executor's log stream can be indexed, grep'd, and shipped to downstream
tooling without parsing free-form text.

The installer is idempotent: repeated calls detect the existing handler
(marked via a sentinel attribute) and return it unchanged. The ``logs/``
directory is created on first call and a pre-existing directory is fine.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# 50 MB per file x 5 backups → ~250 MB cap on disk.
DEFAULT_MAX_BYTES: int = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT: int = 5

DEFAULT_LOG_DIR: Path = Path(__file__).parent.parent / "logs"
DEFAULT_LOG_FILE: str = "executor.jsonl"

_INSTALLED_ATTR = "_tk_structured_handler"

_RESERVED_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class JSONFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` instances as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        run_id = getattr(record, "run_id", None)
        if run_id and run_id != "-":
            payload["run_id"] = run_id
        idea_key = getattr(record, "idea_key", None)
        if idea_key and idea_key != "-":
            payload["idea_key"] = idea_key

        for key, value in record.__dict__.items():
            if (
                key in _RESERVED_RECORD_FIELDS
                or key in payload
                or key.startswith("_")
                or key in ("run_id", "idea_key")
            ):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


def install_file_handler(
    log_dir: Path | str | None = None,
    filename: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    level: int = logging.INFO,
) -> RotatingFileHandler:
    """Attach a rotating JSON-lines file handler to the root logger.

    Idempotent — returns the already-installed handler on repeated calls.
    Creates the target directory on first call; tolerates one that
    already exists.

    Args:
        log_dir: Directory for the log file. Defaults to ``local-agent/logs``.
        filename: Log file name. Defaults to ``executor.jsonl``.
        max_bytes: Rotation threshold in bytes.
        backup_count: Number of rotated backups to retain.
        level: Minimum level for this handler.

    Returns:
        The installed :class:`RotatingFileHandler`.
    """
    log_dir_path = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    log_file = filename or DEFAULT_LOG_FILE
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    for existing in root.handlers:
        if getattr(existing, _INSTALLED_ATTR, False):
            return existing  # type: ignore[return-value]

    handler = RotatingFileHandler(
        log_dir_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    setattr(handler, _INSTALLED_ATTR, True)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())

    # Ensure root passes records of this level through to handlers.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    root.addHandler(handler)
    return handler
