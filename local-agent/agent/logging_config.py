"""
Centralized logging configuration using Python's stdlib logging.

Provides consistent logging across all modules with:
- File rotation to prevent unbounded log growth
- Console output for interactive use
- Per-module loggers for granular control
- Structured format with timestamps and levels
- Request-id correlation via a ``ContextVar`` so concurrent flows
  (Discord on_message, news_digest, executor runs) can be reconstructed
  end-to-end across core → tool → Claude boundaries.
"""

import contextvars
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default log directory (can be overridden)
DEFAULT_LOG_DIR = Path(__file__).parent.parent  # local-agent/

# Log format: timestamp, role, model, level, request id, logger name, message.
# ``%(role)s`` and ``%(model)s`` are supplied by ``_RoleModelFilter`` below.
# ``%(request_id)s`` is supplied by ``RequestIdFilter`` below.
LOG_FORMAT = "%(asctime)s [%(role)s][%(model)s] [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotation settings
MAX_LOG_SIZE = 10_000_000  # 10 MB per file
BACKUP_COUNT = 5  # Keep 5 old log files

# Default value for an unseeded ContextVar — also injected onto LogRecords
# by ``RequestIdFilter`` whenever no caller has set the ContextVar.
DEFAULT_REQUEST_ID = "-"

# Default value for role and model when unset
DEFAULT_ROLE = "-"
DEFAULT_MODEL = "-"

# Process-wide ContextVar for the active request id. Coroutines, threads
# created via ``asyncio.to_thread`` (which calls ``contextvars.copy_context``),
# and ThreadPoolExecutor workers spawned with ``contextvars.copy_context().run``
# all inherit the value automatically.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=DEFAULT_REQUEST_ID
)

# Process-wide ContextVars for role and model.
# Coroutines, threads, and subprocesses all inherit the value automatically.
role_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_role", default=DEFAULT_ROLE
)
model_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "log_model", default=DEFAULT_MODEL
)


def get_request_id() -> str:
    """Return the current request id, or ``DEFAULT_REQUEST_ID`` when unset."""
    return request_id_var.get()


def set_request_id(value: str) -> contextvars.Token:
    """Set the current request id. Returns the token so callers can ``reset``."""
    return request_id_var.set(value)


class RequestIdFilter(logging.Filter):
    """Inject ``request_id`` onto every ``LogRecord`` passing through.

    Installed on every handler created by :func:`setup_logger` and
    :func:`get_logger` so the ``%(request_id)s`` formatter token never
    raises ``KeyError``, even for records that originate from loggers that
    were configured elsewhere and propagated up to a handler we own.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Respect a value already attached to the record (e.g. via
        # ``logger.info(..., extra={"request_id": ...})``); otherwise pull
        # from the ContextVar, falling back to the sentinel default.
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class _RoleModelFilter(logging.Filter):
    """Inject ``role`` and ``model`` onto every ``LogRecord`` passing through.

    Installed on every handler created by :func:`setup_logger` and
    :func:`get_logger` so the ``%(role)s`` and ``%(model)s`` formatter tokens
    never raise ``KeyError``, even for records that originate from loggers that
    were configured elsewhere and propagated up to a handler we own.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Respect values already attached to the record; otherwise pull from
        # the ContextVars, falling back to the sentinel defaults.
        if not hasattr(record, "role"):
            record.role = role_var.get()
        if not hasattr(record, "model"):
            record.model = model_var.get()
        return True


# Singleton filters — handlers can share one instance.
_request_id_filter = RequestIdFilter()
_role_model_filter = _RoleModelFilter()


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (usually module name like 'bot_service')
        log_file: Log file name (e.g., 'service.log'). If None, logs to console only.
        level: Logging level (default INFO)
        log_dir: Directory for log files. Defaults to local-agent/
        console: Whether to also log to console (default True)

    Returns:
        Configured logger instance

    Example:
        logger = setup_logger('bot_service', 'service.log')
        logger.info('Bot started')
        logger.error('Connection failed', exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_request_id_filter)
        console_handler.addFilter(_role_model_filter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = (log_dir or DEFAULT_LOG_DIR) / log_file
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_request_id_filter)
        file_handler.addFilter(_role_model_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    If the logger doesn't exist, creates a basic console-only logger.
    For full configuration, use setup_logger() first.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set up basic console logging if not configured
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handler.addFilter(_request_id_filter)
        handler.addFilter(_role_model_filter)
        logger.addHandler(handler)
    return logger


def set_role(role: str) -> None:
    """Bind the role that every subsequent log record inherits in this
    thread/task.

    Args:
        role: The role name (e.g., "AIM", "AIW", "AIV", "AIMM", "BOT", "HUB").
    """
    role_var.set(role)


def set_model(model: str | None) -> None:
    """Bind the LLM model for records emitted inside the current call.

    Args:
        model: The model name (e.g., "claude-sonnet-4-6", "ollama:qwen3.5:27b").
               Pass ``None`` to unset (default ``"-"``).
    """
    model_var.set(model or DEFAULT_MODEL)


# Pre-configured loggers for main modules
def get_bot_service_logger() -> logging.Logger:
    """Get logger for bot_service.py."""
    return setup_logger("bot_service", "service.log")


def get_safe_update_logger() -> logging.Logger:
    """Get logger for safe_update.py."""
    return setup_logger("safe_update", "safe_update.log")


def get_discord_bot_logger() -> logging.Logger:
    """Get logger for discord_memory_bot.py."""
    return setup_logger("discord_bot", "discord_bot.log")


def get_agent_logger() -> logging.Logger:
    """Get logger for core agent."""
    return setup_logger("agent", "agent.log")
