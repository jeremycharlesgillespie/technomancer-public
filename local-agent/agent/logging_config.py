"""
Centralized logging configuration using Python's stdlib logging.

Provides consistent logging across all modules with:
- File rotation to prevent unbounded log growth
- Console output for interactive use
- Per-module loggers for granular control
- Structured format with timestamps and levels
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default log directory (can be overridden)
DEFAULT_LOG_DIR = Path(__file__).parent.parent  # local-agent/

# Log format: timestamp, level, logger name, message
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotation settings
MAX_LOG_SIZE = 10_000_000  # 10 MB per file
BACKUP_COUNT = 5  # Keep 5 old log files


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
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = (log_dir or DEFAULT_LOG_DIR) / log_file
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
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
        logger.addHandler(handler)
    return logger


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
