"""Tests for the logging_config module — logger setup and configuration."""

import logging
from logging.handlers import RotatingFileHandler

import pytest

from agent.logging_config import (
    BACKUP_COUNT,
    MAX_LOG_SIZE,
    get_agent_logger,
    get_bot_service_logger,
    get_discord_bot_logger,
    get_logger,
    get_safe_update_logger,
    setup_logger,
)


class TestSetupLogger:
    """Test logger creation and configuration."""

    def test_creates_logger(self):
        logger = setup_logger("test_basic", console=True)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_basic"
        assert logger.level == logging.INFO

    def test_console_handler_added(self):
        # Use unique name to avoid handler reuse
        logger = setup_logger("test_console_handler_unique", console=True)
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types

    def test_file_handler_added(self, tmp_path):
        logger = setup_logger(
            "test_file_handler",
            log_file="test.log",
            log_dir=tmp_path,
            console=False,
        )
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "RotatingFileHandler" in handler_types

    def test_custom_level(self):
        logger = setup_logger("test_level_unique", level=logging.DEBUG, console=True)
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        logger = setup_logger("test_nodupe_unique", console=True)
        handler_count = len(logger.handlers)
        # Call again — should not add more handlers
        logger2 = setup_logger("test_nodupe_unique", console=True)
        assert len(logger2.handlers) == handler_count

    def test_writes_to_file(self, tmp_path):
        logger = setup_logger(
            "test_file_write_unique",
            log_file="output.log",
            log_dir=tmp_path,
            console=False,
        )
        logger.info("Test message")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        log_content = (tmp_path / "output.log").read_text()
        assert "Test message" in log_content


class TestGetLogger:
    """Test get_logger retrieval."""

    def test_returns_logger(self):
        logger = get_logger("test_get_unique")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_unique"

    def test_auto_creates_handler(self):
        logger = get_logger("test_auto_handler_unique")
        assert len(logger.handlers) >= 1


class TestPreConfiguredLoggers:
    """Test pre-configured logger factories."""

    def test_bot_service_logger(self):
        logger = get_bot_service_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "bot_service"

    def test_safe_update_logger(self):
        logger = get_safe_update_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "safe_update"

    def test_discord_bot_logger(self):
        logger = get_discord_bot_logger()
        assert isinstance(logger, logging.Logger)

    def test_agent_logger(self):
        logger = get_agent_logger()
        assert isinstance(logger, logging.Logger)


class TestRotatingFileHandler:
    """Verify RotatingFileHandler is configured and actually rotates."""

    def test_rotating_handler_params(self, tmp_path):
        """Handler type is RotatingFileHandler with the specified parameters."""
        logger = setup_logger(
            "test_rotating_params_unique",
            log_file="params.log",
            log_dir=tmp_path,
            console=False,
        )
        rotating_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating_handlers) == 1
        handler = rotating_handlers[0]
        assert handler.maxBytes == 10_000_000
        assert handler.backupCount == 5
        assert handler.encoding == "utf-8"

    def test_module_constants(self):
        """Module constants match the spec (10MB / 5 backups)."""
        assert MAX_LOG_SIZE == 10_000_000
        assert BACKUP_COUNT == 5

    def test_rotating_handler_configured(self, tmp_path):
        """Writing >10MB of log lines triggers rotation; agent.log.1 exists and
        the primary file stays under the configured cap."""
        logger = setup_logger(
            "test_rotating_triggers_unique",
            log_file="agent.log",
            log_dir=tmp_path,
            console=False,
        )
        # Write slightly more than 10MB of log content. Each line is ~130 bytes
        # after formatting, so 100k lines of 100-byte payload comfortably
        # crosses the 10MB threshold.
        payload = "x" * 100
        for i in range(100_000):
            logger.info("%d %s", i, payload)
        for h in logger.handlers:
            h.flush()

        primary = tmp_path / "agent.log"
        backup = tmp_path / "agent.log.1"

        assert primary.exists()
        assert backup.exists(), "expected at least one backup file (agent.log.1)"
        # Primary must be under the cap. Allow a small formatter-overhead
        # margin since rotation is checked after each record is emitted.
        assert primary.stat().st_size <= 10_000_000 + 4096
