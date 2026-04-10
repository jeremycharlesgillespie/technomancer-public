"""Tests for the logging_config module — logger setup and configuration."""

import logging

import pytest

from agent.logging_config import (
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
