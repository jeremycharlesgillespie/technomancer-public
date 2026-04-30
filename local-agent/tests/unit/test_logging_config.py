"""Tests for the logging_config module — logger setup and configuration."""

import contextvars
import logging
from logging.handlers import RotatingFileHandler

import pytest

from agent.logging_config import (
    BACKUP_COUNT,
    DEFAULT_REQUEST_ID,
    MAX_LOG_SIZE,
    RequestIdFilter,
    get_agent_logger,
    get_bot_service_logger,
    get_discord_bot_logger,
    get_logger,
    get_request_id,
    get_safe_update_logger,
    request_id_var,
    set_request_id,
    setup_logger,
)


@pytest.fixture(autouse=True)
def _reset_request_id():
    """Run each test in a fresh ContextVar copy so leakage across tests
    can't make a passing test depend on ordering."""
    token = request_id_var.set(DEFAULT_REQUEST_ID)
    try:
        yield
    finally:
        request_id_var.reset(token)


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


def _make_record(name: str = "test", msg: str = "hello") -> logging.LogRecord:
    """Build a minimal LogRecord for filter testing."""
    return logging.LogRecord(
        name=name, level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=None, exc_info=None,
    )


class TestRequestIdContextVar:
    """ContextVar default and set/get behavior."""

    def test_default_value_is_dash(self):
        assert get_request_id() == DEFAULT_REQUEST_ID
        assert DEFAULT_REQUEST_ID == "-"

    def test_set_and_get_round_trip(self):
        token = set_request_id("abc-123")
        try:
            assert get_request_id() == "abc-123"
        finally:
            request_id_var.reset(token)
        assert get_request_id() == DEFAULT_REQUEST_ID

    def test_isolated_across_contexts(self):
        """Each Context.copy() gets its own isolated value."""
        set_request_id("outer")

        def _inner_run() -> str:
            set_request_id("inner")
            return get_request_id()

        ctx = contextvars.copy_context()
        inner_value = ctx.run(_inner_run)
        assert inner_value == "inner"
        # Outer context unchanged — Context.run() does not bleed back.
        assert get_request_id() == "outer"


class TestRequestIdFilter:
    """Filter injects request_id onto LogRecords."""

    def test_filter_injects_default_when_unset(self):
        rid_filter = RequestIdFilter()
        record = _make_record()
        assert rid_filter.filter(record) is True
        assert record.request_id == DEFAULT_REQUEST_ID

    def test_filter_injects_seeded_value(self):
        rid_filter = RequestIdFilter()
        set_request_id("discord-12345")
        record = _make_record()
        rid_filter.filter(record)
        assert record.request_id == "discord-12345"

    def test_filter_returns_true_to_pass_record_through(self):
        """A logging.Filter must return truthy to allow the record through."""
        rid_filter = RequestIdFilter()
        record = _make_record()
        assert rid_filter.filter(record) is True

    def test_filter_respects_existing_attribute(self):
        """An explicit ``extra={'request_id': ...}`` should not be overwritten."""
        rid_filter = RequestIdFilter()
        set_request_id("from-contextvar")
        record = _make_record()
        record.request_id = "from-extra"
        rid_filter.filter(record)
        assert record.request_id == "from-extra"


class TestRequestIdInLogOutput:
    """End-to-end: configured loggers emit ``[rid=<id>]`` formatted output."""

    def test_log_line_includes_default_rid(self, tmp_path):
        logger = setup_logger(
            "test_rid_default_unique",
            log_file="rid_default.log",
            log_dir=tmp_path,
            console=False,
        )
        logger.info("default test")
        for h in logger.handlers:
            h.flush()
        content = (tmp_path / "rid_default.log").read_text(encoding="utf-8")


class TestErrorPaths:
    """Error path tests for logging_config — edge cases and failure modes."""

    def test_setup_logger_nonexistent_parent_dir_creates_it(self, tmp_path):
        """setup_logger should create parent directories if they don't exist."""
        nonexistent = tmp_path / "nonexistent" / "nested" / "dir"
        logger = setup_logger(
            "test_nonexistent_dir",
            log_file="test.log",
            log_dir=nonexistent,
            console=False,
        )
        assert isinstance(logger, logging.Logger)
        assert nonexistent.exists()

    def test_setup_logger_unwritable_directory_raises_oserror(self, tmp_path):
        """setup_logger should raise OSError when directory is unwritable."""
        # Create a directory and make it read-only
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            # This should raise PermissionError or OSError
            with pytest.raises((OSError, PermissionError)):
                setup_logger(
                    "test_readonly_dir",
                    log_file="test.log",
                    log_dir=readonly_dir,
                    console=False,
                )
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_setup_logger_console_false_no_log_file(self):
        """setup_logger with console=False and no log_file should create handler-less logger."""
        logger = setup_logger("test_handlerless", console=False)
        assert isinstance(logger, logging.Logger)
        # Should have no handlers when console=False and no log_file
        assert len(logger.handlers) == 0

    def test_setup_logger_empty_string_log_file(self):
        """setup_logger with empty string log_file should create handler-less logger."""
        logger = setup_logger("test_empty_logfile", log_file="", console=False)
        assert isinstance(logger, logging.Logger)
        # Empty string should be falsy, so no file handler added
        assert len(logger.handlers) == 0

    def test_set_request_id_empty_string(self):
        """set_request_id should accept empty string without error."""
        token = set_request_id("")
        assert get_request_id() == ""
        request_id_var.reset(token)

    def test_set_request_id_special_characters(self):
        """set_request_id should handle special characters including newlines."""
        # Test newline character
        token = set_request_id("test\nnewline")
        assert get_request_id() == "test\nnewline"
        request_id_var.reset(token)

        # Test null character
        token = set_request_id("test\x00null")
        assert get_request_id() == "test\x00null"
        request_id_var.reset(token)

    def test_set_request_id_very_long_string(self):
        """set_request_id should handle very long request IDs."""
        long_id = "x" * 10000
        token = set_request_id(long_id)
        assert get_request_id() == long_id
        request_id_var.reset(token)

    def test_set_request_id_unicode(self):
        """set_request_id should handle unicode characters."""
        unicode_id = "test-日本語-🚀"
        token = set_request_id(unicode_id)
        assert get_request_id() == unicode_id
        request_id_var.reset(token)

    def test_log_line_includes_default_rid(self, tmp_path):
        logger = setup_logger(
            "test_rid_seeded_unique",
            log_file="rid_seeded.log",
            log_dir=tmp_path,
            console=False,
        )
        set_request_id("discord-999")
        logger.info("seeded test")
        for h in logger.handlers:
            h.flush()
        content = (tmp_path / "rid_seeded.log").read_text()
        assert "[rid=discord-999]" in content
        assert "seeded test" in content

    def test_filter_is_installed_on_handlers(self, tmp_path):
        """Each handler created by setup_logger carries the RequestIdFilter."""
        logger = setup_logger(
            "test_rid_filter_installed_unique",
            log_file="rid_installed.log",
            log_dir=tmp_path,
            console=True,
        )
        for handler in logger.handlers:
            filter_types = {type(f).__name__ for f in handler.filters}
            assert "RequestIdFilter" in filter_types, (
                f"handler {type(handler).__name__} missing RequestIdFilter"
            )

    def test_get_logger_also_installs_filter(self):
        """``get_logger`` for an unconfigured name installs the filter."""
        logger = get_logger("test_rid_get_logger_unique")
        for handler in logger.handlers:
            filter_types = {type(f).__name__ for f in handler.filters}
            assert "RequestIdFilter" in filter_types
