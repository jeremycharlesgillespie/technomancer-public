"""Tests for agent.structured_logging — JSON formatter and rotating file handler."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from agent.structured_logging import (
    DEFAULT_LOG_FILE,
    JSONFormatter,
    _INSTALLED_ATTR,
    install_file_handler,
)


@pytest.fixture(autouse=True)
def _clean_root_handler():
    """Remove any handler installed by install_file_handler before & after each test.

    install_file_handler mutates the global root logger; isolation is
    mandatory or state leaks across tests.
    """
    root = logging.getLogger()
    orig_level = root.level

    def _prune():
        for h in list(root.handlers):
            if getattr(h, _INSTALLED_ATTR, False):
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

    _prune()
    yield
    _prune()
    root.setLevel(orig_level)


def _make_record(**overrides) -> logging.LogRecord:
    defaults = {
        "name": "test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "hello",
        "args": (),
        "exc_info": None,
    }
    defaults.update(overrides)
    return logging.LogRecord(**defaults)


class TestJSONFormatter:
    def test_produces_valid_json(self):
        record = _make_record(msg="hello %s", args=("world",))
        line = JSONFormatter().format(record)
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test"
        assert payload["message"] == "hello world"
        assert "timestamp" in payload

    def test_includes_run_id_and_idea_key_when_set(self):
        record = _make_record()
        record.run_id = "r1"
        record.idea_key = "TK-42"
        payload = json.loads(JSONFormatter().format(record))
        assert payload["run_id"] == "r1"
        assert payload["idea_key"] == "TK-42"

    def test_omits_run_id_defaults(self):
        record = _make_record()
        record.run_id = "-"
        record.idea_key = "-"
        payload = json.loads(JSONFormatter().format(record))
        assert "run_id" not in payload
        assert "idea_key" not in payload

    def test_includes_extra_fields(self):
        record = _make_record()
        record.custom_field = {"a": 1}
        payload = json.loads(JSONFormatter().format(record))
        assert payload["custom_field"] == {"a": 1}

    def test_coerces_non_serializable(self):
        record = _make_record()
        record.obj = object()
        payload = json.loads(JSONFormatter().format(record))
        assert "obj" in payload
        assert isinstance(payload["obj"], str)

    def test_includes_exception_info(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record(level=logging.ERROR, msg="failed", exc_info=sys.exc_info())
        payload = json.loads(JSONFormatter().format(record))
        assert "ValueError" in payload["exc_info"]
        assert "boom" in payload["exc_info"]


class TestInstallFileHandler:
    def test_creates_logs_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        install_file_handler(log_dir=log_dir)
        assert log_dir.is_dir()

    def test_tolerates_pre_existing_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "leftover.txt").write_text("unrelated")
        handler = install_file_handler(log_dir=log_dir)
        assert handler is not None
        assert (log_dir / "leftover.txt").exists()

    def test_idempotent_attaches_only_once(self, tmp_path):
        log_dir = tmp_path / "logs"
        h1 = install_file_handler(log_dir=log_dir)
        h2 = install_file_handler(log_dir=log_dir)
        assert h1 is h2
        root = logging.getLogger()
        installed = [h for h in root.handlers if getattr(h, _INSTALLED_ATTR, False)]
        assert len(installed) == 1

    def test_writes_valid_json_per_line(self, tmp_path):
        log_dir = tmp_path / "logs"
        install_file_handler(log_dir=log_dir)
        logger = logging.getLogger("tk.structured.test")
        logger.info("first message")
        logger.warning("second message")
        for h in logging.getLogger().handlers:
            h.flush()

        log_file = log_dir / DEFAULT_LOG_FILE
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        # At minimum our two messages should appear; other stray root-logger
        # traffic is fine as long as every line is valid JSON.
        parsed = [json.loads(line) for line in lines]
        messages = [p["message"] for p in parsed]
        assert "first message" in messages
        assert "second message" in messages
        first = next(p for p in parsed if p["message"] == "first message")
        second = next(p for p in parsed if p["message"] == "second message")
        assert first["level"] == "INFO"
        assert second["level"] == "WARNING"

    def test_rotates_at_size_limit(self, tmp_path):
        log_dir = tmp_path / "logs"
        install_file_handler(log_dir=log_dir, max_bytes=500, backup_count=3)
        logger = logging.getLogger("tk.structured.rotation")
        for i in range(200):
            logger.info("line %d with filler content to fatten the record size", i)
        for h in logging.getLogger().handlers:
            h.flush()

        base = log_dir / DEFAULT_LOG_FILE
        backup = log_dir / f"{DEFAULT_LOG_FILE}.1"
        assert base.exists(), "primary log file missing after rotation"
        assert backup.exists(), "rotation did not produce a .1 backup"
        # Backup count caps retained files at 3 rolled + the primary.
        all_rotated = sorted(log_dir.glob(f"{DEFAULT_LOG_FILE}.*"))
        assert len(all_rotated) <= 3
