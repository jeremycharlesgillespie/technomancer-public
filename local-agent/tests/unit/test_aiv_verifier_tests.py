"""Tests for aiv.verifiers.tests_only — executor pytest-log tail verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiv.verifiers.tests_only import MAX_TAIL_CHARS, capture


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCaptureSuccess:
    def test_small_log_returned_verbatim(self, tmp_path):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        log = logs_dir / "TK-688.log"
        log.write_text("== 5 passed in 1.23s ==\n", encoding="utf-8")

        result = capture("TK-688", logs_dir=logs_dir)

        assert result["exit_code"] == 0
        assert result["pytest_output"] == "== 5 passed in 1.23s ==\n"
        assert result["log_path"] == str(log)
        assert "error" not in result

    def test_tail_matches_last_8k_of_large_log(self, tmp_path):
        """Acceptance criterion: fake execution_logs/TK-X.log → returned
        tail matches the file's last 8K."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        log = logs_dir / "TK-100.log"

        # Build content that's clearly larger than the cap so truncation
        # is observable, with a unique prefix we expect to be dropped
        # and a unique suffix we expect to be kept.
        prefix = "PREFIX-DROPPED\n"
        filler = "x" * (MAX_TAIL_CHARS * 2)
        suffix = "SUFFIX-KEPT\n"
        log.write_text(prefix + filler + suffix, encoding="utf-8")

        result = capture("TK-100", logs_dir=logs_dir)

        assert result["exit_code"] == 0
        assert len(result["pytest_output"]) == MAX_TAIL_CHARS
        # Matches the raw file tail byte-for-byte.
        full_text = log.read_text(encoding="utf-8")
        assert result["pytest_output"] == full_text[-MAX_TAIL_CHARS:]
        assert "SUFFIX-KEPT" in result["pytest_output"]
        assert "PREFIX-DROPPED" not in result["pytest_output"]

    def test_log_exactly_at_cap_returned_whole(self, tmp_path):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        log = logs_dir / "TK-77.log"
        content = "y" * MAX_TAIL_CHARS
        log.write_text(content, encoding="utf-8")

        result = capture("TK-77", logs_dir=logs_dir)

        assert result["pytest_output"] == content
        assert len(result["pytest_output"]) == MAX_TAIL_CHARS

    def test_empty_log_file(self, tmp_path):
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        (logs_dir / "TK-0.log").write_text("", encoding="utf-8")

        result = capture("TK-0", logs_dir=logs_dir)

        assert result["exit_code"] == 0
        assert result["pytest_output"] == ""
        assert "error" not in result


# ---------------------------------------------------------------------------
# Failure paths — must return a dict, never raise
# ---------------------------------------------------------------------------


class TestCaptureMissingLog:
    def test_missing_log_returns_error_dict(self, tmp_path):
        """Acceptance criterion: missing log file → dict with error, not raise."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()

        result = capture("TK-404", logs_dir=logs_dir)

        assert result["error"] == "log not found"
        assert result["exit_code"] == -1
        assert result["log_path"] == str(logs_dir / "TK-404.log")
        assert "pytest_output" not in result

    def test_missing_logs_dir_returns_error_dict(self, tmp_path):
        """Even if the whole directory doesn't exist, don't raise."""
        nonexistent = tmp_path / "never_made" / "execution_logs"

        result = capture("TK-999", logs_dir=nonexistent)

        assert result["error"] == "log not found"
        assert result["exit_code"] == -1
        assert str(nonexistent) in result["log_path"]


class TestCaptureDefaultLogsDir:
    def test_default_dir_used_when_override_omitted(self, monkeypatch, tmp_path):
        """With no ``logs_dir`` override, the module-level default is used."""
        from aiv.verifiers import tests_only

        monkeypatch.setattr(tests_only, "EXECUTION_LOGS_DIR", tmp_path)
        (tmp_path / "TK-42.log").write_text("hello", encoding="utf-8")

        result = tests_only.capture("TK-42")

        assert result["pytest_output"] == "hello"
        assert result["exit_code"] == 0


class TestCaptureNonUtf8Bytes:
    def test_invalid_utf8_does_not_raise(self, tmp_path):
        """Executor logs are plain ASCII in practice, but a stray non-UTF8
        byte (e.g. a Windows console codepage artifact) must not crash
        the verifier."""
        logs_dir = tmp_path / "execution_logs"
        logs_dir.mkdir()
        log = logs_dir / "TK-bin.log"
        log.write_bytes(b"before\xff\xfeafter")

        result = capture("TK-bin", logs_dir=logs_dir)

        assert result["exit_code"] == 0
        assert "before" in result["pytest_output"]
        assert "after" in result["pytest_output"]
