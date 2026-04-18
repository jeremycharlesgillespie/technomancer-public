"""Tests for aimm.decisions — append-only JSONL audit log."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from aimm import decisions as aimm_decisions
from aimm.decisions import log_decision, read_recent, set_log_path


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    """Redirect the decisions log to a per-test temp file."""
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(aimm_decisions, "LOG_DIR", tmp_path)
    monkeypatch.setattr(aimm_decisions, "LOG_FILE", log_file)
    return log_file


# ---------------------------------------------------------------------------
# log_decision — basic append + schema
# ---------------------------------------------------------------------------

class TestLogDecision:
    def test_three_calls_produce_three_lines(self, _isolated_log):
        """Acceptance: 3 log_decision calls produce a 3-line file,
        each line valid JSON with the required keys."""
        log_decision("TK-1", "approve", "safe category")
        log_decision("TK-2", "archive", "stale")
        log_decision("TK-3", "draft", "new idea")

        lines = _isolated_log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

        required = {"timestamp", "cycle_id", "story_key", "action", "reason", "metadata"}
        for line in lines:
            record = json.loads(line)
            assert required.issubset(record.keys())

    def test_writes_expected_values(self, _isolated_log):
        log_decision(
            "TK-42",
            "approve",
            "matches cat:quality",
            metadata={"labels": ["cat:quality"]},
            cycle_id="cycle-001",
        )
        record = json.loads(_isolated_log.read_text(encoding="utf-8"))
        assert record["story_key"] == "TK-42"
        assert record["action"] == "approve"
        assert record["reason"] == "matches cat:quality"
        assert record["metadata"] == {"labels": ["cat:quality"]}
        assert record["cycle_id"] == "cycle-001"

    def test_timestamp_is_iso_8601(self, _isolated_log):
        log_decision("TK-1", "leave", "no change")
        record = json.loads(_isolated_log.read_text(encoding="utf-8"))
        # Parses without error → valid ISO format.
        datetime.fromisoformat(record["timestamp"])

    def test_default_metadata_is_empty_dict(self, _isolated_log):
        log_decision("TK-1", "approve", "ok")
        record = json.loads(_isolated_log.read_text(encoding="utf-8"))
        assert record["metadata"] == {}

    def test_default_cycle_id_is_empty_string(self, _isolated_log):
        log_decision("TK-1", "approve", "ok")
        record = json.loads(_isolated_log.read_text(encoding="utf-8"))
        assert record["cycle_id"] == ""

    def test_rate_limited_action_accepted(self, _isolated_log):
        """Rate-limited skips are audit events too."""
        log_decision("TK-1", "rate-limited", "budget exhausted")
        record = json.loads(_isolated_log.read_text(encoding="utf-8"))
        assert record["action"] == "rate-limited"

    def test_appends_rather_than_overwrites(self, _isolated_log):
        log_decision("TK-1", "approve", "first")
        log_decision("TK-2", "approve", "second")
        lines = _isolated_log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["story_key"] == "TK-1"
        assert json.loads(lines[1])["story_key"] == "TK-2"

    def test_creates_parent_directory_if_missing(self, tmp_path, monkeypatch):
        nested = tmp_path / "does" / "not" / "exist" / "decisions.jsonl"
        monkeypatch.setattr(aimm_decisions, "LOG_FILE", nested)
        monkeypatch.setattr(aimm_decisions, "LOG_DIR", nested.parent)

        log_decision("TK-1", "approve", "ok")
        assert nested.exists()


# ---------------------------------------------------------------------------
# log_decision — resilience (acceptance: never raises)
# ---------------------------------------------------------------------------

class TestLogDecisionNeverRaises:
    def test_swallows_oserror_from_open(self, _isolated_log):
        """If the filesystem refuses to open the file, we log and move on."""
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            log_decision("TK-1", "approve", "ok")

    def test_swallows_non_json_serializable_metadata(self, _isolated_log):
        class NotSerializable:
            pass

        # metadata contains an object json.dumps cannot handle.
        log_decision("TK-1", "approve", "ok", metadata={"bad": NotSerializable()})

        # Nothing should have been written.
        assert not _isolated_log.exists() or _isolated_log.read_text() == ""

    def test_swallows_mkdir_oserror(self, _isolated_log):
        with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
            log_decision("TK-1", "approve", "ok")


# ---------------------------------------------------------------------------
# read_recent
# ---------------------------------------------------------------------------

class TestReadRecent:
    def test_returns_last_two_newest_first(self, _isolated_log):
        """Acceptance: read_recent(2) returns the last 2 records newest-first."""
        log_decision("TK-1", "approve", "first")
        log_decision("TK-2", "approve", "second")
        log_decision("TK-3", "approve", "third")

        recent = list(read_recent(2))
        assert len(recent) == 2
        assert recent[0]["story_key"] == "TK-3"
        assert recent[1]["story_key"] == "TK-2"

    def test_missing_file_yields_nothing(self, _isolated_log):
        assert not _isolated_log.exists()
        assert list(read_recent(10)) == []

    def test_zero_or_negative_n_yields_nothing(self, _isolated_log):
        log_decision("TK-1", "approve", "ok")
        assert list(read_recent(0)) == []
        assert list(read_recent(-5)) == []

    def test_n_larger_than_records_returns_all(self, _isolated_log):
        log_decision("TK-1", "approve", "a")
        log_decision("TK-2", "approve", "b")
        recent = list(read_recent(100))
        assert len(recent) == 2
        assert [r["story_key"] for r in recent] == ["TK-2", "TK-1"]

    def test_skips_malformed_lines(self, _isolated_log):
        log_decision("TK-1", "approve", "good")
        with _isolated_log.open("a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
        log_decision("TK-2", "approve", "good2")

        recent = list(read_recent(10))
        assert len(recent) == 2
        assert recent[0]["story_key"] == "TK-2"
        assert recent[1]["story_key"] == "TK-1"

    def test_skips_blank_lines(self, _isolated_log):
        log_decision("TK-1", "approve", "ok")
        with _isolated_log.open("a", encoding="utf-8") as fh:
            fh.write("\n   \n")
        log_decision("TK-2", "approve", "ok")

        recent = list(read_recent(10))
        assert [r["story_key"] for r in recent] == ["TK-2", "TK-1"]


# ---------------------------------------------------------------------------
# set_log_path
# ---------------------------------------------------------------------------

class TestSetLogPath:
    def test_redirects_writes(self, tmp_path):
        target = tmp_path / "custom" / "audit.jsonl"
        set_log_path(target)
        try:
            log_decision("TK-1", "approve", "ok")
            assert target.exists()
            record = json.loads(target.read_text(encoding="utf-8"))
            assert record["story_key"] == "TK-1"
        finally:
            # Restore module-level defaults so the autouse fixture in the
            # next test doesn't inherit our redirect.
            set_log_path(Path(aimm_decisions.__file__).parent / "decisions.jsonl")

    def test_creates_missing_parent_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "audit.jsonl"
        assert not target.parent.exists()
        set_log_path(target)
        try:
            assert target.parent.is_dir()
        finally:
            set_log_path(Path(aimm_decisions.__file__).parent / "decisions.jsonl")

    def test_accepts_string_path(self, tmp_path):
        target = tmp_path / "as_str.jsonl"
        set_log_path(str(target))
        try:
            log_decision("TK-1", "approve", "ok")
            assert target.exists()
        finally:
            set_log_path(Path(aimm_decisions.__file__).parent / "decisions.jsonl")
