"""Tests for GET /api/memory/integrity — memory compaction health snapshot."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from idea_board.web import _collect_memory_integrity, app


REQUIRED_FIELDS = (
    "last_compaction_at",
    "last_verify_passed",
    "backup_count_hourly",
    "backup_count_daily",
    "newest_backup_age_seconds",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def stub_vault(tmp_path, monkeypatch):
    """Point settings.vault_path at tmp_path for the route under test.

    Returns (vault_root, backups_root, crash_log_path) with directories
    pre-created so individual tests only have to add files.
    """
    from idea_board import web as web_module

    monkeypatch.setattr(web_module.settings, "vault_path", tmp_path)
    backups_root = tmp_path / "Backups" / "memory"
    backups_root.mkdir(parents=True)
    crash_log = tmp_path / "LLM Memory" / "Permanent" / "crash_log.md"
    crash_log.parent.mkdir(parents=True)
    return tmp_path, backups_root, crash_log


def _make_snapshot(
    backups_root: Path,
    name: str,
    *,
    with_hourly: bool = True,
    with_daily: bool = True,
    with_memories: bool = False,
    mtime: float | None = None,
) -> Path:
    """Create a backup dir with the requested files and optional mtime."""
    snap = backups_root / name
    snap.mkdir()
    if with_hourly:
        (snap / "hourly.md").write_text("hourly")
    if with_daily:
        (snap / "daily.md").write_text("daily")
    if with_memories:
        (snap / "memories.md").write_text("mem")
    if mtime is not None:
        os.utime(snap, (mtime, mtime))
    return snap


# ---------------------------------------------------------------------------
# Route tests — acceptance criterion from the story
# ---------------------------------------------------------------------------


class TestMemoryIntegrityRoute:
    def test_returns_200_and_all_five_fields(self, client, stub_vault):
        """Acceptance: endpoint returns 200 with all five fields."""
        resp = client.get("/api/memory/integrity")
        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        data = resp.get_json()
        for field in REQUIRED_FIELDS:
            assert field in data, f"missing field: {field}"

    def test_empty_vault_returns_defaults(self, client, stub_vault):
        """No backups + no crash log -> all null/zero defaults."""
        resp = client.get("/api/memory/integrity")
        data = resp.get_json()
        assert data["last_compaction_at"] is None
        assert data["last_verify_passed"] is None
        assert data["backup_count_hourly"] == 0
        assert data["backup_count_daily"] == 0
        assert data["newest_backup_age_seconds"] is None

    def test_missing_backups_dir_is_not_an_error(self, client, tmp_path, monkeypatch):
        """If Backups/memory doesn't exist at all, route still returns 200."""
        from idea_board import web as web_module

        monkeypatch.setattr(web_module.settings, "vault_path", tmp_path)
        resp = client.get("/api/memory/integrity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["backup_count_hourly"] == 0
        assert data["backup_count_daily"] == 0


# ---------------------------------------------------------------------------
# Collector tests — drive logic directly with a stubbed tree
# ---------------------------------------------------------------------------


class TestBackupCounts:
    def test_stubbed_backup_dir_produces_expected_counts(self, tmp_path):
        """Acceptance: a stubbed backup dir produces the expected counts."""
        backups = tmp_path / "Backups" / "memory"
        backups.mkdir(parents=True)
        crash_log = tmp_path / "crash_log.md"

        # 3 snapshots with hourly, 2 with daily, 1 with only memories.md
        _make_snapshot(backups, "20260416-120000", with_hourly=True, with_daily=True)
        _make_snapshot(backups, "20260416-123000", with_hourly=True, with_daily=True)
        _make_snapshot(backups, "20260416-130000", with_hourly=True, with_daily=False)
        _make_snapshot(
            backups, "20260416-133000",
            with_hourly=False, with_daily=False, with_memories=True,
        )

        payload = _collect_memory_integrity(backups, crash_log)
        assert payload["backup_count_hourly"] == 3
        assert payload["backup_count_daily"] == 2

    def test_files_outside_snapshot_dirs_are_ignored(self, tmp_path):
        """Stray files at the Backups/memory root don't bump counts."""
        backups = tmp_path / "Backups" / "memory"
        backups.mkdir(parents=True)
        (backups / "hourly.md").write_text("stray")
        (backups / "daily.md").write_text("stray")

        payload = _collect_memory_integrity(backups, tmp_path / "nope.md")
        assert payload["backup_count_hourly"] == 0
        assert payload["backup_count_daily"] == 0
        assert payload["last_compaction_at"] is None

    def test_snapshot_with_collision_suffix_counted(self, tmp_path):
        """Dirs like 20260416-120000-1 are valid snapshot dirs."""
        backups = tmp_path / "backups"
        backups.mkdir()
        _make_snapshot(backups, "20260416-120000")
        _make_snapshot(backups, "20260416-120000-1")

        payload = _collect_memory_integrity(backups, tmp_path / "no.md")
        assert payload["backup_count_hourly"] == 2
        assert payload["backup_count_daily"] == 2


class TestLastCompactionAt:
    def test_picks_newest_snapshot_by_name(self, tmp_path):
        backups = tmp_path / "backups"
        backups.mkdir()
        _make_snapshot(backups, "20260401-080000")
        _make_snapshot(backups, "20260416-120000")  # newest
        _make_snapshot(backups, "20260410-100000")

        payload = _collect_memory_integrity(backups, tmp_path / "no.md")
        assert payload["last_compaction_at"] == datetime(
            2026, 4, 16, 12, 0, 0
        ).isoformat()

    def test_unparseable_name_falls_back_to_mtime(self, tmp_path):
        backups = tmp_path / "backups"
        backups.mkdir()
        snap = _make_snapshot(backups, "not-a-timestamp", mtime=1_700_000_000.0)

        payload = _collect_memory_integrity(backups, tmp_path / "no.md")
        # Fallback uses the directory's mtime; just confirm it's populated
        # with the expected ISO string.
        assert payload["last_compaction_at"] == datetime.fromtimestamp(
            snap.stat().st_mtime
        ).isoformat()


class TestNewestBackupAge:
    def test_age_uses_mtime_and_is_non_negative(self, tmp_path):
        backups = tmp_path / "backups"
        backups.mkdir()
        now = time.time()
        _make_snapshot(backups, "20260416-120000", mtime=now - 90.0)

        payload = _collect_memory_integrity(backups, tmp_path / "no.md")
        age = payload["newest_backup_age_seconds"]
        assert age is not None
        # Allow a couple seconds of slack for test runtime
        assert 85 <= age <= 120

    def test_future_mtime_clamps_to_zero(self, tmp_path):
        """A snapshot with a future mtime (clock skew) still reports >= 0."""
        backups = tmp_path / "backups"
        backups.mkdir()
        _make_snapshot(backups, "20260416-120000", mtime=time.time() + 3600.0)

        payload = _collect_memory_integrity(backups, tmp_path / "no.md")
        assert payload["newest_backup_age_seconds"] == 0


class TestLastVerifyPassed:
    def test_passed_marker_returns_true(self, tmp_path):
        log = tmp_path / "crash_log.md"
        log.write_text("# Report\n**Compaction Verify:** PASSED\n")

        payload = _collect_memory_integrity(tmp_path / "no-backups", log)
        assert payload["last_verify_passed"] is True

    def test_failed_marker_returns_false(self, tmp_path):
        log = tmp_path / "crash_log.md"
        log.write_text("# Report\n**Compaction Verify:** FAILED\n")

        payload = _collect_memory_integrity(tmp_path / "no-backups", log)
        assert payload["last_verify_passed"] is False

    def test_last_marker_wins(self, tmp_path):
        """When multiple markers exist, the most recent one is reported."""
        log = tmp_path / "crash_log.md"
        log.write_text(
            "**Compaction Verify:** PASSED\n"
            "...later...\n"
            "**Compaction Verify:** FAILED\n"
        )

        payload = _collect_memory_integrity(tmp_path / "no-backups", log)
        assert payload["last_verify_passed"] is False

    def test_no_marker_returns_none(self, tmp_path):
        log = tmp_path / "crash_log.md"
        log.write_text("# Bot Crash Report\nsomething unrelated\n")

        payload = _collect_memory_integrity(tmp_path / "no-backups", log)
        assert payload["last_verify_passed"] is None

    def test_missing_crash_log_returns_none(self, tmp_path):
        payload = _collect_memory_integrity(
            tmp_path / "no-backups", tmp_path / "missing.md"
        )
        assert payload["last_verify_passed"] is None

    def test_marker_case_insensitive(self, tmp_path):
        log = tmp_path / "crash_log.md"
        log.write_text("**compaction verify:** passed\n")

        payload = _collect_memory_integrity(tmp_path / "no-backups", log)
        assert payload["last_verify_passed"] is True


# ---------------------------------------------------------------------------
# End-to-end: route + stubbed vault tree
# ---------------------------------------------------------------------------


class TestMemoryIntegrityEndToEnd:
    def test_full_payload_with_realistic_tree(self, client, stub_vault):
        vault, backups, crash_log = stub_vault
        _make_snapshot(backups, "20260416-120000")
        _make_snapshot(backups, "20260416-123000")
        crash_log.write_text(
            "# Bot Crash Report\n"
            "**Timestamp:** 2026-04-16 12:30:05\n"
            "**Compaction Verify:** PASSED\n"
        )

        resp = client.get("/api/memory/integrity")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["backup_count_hourly"] == 2
        assert data["backup_count_daily"] == 2
        assert data["last_verify_passed"] is True
        assert data["last_compaction_at"] == datetime(
            2026, 4, 16, 12, 30, 0
        ).isoformat()
        assert data["newest_backup_age_seconds"] is not None
        assert data["newest_backup_age_seconds"] >= 0
