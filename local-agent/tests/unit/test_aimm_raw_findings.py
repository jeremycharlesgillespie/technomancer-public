"""Tests for aimm.raw_findings — findings storage with story_key dedup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aimm.raw_findings import RawFindings


@pytest.fixture
def findings_path(tmp_path: Path) -> Path:
    return tmp_path / "raw_findings.md"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_missing_file(self, findings_path: Path) -> None:
        assert not findings_path.exists()
        RawFindings(findings_path)
        assert findings_path.exists()
        assert findings_path.read_text(encoding="utf-8") == ""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "docs" / "aimm" / "raw_findings.md"
        assert not nested.parent.exists()
        RawFindings(nested)
        assert nested.exists()
        assert nested.parent.is_dir()

    def test_accepts_str_path(self, findings_path: Path) -> None:
        store = RawFindings(str(findings_path))
        assert store.path == findings_path

    def test_leaves_existing_file_untouched(self, findings_path: Path) -> None:
        findings_path.write_text("pre-existing content", encoding="utf-8")
        RawFindings(findings_path)
        assert findings_path.read_text(encoding="utf-8") == "pre-existing content"

    def test_loads_existing_story_keys_from_disk(self, findings_path: Path) -> None:
        findings_path.write_text(
            "### 2026-04-18 12:00 UTC — TK-1\n\n"
            "- story_key: TK-1\n"
            "- theme: perf\n\n---\n\n"
            "### 2026-04-18 12:05 UTC — TK-2\n\n"
            "- story_key: TK-2\n\n---\n",
            encoding="utf-8",
        )
        store = RawFindings(findings_path)
        assert store.story_keys == {"TK-1", "TK-2"}

    def test_empty_existing_file_yields_empty_seen(self, findings_path: Path) -> None:
        findings_path.write_text("", encoding="utf-8")
        store = RawFindings(findings_path)
        assert store.story_keys == set()

    def test_malformed_file_does_not_crash(self, findings_path: Path) -> None:
        findings_path.write_text(
            "random prose with no story_key markers at all", encoding="utf-8"
        )
        store = RawFindings(findings_path)
        assert store.story_keys == set()


# ---------------------------------------------------------------------------
# append_finding
# ---------------------------------------------------------------------------


class TestAppendFinding:
    def test_appends_new_finding_and_returns_true(
        self, findings_path: Path
    ) -> None:
        store = RawFindings(findings_path)
        assert store.append_finding({"story_key": "TK-1", "theme": "perf"}) is True
        assert store.has("TK-1")
        text = findings_path.read_text(encoding="utf-8")
        assert "- story_key: TK-1" in text
        assert "- theme: perf" in text

    def test_duplicate_story_key_is_skipped(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        assert store.append_finding({"story_key": "TK-100", "theme": "a"}) is True
        assert store.append_finding({"story_key": "TK-100", "theme": "b"}) is False
        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- story_key: TK-100") == 1
        # Second payload's theme must not have been written.
        assert "- theme: b" not in text

    def test_three_unique_two_duplicates_yields_three_entries(
        self, findings_path: Path
    ) -> None:
        """Acceptance criterion: 3 unique + 2 duplicates → 3 entries."""
        store = RawFindings(findings_path)
        assert store.append_finding({"story_key": "TK-1"}) is True
        assert store.append_finding({"story_key": "TK-2"}) is True
        assert store.append_finding({"story_key": "TK-3"}) is True
        assert store.append_finding({"story_key": "TK-1"}) is False
        assert store.append_finding({"story_key": "TK-2"}) is False

        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- story_key: TK-") == 3
        assert store.story_keys == {"TK-1", "TK-2", "TK-3"}

    def test_missing_story_key_returns_false(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        assert store.append_finding({"theme": "x"}) is False
        assert store.append_finding({"story_key": ""}) is False
        assert store.append_finding({"story_key": "   "}) is False
        assert findings_path.read_text(encoding="utf-8") == ""

    def test_non_dict_payload_returns_false(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        assert store.append_finding("not a dict") is False  # type: ignore[arg-type]
        assert store.append_finding(None) is False  # type: ignore[arg-type]
        assert store.append_finding(123) is False  # type: ignore[arg-type]
        assert findings_path.read_text(encoding="utf-8") == ""

    def test_dedup_survives_across_instances(self, findings_path: Path) -> None:
        first = RawFindings(findings_path)
        assert first.append_finding({"story_key": "TK-9", "theme": "x"}) is True

        second = RawFindings(findings_path)
        assert second.has("TK-9")
        assert second.append_finding({"story_key": "TK-9", "theme": "y"}) is False

        text = findings_path.read_text(encoding="utf-8")
        assert text.count("- story_key: TK-9") == 1

    def test_entry_flattens_multiline_values(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        store.append_finding(
            {"story_key": "TK-5", "note": "line1\nline2\r\nline3"}
        )
        text = findings_path.read_text(encoding="utf-8")
        # The note value must fit on one line so our story_key regex
        # never matches accidental prose.
        assert "- note: line1 line2  line3" in text
        # Re-loading should still show exactly one story_key.
        reloaded = RawFindings(findings_path)
        assert reloaded.story_keys == {"TK-5"}

    def test_extra_keys_are_serialised(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        store.append_finding(
            {
                "story_key": "TK-42",
                "finding_type": "observation",
                "theme": "latency",
                "why_it_matters": "users noticed",
            }
        )
        text = findings_path.read_text(encoding="utf-8")
        assert "- finding_type: observation" in text
        assert "- theme: latency" in text
        assert "- why_it_matters: users noticed" in text

    def test_os_error_on_append_returns_false(self, findings_path: Path) -> None:
        store = RawFindings(findings_path)
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            assert (
                store.append_finding({"story_key": "TK-1", "theme": "x"}) is False
            )
        # Failed write must not have added to the in-memory set —
        # otherwise a later retry would be silently dedup'd.
        assert not store.has("TK-1")


# ---------------------------------------------------------------------------
# story_keys snapshot semantics
# ---------------------------------------------------------------------------


class TestStoryKeysSnapshot:
    def test_snapshot_is_decoupled_from_internal_state(
        self, findings_path: Path
    ) -> None:
        store = RawFindings(findings_path)
        store.append_finding({"story_key": "TK-1"})

        snap = store.story_keys
        snap.add("TK-2")  # caller mutation must not leak back in
        snap.discard("TK-1")

        assert store.has("TK-1")
        assert not store.has("TK-2")
