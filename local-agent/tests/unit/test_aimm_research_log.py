"""Tests for aimm.research_log — append AIMM cycle summaries.

Primary acceptance test: three cycles on the same UTC day must produce
exactly one ``## YYYY-MM-DD`` heading in ``docs/research_notes.md`` with
three entries listed below it. This is the behaviour the production
caller (aimm/manager.py) has been failing to achieve, so it deserves its
own named test.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from aimm.research_log import (
    DEFAULT_RESEARCH_NOTES_PATH,
    TITLE_HEADING,
    CycleSummary,
    append_cycle_summary,
    find_or_create_date_section,
    format_cycle_entry,
)


@pytest.fixture
def notes_path(tmp_path: Path) -> Path:
    return tmp_path / "research_notes.md"


def _count_date_headings(content: str, date_str: str) -> int:
    pattern = re.compile(rf"^## {re.escape(date_str)}\s*$", re.MULTILINE)
    return len(pattern.findall(content))


def _count_entries(content: str, date_str: str) -> int:
    """Count ### entries that fall under the given date section."""
    date_re = re.compile(rf"^## {re.escape(date_str)}\s*$", re.MULTILINE)
    m = date_re.search(content)
    if not m:
        return 0
    start = m.end()
    next_section = re.search(r"(?m)^## ", content[start:])
    body = content[start:] if next_section is None else content[start : start + next_section.start()]
    return len(re.findall(r"(?m)^### ", body))


# ---------------------------------------------------------------------------
# CycleSummary
# ---------------------------------------------------------------------------


class TestCycleSummary:
    def test_defaults(self) -> None:
        s = CycleSummary()
        assert s.observed == 0
        assert s.findings_logged == 0
        assert s.suggestions_logged == 0
        assert s.theme == ""
        assert s.cycle_id == ""
        assert s.notes == ""
        assert s.extras == {}

    def test_populated(self) -> None:
        s = CycleSummary(
            observed=5,
            findings_logged=2,
            suggestions_logged=1,
            theme="failure-modes",
            cycle_id="abc-123",
            notes="first run post-deploy",
        )
        assert s.observed == 5
        assert s.theme == "failure-modes"


# ---------------------------------------------------------------------------
# format_cycle_entry — markdown rendering
# ---------------------------------------------------------------------------


class TestFormatCycleEntry:
    def test_renders_required_fields(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(observed=5, findings_logged=2, suggestions_logged=1),
            "14:35 UTC",
        )
        assert "### 14:35 UTC — cycle" in entry
        assert "- observed: 5" in entry
        assert "- findings_logged: 2" in entry
        assert "- suggestions_logged: 1" in entry
        assert entry.rstrip().endswith("---")

    def test_uses_theme_in_headline_when_provided(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(theme="failure-modes"), "10:00 UTC"
        )
        assert "### 10:00 UTC — failure-modes" in entry

    def test_cycle_id_headline_when_theme_missing(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(cycle_id="abc-123"), "10:00 UTC"
        )
        assert "### 10:00 UTC — abc-123" in entry

    def test_emits_optional_fields(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                theme="t1",
                cycle_id="c1",
                notes="something happened",
                extras={"extra_key": "extra_val"},
            ),
            "10:00 UTC",
        )
        assert "- theme: t1" in entry
        assert "- cycle_id: c1" in entry
        assert "- notes: something happened" in entry
        assert "- extra_key: extra_val" in entry

    def test_flattens_multiline_values(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(notes="line one\nline two"), "10:00 UTC"
        )
        assert "- notes: line one line two" in entry
        assert "line one\nline two" not in entry


# ---------------------------------------------------------------------------
# find_or_create_date_section — insertion-point logic
# ---------------------------------------------------------------------------


class TestFindOrCreateDateSection:
    def test_creates_section_and_title_in_empty_content(self) -> None:
        content, idx = find_or_create_date_section("", "2026-04-18")
        assert TITLE_HEADING in content
        assert "## 2026-04-18" in content
        assert content[:idx].endswith("## 2026-04-18")

    def test_reuses_existing_date_section(self) -> None:
        original = (
            f"{TITLE_HEADING}\n\nSome preamble.\n\n## 2026-04-18\n\n"
            "### 10:00 UTC — a\n- observed: 1\n\n---\n"
        )
        content, idx = find_or_create_date_section(original, "2026-04-18")
        # No new date heading should have been introduced.
        assert _count_date_headings(content, "2026-04-18") == 1
        # Insertion point lives inside the existing section.
        date_pos = content.index("## 2026-04-18")
        assert idx > date_pos

    def test_inserts_before_next_section(self) -> None:
        original = (
            f"{TITLE_HEADING}\n\n## 2026-04-18\n\n"
            "### 10:00 UTC — a\n- observed: 1\n\n---\n\n"
            "## 2026-04-19\n\n### 09:00 UTC — b\n- observed: 2\n\n---\n"
        )
        content, idx = find_or_create_date_section(original, "2026-04-18")
        # Insertion offset should sit before the next date heading.
        next_heading_pos = content.index("## 2026-04-19")
        assert idx <= next_heading_pos

    def test_appends_new_date_section_for_new_day(self) -> None:
        original = (
            f"{TITLE_HEADING}\n\n## 2026-04-18\n\n"
            "### 10:00 UTC — a\n- observed: 1\n\n---\n"
        )
        content, idx = find_or_create_date_section(original, "2026-04-19")
        assert _count_date_headings(content, "2026-04-18") == 1
        assert _count_date_headings(content, "2026-04-19") == 1
        # Index is positioned just after the new heading we created.
        assert content[:idx].endswith("## 2026-04-19")


# ---------------------------------------------------------------------------
# append_cycle_summary — the public API (primary acceptance criteria)
# ---------------------------------------------------------------------------


class TestSingleCycleNewFile:
    def test_creates_file_header_and_entry(self, notes_path: Path) -> None:
        cycle = CycleSummary(
            observed=5,
            findings_logged=2,
            suggestions_logged=1,
            theme="failure-modes",
            cycle_id="c-001",
        )
        ok = append_cycle_summary(
            cycle,
            date=datetime(2026, 4, 18, 14, 35, tzinfo=timezone.utc),
            notes_path=notes_path,
        )
        assert ok is True
        assert notes_path.exists()

        content = notes_path.read_text(encoding="utf-8")
        assert TITLE_HEADING in content
        assert _count_date_headings(content, "2026-04-18") == 1
        assert _count_entries(content, "2026-04-18") == 1
        assert "### 14:35 UTC — failure-modes" in content
        assert "- observed: 5" in content
        assert "- findings_logged: 2" in content
        assert "- suggestions_logged: 1" in content

    def test_returns_false_for_non_cyclesummary(self, notes_path: Path) -> None:
        ok = append_cycle_summary(
            {"observed": 1},  # type: ignore[arg-type]
            notes_path=notes_path,
        )
        assert ok is False
        assert not notes_path.exists()


class TestSingleCycleExistingDate:
    def test_reuses_existing_date_header(self, notes_path: Path) -> None:
        # Seed the file with a prior entry on the same day.
        append_cycle_summary(
            CycleSummary(observed=1, theme="t1"),
            date=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )
        append_cycle_summary(
            CycleSummary(observed=2, theme="t2"),
            date=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )

        content = notes_path.read_text(encoding="utf-8")
        assert _count_date_headings(content, "2026-04-18") == 1
        assert _count_entries(content, "2026-04-18") == 2
        # Both entries present and ordered by append sequence.
        assert content.index("### 10:00 UTC — t1") < content.index(
            "### 12:00 UTC — t2"
        )


class TestThreeCyclesSameDaySingleDateHeader:
    """The failing scenario the acceptance criteria explicitly calls out."""

    def test_three_cycles_produce_one_header_three_entries(
        self, notes_path: Path
    ) -> None:
        base = datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc)
        for i in range(3):
            ok = append_cycle_summary(
                CycleSummary(
                    observed=i + 1,
                    findings_logged=i,
                    suggestions_logged=0,
                    theme=f"theme-{i}",
                    cycle_id=f"cycle-{i}",
                ),
                date=base + timedelta(hours=i),
                notes_path=notes_path,
            )
            assert ok is True

        content = notes_path.read_text(encoding="utf-8")

        # 1 date header, 3 entries below it.
        assert _count_date_headings(content, "2026-04-18") == 1
        assert _count_entries(content, "2026-04-18") == 3

        # Each entry's distinguishing headline is present.
        assert "### 09:00 UTC — theme-0" in content
        assert "### 10:00 UTC — theme-1" in content
        assert "### 11:00 UTC — theme-2" in content

        # Order is preserved: earliest cycle first.
        idx0 = content.index("09:00 UTC — theme-0")
        idx1 = content.index("10:00 UTC — theme-1")
        idx2 = content.index("11:00 UTC — theme-2")
        assert idx0 < idx1 < idx2

        # Each entry has its own observed-count line.
        assert "- observed: 1" in content
        assert "- observed: 2" in content
        assert "- observed: 3" in content


class TestMultiDayOrdering:
    def test_new_day_creates_new_section_preserving_old(
        self, notes_path: Path
    ) -> None:
        append_cycle_summary(
            CycleSummary(observed=1, theme="first-day"),
            date=datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )
        append_cycle_summary(
            CycleSummary(observed=2, theme="second-day"),
            date=datetime(2026, 4, 19, 9, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )

        content = notes_path.read_text(encoding="utf-8")
        assert _count_date_headings(content, "2026-04-18") == 1
        assert _count_date_headings(content, "2026-04-19") == 1
        # Append order: older day section first.
        assert content.index("## 2026-04-18") < content.index(
            "## 2026-04-19"
        )


class TestNaiveAndTimezonedDates:
    def test_naive_datetime_treated_as_utc(self, notes_path: Path) -> None:
        append_cycle_summary(
            CycleSummary(observed=1, theme="t"),
            date=datetime(2026, 4, 18, 14, 0),  # no tzinfo
            notes_path=notes_path,
        )
        content = notes_path.read_text(encoding="utf-8")
        assert "## 2026-04-18" in content
        assert "### 14:00 UTC — t" in content

    def test_non_utc_datetime_converted(self, notes_path: Path) -> None:
        tz = timezone(timedelta(hours=-5))  # EST
        # 18:00 EST == 23:00 UTC on the same date.
        append_cycle_summary(
            CycleSummary(observed=1, theme="t"),
            date=datetime(2026, 4, 18, 18, 0, tzinfo=tz),
            notes_path=notes_path,
        )
        content = notes_path.read_text(encoding="utf-8")
        assert "### 23:00 UTC — t" in content


class TestDefaultPath:
    def test_uses_default_path_when_override_omitted(
        self, tmp_path: Path
    ) -> None:
        fake_default = tmp_path / "research_notes.md"
        with patch(
            "aimm.research_log.DEFAULT_RESEARCH_NOTES_PATH", fake_default
        ):
            ok = append_cycle_summary(
                CycleSummary(observed=1, theme="t"),
                date=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            )
        assert ok is True
        assert fake_default.exists()


class TestWriteFailureSwallowed:
    def test_returns_false_on_write_failure(self, notes_path: Path) -> None:
        with patch(
            "aimm.research_log._atomic_write", return_value=False
        ):
            ok = append_cycle_summary(
                CycleSummary(observed=1, theme="t"),
                date=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
                notes_path=notes_path,
            )
        assert ok is False


class TestDefaultPathConstant:
    def test_default_path_under_docs(self) -> None:
        assert DEFAULT_RESEARCH_NOTES_PATH.name == "research_notes.md"
        assert DEFAULT_RESEARCH_NOTES_PATH.parent.name == "docs"
