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
        assert s.findings == []
        assert s.suggestions == []
        assert s.hypotheses == []
        assert s.theme_coverage_delta == ""

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

    def test_no_leading_or_trailing_newlines(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(observed=1, theme="t"), "10:00 UTC"
        )
        assert not entry.startswith("\n")
        assert not entry.endswith("\n")
        assert entry.startswith("### 10:00 UTC — t")
        assert entry.endswith("---")

    def test_renders_findings_as_bullet_list(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                findings=["Finding A", "Finding B"],
            ),
            "10:00 UTC",
        )
        assert "**Findings:**" in entry
        assert "- Finding A" in entry
        assert "- Finding B" in entry
        # Findings block appears after the counts section.
        assert entry.index("- observed: 0") < entry.index("**Findings:**")

    def test_renders_suggestions_as_bullet_list(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                suggestions=["Try X", "Consider Y"],
            ),
            "10:00 UTC",
        )
        assert "**Suggestions:**" in entry
        assert "- Try X" in entry
        assert "- Consider Y" in entry

    def test_renders_hypotheses_as_bullet_list(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                hypotheses=["H1: gravity wins", "H2: caching helps"],
            ),
            "10:00 UTC",
        )
        assert "**Hypotheses:**" in entry
        assert "- H1: gravity wins" in entry
        assert "- H2: caching helps" in entry

    def test_renders_theme_coverage_delta_inline(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                theme_coverage_delta="+2 themes added (failure-modes, latency)",
            ),
            "10:00 UTC",
        )
        assert (
            "- theme_coverage_delta: +2 themes added (failure-modes, latency)"
            in entry
        )

    def test_omits_empty_bullet_sections(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(observed=1),
            "10:00 UTC",
        )
        assert "**Findings:**" not in entry
        assert "**Suggestions:**" not in entry
        assert "**Hypotheses:**" not in entry
        assert "theme_coverage_delta" not in entry

    def test_flattens_bullet_list_items(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                findings=["line one\nline two", "  spaced  "],
            ),
            "10:00 UTC",
        )
        assert "- line one line two" in entry
        assert "- spaced" in entry
        # Flattened items must not introduce embedded newlines that
        # would break sibling bullet rendering.
        findings_block = entry.split("**Findings:**")[1]
        for line in findings_block.splitlines():
            if line.startswith("- "):
                assert "\n" not in line

    def test_skips_blank_bullet_items(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(findings=["", "  ", "kept"]),
            "10:00 UTC",
        )
        assert "- kept" in entry
        # No empty bullets emitted.
        assert "\n- \n" not in entry
        assert "\n-  \n" not in entry

    def test_renders_all_rich_fields_together(self) -> None:
        entry = format_cycle_entry(
            CycleSummary(
                observed=4,
                findings_logged=2,
                suggestions_logged=1,
                theme="failure-modes",
                cycle_id="c-42",
                findings=["Race in queue drain"],
                suggestions=["Add backpressure"],
                hypotheses=["Contention grows with worker count"],
                theme_coverage_delta="+1 theme",
            ),
            "14:35 UTC",
        )
        # Headline + counts
        assert "### 14:35 UTC — failure-modes" in entry
        assert "- observed: 4" in entry
        # Coverage delta inline, before the rich sections
        delta_idx = entry.index("- theme_coverage_delta: +1 theme")
        findings_idx = entry.index("**Findings:**")
        assert delta_idx < findings_idx
        # All three rich sections render in the documented order.
        assert findings_idx < entry.index("**Suggestions:**")
        assert entry.index("**Suggestions:**") < entry.index("**Hypotheses:**")
        # Each section's bullet is present.
        assert "- Race in queue drain" in entry
        assert "- Add backpressure" in entry
        assert "- Contention grows with worker count" in entry
        # Still ends with the separator.
        assert entry.endswith("---")


# ---------------------------------------------------------------------------
# find_or_create_date_section — insertion-point logic
# ---------------------------------------------------------------------------


class TestFindOrCreateDateSection:
    """Pure function: parses content, returns ``(start, end, exists)``.

    Covers the five mock file states called out in the acceptance
    criteria — empty, one header, multiple headers, target in middle,
    target at end — plus a handful of related edge cases.
    """

    DATE_TARGET = datetime(2026, 4, 18, tzinfo=timezone.utc)

    # --- empty content -----------------------------------------------------

    def test_empty_content_returns_zero_length_insert(self) -> None:
        """State: empty file. Target: any date. Exists=False; insert at 0."""
        start, end, exists = find_or_create_date_section("", self.DATE_TARGET)
        assert exists is False
        assert start == 0
        assert end == 0

    def test_content_without_any_headings_returns_eof_insert(self) -> None:
        content = f"{TITLE_HEADING}\n\nSome preamble only.\n"
        start, end, exists = find_or_create_date_section(
            content, self.DATE_TARGET
        )
        assert exists is False
        assert start == len(content)
        assert end == len(content)

    # --- one header --------------------------------------------------------

    def test_one_header_exact_match(self) -> None:
        """State: file with exactly one date heading that matches."""
        content = (
            f"{TITLE_HEADING}\n\nSome preamble.\n\n## 2026-04-18\n\n"
            "### 10:00 UTC — a\n- observed: 1\n\n---\n"
        )
        start, end, exists = find_or_create_date_section(
            content, self.DATE_TARGET
        )
        assert exists is True
        assert content[start : start + len("## 2026-04-18")] == "## 2026-04-18"
        # No later section → section extends to EOF.
        assert end == len(content)
        # Section body is fully captured inside [start, end).
        assert "### 10:00 UTC — a" in content[start:end]

    def test_one_header_no_match_newer_target_inserts_at_top(self) -> None:
        """State: file has one older heading; target is newer → insert above it."""
        content = (
            f"{TITLE_HEADING}\n\n## 2026-04-18\n\n"
            "### 10:00 UTC — a\n- observed: 1\n\n---\n"
        )
        target = datetime(2026, 4, 19, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is False
        assert start == end
        # Insert position is at the existing (older) heading so the new
        # section ends up above it (reverse-chronological).
        assert start == content.index("## 2026-04-18")

    def test_one_header_no_match_older_target_inserts_at_eof(self) -> None:
        """State: one newer heading; target is older → append at end."""
        content = (
            f"{TITLE_HEADING}\n\n## 2026-04-19\n\n"
            "### 09:00 UTC — a\n- observed: 1\n\n---\n"
        )
        target = datetime(2026, 4, 17, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is False
        assert start == len(content)
        assert end == len(content)

    # --- multiple headers --------------------------------------------------

    def test_multiple_headers_match_bounds_middle_section(self) -> None:
        """Target matches a heading sandwiched between two others."""
        content = (
            f"{TITLE_HEADING}\n\n"
            "## 2026-04-20\n\n### 09:00 UTC — c\n- observed: 3\n\n---\n\n"
            "## 2026-04-19\n\n### 09:00 UTC — b\n- observed: 2\n\n---\n\n"
            "## 2026-04-18\n\n### 09:00 UTC — a\n- observed: 1\n\n---\n"
        )
        target = datetime(2026, 4, 19, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is True
        assert start == content.index("## 2026-04-19")
        # The section ends just before the next ## heading.
        assert end == content.index("## 2026-04-18")
        section_body = content[start:end]
        assert "### 09:00 UTC — b" in section_body
        # Must NOT swallow neighbouring sections.
        assert "### 09:00 UTC — a" not in section_body
        assert "### 09:00 UTC — c" not in section_body

    def test_target_date_in_middle_of_existing_range_inserts_between(self) -> None:
        """Target between two existing dates → insert before the older one."""
        content = (
            f"{TITLE_HEADING}\n\n"
            "## 2026-04-20\n\n### 09:00 UTC — c\n---\n\n"
            "## 2026-04-18\n\n### 09:00 UTC — a\n---\n"
        )
        # 2026-04-19 sits between 20 (newer) and 18 (older).
        target = datetime(2026, 4, 19, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is False
        assert start == end
        # Inserts just before the first OLDER heading (2026-04-18).
        assert start == content.index("## 2026-04-18")
        # Insertion offset sits after the newer (2026-04-20) section.
        assert start > content.index("## 2026-04-20")

    def test_target_older_than_all_appends_at_eof(self) -> None:
        """Target older than every existing heading → insert at EOF."""
        content = (
            f"{TITLE_HEADING}\n\n"
            "## 2026-04-20\n\n### 09:00 UTC — c\n---\n\n"
            "## 2026-04-19\n\n### 09:00 UTC — b\n---\n"
        )
        target = datetime(2026, 4, 10, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is False
        assert start == len(content)
        assert end == len(content)

    def test_target_newer_than_all_inserts_before_first_heading(self) -> None:
        """Target newer than every existing heading → insert before first."""
        content = (
            f"{TITLE_HEADING}\n\n"
            "## 2026-04-18\n\n### 10:00 UTC — a\n---\n\n"
            "## 2026-04-17\n\n### 10:00 UTC — z\n---\n"
        )
        target = datetime(2026, 4, 19, tzinfo=timezone.utc)
        start, end, exists = find_or_create_date_section(content, target)
        assert exists is False
        assert start == end
        # Insertion is at the first (newest existing) heading position.
        assert start == content.index("## 2026-04-18")

    # --- matching + section_end edge cases ---------------------------------

    def test_match_section_ends_at_next_h2_even_if_non_date(self) -> None:
        """Any ``## `` heading — even non-date — terminates a section."""
        content = (
            f"{TITLE_HEADING}\n\n## 2026-04-18\n\n### 10:00 UTC — a\n---\n\n"
            "## Appendix\n\nNotes.\n"
        )
        start, end, exists = find_or_create_date_section(
            content, self.DATE_TARGET
        )
        assert exists is True
        assert start == content.index("## 2026-04-18")
        assert end == content.index("## Appendix")

    def test_accepts_naive_datetime_as_utc(self) -> None:
        content = f"{TITLE_HEADING}\n\n## 2026-04-18\n\n### 10:00 UTC — a\n---\n"
        naive = datetime(2026, 4, 18, 12, 0)  # no tzinfo
        start, _end, exists = find_or_create_date_section(content, naive)
        assert exists is True
        assert start == content.index("## 2026-04-18")

    def test_accepts_tz_aware_datetime_converted_to_utc(self) -> None:
        # 23:00 EST on 2026-04-17 == 04:00 UTC on 2026-04-18.
        est = timezone(timedelta(hours=-5))
        content = f"{TITLE_HEADING}\n\n## 2026-04-18\n\n### a\n---\n"
        aware = datetime(2026, 4, 17, 23, 0, tzinfo=est)
        _start, _end, exists = find_or_create_date_section(content, aware)
        assert exists is True

    def test_accepts_plain_date_object(self) -> None:
        from datetime import date as date_cls

        content = f"{TITLE_HEADING}\n\n## 2026-04-18\n\n### a\n---\n"
        start, _end, exists = find_or_create_date_section(
            content, date_cls(2026, 4, 18)
        )
        assert exists is True
        assert start == content.index("## 2026-04-18")

    def test_accepts_date_string(self) -> None:
        content = f"{TITLE_HEADING}\n\n## 2026-04-18\n\n### a\n---\n"
        start, _end, exists = find_or_create_date_section(content, "2026-04-18")
        assert exists is True
        assert start == content.index("## 2026-04-18")


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
    def test_newer_day_stacks_on_top_of_older(
        self, notes_path: Path
    ) -> None:
        """Reverse-chronological: newer date ends up above the older one."""
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
        # Reverse-chronological: newer day first.
        assert content.index("## 2026-04-19") < content.index(
            "## 2026-04-18"
        )

    def test_older_day_inserted_last_goes_below_existing(
        self, notes_path: Path
    ) -> None:
        """Insert newer then older — older still ends up at the bottom."""
        append_cycle_summary(
            CycleSummary(observed=1, theme="newer"),
            date=datetime(2026, 4, 19, 10, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )
        append_cycle_summary(
            CycleSummary(observed=2, theme="older"),
            date=datetime(2026, 4, 18, 9, 0, tzinfo=timezone.utc),
            notes_path=notes_path,
        )

        content = notes_path.read_text(encoding="utf-8")
        assert content.index("## 2026-04-19") < content.index(
            "## 2026-04-18"
        )
        # Both entries present and unharmed.
        assert "### 10:00 UTC — newer" in content
        assert "### 09:00 UTC — older" in content

    def test_three_days_stack_newest_first(self, notes_path: Path) -> None:
        """Regardless of append order, dates render newest → oldest."""
        # Append out-of-order on purpose.
        for d, theme in [
            (datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc), "middle"),
            (datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc), "oldest"),
            (datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc), "newest"),
        ]:
            append_cycle_summary(
                CycleSummary(observed=1, theme=theme),
                date=d,
                notes_path=notes_path,
            )

        content = notes_path.read_text(encoding="utf-8")
        idx_18 = content.index("## 2026-04-18")
        idx_17 = content.index("## 2026-04-17")
        idx_16 = content.index("## 2026-04-16")
        assert idx_18 < idx_17 < idx_16


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
