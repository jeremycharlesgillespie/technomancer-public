"""Tests for ``agent.paper_themes`` — the docs/paper_themes.md loader.

Covers the happy path, the missing-file error path, and the malformed-entry
tolerance behavior. Tests use ``tmp_path`` + an explicit ``path=`` argument
to isolate file I/O from the real ``docs/paper_themes.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent.paper_themes import THEMES_PATH, load_paper_themes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def themes_file(tmp_path: Path) -> Path:
    """Path to a writable themes file inside tmp_path (not yet created)."""
    return tmp_path / "paper_themes.md"


VALID_THEMES_MD = """\
# Paper Themes

Preamble paragraph that should be ignored.

---

## failure-mode discoveries
- target_per_week: 5

Stories that surface real failure modes.

Keywords: crash, stall.

---

## measurement + benchmarks
- target_per_week: 3

Stories that add metrics or benchmarks.

---

## novel autonomy mechanisms
- target_per_week: 7

Stories that ship a new self-direction mechanism.

---

## Editing notes

This appendix has no target_per_week and must be ignored.
"""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidInput:
    def test_returns_list_of_themes(self, themes_file: Path) -> None:
        themes_file.write_text(VALID_THEMES_MD, encoding="utf-8")

        themes = load_paper_themes(path=themes_file)

        assert isinstance(themes, list)
        assert len(themes) == 3

    def test_each_theme_has_required_keys(self, themes_file: Path) -> None:
        themes_file.write_text(VALID_THEMES_MD, encoding="utf-8")

        themes = load_paper_themes(path=themes_file)

        for theme in themes:
            assert set(theme.keys()) == {"name", "target_per_week"}
            assert isinstance(theme["name"], str)
            assert isinstance(theme["target_per_week"], int)

    def test_preserves_file_order(self, themes_file: Path) -> None:
        themes_file.write_text(VALID_THEMES_MD, encoding="utf-8")

        themes = load_paper_themes(path=themes_file)

        assert [t["name"] for t in themes] == [
            "failure-mode discoveries",
            "measurement + benchmarks",
            "novel autonomy mechanisms",
        ]

    def test_parses_target_per_week_as_int(self, themes_file: Path) -> None:
        themes_file.write_text(VALID_THEMES_MD, encoding="utf-8")

        themes = load_paper_themes(path=themes_file)
        by_name = {t["name"]: t["target_per_week"] for t in themes}

        assert by_name["failure-mode discoveries"] == 5
        assert by_name["measurement + benchmarks"] == 3
        assert by_name["novel autonomy mechanisms"] == 7

    def test_editing_notes_appendix_is_excluded(self, themes_file: Path) -> None:
        themes_file.write_text(VALID_THEMES_MD, encoding="utf-8")

        themes = load_paper_themes(path=themes_file)

        assert not any(t["name"].lower() == "editing notes" for t in themes)

    def test_real_docs_file_loads(self) -> None:
        """The actual committed docs/paper_themes.md should parse cleanly."""
        if not THEMES_PATH.exists():
            pytest.skip(f"{THEMES_PATH} missing in this checkout")

        themes = load_paper_themes()

        assert len(themes) >= 1
        for theme in themes:
            assert theme["name"]
            assert theme["target_per_week"] >= 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.md"

        with pytest.raises(ValueError, match="paper_themes.md not found"):
            load_paper_themes(path=missing)

    def test_missing_file_error_includes_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.md"

        with pytest.raises(ValueError) as exc:
            load_paper_themes(path=missing)

        assert str(missing) in str(exc.value)


# ---------------------------------------------------------------------------
# Malformed input (warn + skip, don't raise)
# ---------------------------------------------------------------------------


class TestMalformedEntries:
    def test_non_integer_target_is_skipped_with_warning(
        self, themes_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        themes_file.write_text(
            """\
## good theme
- target_per_week: 4

## bad theme
- target_per_week: abc
""",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING, logger="agent.paper_themes")

        themes = load_paper_themes(path=themes_file)

        assert [t["name"] for t in themes] == ["good theme"]
        assert "bad theme" in caplog.text
        assert "non-integer" in caplog.text

    def test_empty_target_value_is_skipped_with_warning(
        self, themes_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        themes_file.write_text(
            """\
## empty theme
- target_per_week:

## ok theme
- target_per_week: 2
""",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING, logger="agent.paper_themes")

        themes = load_paper_themes(path=themes_file)

        assert [t["name"] for t in themes] == ["ok theme"]
        assert "empty theme" in caplog.text

    def test_negative_target_is_skipped_with_warning(
        self, themes_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        themes_file.write_text(
            """\
## negative theme
- target_per_week: -3

## positive theme
- target_per_week: 5
""",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING, logger="agent.paper_themes")

        themes = load_paper_themes(path=themes_file)

        assert [t["name"] for t in themes] == ["positive theme"]
        assert "negative theme" in caplog.text

    def test_section_without_target_bullet_is_silently_skipped(
        self, themes_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sections like 'Editing notes' that have no target_per_week bullet
        are not themes — they should be dropped without a warning, so the
        standard appendix doesn't spam logs every cycle."""
        themes_file.write_text(
            """\
## real theme
- target_per_week: 5

## just a note

Free-form text with no target bullet.
""",
            encoding="utf-8",
        )
        caplog.set_level(logging.WARNING, logger="agent.paper_themes")

        themes = load_paper_themes(path=themes_file)

        assert [t["name"] for t in themes] == ["real theme"]
        assert "just a note" not in caplog.text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file_returns_empty_list(self, themes_file: Path) -> None:
        themes_file.write_text("", encoding="utf-8")

        assert load_paper_themes(path=themes_file) == []

    def test_no_themes_only_top_level_heading(self, themes_file: Path) -> None:
        themes_file.write_text("# Paper Themes\n\nNo sections.\n", encoding="utf-8")

        assert load_paper_themes(path=themes_file) == []

    def test_theme_names_are_stripped(self, themes_file: Path) -> None:
        themes_file.write_text(
            "##   spaced theme   \n- target_per_week: 1\n", encoding="utf-8"
        )

        themes = load_paper_themes(path=themes_file)

        assert themes == [{"name": "spaced theme", "target_per_week": 1}]

    def test_zero_target_is_allowed(self, themes_file: Path) -> None:
        """target_per_week: 0 means 'suppress this theme for now' — valid, not a warning."""
        themes_file.write_text(
            "## paused theme\n- target_per_week: 0\n", encoding="utf-8"
        )

        themes = load_paper_themes(path=themes_file)

        assert themes == [{"name": "paused theme", "target_per_week": 0}]

    def test_target_per_week_is_case_insensitive(self, themes_file: Path) -> None:
        themes_file.write_text(
            "## loud theme\n- Target_Per_Week: 4\n", encoding="utf-8"
        )

        themes = load_paper_themes(path=themes_file)

        assert themes == [{"name": "loud theme", "target_per_week": 4}]
