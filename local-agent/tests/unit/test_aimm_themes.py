"""Tests for aimm.themes — theme parser + gap analysis.

Verifies the two acceptance criteria from TK-663:

* :func:`load_themes` parses a 3-theme markdown file into 3 :class:`Theme`
  objects with names, targets, and keyword tuples.
* :func:`theme_gap` returns under-served themes sorted worst-first when
  given a mocked 7-day completion history.

Plus the surrounding edge cases that production will hit immediately:
malformed sections, missing files, provider failures, multi-theme matches,
zero-target themes, and case-insensitive keyword matching.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aimm.themes import (
    DEFAULT_LOOKBACK_DAYS,
    Theme,
    load_themes,
    theme_gap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


THREE_THEME_MD = """\
# Paper Themes

Editable priority list.

---

## failure-mode discoveries
- target_per_week: 5

Stories that surface a real, previously-unknown failure mode.

Keywords: crash, stall, orphan, race, silent-failure, regression.

---

## measurement + benchmarks
- target_per_week: 5

Stories that ship an observable metric or benchmark.

Keywords: metric, benchmark, dashboard, telemetry, profiling.

---

## novel autonomy mechanisms
- target_per_week: 5

Stories that ship a new self-direction mechanism.

Keywords: self-healing, auto-split, auto-resolve, feedback-loop.

---

## Editing notes

Theme name is the markdown heading text. (No target — not a theme.)
"""


@pytest.fixture
def themes_file(tmp_path: Path) -> Path:
    path = tmp_path / "paper_themes.md"
    path.write_text(THREE_THEME_MD, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_themes
# ---------------------------------------------------------------------------


class TestLoadThemes:
    def test_returns_three_themes_for_three_theme_file(self, themes_file: Path) -> None:
        """Acceptance criterion 1: 3 themes in → 3 Theme objects out.

        The 'Editing notes' appendix has no target_per_week and must be
        ignored, otherwise production would treat documentation prose as
        a theme.
        """
        themes = load_themes(themes_file)
        assert len(themes) == 3
        assert all(isinstance(t, Theme) for t in themes)

    def test_theme_fields_are_populated(self, themes_file: Path) -> None:
        themes = load_themes(themes_file)
        first = themes[0]
        assert first.name == "failure-mode discoveries"
        assert first.target_per_week == 5
        assert "crash" in first.keywords
        assert "stall" in first.keywords
        assert "regression" in first.keywords

    def test_theme_order_matches_file_order(self, themes_file: Path) -> None:
        names = [t.name for t in load_themes(themes_file)]
        assert names == [
            "failure-mode discoveries",
            "measurement + benchmarks",
            "novel autonomy mechanisms",
        ]

    def test_keywords_are_lowercased(self, tmp_path: Path) -> None:
        path = tmp_path / "themes.md"
        path.write_text(
            "## test\n- target_per_week: 3\n\nKeywords: CRASH, Stall, REGRESSION.\n",
            encoding="utf-8",
        )
        theme = load_themes(path)[0]
        assert theme.keywords == ("crash", "stall", "regression")

    def test_section_without_keywords_returns_empty_tuple(self, tmp_path: Path) -> None:
        path = tmp_path / "themes.md"
        path.write_text("## naked\n- target_per_week: 2\n\nNo keywords here.\n", encoding="utf-8")
        theme = load_themes(path)[0]
        assert theme.keywords == ()

    def test_missing_file_raises_valueerror(self, tmp_path: Path) -> None:
        """ValueError, not FileNotFoundError — callers treat it as config."""
        with pytest.raises(ValueError, match="not found"):
            load_themes(tmp_path / "absent.md")

    def test_section_without_target_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "themes.md"
        path.write_text(
            "## not-a-theme\n\nJust prose, no target bullet.\n\n"
            "## real\n- target_per_week: 2\n",
            encoding="utf-8",
        )
        themes = load_themes(path)
        assert [t.name for t in themes] == ["real"]

    def test_negative_target_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "themes.md"
        path.write_text(
            "## bad\n- target_per_week: -1\n\n## good\n- target_per_week: 1\n",
            encoding="utf-8",
        )
        assert [t.name for t in load_themes(path)] == ["good"]

    def test_non_integer_target_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "themes.md"
        path.write_text(
            "## bad\n- target_per_week: many\n\n## good\n- target_per_week: 1\n",
            encoding="utf-8",
        )
        assert [t.name for t in load_themes(path)] == ["good"]


# ---------------------------------------------------------------------------
# theme_gap
# ---------------------------------------------------------------------------


def _story(summary: str, description: str = "", labels: list[str] | None = None) -> dict:
    return {
        "summary": summary,
        "description": description,
        "labels": labels or [],
    }


class TestThemeGap:
    def test_returns_only_under_served_theme(self) -> None:
        """Acceptance criterion 2: theme A has 2 stories (target 5),
        theme B has 5 stories (target 5) → gap returns [A]."""
        theme_a = Theme(name="a", target_per_week=5, keywords=("alpha",))
        theme_b = Theme(name="b", target_per_week=5, keywords=("beta",))

        stories = [
            _story("alpha story one"),
            _story("alpha story two"),
            _story("beta one"),
            _story("beta two"),
            _story("beta three"),
            _story("beta four"),
            _story("beta five"),
        ]
        provider = MagicMock(return_value=stories)

        gaps = theme_gap(provider, [theme_a, theme_b])

        assert gaps == [theme_a]

    def test_provider_is_called_with_default_lookback(self) -> None:
        provider = MagicMock(return_value=[])
        theme_gap(provider, [Theme("x", 1, ("x",))])
        provider.assert_called_once_with(days=DEFAULT_LOOKBACK_DAYS)

    def test_lookback_override_is_propagated(self) -> None:
        provider = MagicMock(return_value=[])
        theme_gap(provider, [Theme("x", 1, ("x",))], days=14)
        provider.assert_called_once_with(days=14)

    def test_provider_without_days_kwarg_still_works(self) -> None:
        """Falls back to bare call when provider rejects ``days=``."""
        calls: list[tuple] = []

        def provider() -> list:
            calls.append(())
            return []

        theme_gap(provider, [Theme("x", 1, ("x",))])
        assert calls == [()]

    def test_provider_object_with_method(self) -> None:
        class Reader:
            def get_recent_completions(self, days: int) -> list[dict]:
                return [_story("alpha thing")]

        gaps = theme_gap(Reader(), [Theme("a", 2, ("alpha",))])
        # 1 match, target 2 → deficit 1, returned.
        assert len(gaps) == 1

    def test_worst_gap_is_sorted_first(self) -> None:
        """Deficit 4 must appear before deficit 1."""
        theme_low = Theme("low", target_per_week=5, keywords=("low",))   # 0 matches → 5
        theme_high = Theme("high", target_per_week=5, keywords=("high",))  # 4 matches → 1
        stories = [_story("high one"), _story("high two"), _story("high three"), _story("high four")]
        provider = MagicMock(return_value=stories)

        gaps = theme_gap(provider, [theme_low, theme_high])

        assert gaps == [theme_low, theme_high]

    def test_themes_at_or_above_target_are_excluded(self) -> None:
        theme = Theme("ok", target_per_week=2, keywords=("ok",))
        provider = MagicMock(return_value=[_story("ok one"), _story("ok two")])
        assert theme_gap(provider, [theme]) == []

    def test_zero_target_theme_is_never_a_gap(self) -> None:
        theme = Theme("optional", target_per_week=0, keywords=("nothing-here",))
        provider = MagicMock(return_value=[])
        assert theme_gap(provider, [theme]) == []

    def test_keyword_matching_is_case_insensitive(self) -> None:
        theme = Theme("crash", target_per_week=2, keywords=("crash",))
        provider = MagicMock(return_value=[_story("System CRASH detected")])
        # 1 match, target 2 → still a gap of 1
        gaps = theme_gap(provider, [theme])
        assert gaps == [theme]

    def test_keyword_matches_against_description(self) -> None:
        theme = Theme("a", target_per_week=1, keywords=("metric",))
        provider = MagicMock(return_value=[_story("unrelated", description="ships a new metric")])
        assert theme_gap(provider, [theme]) == []  # 1 match meets target 1

    def test_keyword_matches_against_labels(self) -> None:
        theme = Theme("a", target_per_week=1, keywords=("autonomy",))
        provider = MagicMock(return_value=[_story("x", labels=["cat:autonomy", "src:planning"])])
        assert theme_gap(provider, [theme]) == []

    def test_one_story_can_match_multiple_themes(self) -> None:
        theme_a = Theme("crash", target_per_week=1, keywords=("crash",))
        theme_b = Theme("metric", target_per_week=1, keywords=("metric",))
        provider = MagicMock(return_value=[_story("crash metric reported")])
        # Both themes hit their target from a single story — neither is a gap.
        assert theme_gap(provider, [theme_a, theme_b]) == []

    def test_one_story_increments_each_matched_theme_only_once(self) -> None:
        """A story with two keywords for the same theme counts once."""
        theme = Theme("a", target_per_week=2, keywords=("crash", "stall"))
        provider = MagicMock(return_value=[_story("crash and stall in one go")])
        # One story → count 1, not 2. Still a gap of 1.
        assert theme_gap(provider, [theme]) == [theme]

    def test_provider_returning_none_yields_no_data(self) -> None:
        provider = MagicMock(return_value=None)
        theme = Theme("a", target_per_week=3, keywords=("a",))
        assert theme_gap(provider, [theme]) == [theme]

    def test_provider_exception_is_swallowed(self) -> None:
        provider = MagicMock(side_effect=RuntimeError("jira down"))
        theme = Theme("a", target_per_week=3, keywords=("a",))
        # Provider failure → 0 completions → every positive-target theme is a gap.
        assert theme_gap(provider, [theme]) == [theme]

    def test_empty_themes_returns_empty_list(self) -> None:
        provider = MagicMock(return_value=[])
        assert theme_gap(provider, []) == []

    def test_theme_without_keywords_is_always_a_gap(self) -> None:
        """No keywords means nothing can ever match — the theme is
        chronically under-served until keywords are added."""
        theme = Theme("naked", target_per_week=1, keywords=())
        provider = MagicMock(return_value=[_story("anything goes")])
        assert theme_gap(provider, [theme]) == [theme]

    def test_malformed_story_entries_are_ignored(self) -> None:
        """Provider returning non-dict items must not crash."""
        theme = Theme("a", target_per_week=1, keywords=("alpha",))
        provider = MagicMock(return_value=["not a dict", None, _story("alpha hit")])
        assert theme_gap(provider, [theme]) == []
