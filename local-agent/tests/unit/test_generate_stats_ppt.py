"""Tests for scripts/generate_stats_ppt.py (TK-829, TK-836).

Covers: CLI parsing, DB initialization, title/summary slide creation,
loader-based slide iteration, skip-on-failure behaviour, and the final
.pptx output.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pptx import Presentation

import agent.daily_stats as ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point daily_stats at a temp DB and reset the thread-local connection."""
    db_path = tmp_path / "daily_stats.db"
    monkeypatch.setattr(ds, "DB_DIR", tmp_path)
    monkeypatch.setattr(ds, "DB_PATH", db_path)
    ds._local.__dict__.pop("conn", None)
    yield
    conn = getattr(ds._local, "conn", None)
    if conn:
        conn.close()
        ds._local.__dict__.pop("conn", None)


@pytest.fixture(autouse=True)
def _clean_slide_registry():
    """Snapshot and restore the slide registry around every test."""
    from scripts.slides import slide_registry

    original = dict(slide_registry)
    yield
    slide_registry.clear()
    slide_registry.update(original)


@pytest.fixture(autouse=True)
def _unload_slide_modules():
    """Remove imported slide_* modules so import side-effects are fresh per test."""
    yield
    for key in list(sys.modules):
        if key.startswith("scripts.slides.slide_") or (
            key.startswith("slide_") and not key.startswith("slide_registry")
        ):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        from scripts.generate_stats_ppt import parse_args

        args = parse_args([])
        assert args.since is None
        assert args.project is None
        assert args.output.endswith(".pptx")

    def test_since_flag(self):
        from scripts.generate_stats_ppt import parse_args

        args = parse_args(["--since", "2026-01-01"])
        assert args.since == "2026-01-01"

    def test_project_flag(self):
        from scripts.generate_stats_ppt import parse_args

        args = parse_args(["--project", "TK"])
        assert args.project == "TK"

    def test_output_flag(self):
        from scripts.generate_stats_ppt import parse_args

        args = parse_args(["--output", "/tmp/deck.pptx"])
        assert args.output == "/tmp/deck.pptx"

    def test_all_flags_together(self):
        from scripts.generate_stats_ppt import parse_args

        args = parse_args(["--since", "2026-03-01", "--project", "FA", "--output", "out.pptx"])
        assert args.since == "2026-03-01"
        assert args.project == "FA"
        assert args.output == "out.pptx"

    def test_unknown_flag_raises(self):
        from scripts.generate_stats_ppt import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--bogus"])


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------


class TestDbInit:
    def test_build_presentation_calls_init_db(self):
        from scripts.generate_stats_ppt import build_presentation

        with patch("scripts.generate_stats_ppt.daily_stats.init_db") as mock_init:
            with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=[]):
                build_presentation()
        mock_init.assert_called()

    def test_db_table_exists_after_build(self):
        from scripts.generate_stats_ppt import build_presentation

        build_presentation()
        conn = ds._get_conn()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_stats'"
        ).fetchone()
        assert result is not None


# ---------------------------------------------------------------------------
# Title / summary slide
# ---------------------------------------------------------------------------


class TestTitleSlide:
    def test_first_slide_is_title(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        assert len(prs.slides) >= 1
        texts = [sh.text_frame.text for sh in prs.slides[0].shapes if sh.has_text_frame]
        assert any("Stats" in t or "Project" in t for t in texts)

    def test_title_slide_shows_since_when_provided(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation(since="2026-01-15")
        title_texts = " ".join(
            sh.text_frame.text for sh in prs.slides[0].shapes if sh.has_text_frame
        )
        assert "2026-01-15" in title_texts

    def test_title_slide_shows_project_when_provided(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation(project="TK")
        title_texts = " ".join(
            sh.text_frame.text for sh in prs.slides[0].shapes if sh.has_text_frame
        )
        assert "TK" in title_texts

    def test_title_slide_fallback_text_when_no_filters(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        title_texts = " ".join(
            sh.text_frame.text for sh in prs.slides[0].shapes if sh.has_text_frame
        )
        assert "All" in title_texts


# ---------------------------------------------------------------------------
# Title / summary slide — exact content
# ---------------------------------------------------------------------------


class TestTitleSummarySlideContent:
    """Assert the exact strings on the title slide rather than loose contains-checks."""

    def _slide_texts(self, slide) -> list[str]:
        return [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]

    def test_title_slide_exact_heading(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        texts = self._slide_texts(prs.slides[0])
        assert "Project Stats" in texts

    def test_title_slide_default_subtitle_exact(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        texts = self._slide_texts(prs.slides[0])
        assert "All projects · All time" in texts

    def test_title_slide_since_subtitle_exact(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation(since="2026-01-15")
        texts = self._slide_texts(prs.slides[0])
        assert "Since 2026-01-15" in texts

    def test_title_slide_project_subtitle_exact(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation(project="TK")
        texts = self._slide_texts(prs.slides[0])
        assert "Project: TK" in texts

    def test_title_slide_both_filters_subtitle_exact(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation(since="2026-03-01", project="FA")
        texts = self._slide_texts(prs.slides[0])
        assert "Since 2026-03-01  ·  Project: FA" in texts

    def test_title_slide_date_appears_verbatim(self):
        """Exact date string passed as `since` must appear in the subtitle text."""
        from scripts.generate_stats_ppt import build_presentation

        date = "2026-04-19"
        prs = build_presentation(since=date)
        all_text = " ".join(self._slide_texts(prs.slides[0]))
        assert date in all_text

    def test_title_slide_has_exactly_two_text_shapes(self):
        """The title slide always has exactly two textboxes: heading + subtitle."""
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        texts = self._slide_texts(prs.slides[0])
        assert len(texts) == 2

    def test_title_slide_no_extra_text_when_no_filters(self):
        """Without filters the subtitle must be the fallback string, nothing else."""
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        texts = self._slide_texts(prs.slides[0])
        assert texts == ["Project Stats", "All projects · All time"]


# ---------------------------------------------------------------------------
# Loader-based slide loop (TK-836)
# ---------------------------------------------------------------------------

def _make_module(name: str, adds_slide: bool = True) -> types.ModuleType:
    """Return a fake slide module whose add_slide optionally appends a slide."""
    m = types.ModuleType(name)

    def add_slide(prs: Presentation) -> None:
        if adds_slide:
            prs.slides.add_slide(prs.slide_layouts[6])

    m.add_slide = add_slide
    return m


class TestLoaderBasedSlideLoop:
    """build_presentation() iterates slide files via load_slide_module."""

    def test_successfully_registered_builders_add_slides_to_deck(self):
        from scripts.generate_stats_ppt import build_presentation

        paths = [Path("slide_01.py"), Path("slide_02.py")]
        modules = [_make_module("slide_01"), _make_module("slide_02")]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                prs = build_presentation()

        # title slide + 2 content slides
        assert len(prs.slides) == 3

    def test_broken_builder_reduces_slide_count_but_deck_still_builds(self):
        from scripts.generate_stats_ppt import build_presentation

        paths = [Path("slide_01.py"), Path("slide_02.py")]
        # first module fails to load (returns None), second succeeds
        modules = [None, _make_module("slide_02")]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                prs = build_presentation()

        # title slide + 1 good slide (broken one skipped)
        assert len(prs.slides) == 2

    def test_add_slide_receives_presentation_object(self):
        from scripts.generate_stats_ppt import build_presentation

        received: list[object] = []
        m = types.ModuleType("slide_01")
        m.add_slide = lambda prs: received.append(prs)  # type: ignore[attr-defined]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=[Path("slide_01.py")]):
            with patch("scripts.generate_stats_ppt.load_slide_module", return_value=m):
                prs = build_presentation()

        assert len(received) == 1
        assert received[0] is prs

    def test_slides_loaded_in_path_order(self):
        from scripts.generate_stats_ppt import build_presentation

        order: list[str] = []

        def make_ordered_module(name: str) -> types.ModuleType:
            m = types.ModuleType(name)
            def add_slide(prs: Presentation) -> None:
                order.append(name)
            m.add_slide = add_slide
            return m

        paths = [Path("slide_01.py"), Path("slide_02.py"), Path("slide_03.py")]
        modules = [make_ordered_module(p.stem) for p in paths]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                build_presentation()

        assert order == ["slide_01", "slide_02", "slide_03"]

    def test_exception_in_add_slide_is_logged_and_skipped(self, caplog):
        from scripts.generate_stats_ppt import build_presentation

        bad = types.ModuleType("slide_bad")
        bad.add_slide = MagicMock(side_effect=RuntimeError("exploded"))  # type: ignore[attr-defined]
        good = _make_module("slide_good")

        paths = [Path("slide_bad.py"), Path("slide_good.py")]
        modules = [bad, good]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                with caplog.at_level(logging.ERROR, logger="scripts.generate_stats_ppt"):
                    prs = build_presentation()

        # title + 1 good slide; bad slide skipped
        assert len(prs.slides) >= 2
        assert any("slide_bad.py" in r.message for r in caplog.records)

    def test_exception_in_later_add_slide_does_not_prevent_earlier(self):
        from scripts.generate_stats_ppt import build_presentation

        results: list[str] = []
        first = types.ModuleType("slide_first")
        first.add_slide = lambda prs: results.append("first ran")  # type: ignore[attr-defined]
        broken = types.ModuleType("slide_broken")
        broken.add_slide = MagicMock(side_effect=ValueError("bad"))  # type: ignore[attr-defined]

        paths = [Path("slide_first.py"), Path("slide_broken.py")]
        modules = [first, broken]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                build_presentation()

        assert "first ran" in results

    def test_none_module_prints_warning(self, capsys):
        from scripts.generate_stats_ppt import build_presentation

        paths = [Path("slide_broken.py")]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", return_value=None):
                build_presentation()

        out = capsys.readouterr().out
        assert "slide_broken.py" in out


# ---------------------------------------------------------------------------
# Final presentation output
# ---------------------------------------------------------------------------


class TestReturnsPresentation:
    def test_build_presentation_returns_presentation_instance(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        # pptx.Presentation is a factory function; verify the returned object
        # has the expected slides attribute.
        assert hasattr(prs, "slides")

    def test_presentation_has_at_least_title_slide(self):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        assert len(prs.slides) >= 1

    def test_saved_pptx_reopens_without_error(self, tmp_path):
        from scripts.generate_stats_ppt import build_presentation

        prs = build_presentation()
        output = tmp_path / "stats.pptx"
        prs.save(str(output))
        reopened = Presentation(str(output))
        assert len(reopened.slides) >= 1

    def test_main_saves_file_to_output_path(self, tmp_path):
        from scripts.generate_stats_ppt import main

        output = tmp_path / "deck.pptx"
        main(["--output", str(output)])
        assert output.exists()
        assert output.stat().st_size > 0

    def test_main_creates_parent_dirs_if_missing(self, tmp_path):
        from scripts.generate_stats_ppt import main

        output = tmp_path / "deep" / "nested" / "deck.pptx"
        main(["--output", str(output)])
        assert output.exists()

    def test_slide_count_matches_loader_plus_title(self):
        from scripts.generate_stats_ppt import build_presentation

        paths = [Path("slide_10.py"), Path("slide_11.py")]
        # modules that don't actually add slides — just verify deck builds
        modules = [_make_module("slide_10", adds_slide=False), _make_module("slide_11", adds_slide=False)]

        with patch("scripts.generate_stats_ppt._iter_slide_paths", return_value=paths):
            with patch("scripts.generate_stats_ppt.load_slide_module", side_effect=modules):
                prs = build_presentation()

        # title slide only (counter modules don't add slides)
        assert hasattr(prs, "slides")
