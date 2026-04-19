"""Tests for scripts/slides/loader.py — load_slide_module."""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

from scripts.slides.loader import load_slide_module


@pytest.fixture()
def good_slide(tmp_path: Path) -> Path:
    """A valid Python module file."""
    p = tmp_path / "good_slide.py"
    p.write_text("value = 42\n")
    return p


@pytest.fixture()
def broken_slide(tmp_path: Path) -> Path:
    """A Python file with a SyntaxError."""
    p = tmp_path / "broken_slide.py"
    p.write_text("def foo(:\n    pass\n")
    return p


@pytest.fixture()
def import_error_slide(tmp_path: Path) -> Path:
    """A Python file that raises ImportError at import time."""
    p = tmp_path / "import_error_slide.py"
    p.write_text("import _nonexistent_package_xyz\n")
    return p


@pytest.fixture()
def attribute_error_slide(tmp_path: Path) -> Path:
    """A Python file that raises AttributeError at import time."""
    p = tmp_path / "attribute_error_slide.py"
    p.write_text(textwrap.dedent("""\
        class _Stub:
            pass

        _Stub().nonexistent_attr.something
    """))
    return p


class TestLoadSuccess:
    def test_load_success(self, good_slide: Path) -> None:
        """load_slide_module returns the module object for a valid file."""
        mod = load_slide_module(str(good_slide))
        assert mod is not None
        assert isinstance(mod, types.ModuleType)

    def test_module_attributes_accessible(self, good_slide: Path) -> None:
        """Attributes defined in the slide file are accessible on the returned module."""
        mod = load_slide_module(str(good_slide))
        assert mod is not None
        assert mod.value == 42


class TestLoadFailure:
    def test_load_failure_logs_and_returns_none_syntax(
        self, broken_slide: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SyntaxError → returns None and logs a WARNING."""
        import logging

        with caplog.at_level(logging.WARNING, logger="scripts.slides.loader"):
            result = load_slide_module(str(broken_slide))

        assert result is None
        assert any("broken_slide.py" in r.message for r in caplog.records)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_load_failure_logs_and_returns_none_import(
        self, import_error_slide: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ImportError → returns None and logs a WARNING."""
        import logging

        with caplog.at_level(logging.WARNING, logger="scripts.slides.loader"):
            result = load_slide_module(str(import_error_slide))

        assert result is None
        assert any("import_error_slide.py" in r.message for r in caplog.records)

    def test_load_failure_logs_and_returns_none_attribute(
        self, attribute_error_slide: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AttributeError → returns None and logs a WARNING."""
        import logging

        with caplog.at_level(logging.WARNING, logger="scripts.slides.loader"):
            result = load_slide_module(str(attribute_error_slide))

        assert result is None
        assert any("attribute_error_slide.py" in r.message for r in caplog.records)

    def test_missing_file_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A path that does not exist → returns None and logs a WARNING."""
        import logging

        missing = tmp_path / "missing_slide.py"
        with caplog.at_level(logging.WARNING, logger="scripts.slides.loader"):
            result = load_slide_module(str(missing))

        assert result is None
        assert any("missing_slide.py" in r.message for r in caplog.records)
