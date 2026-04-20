"""Tests for scripts/load_slide_loader.py — load_slide()."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from scripts.load_slide_loader import load_slide


class TestLoadSlide:
    def test_load_slide_returns_module_on_good_file(self, tmp_path: Path) -> None:
        """load_slide returns a ModuleType for a valid Python file."""
        p = tmp_path / "good_slide.py"
        p.write_text("value = 99\n")
        result = load_slide(str(p))
        assert result is not None
        assert isinstance(result, types.ModuleType)
        assert result.value == 99

    def test_load_slide_returns_none_on_bad_module(self, tmp_path: Path) -> None:
        """load_slide returns None for a file with a SyntaxError."""
        p = tmp_path / "bad_slide.py"
        p.write_text("def broken(:\n    pass\n")
        result = load_slide(str(p))
        assert result is None
