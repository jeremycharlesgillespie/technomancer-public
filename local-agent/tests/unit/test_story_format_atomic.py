"""Tests for the ATOMIC_LABEL constant in agent.story_format."""

from pathlib import Path

from agent import story_format
from agent.story_format import ATOMIC_LABEL


def test_atomic_label_value():
    assert ATOMIC_LABEL == "atomic"


def test_atomic_label_is_string():
    assert isinstance(ATOMIC_LABEL, str)


def test_atomic_label_importable_from_module_namespace():
    assert story_format.ATOMIC_LABEL == "atomic"


def test_atomic_stories_doc_exists():
    repo_root = Path(__file__).resolve().parents[3]
    doc = repo_root / "docs" / "atomic_stories.md"
    assert doc.exists(), f"expected doc at {doc}"

    body = doc.read_text(encoding="utf-8")
    assert "atomic" in body.lower()
    # The doc must define all four criteria bullets.
    assert body.count("\n1. ") >= 1
    assert body.count("\n2. ") >= 1
    assert body.count("\n3. ") >= 1
    assert body.count("\n4. ") >= 1
