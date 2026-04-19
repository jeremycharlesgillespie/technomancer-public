"""Tests for agent/incidents.py — incidents log append helper."""

from pathlib import Path

import pytest

from agent.incidents import INCIDENTS_PATH, _TABLE_HEADER, _TABLE_SEP, append_incident


@pytest.fixture
def incidents_file(tmp_path) -> Path:
    """Return a temp incidents.md pre-seeded with the standard table header."""
    f = tmp_path / "incidents.md"
    f.write_text(
        "# Incidents Log\n\n"
        f"{_TABLE_HEADER}\n"
        f"{_TABLE_SEP}\n"
        "| 2026-04-17 | something broke | logs | abc123 |\n",
        encoding="utf-8",
    )
    return f


def test_append_adds_row(incidents_file):
    result = append_incident(
        "2026-04-18",
        "new bug",
        "alert fired",
        "def456",
        path=incidents_file,
    )
    assert result is True
    text = incidents_file.read_text(encoding="utf-8")
    assert "| 2026-04-18 | new bug | alert fired | def456 |" in text


def test_append_preserves_existing_rows(incidents_file):
    append_incident("2026-04-18", "new bug", "alert fired", "def456", path=incidents_file)
    text = incidents_file.read_text(encoding="utf-8")
    assert "| 2026-04-17 | something broke | logs | abc123 |" in text
    assert "| 2026-04-18 | new bug | alert fired | def456 |" in text


def test_append_idempotent_on_duplicate(incidents_file):
    # Same (date, what) twice — second call returns False, no duplicate row
    append_incident("2026-04-18", "new bug", "alert fired", "def456", path=incidents_file)
    result = append_incident("2026-04-18", "new bug", "alert fired", "def456", path=incidents_file)
    assert result is False
    text = incidents_file.read_text(encoding="utf-8")
    assert text.count("new bug") == 1


def test_append_idempotent_on_existing_seed_row(incidents_file):
    result = append_incident(
        "2026-04-17",
        "something broke",
        "logs",
        "abc123",
        path=incidents_file,
    )
    assert result is False
    text = incidents_file.read_text(encoding="utf-8")
    assert text.count("something broke") == 1


def test_append_to_empty_file(tmp_path):
    f = tmp_path / "incidents.md"
    result = append_incident("2026-04-19", "first bug", "manual", "xyz", path=f)
    assert result is True
    text = f.read_text(encoding="utf-8")
    assert _TABLE_HEADER in text
    assert "| 2026-04-19 | first bug | manual | xyz |" in text


def test_append_to_nonexistent_file(tmp_path):
    f = tmp_path / "subdir" / "incidents.md"
    result = append_incident("2026-04-19", "bug", "check", "fix", path=f)
    assert result is True
    assert f.exists()


def test_append_multiple_rows_in_order(incidents_file):
    append_incident("2026-04-18", "bug A", "ci", "c1", path=incidents_file)
    append_incident("2026-04-19", "bug B", "alert", "c2", path=incidents_file)
    lines = [
        ln for ln in incidents_file.read_text(encoding="utf-8").splitlines()
        if ln.startswith("|") and "Date" not in ln and "---" not in ln
    ]
    assert len(lines) == 3
    assert "bug A" in lines[1]
    assert "bug B" in lines[2]


def test_incidents_path_points_to_docs(tmp_path):
    # Sanity-check the default path is under docs/
    assert INCIDENTS_PATH.parts[-2] == "docs"
    assert INCIDENTS_PATH.name == "incidents.md"
