"""Tests for /quality filter + sort + pagination UI (TK-690).

Covers:
- GET /quality?project=TK returns only rows whose story_key starts with TK-.
- GET /quality?sort=score returns rows ordered ASC by overall_score.
- Pagination: 50 rows per page, page 2 shows the next 50.
- Toolbar widgets render (dropdown + sort selector).
- Invalid sort / page values fall back gracefully.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent import aiv_schema
from idea_board.web import app


@pytest.fixture(autouse=True)
def _isolate_aiv_db(tmp_path, monkeypatch):
    """Point aiv_schema at a per-test temporary SQLite DB."""
    db_path = tmp_path / "aiv.db"
    monkeypatch.setattr(aiv_schema, "DB_DIR", tmp_path)
    monkeypatch.setattr(aiv_schema, "DB_PATH", db_path)
    aiv_schema._local.__dict__.pop("conn", None)
    yield
    conn = getattr(aiv_schema._local, "conn", None)
    if conn is not None:
        conn.close()
        aiv_schema._local.conn = None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _insert_row(
    *,
    story_key: str,
    story_title: str = "Some story",
    validated_at: str | None = None,
    overall: float = 8.0,
    red_flags: list[str] | None = None,
) -> None:
    """Insert a synthetic story_quality row for filter/sort tests."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "story_key": story_key,
        "story_title": story_title,
        "merged_at": validated_at,
        "validated_at": validated_at,
        "meets_requirements": 8,
        "code_quality": 8,
        "test_quality": 8,
        "security_safety": 8,
        "scope_discipline": 8,
        "edge_cases": 8,
        "product_impact": 8,
        "overall_score": overall,
        "red_flags_json": json.dumps(red_flags or []),
        "verification_method": "tests-only",
        "verification_output": "",
        "reasoning_json": json.dumps({}),
        "error": None,
    }
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(f":{k}" for k in payload)
    conn.execute(
        f"INSERT INTO story_quality ({columns}) VALUES ({placeholders})",
        payload,
    )
    conn.commit()


def _row_order(body: str, keys: list[str]) -> list[str]:
    """Return ``keys`` sorted by their first appearance in ``body``.

    Keys that don't appear are dropped. Used to assert server-rendered row
    order without relying on fragile HTML parsing.
    """
    positions = []
    for k in keys:
        idx = body.find(k)
        if idx >= 0:
            positions.append((idx, k))
    positions.sort()
    return [k for _, k in positions]


class TestProjectFilter:
    """Acceptance: ``GET /quality?project=TK`` returns only TK rows."""

    def test_project_filter_scopes_rows_to_prefix(self, client):
        _insert_row(story_key="TK-100", story_title="TK story A")
        _insert_row(story_key="TK-101", story_title="TK story B")
        _insert_row(story_key="FA-200", story_title="FA story")

        resp = client.get("/quality?project=TK")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "TK-100" in body
        assert "TK-101" in body
        assert "FA-200" not in body

    def test_project_filter_empty_returns_all(self, client):
        """No ``project`` param => every project's rows appear."""
        _insert_row(story_key="TK-300", story_title="TK row")
        _insert_row(story_key="FA-400", story_title="FA row")

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert "TK-300" in body
        assert "FA-400" in body

    def test_project_filter_unknown_project_renders_empty(self, client):
        """Filter matching no rows still renders the page, not a 500."""
        _insert_row(story_key="TK-500")

        resp = client.get("/quality?project=ZZ")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "TK-500" not in body
        # Page still renders its heading.
        assert "Recent Validations" in body

    def test_project_filter_dropdown_reflects_current_value(self, client):
        """The <select name='project'> marks the chosen project as selected."""
        _insert_row(story_key="TK-600")
        _insert_row(story_key="FA-700")

        resp = client.get("/quality?project=FA")
        body = resp.get_data(as_text=True)

        # The dropdown should include TK and FA options and mark FA selected.
        assert 'value="FA" selected' in body or "value='FA' selected" in body


class TestSortByScore:
    """Acceptance: ``GET /quality?sort=score`` orders rows by overall_score ASC."""

    def test_sort_by_score_ascending(self, client):
        _insert_row(story_key="TK-800", overall=9.0)
        _insert_row(story_key="TK-801", overall=3.0)
        _insert_row(story_key="TK-802", overall=6.0)

        resp = client.get("/quality?sort=score")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        order = _row_order(body, ["TK-800", "TK-801", "TK-802"])
        assert order == ["TK-801", "TK-802", "TK-800"]

    def test_sort_default_is_newest_first(self, client):
        """With no ``sort`` param, default remains validated_at DESC."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-900",
            validated_at=(now - timedelta(hours=3)).isoformat(),
            overall=1.0,
        )
        _insert_row(
            story_key="TK-901",
            validated_at=now.isoformat(),
            overall=9.0,
        )

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)
        order = _row_order(body, ["TK-900", "TK-901"])
        # Newest first — TK-901 before TK-900 even though its score is higher.
        assert order == ["TK-901", "TK-900"]

    def test_sort_invalid_value_falls_back_to_recent(self, client):
        """An unknown sort value does not 500 and uses the default order."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-950",
            validated_at=(now - timedelta(hours=1)).isoformat(),
            overall=2.0,
        )
        _insert_row(
            story_key="TK-951",
            validated_at=now.isoformat(),
            overall=9.0,
        )

        resp = client.get("/quality?sort=banana")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        order = _row_order(body, ["TK-950", "TK-951"])
        assert order == ["TK-951", "TK-950"]


class TestPagination:
    """50 rows per page; ``?page=N`` advances through the result set."""

    def test_first_page_caps_at_50_rows(self, client):
        now = datetime.now(timezone.utc)
        for i in range(55):
            _insert_row(
                story_key=f"TK-{1000 + i}",
                validated_at=(now - timedelta(minutes=i)).isoformat(),
            )

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        # The 50 most recent rows appear; older ones are on page 2.
        assert "TK-1000" in body  # newest (i=0)
        assert "TK-1049" in body  # 50th most recent
        assert "TK-1050" not in body  # pushed to page 2

    def test_second_page_shows_remaining_rows(self, client):
        now = datetime.now(timezone.utc)
        for i in range(55):
            _insert_row(
                story_key=f"TK-{2000 + i}",
                validated_at=(now - timedelta(minutes=i)).isoformat(),
            )

        resp = client.get("/quality?page=2")
        body = resp.get_data(as_text=True)

        assert "TK-2000" not in body  # newest is on page 1
        assert "TK-2050" in body  # page-2 rows appear
        assert "TK-2054" in body

    def test_invalid_page_falls_back_to_first_page(self, client):
        _insert_row(story_key="TK-3000")

        resp = client.get("/quality?page=not-a-number")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "TK-3000" in body


class TestToolbarRendering:
    """The filter+sort toolbar must be present in the rendered page."""

    def test_toolbar_exposes_project_dropdown(self, client):
        _insert_row(story_key="TK-4000")
        _insert_row(story_key="FA-4001")

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert 'name="project"' in body or "name='project'" in body
        # Both project prefixes show as dropdown options.
        assert ">TK<" in body
        assert ">FA<" in body

    def test_toolbar_exposes_sort_selector(self, client):
        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert 'name="sort"' in body or "name='sort'" in body
        assert 'value="recent"' in body or "value='recent'" in body
        assert 'value="score"' in body or "value='score'" in body
