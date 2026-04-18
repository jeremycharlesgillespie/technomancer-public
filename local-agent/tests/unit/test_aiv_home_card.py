"""Tests for the 'Recent Validation' card on the hub home page (TK-691).

Acceptance criteria:
- Card HTML shows the 5 most-recent story_quality rows + a '24h avg: X.X' label.
- Empty DB → card shows 'no validations yet'.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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


@pytest.fixture(autouse=True)
def _empty_ideas():
    """Keep the hub render path independent of real board data."""
    with patch("idea_board.web.load_ideas", return_value=[]):
        yield


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
) -> None:
    """Insert a synthetic story_quality row for the card to read."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "story_key": story_key,
        "story_title": story_title,
        "merged_at": validated_at,
        "validated_at": validated_at,
        "meets_requirements": 9,
        "code_quality": 8,
        "test_quality": 7,
        "security_safety": 10,
        "scope_discipline": 9,
        "edge_cases": 6,
        "product_impact": 8,
        "overall_score": overall,
        "red_flags_json": json.dumps([]),
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


def _card_html(body: str) -> str:
    """Slice the 'Recent Validation' card out of the full hub HTML."""
    idx = body.index("Recent Validation")
    start = body.rfind("<a ", 0, idx)
    assert start != -1, "Expected an <a> wrapping the Recent Validation card"
    end = body.index("</a>", idx) + len("</a>")
    return body[start:end]


class TestRecentValidationCardEmpty:
    """Empty AIV DB → card renders an empty-state message."""

    def test_card_appears_with_empty_state_when_no_rows(self, client):
        body = client.get("/").get_data(as_text=True)

        assert "Recent Validation" in body
        card = _card_html(body)
        assert "no validations yet" in card

    def test_empty_card_links_to_quality(self, client):
        body = client.get("/").get_data(as_text=True)

        card = _card_html(body)
        assert 'href="/quality"' in card

    def test_empty_card_omits_24h_avg_label(self, client):
        body = client.get("/").get_data(as_text=True)

        card = _card_html(body)
        assert "24h avg" not in card


class TestRecentValidationCardWithRows:
    """Acceptance: card shows last 5 rows + '24h avg: X.X' label."""

    def test_card_shows_24h_avg_label(self, client):
        now = datetime.now(timezone.utc)
        for i, score in enumerate([7.0, 8.0, 9.0]):
            _insert_row(
                story_key=f"TK-1{i:03d}",
                validated_at=(now - timedelta(minutes=i)).isoformat(),
                overall=score,
            )

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        assert "24h avg: 8.0" in card

    def test_card_shows_only_5_most_recent_rows(self, client):
        now = datetime.now(timezone.utc)
        for i in range(7):
            _insert_row(
                story_key=f"TK-2{i:03d}",
                validated_at=(now - timedelta(minutes=i)).isoformat(),
                overall=8.0,
            )

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        # 5 newest (i=0..4) appear; older 2 (i=5,6) do not.
        for i in range(5):
            assert f"TK-2{i:03d}" in card
        for i in range(5, 7):
            assert f"TK-2{i:03d}" not in card

    def test_card_orders_newest_first(self, client):
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-3001",
            validated_at=(now - timedelta(hours=2)).isoformat(),
        )
        _insert_row(
            story_key="TK-3002",
            validated_at=now.isoformat(),
        )

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        assert card.index("TK-3002") < card.index("TK-3001")

    def test_card_links_to_quality(self, client):
        _insert_row(story_key="TK-4001")

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        assert 'href="/quality"' in card

    def test_24h_avg_excludes_older_rows(self, client):
        """Stories validated >24h ago are not part of the average."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-5001",
            validated_at=(now - timedelta(hours=48)).isoformat(),
            overall=2.0,
        )
        _insert_row(
            story_key="TK-5002",
            validated_at=(now - timedelta(hours=1)).isoformat(),
            overall=10.0,
        )

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        # Average must reflect only the in-window row (10.0), not the 48h-old 2.0.
        assert "24h avg: 10.0" in card

    def test_24h_avg_ignores_negative_sentinels(self, client):
        """Sentinel -1 overall scores are excluded from the mean."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-6001",
            validated_at=now.isoformat(),
            overall=-1.0,
        )
        _insert_row(
            story_key="TK-6002",
            validated_at=now.isoformat(),
            overall=8.0,
        )

        body = client.get("/").get_data(as_text=True)
        card = _card_html(body)

        assert "24h avg: 8.0" in card


class TestRecentValidationCardSurvivesMissingDb:
    """A missing AIV DB file must not 500 the hub home."""

    def test_hub_renders_when_db_missing(self, client, tmp_path, monkeypatch):
        ghost = tmp_path / "does-not-exist.db"
        monkeypatch.setattr(aiv_schema, "DB_PATH", ghost)

        resp = client.get("/")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        card = _card_html(body)
        assert "no validations yet" in card
