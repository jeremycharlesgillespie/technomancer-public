"""Tests for /quality + /api/aiv/recent + /api/aiv/flags (TK-684).

Covers the dashboard that makes AIV's seven-axis quality scores visible:

- GET /quality renders HTML with the 'Recent Validations' heading and every
  validated story as a row.
- Rows for stories with at least one red flag carry ``class='flagged'`` so
  the operator can spot failing work at a glance.
- GET /api/aiv/recent returns a JSON envelope with every documented key
  per story.
- GET /api/aiv/flags returns 24h flag counts grouped by type.
- Missing AIV DB collapses to empty responses, not a 500.
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
    merged_at: str | None = None,
    scores: tuple[int, int, int, int, int, int, int] = (9, 8, 7, 10, 9, 6, 8),
    overall: float = 8.1,
    red_flags: list[str] | None = None,
    verification_method: str = "tests-only",
    error: str | None = None,
) -> None:
    """Insert a synthetic story_quality row for use in tests."""
    aiv_schema.init_db()
    conn = aiv_schema._get_conn()
    if validated_at is None:
        validated_at = datetime.now(timezone.utc).isoformat()
    if merged_at is None:
        merged_at = validated_at
    payload = {
        "story_key": story_key,
        "story_title": story_title,
        "merged_at": merged_at,
        "validated_at": validated_at,
        "meets_requirements": scores[0],
        "code_quality": scores[1],
        "test_quality": scores[2],
        "security_safety": scores[3],
        "scope_discipline": scores[4],
        "edge_cases": scores[5],
        "product_impact": scores[6],
        "overall_score": overall,
        "red_flags_json": json.dumps(red_flags or []),
        "verification_method": verification_method,
        "verification_output": "",
        "reasoning_json": json.dumps({}),
        "error": error,
    }
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(f":{k}" for k in payload)
    conn.execute(
        f"INSERT INTO story_quality ({columns}) VALUES ({placeholders})",
        payload,
    )
    conn.commit()


class TestQualityPage:
    """GET /quality — HTML dashboard."""

    def test_quality_page_returns_200_with_heading(self, client):
        """Acceptance: GET /quality returns 200 + HTML with 'Recent Validations'."""
        resp = client.get("/quality")

        assert resp.status_code == 200
        assert "text/html" in resp.content_type
        body = resp.get_data(as_text=True)
        assert "Recent Validations" in body

    def test_quality_page_renders_empty_state_when_no_rows(self, client):
        """No validated stories yet — page still renders, shows empty message."""
        resp = client.get("/quality")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "Recent Validations" in body
        assert "No validated stories yet" in body

    def test_quality_page_row_with_red_flag_has_flagged_class(self, client):
        """Acceptance: a row with any red flag carries class='flagged'."""
        _insert_row(
            story_key="TK-9001",
            story_title="Story with a flag",
            red_flags=["broad_except"],
        )

        resp = client.get("/quality")

        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "TK-9001" in body
        # The row for TK-9001 should be in a <tr class='flagged'>.
        # Locate the row by key and check the surrounding <tr> class.
        key_idx = body.index("TK-9001")
        preceding = body[:key_idx]
        last_tr = preceding.rfind("<tr")
        assert last_tr != -1, "Expected a <tr> before the TK-9001 cell"
        tr_open = body[last_tr:key_idx]
        assert "class='flagged'" in tr_open or 'class="flagged"' in tr_open

    def test_quality_page_clean_row_is_not_flagged(self, client):
        """A row with no red flags must NOT carry the flagged class."""
        _insert_row(
            story_key="TK-9002",
            story_title="Spotless story",
            red_flags=[],
        )

        resp = client.get("/quality")

        body = resp.get_data(as_text=True)
        key_idx = body.index("TK-9002")
        preceding = body[:key_idx]
        last_tr = preceding.rfind("<tr")
        tr_open = body[last_tr:key_idx]
        assert "flagged" not in tr_open

    def test_quality_page_orders_newest_first(self, client):
        """Rows must appear in validated_at DESC order in the HTML body."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-9010",
            validated_at=(now - timedelta(hours=2)).isoformat(),
        )
        _insert_row(
            story_key="TK-9011",
            validated_at=now.isoformat(),
        )

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert body.index("TK-9011") < body.index("TK-9010")

    def test_quality_page_renders_red_flag_summary_panel(self, client):
        """Flag summary panel must surface flag types seen in the last 24h."""
        _insert_row(
            story_key="TK-9020",
            red_flags=["broad_except", "hardcoded_secret"],
        )

        resp = client.get("/quality")
        body = resp.get_data(as_text=True)

        assert "broad_except" in body
        assert "hardcoded_secret" in body


class TestApiAivRecent:
    """GET /api/aiv/recent — JSON list, newest first."""

    EXPECTED_KEYS = {
        "story_key",
        "story_title",
        "merged_at",
        "validated_at",
        "meets_requirements",
        "code_quality",
        "test_quality",
        "security_safety",
        "scope_discipline",
        "edge_cases",
        "product_impact",
        "overall_score",
        "red_flags",
        "verification_method",
        "error",
    }

    def test_api_recent_returns_200_with_empty_list_when_no_rows(self, client):
        """Empty DB — endpoint still returns the envelope with recent: []."""
        resp = client.get("/api/aiv/recent")

        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        body = resp.get_json()
        assert body == {"recent": [], "limit": 100}

    def test_api_recent_returns_all_expected_keys(self, client):
        """Acceptance: each entry exposes the full documented key set."""
        _insert_row(
            story_key="TK-9100",
            story_title="API shape check",
            scores=(9, 8, 7, 10, 9, 6, 8),
            overall=8.1,
            red_flags=["magic_number"],
        )

        resp = client.get("/api/aiv/recent")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "recent" in body
        assert isinstance(body["recent"], list)
        assert len(body["recent"]) == 1
        entry = body["recent"][0]
        assert set(entry.keys()) == self.EXPECTED_KEYS

        assert entry["story_key"] == "TK-9100"
        assert entry["story_title"] == "API shape check"
        assert entry["meets_requirements"] == 9
        assert entry["overall_score"] == pytest.approx(8.1)
        assert entry["red_flags"] == ["magic_number"]

    def test_api_recent_orders_newest_first(self, client):
        """Entries are emitted in validated_at DESC order."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-9110",
            validated_at=(now - timedelta(hours=3)).isoformat(),
        )
        _insert_row(
            story_key="TK-9111",
            validated_at=(now - timedelta(minutes=5)).isoformat(),
        )
        _insert_row(
            story_key="TK-9112",
            validated_at=now.isoformat(),
        )

        resp = client.get("/api/aiv/recent")
        keys = [e["story_key"] for e in resp.get_json()["recent"]]

        assert keys == ["TK-9112", "TK-9111", "TK-9110"]

    def test_api_recent_respects_limit_param(self, client):
        """?limit=N caps the row count; values are clamped to [1, 500]."""
        for i in range(5):
            _insert_row(
                story_key=f"TK-92{i:02d}",
                validated_at=(
                    datetime.now(timezone.utc) - timedelta(minutes=i)
                ).isoformat(),
            )

        resp = client.get("/api/aiv/recent?limit=2")
        body = resp.get_json()

        assert body["limit"] == 2
        assert len(body["recent"]) == 2

    def test_api_recent_bad_limit_falls_back_to_default(self, client):
        """Non-integer limit does not 500 — defaults to 100."""
        _insert_row(story_key="TK-9300")

        resp = client.get("/api/aiv/recent?limit=not-a-number")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["limit"] == 100

    def test_api_recent_parses_red_flags_to_list(self, client):
        """red_flags must land in the JSON as an actual list, not a JSON string."""
        _insert_row(
            story_key="TK-9400",
            red_flags=["broad_except", "todo_left_in_code"],
        )

        resp = client.get("/api/aiv/recent")
        entry = resp.get_json()["recent"][0]

        assert entry["red_flags"] == ["broad_except", "todo_left_in_code"]

    def test_api_recent_survives_missing_db(self, client, tmp_path, monkeypatch):
        """DB file doesn't exist — endpoint returns empty list, not 500."""
        ghost = tmp_path / "does-not-exist.db"
        monkeypatch.setattr(aiv_schema, "DB_PATH", ghost)

        resp = client.get("/api/aiv/recent")

        assert resp.status_code == 200
        assert resp.get_json() == {"recent": [], "limit": 100}


class TestApiAivFlags:
    """GET /api/aiv/flags — 24h red-flag tally."""

    def test_api_flags_returns_zero_counts_when_no_rows(self, client):
        """No rows — envelope still has the documented keys with zero totals."""
        resp = client.get("/api/aiv/flags")

        assert resp.status_code == 200
        assert "application/json" in resp.content_type
        body = resp.get_json()
        assert body == {
            "window_hours": 24,
            "counts": {},
            "total_flags": 0,
            "stories_with_flags": 0,
        }

    def test_api_flags_counts_flag_occurrences(self, client):
        """Each flag occurrence across recent stories should be tallied."""
        _insert_row(
            story_key="TK-9500",
            red_flags=["broad_except", "magic_number"],
        )
        _insert_row(
            story_key="TK-9501",
            red_flags=["broad_except"],
        )

        resp = client.get("/api/aiv/flags")
        body = resp.get_json()

        assert body["counts"]["broad_except"] == 2
        assert body["counts"]["magic_number"] == 1
        assert body["total_flags"] == 3
        assert body["stories_with_flags"] == 2
        assert body["window_hours"] == 24

    def test_api_flags_excludes_rows_outside_window(self, client):
        """Stories validated before the window must not contribute flags."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-9600",
            validated_at=(now - timedelta(hours=48)).isoformat(),
            red_flags=["old_flag"],
        )
        _insert_row(
            story_key="TK-9601",
            validated_at=(now - timedelta(hours=2)).isoformat(),
            red_flags=["fresh_flag"],
        )

        resp = client.get("/api/aiv/flags?hours=24")
        body = resp.get_json()

        assert "old_flag" not in body["counts"]
        assert body["counts"].get("fresh_flag") == 1
        assert body["total_flags"] == 1

    def test_api_flags_custom_window_hours(self, client):
        """?hours=N selects the tally window; values clamped to [1, 720]."""
        now = datetime.now(timezone.utc)
        _insert_row(
            story_key="TK-9700",
            validated_at=(now - timedelta(hours=48)).isoformat(),
            red_flags=["old_flag"],
        )

        resp = client.get("/api/aiv/flags?hours=72")
        body = resp.get_json()

        assert body["window_hours"] == 72
        assert body["counts"].get("old_flag") == 1

    def test_api_flags_survives_missing_db(self, client, tmp_path, monkeypatch):
        """DB file doesn't exist — endpoint returns zero counts, not 500."""
        ghost = tmp_path / "does-not-exist.db"
        monkeypatch.setattr(aiv_schema, "DB_PATH", ghost)

        resp = client.get("/api/aiv/flags")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["total_flags"] == 0
        assert body["stories_with_flags"] == 0
