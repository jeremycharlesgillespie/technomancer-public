"""Tests for idea_board/news_config.py Flask route handlers (TK-593).

Verifies the transactional read-modify-write refactor: each API handler
that mutates prefs now routes through ``mutate_news_config`` so
concurrent callers can't interleave reads and writes.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest
from flask import Flask

from agent import news_prefs_db
from idea_board.news_config import news_bp


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point news_prefs_db at a temporary DB per test.

    Pre-initializes the DB file in WAL mode using a bootstrap connection
    so that the parallel workers in ``TestConcurrentDedup`` don't race on
    the initial ``PRAGMA journal_mode=WAL`` handshake — setting WAL on a
    fresh DB needs EXCLUSIVE access, and under xdist load several worker
    threads opening connections at the same time can collide and raise
    ``sqlite3.OperationalError: database is locked`` (TK-709).
    """
    db_path = tmp_path / "news_prefs.db"
    legacy_path = tmp_path / "news_config.json"
    monkeypatch.setattr(news_prefs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(news_prefs_db, "DB_PATH", db_path)
    monkeypatch.setattr(news_prefs_db, "LEGACY_JSON_PATH", legacy_path)
    news_prefs_db._local.__dict__.pop("conn", None)

    # Bootstrap the DB file into WAL mode before any production code path
    # opens a connection. WAL is persisted on the DB file, so every
    # connection opened afterwards inherits it for free.
    _bootstrap = sqlite3.connect(str(db_path), timeout=5)
    try:
        _bootstrap.execute("PRAGMA busy_timeout=5000")
        _bootstrap.execute("PRAGMA journal_mode=WAL")
        _bootstrap.commit()
    finally:
        _bootstrap.close()

    # Ensure path-logged flag resets so tests are order-independent.
    import idea_board.news_config as news_config
    news_config._path_logged = False

    yield

    conn = getattr(news_prefs_db._local, "conn", None)
    if conn is not None:
        conn.close()
        news_prefs_db._local.conn = None


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(news_bp)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestSchedule:
    def test_put_updates_schedule(self, client):
        resp = client.put(
            "/api/news/config",
            json={"start_hour": 6, "end_hour": 20, "max_articles_per_hour": 3},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["start_hour"] == 6
        assert data["end_hour"] == 20
        assert data["max_articles_per_hour"] == 3

    def test_put_clamps_out_of_range(self, client):
        resp = client.put("/api/news/config", json={"start_hour": 99, "end_hour": 0})
        data = resp.get_json()
        assert data["start_hour"] == 23
        assert data["end_hour"] == 1


class TestLikesDedup:
    def test_add_like_twice_no_duplicate(self, client):
        client.post("/api/news/likes", json={"topic": "python"})
        resp = client.post("/api/news/likes", json={"topic": "python"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["likes"].count("python") == 1

    def test_add_like_removes_from_dislikes(self, client):
        client.post("/api/news/dislikes", json={"topic": "crypto"})
        client.post("/api/news/likes", json={"topic": "crypto"})
        resp = client.get("/api/news/config")
        data = resp.get_json()
        assert "crypto" in data["likes"]
        assert "crypto" not in data["dislikes"]

    def test_remove_like(self, client):
        client.post("/api/news/likes", json={"topic": "python"})
        resp = client.delete("/api/news/likes", json={"topic": "python"})
        assert resp.status_code == 200
        assert "python" not in resp.get_json()["likes"]

    def test_add_like_empty_topic_400(self, client):
        resp = client.post("/api/news/likes", json={"topic": "   "})
        assert resp.status_code == 400


class TestDislikesDedup:
    def test_add_dislike_twice_no_duplicate(self, client):
        client.post("/api/news/dislikes", json={"topic": "sports"})
        resp = client.post("/api/news/dislikes", json={"topic": "sports"})
        data = resp.get_json()
        assert data["dislikes"].count("sports") == 1

    def test_add_dislike_removes_from_likes(self, client):
        client.post("/api/news/likes", json={"topic": "sports"})
        client.post("/api/news/dislikes", json={"topic": "sports"})
        resp = client.get("/api/news/config")
        data = resp.get_json()
        assert "sports" in data["dislikes"]
        assert "sports" not in data["likes"]


class TestFeeds:
    def test_add_feed(self, client):
        resp = client.post(
            "/api/news/feeds",
            json={"name": "Custom", "url": "http://custom.example.com/feed", "category": "ai"},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["feed"]["name"] == "Custom"

    def test_add_feed_duplicate_url_409(self, client):
        payload = {"name": "Custom", "url": "http://dup.example.com/feed"}
        client.post("/api/news/feeds", json=payload)
        resp = client.post("/api/news/feeds", json=payload)
        assert resp.status_code == 409
        assert "already exists" in resp.get_json()["error"]

    def test_add_feed_missing_fields_400(self, client):
        resp = client.post("/api/news/feeds", json={"name": "x"})
        assert resp.status_code == 400

    def test_delete_feed_bad_index_404(self, client):
        resp = client.delete("/api/news/feeds/9999")
        assert resp.status_code == 404

    def test_toggle_feed_bad_index_404(self, client):
        resp = client.post("/api/news/feeds/9999/toggle")
        assert resp.status_code == 404

    def test_toggle_feed_flips_enabled(self, client):
        # Default feeds are loaded on first GET; first feed starts enabled.
        client.get("/api/news/config")
        resp = client.post("/api/news/feeds/0/toggle")
        assert resp.status_code == 200
        assert resp.get_json()["feed"]["enabled"] is False
        resp2 = client.post("/api/news/feeds/0/toggle")
        assert resp2.get_json()["feed"]["enabled"] is True


class TestReset:
    def test_reset_wipes_likes_and_dislikes(self, client):
        client.post("/api/news/likes", json={"topic": "python"})
        client.post("/api/news/dislikes", json={"topic": "sports"})
        resp = client.post("/api/news/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["likes"] == []
        assert data["dislikes"] == []


class TestConcurrentDedup:
    """Integration: under concurrent API hits, dedup holds."""

    def test_fixture_enables_wal_mode(self):
        """TK-709: the _isolate_db fixture must pre-set WAL mode so
        concurrent worker threads don't race on the initial PRAGMA."""
        conn = news_prefs_db._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_parallel_add_like_yields_single_entry(self, client):
        """8 concurrent POSTs of the same like must leave one copy.

        Note: Flask test_client is not thread-safe across requests, so
        we exercise the layer directly via ``mutate_news_config`` rather
        than through the HTTP client — the route handler body is a thin
        wrapper around that call.
        """
        from idea_board.news_config import load_news_config, mutate_news_config

        errors: list[BaseException] = []

        def _worker():
            try:
                mutate_news_config(_add_python)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        def _add_python(cfg):
            if "python" not in cfg.likes:
                cfg.likes.append("python")
            return cfg

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent workers failed: {errors}"
        final = load_news_config()
        assert final.likes.count("python") == 1
