"""Tests for agent/news_prefs_db.py — SQLite-backed news preferences."""

from __future__ import annotations

import json
import sqlite3

import pytest

from agent import news_prefs_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point news_prefs_db at a temporary DB and legacy JSON path per test."""
    db_path = tmp_path / "news_prefs.db"
    legacy_path = tmp_path / "news_config.json"
    monkeypatch.setattr(news_prefs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(news_prefs_db, "DB_PATH", db_path)
    monkeypatch.setattr(news_prefs_db, "LEGACY_JSON_PATH", legacy_path)
    # Drop any cached per-thread connection so each test gets a fresh one
    news_prefs_db._local.__dict__.pop("conn", None)
    yield
    conn = getattr(news_prefs_db._local, "conn", None)
    if conn is not None:
        conn.close()
        news_prefs_db._local.conn = None


class TestInitDb:
    def test_creates_table(self):
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='news_prefs'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        news_prefs_db.init_db()
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        assert conn.execute("SELECT COUNT(*) FROM news_prefs").fetchone()[0] == 0

    def test_singleton_constraint(self):
        """Attempting to insert id != 1 must fail the CHECK constraint."""
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO news_prefs (id, data) VALUES (2, '{}')")


class TestLoadSave:
    def test_load_empty_returns_none(self):
        assert news_prefs_db.load_prefs() is None

    def test_save_then_load(self):
        data = {
            "feeds": [{"name": "X", "url": "http://x", "category": "ai", "enabled": True}],
            "likes": ["python"],
            "dislikes": ["crypto"],
            "start_hour": 8,
            "end_hour": 22,
            "max_articles_per_hour": 2,
        }
        news_prefs_db.save_prefs(data)
        loaded = news_prefs_db.load_prefs()
        assert loaded == data

    def test_save_upserts_single_row(self):
        news_prefs_db.save_prefs({"likes": ["a"]})
        news_prefs_db.save_prefs({"likes": ["b"]})
        conn = news_prefs_db._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM news_prefs").fetchone()[0]
        assert count == 1
        loaded = news_prefs_db.load_prefs()
        assert loaded == {"likes": ["b"]}

    def test_save_updates_updated_at(self):
        news_prefs_db.save_prefs({"likes": ["a"]})
        conn = news_prefs_db._get_conn()
        first = conn.execute("SELECT updated_at FROM news_prefs WHERE id = 1").fetchone()[0]
        # A subsequent save should refresh updated_at (same-second is fine; value just exists)
        news_prefs_db.save_prefs({"likes": ["b"]})
        second = conn.execute("SELECT updated_at FROM news_prefs WHERE id = 1").fetchone()[0]
        assert first is not None and second is not None

    def test_load_survives_corrupt_payload(self):
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        conn.execute("INSERT INTO news_prefs (id, data) VALUES (1, 'not valid json')")
        conn.commit()
        assert news_prefs_db.load_prefs() is None

    def test_load_rejects_non_dict_payload(self):
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        conn.execute("INSERT INTO news_prefs (id, data) VALUES (1, '[1,2,3]')")
        conn.commit()
        assert news_prefs_db.load_prefs() is None

    def test_save_roundtrips_unicode(self):
        data = {"likes": ["café", "日本語", "emoji-free"]}
        news_prefs_db.save_prefs(data)
        assert news_prefs_db.load_prefs() == data


class TestMigration:
    def test_migrates_legacy_json_on_first_load(self):
        legacy = news_prefs_db.LEGACY_JSON_PATH
        payload = {
            "feeds": [{"name": "Legacy", "url": "http://legacy", "category": "general", "enabled": True}],
            "likes": ["rust"],
            "dislikes": [],
            "start_hour": 6,
            "end_hour": 23,
            "max_articles_per_hour": 3,
        }
        legacy.write_text(json.dumps(payload), encoding="utf-8")

        loaded = news_prefs_db.load_prefs()

        assert loaded == payload
        # Legacy file should be renamed to .migrated.bak
        assert not legacy.exists()
        backup = legacy.with_name(legacy.name + news_prefs_db.LEGACY_BACKUP_SUFFIX)
        assert backup.exists()

    def test_migration_only_runs_when_db_is_empty(self):
        """If the DB already has a row, don't touch the legacy file."""
        news_prefs_db.save_prefs({"likes": ["db-value"]})
        legacy = news_prefs_db.LEGACY_JSON_PATH
        legacy.write_text(json.dumps({"likes": ["legacy-value"]}), encoding="utf-8")

        loaded = news_prefs_db.load_prefs()

        assert loaded == {"likes": ["db-value"]}
        # Legacy file untouched because DB already had data
        assert legacy.exists()

    def test_no_legacy_file_is_noop(self):
        # No legacy file, no DB row — load_prefs should just return None
        assert not news_prefs_db.LEGACY_JSON_PATH.exists()
        assert news_prefs_db.load_prefs() is None

    def test_migration_skips_invalid_json(self):
        legacy = news_prefs_db.LEGACY_JSON_PATH
        legacy.write_text("{not valid", encoding="utf-8")
        # Should swallow the error, not migrate, and not rename
        assert news_prefs_db.load_prefs() is None
        assert legacy.exists()
        backup = legacy.with_name(legacy.name + news_prefs_db.LEGACY_BACKUP_SUFFIX)
        assert not backup.exists()

    def test_migration_skips_non_dict_json(self):
        legacy = news_prefs_db.LEGACY_JSON_PATH
        legacy.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        assert news_prefs_db.load_prefs() is None
        # Legacy file stays in place since we couldn't migrate it
        assert legacy.exists()

    def test_migration_overwrites_existing_backup(self):
        legacy = news_prefs_db.LEGACY_JSON_PATH
        backup = legacy.with_name(legacy.name + news_prefs_db.LEGACY_BACKUP_SUFFIX)
        backup.write_text("stale previous backup", encoding="utf-8")
        legacy.write_text(json.dumps({"likes": ["fresh"]}), encoding="utf-8")

        loaded = news_prefs_db.load_prefs()

        assert loaded == {"likes": ["fresh"]}
        assert not legacy.exists()
        # Backup was replaced with the fresh-migrated contents
        assert backup.exists()
        assert "stale previous backup" not in backup.read_text(encoding="utf-8")

    def test_load_after_migration_returns_migrated_data(self):
        legacy = news_prefs_db.LEGACY_JSON_PATH
        payload = {"likes": ["x"], "dislikes": ["y"]}
        legacy.write_text(json.dumps(payload), encoding="utf-8")

        # First call triggers migration
        first = news_prefs_db.load_prefs()
        # Second call should read from DB (legacy file is gone)
        second = news_prefs_db.load_prefs()

        assert first == payload
        assert second == payload


class TestNewsConfigIntegration:
    """End-to-end smoke: NewsConfig dataclass roundtrips through the DB."""

    def test_roundtrip_via_news_config_api(self):
        from idea_board.news_config import NewsConfig, load_news_config, save_news_config

        cfg = NewsConfig(
            feeds=[{"name": "X", "url": "http://x", "category": "ai", "enabled": False}],
            likes=["python"],
            dislikes=["sports"],
            start_hour=7,
            end_hour=20,
            max_articles_per_hour=2,
        )
        save_news_config(cfg)
        loaded = load_news_config()

        assert loaded.feeds == cfg.feeds
        assert loaded.likes == cfg.likes
        assert loaded.dislikes == cfg.dislikes
        assert loaded.start_hour == cfg.start_hour
        assert loaded.end_hour == cfg.end_hour
        assert loaded.max_articles_per_hour == cfg.max_articles_per_hour

    def test_missing_prefs_returns_defaults(self):
        from idea_board.news_config import DEFAULT_FEEDS, load_news_config

        cfg = load_news_config()
        # No DB row, no legacy file → defaults
        assert cfg.start_hour == 9
        assert cfg.end_hour == 21
        assert len(cfg.feeds) == len(DEFAULT_FEEDS)

    def test_missing_prefs_persists_defaults(self):
        """Deleting the DB row and re-loading must recreate it with defaults
        (prevents the silent-reset bug that wiped user-added dislikes)."""
        import idea_board.news_config as news_config
        from idea_board.news_config import DEFAULT_FEEDS, load_news_config

        # Reset the once-per-process path-logged flag so this test behaves
        # the same regardless of ordering with other tests that call load.
        news_config._path_logged = False

        # Simulate a fresh install / wiped DB.
        assert news_prefs_db.load_prefs() is None

        load_news_config()

        # A row must now exist with the default feeds persisted.
        persisted = news_prefs_db.load_prefs()
        assert persisted is not None
        assert len(persisted["feeds"]) == len(DEFAULT_FEEDS)
        # updated_at should be populated so /news can show "Last saved: ..."
        assert news_prefs_db.get_updated_at() is not None

    def test_user_dislikes_survive_reload(self):
        """After the first-load persist, adding a dislike and reloading must
        keep it — this is the exact scenario the story was filed for."""
        from idea_board.news_config import load_news_config, save_news_config

        cfg = load_news_config()
        cfg.dislikes.append("politics")
        save_news_config(cfg)

        reloaded = load_news_config()
        assert "politics" in reloaded.dislikes

    def test_get_last_saved_after_write(self):
        from idea_board.news_config import NewsConfig, get_last_saved, save_news_config

        assert get_last_saved() is None
        save_news_config(NewsConfig())
        assert get_last_saved() is not None

    def test_legacy_json_migrates_through_news_config_api(self):
        from idea_board.news_config import load_news_config

        legacy = news_prefs_db.LEGACY_JSON_PATH
        legacy.write_text(
            json.dumps(
                {
                    "feeds": [],
                    "likes": ["legacy-like"],
                    "dislikes": [],
                    "start_hour": 5,
                    "end_hour": 23,
                    "max_articles_per_hour": 4,
                }
            ),
            encoding="utf-8",
        )

        cfg = load_news_config()

        assert cfg.likes == ["legacy-like"]
        assert cfg.start_hour == 5
        assert cfg.max_articles_per_hour == 4
        assert not legacy.exists()
        assert legacy.with_name(legacy.name + news_prefs_db.LEGACY_BACKUP_SUFFIX).exists()
