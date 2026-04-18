"""Tests for agent/news_prefs_db.py — SQLite-backed news preferences."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from agent import news_prefs_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point news_prefs_db at a temporary DB and legacy JSON path per test.

    Pre-initializes the DB file in WAL mode using a bootstrap connection
    so that concurrent worker threads (e.g. ``TestMutatePrefs`` and
    ``TestNewsConfigIntegration.test_mutate_news_config_dedup_under_contention``)
    don't race on the initial ``PRAGMA journal_mode=WAL`` handshake —
    setting WAL on a fresh DB needs EXCLUSIVE access, and under xdist
    load several worker threads opening connections at the same time can
    collide and raise ``sqlite3.OperationalError: database is locked``
    (TK-716, same bug family as TK-709).
    """
    db_path = tmp_path / "news_prefs.db"
    legacy_path = tmp_path / "news_config.json"
    monkeypatch.setattr(news_prefs_db, "DB_DIR", tmp_path)
    monkeypatch.setattr(news_prefs_db, "DB_PATH", db_path)
    monkeypatch.setattr(news_prefs_db, "LEGACY_JSON_PATH", legacy_path)
    # Drop any cached per-thread connection so each test gets a fresh one
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


class TestMutatePrefs:
    """Tests for the atomic read-modify-write helper (TK-593)."""

    def test_creates_row_from_default_factory_when_missing(self):
        result = news_prefs_db.mutate_prefs(
            lambda d: {**d, "likes": [*d.get("likes", []), "rust"]},
            default_factory=lambda: {"likes": [], "dislikes": []},
        )
        assert result == {"likes": ["rust"], "dislikes": []}
        assert news_prefs_db.load_prefs() == {"likes": ["rust"], "dislikes": []}

    def test_uses_empty_dict_when_no_default_factory(self):
        result = news_prefs_db.mutate_prefs(lambda d: {**d, "k": "v"})
        assert result == {"k": "v"}
        assert news_prefs_db.load_prefs() == {"k": "v"}

    def test_reads_existing_row_before_mutating(self):
        news_prefs_db.save_prefs({"likes": ["python"], "dislikes": []})
        result = news_prefs_db.mutate_prefs(
            lambda d: {**d, "likes": [*d["likes"], "rust"]}
        )
        assert result == {"likes": ["python", "rust"], "dislikes": []}

    def test_mutator_exception_rolls_back(self):
        news_prefs_db.save_prefs({"likes": ["python"]})

        def _bad_mutator(_data):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            news_prefs_db.mutate_prefs(_bad_mutator)

        # Original row must be untouched
        assert news_prefs_db.load_prefs() == {"likes": ["python"]}

    def test_mutator_must_return_dict(self):
        news_prefs_db.save_prefs({"likes": []})
        with pytest.raises(TypeError, match="must return dict"):
            news_prefs_db.mutate_prefs(lambda _d: ["not", "a", "dict"])  # type: ignore[arg-type,return-value]
        # Row unchanged
        assert news_prefs_db.load_prefs() == {"likes": []}

    def test_updated_at_refreshes(self):
        news_prefs_db.save_prefs({"likes": []})
        first = news_prefs_db.get_updated_at()
        # Mutate — updated_at should become non-None and be set by the upsert
        news_prefs_db.mutate_prefs(lambda d: {**d, "likes": ["rust"]})
        second = news_prefs_db.get_updated_at()
        assert first is not None
        assert second is not None

    def test_handles_corrupt_payload_with_default_factory(self):
        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        conn.execute("INSERT INTO news_prefs (id, data) VALUES (1, 'not valid json')")
        conn.commit()

        result = news_prefs_db.mutate_prefs(
            lambda d: {**d, "recovered": True},
            default_factory=lambda: {"likes": ["fallback"]},
        )
        assert result == {"likes": ["fallback"], "recovered": True}

    def test_dedup_same_topic_twice_under_contention(self):
        """Two concurrent "add like" mutations with the same topic must
        only append once — the transaction forces the second caller to
        observe the first caller's write."""
        news_prefs_db.save_prefs({"likes": [], "dislikes": []})

        def _add_rust(current: dict) -> dict:
            likes = list(current.get("likes", []))
            if "rust" not in likes:
                likes.append("rust")
            return {**current, "likes": likes}

        errors: list[BaseException] = []

        def _worker():
            try:
                news_prefs_db.mutate_prefs(_add_rust)
            except BaseException as e:  # pragma: no cover - surfaces in assert
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"mutator raised under contention: {errors}"
        final = news_prefs_db.load_prefs()
        assert final is not None
        # Exactly one copy of "rust" despite 8 concurrent adds
        assert final["likes"].count("rust") == 1

    def test_concurrent_independent_mutations_both_persist(self):
        """Adding a like and a dislike concurrently must not lose either.

        This is the TOCTOU scenario the old code had: thread A reads
        prefs, thread B reads prefs, A appends its like and saves, B
        appends its dislike and saves (overwriting A's change).
        """
        news_prefs_db.save_prefs({"likes": [], "dislikes": []})

        def _add_like(current: dict) -> dict:
            likes = list(current.get("likes", []))
            likes.append("python")
            time.sleep(0.01)  # widen the race window
            return {**current, "likes": likes}

        def _add_dislike(current: dict) -> dict:
            dislikes = list(current.get("dislikes", []))
            dislikes.append("crypto")
            time.sleep(0.01)
            return {**current, "dislikes": dislikes}

        errors: list[BaseException] = []

        def _run(fn):
            def _inner():
                try:
                    news_prefs_db.mutate_prefs(fn)
                except BaseException as e:  # pragma: no cover
                    errors.append(e)
            return _inner

        t1 = threading.Thread(target=_run(_add_like))
        t2 = threading.Thread(target=_run(_add_dislike))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"concurrent mutators failed: {errors}"
        final = news_prefs_db.load_prefs()
        assert final == {"likes": ["python"], "dislikes": ["crypto"]}


class TestNewsConfigIntegration:
    """End-to-end smoke: NewsConfig dataclass roundtrips through the DB."""

    def test_fixture_enables_wal_mode(self):
        """TK-716: the _isolate_db fixture must pre-set WAL mode so
        concurrent worker threads don't race on the initial PRAGMA."""
        conn = news_prefs_db._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

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

    def test_dislikes_persist_when_db_wiped_between_saves(self):
        """Regression for the silent-drop bug: the prefs row disappears
        between two save_news_config calls (simulates an OneDrive-sync
        conflict or a stray manual delete). The post-fix behaviour is that
        the second save still lands on disk and a subsequent reload sees
        it — before the fix, load_news_config returned a transient empty
        NewsConfig and nothing ever persisted after the wipe.
        """
        import idea_board.news_config as news_config
        from idea_board.news_config import load_news_config, save_news_config

        news_config._path_logged = False

        cfg = load_news_config()
        cfg.dislikes.append("politics")
        save_news_config(cfg)
        assert "politics" in (news_prefs_db.load_prefs() or {}).get("dislikes", [])

        # Wipe the singleton row mid-flight to mimic the file-missing scenario.
        conn = news_prefs_db._get_conn()
        conn.execute("DELETE FROM news_prefs WHERE id = 1")
        conn.commit()
        assert news_prefs_db.load_prefs() is None

        cfg2 = load_news_config()
        cfg2.dislikes.append("sports")
        save_news_config(cfg2)

        reloaded = load_news_config()
        assert "sports" in reloaded.dislikes
        assert news_prefs_db.get_updated_at() is not None

    def test_logs_db_path_on_first_load_even_with_existing_row(self, caplog):
        """The DB path must appear in startup logs on every boot, not only
        when the fresh-install path creates the row. Operators use this to
        rule out "am I looking at the right file?" on restart."""
        import logging

        import idea_board.news_config as news_config
        from idea_board.news_config import load_news_config, save_news_config

        # Seed an existing row so the defaults-persist branch does NOT run.
        save_news_config(news_config.NewsConfig(dislikes=["politics"]))
        news_config._path_logged = False

        with caplog.at_level(logging.INFO, logger="idea_board.news_config"):
            load_news_config()

        assert any("News prefs DB at" in r.message for r in caplog.records)

    def test_get_last_saved_after_write(self):
        from idea_board.news_config import NewsConfig, get_last_saved, save_news_config

        assert get_last_saved() is None
        save_news_config(NewsConfig())
        assert get_last_saved() is not None

    def test_mutate_news_config_roundtrips_dataclass(self):
        from idea_board.news_config import mutate_news_config

        def _apply(cfg):
            cfg.likes.append("python")
            cfg.start_hour = 6
            return cfg

        updated = mutate_news_config(_apply)
        assert "python" in updated.likes
        assert updated.start_hour == 6

        # Persisted
        from idea_board.news_config import load_news_config
        reloaded = load_news_config()
        assert "python" in reloaded.likes
        assert reloaded.start_hour == 6

    def test_mutate_news_config_dedup_under_contention(self):
        """Multiple threads adding the same like only produce one entry."""
        from idea_board.news_config import mutate_news_config

        def _add_same_like(cfg):
            if "python" not in cfg.likes:
                cfg.likes.append("python")
            return cfg

        errors: list[BaseException] = []

        def _worker():
            try:
                mutate_news_config(_add_same_like)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        from idea_board.news_config import load_news_config
        final = load_news_config()
        assert final.likes.count("python") == 1

    def test_mutate_news_config_corrupt_payload_uses_defaults(self):
        """If the stored payload is malformed, the mutator sees a fresh
        NewsConfig instead of raising — the API endpoints should keep
        working after a corrupt write."""
        from idea_board.news_config import DEFAULT_FEEDS, mutate_news_config

        news_prefs_db.init_db()
        conn = news_prefs_db._get_conn()
        conn.execute("INSERT INTO news_prefs (id, data) VALUES (1, 'garbage')")
        conn.commit()

        def _apply(cfg):
            cfg.likes.append("recovered")
            return cfg

        updated = mutate_news_config(_apply)
        assert "recovered" in updated.likes
        assert len(updated.feeds) == len(DEFAULT_FEEDS)

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
