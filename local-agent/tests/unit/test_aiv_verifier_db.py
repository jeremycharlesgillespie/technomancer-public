"""Tests for aiv.verifiers.db_query — SQLite schema-change verifier."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aiv.verifiers.db_query import (
    SAMPLE_ROW_LIMIT,
    _extract_table_names,
    _resolve_db_path,
    capture,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(db_path: Path, table: str, columns_sql: str, rows: list[tuple]) -> None:
    """Create ``db_path`` with ``table`` and insert ``rows``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"CREATE TABLE {table} ({columns_sql})")
        if rows:
            placeholders = ",".join("?" for _ in rows[0])
            conn.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})", rows
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _extract_table_names
# ---------------------------------------------------------------------------


class TestExtractTableNames:
    def test_simple_create_table(self):
        source = "CREATE TABLE users (id INTEGER);"
        assert _extract_table_names(source) == ["users"]

    def test_create_table_if_not_exists(self):
        source = "CREATE TABLE IF NOT EXISTS widgets (id INTEGER);"
        assert _extract_table_names(source) == ["widgets"]

    def test_case_insensitive(self):
        source = "create table Orders (id INTEGER);"
        assert _extract_table_names(source) == ["Orders"]

    def test_alter_table(self):
        source = "ALTER TABLE users ADD COLUMN email TEXT;"
        assert _extract_table_names(source) == ["users"]

    def test_double_quoted_identifier(self):
        source = 'CREATE TABLE "users" (id INTEGER);'
        assert _extract_table_names(source) == ["users"]

    def test_backtick_identifier(self):
        source = "CREATE TABLE `users` (id INTEGER);"
        assert _extract_table_names(source) == ["users"]

    def test_bracket_identifier(self):
        source = "CREATE TABLE [users] (id INTEGER);"
        assert _extract_table_names(source) == ["users"]

    def test_schema_prefix_extracts_table_not_schema(self):
        source = "CREATE TABLE main.users (id INTEGER);"
        assert _extract_table_names(source) == ["users"]

    def test_multiple_tables_preserve_order(self):
        source = """
            CREATE TABLE users (id INTEGER);
            CREATE TABLE posts (id INTEGER);
            ALTER TABLE users ADD COLUMN email TEXT;
        """
        assert _extract_table_names(source) == ["users", "posts"]

    def test_deduplicates_case_insensitively(self):
        source = """
            CREATE TABLE users (id INTEGER);
            ALTER TABLE Users ADD COLUMN email TEXT;
        """
        assert _extract_table_names(source) == ["users"]

    def test_no_matches_returns_empty(self):
        assert _extract_table_names("SELECT * FROM users;") == []
        assert _extract_table_names("") == []

    def test_embedded_in_python_string(self):
        source = '''
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aiv_pending (
                    story_key TEXT PRIMARY KEY
                )
            """)
        '''
        assert _extract_table_names(source) == ["aiv_pending"]


# ---------------------------------------------------------------------------
# _resolve_db_path
# ---------------------------------------------------------------------------


class TestResolveDbPath:
    def test_python_source_with_db_path_line(self, tmp_path):
        data_dir = tmp_path / "data"
        source = 'DB_PATH = DB_DIR / "widgets.db"\n'
        schema_file = tmp_path / "agent" / "widgets_schema.py"
        resolved = _resolve_db_path(schema_file, source, data_dir=data_dir)
        assert resolved == data_dir / "widgets.db"

    def test_python_source_single_quotes(self, tmp_path):
        data_dir = tmp_path / "data"
        source = "DB_PATH = DB_DIR / 'widgets.db'\n"
        schema_file = tmp_path / "agent" / "widgets_schema.py"
        resolved = _resolve_db_path(schema_file, source, data_dir=data_dir)
        assert resolved == data_dir / "widgets.db"

    def test_default_data_dir_is_two_up_plus_data(self, tmp_path):
        source = 'DB_PATH = DB_DIR / "foo.db"\n'
        schema_file = tmp_path / "agent" / "foo_schema.py"
        resolved = _resolve_db_path(schema_file, source, data_dir=None)
        assert resolved == tmp_path / "data" / "foo.db"

    def test_sql_file_with_sibling_db(self, tmp_path):
        sql_file = tmp_path / "migration.sql"
        sql_file.write_text("CREATE TABLE x (id INTEGER);")
        sibling = tmp_path / "migration.db"
        sibling.touch()
        resolved = _resolve_db_path(sql_file, sql_file.read_text())
        assert resolved == sibling

    def test_sql_file_without_sibling_returns_none(self, tmp_path):
        sql_file = tmp_path / "migration.sql"
        resolved = _resolve_db_path(sql_file, "CREATE TABLE x (id INTEGER);")
        assert resolved is None

    def test_python_without_db_path_line_returns_none(self, tmp_path):
        schema_file = tmp_path / "agent" / "broken.py"
        resolved = _resolve_db_path(schema_file, "print('hi')")
        assert resolved is None


# ---------------------------------------------------------------------------
# capture — success paths
# ---------------------------------------------------------------------------


class TestCaptureSuccess:
    def test_sql_file_with_populated_table(self, tmp_path):
        """The acceptance-criterion case: synthetic SQL file with
        CREATE TABLE x → PRAGMA output showing the columns."""
        sql_file = tmp_path / "001_add_widgets.sql"
        sql_file.write_text("CREATE TABLE widgets (id INTEGER, name TEXT);")
        db_path = tmp_path / "app.db"
        _make_db(db_path, "widgets", "id INTEGER, name TEXT", [(1, "a"), (2, "b")])

        result = capture(sql_file, db_path=db_path)

        assert result["table"] == "widgets"
        assert result["db_path"] == str(db_path)
        assert "error" not in result

        pragma = result["pragma"]
        assert len(pragma) == 2
        assert [col["name"] for col in pragma] == ["id", "name"]
        assert [col["type"] for col in pragma] == ["INTEGER", "TEXT"]

        assert result["sample_rows"] == [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]

    def test_empty_table_returns_pragma_and_empty_sample(self, tmp_path):
        sql_file = tmp_path / "002_empty.sql"
        sql_file.write_text("CREATE TABLE events (ts TEXT, payload TEXT);")
        db_path = tmp_path / "app.db"
        _make_db(db_path, "events", "ts TEXT, payload TEXT", [])

        result = capture(sql_file, db_path=db_path)

        assert [col["name"] for col in result["pragma"]] == ["ts", "payload"]
        assert result["sample_rows"] == []

    def test_sample_rows_capped_at_limit(self, tmp_path):
        sql_file = tmp_path / "003.sql"
        sql_file.write_text("CREATE TABLE big (n INTEGER);")
        db_path = tmp_path / "app.db"
        rows = [(i,) for i in range(SAMPLE_ROW_LIMIT + 10)]
        _make_db(db_path, "big", "n INTEGER", rows)

        result = capture(sql_file, db_path=db_path)

        assert len(result["sample_rows"]) == SAMPLE_ROW_LIMIT

    def test_alter_table_picks_up_existing_table(self, tmp_path):
        sql_file = tmp_path / "004_alter.sql"
        sql_file.write_text("ALTER TABLE widgets ADD COLUMN color TEXT;")
        db_path = tmp_path / "app.db"
        _make_db(db_path, "widgets", "id INTEGER, color TEXT", [(1, "red")])

        result = capture(sql_file, db_path=db_path)

        assert result["table"] == "widgets"
        assert "color" in [col["name"] for col in result["pragma"]]
        assert result["sample_rows"] == [{"id": 1, "color": "red"}]

    def test_python_schema_file_with_db_path_line(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        schema_file = tmp_path / "agent" / "widgets_schema.py"
        schema_file.parent.mkdir()
        schema_file.write_text(
            'DB_PATH = DB_DIR / "widgets.db"\n'
            '\n'
            'conn.execute("""\n'
            '    CREATE TABLE IF NOT EXISTS widgets (\n'
            '        id INTEGER PRIMARY KEY,\n'
            '        label TEXT\n'
            '    )\n'
            '""")\n'
        )
        db_path = data_dir / "widgets.db"
        _make_db(db_path, "widgets", "id INTEGER PRIMARY KEY, label TEXT", [(1, "x")])

        result = capture(schema_file, data_dir=data_dir)

        assert result["db_path"] == str(db_path)
        assert result["table"] == "widgets"
        assert [col["name"] for col in result["pragma"]] == ["id", "label"]
        assert result["sample_rows"] == [{"id": 1, "label": "x"}]


# ---------------------------------------------------------------------------
# capture — failure paths
# ---------------------------------------------------------------------------


class TestCaptureTableNotFound:
    def test_db_exists_but_table_missing(self, tmp_path):
        """Acceptance criterion: migration didn't run → table not found."""
        sql_file = tmp_path / "005.sql"
        sql_file.write_text("CREATE TABLE forgotten (id INTEGER);")
        db_path = tmp_path / "app.db"
        # Create a DB with a *different* table, so the file exists but
        # the migration visibly didn't apply.
        _make_db(db_path, "something_else", "id INTEGER", [])

        result = capture(sql_file, db_path=db_path)

        assert result["error"] == "table not found"
        assert result["table"] == "forgotten"
        assert result["db_path"] == str(db_path)
        assert "pragma" not in result
        assert "sample_rows" not in result

    def test_db_file_does_not_exist(self, tmp_path):
        sql_file = tmp_path / "006.sql"
        sql_file.write_text("CREATE TABLE never_made (id INTEGER);")
        missing_db = tmp_path / "never_created.db"

        result = capture(sql_file, db_path=missing_db)

        assert result["error"] == "table not found"
        assert result["table"] == "never_made"
        assert result["db_path"] == str(missing_db)


class TestCaptureNoStatements:
    def test_empty_source_file(self, tmp_path):
        sql_file = tmp_path / "empty.sql"
        sql_file.write_text("")
        result = capture(sql_file, db_path=tmp_path / "ignored.db")
        assert result == {"error": "no CREATE/ALTER statements found"}

    def test_source_with_only_selects(self, tmp_path):
        sql_file = tmp_path / "readonly.sql"
        sql_file.write_text("SELECT * FROM users;")
        result = capture(sql_file, db_path=tmp_path / "ignored.db")
        assert result == {"error": "no CREATE/ALTER statements found"}


class TestCaptureCannotResolveDbPath:
    def test_sql_without_sibling_db_and_no_override(self, tmp_path):
        sql_file = tmp_path / "lonely.sql"
        sql_file.write_text("CREATE TABLE x (id INTEGER);")
        result = capture(sql_file)
        assert result["error"] == "could not resolve DB path"
        assert result["table"] == "x"


class TestCaptureSourceReadErrors:
    def test_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.sql"
        result = capture(missing, db_path=tmp_path / "ignored.db")
        assert "error" in result
        assert "could not read source" in result["error"]


class TestCaptureMultipleTables:
    def test_first_table_wins(self, tmp_path):
        """MVP: when the diff creates multiple tables, capture the
        first one. Documenting the current behavior so a future
        multi-table shape doesn't silently change it."""
        sql_file = tmp_path / "multi.sql"
        sql_file.write_text(
            "CREATE TABLE first (id INTEGER); "
            "CREATE TABLE second (id INTEGER);"
        )
        db_path = tmp_path / "app.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE first (id INTEGER)")
            conn.execute("CREATE TABLE second (id INTEGER)")
            conn.execute("INSERT INTO first VALUES (1)")
            conn.commit()
        finally:
            conn.close()

        result = capture(sql_file, db_path=db_path)
        assert result["table"] == "first"
        assert result["sample_rows"] == [{"id": 1}]
