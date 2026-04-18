"""Tests for aiv.classifier — diff-path → verification method mapping."""

from __future__ import annotations

import pytest

from aiv.classifier import classify_diff


class TestClassifyDiff:
    def test_web_render_when_idea_board_web_changed(self):
        assert classify_diff(["local-agent/idea_board/web.py"]) == "web-render"

    def test_api_call_when_path_contains_api_dir(self):
        assert classify_diff(["src/api/endpoints.py"]) == "api-call"

    def test_db_query_for_sql_file(self):
        assert classify_diff(["migrations/001_add_users.sql"]) == "db-query"

    def test_db_query_for_schema_py(self):
        assert classify_diff(["local-agent/agent/aiv_schema.py"]) == "db-query"

    def test_tests_only_for_ambiguous_diff(self):
        assert classify_diff(["docs/README.md", "CHANGELOG.md"]) == "tests-only"

    def test_empty_paths_returns_tests_only(self):
        assert classify_diff([]) == "tests-only"

    def test_tests_only_when_only_test_files_changed(self):
        assert (
            classify_diff(["local-agent/tests/unit/test_foo.py"]) == "tests-only"
        )

    def test_web_render_wins_over_other_categories(self):
        paths = [
            "local-agent/idea_board/web.py",
            "src/api/handler.py",
            "migrations/002.sql",
        ]
        assert classify_diff(paths) == "web-render"

    def test_api_call_wins_over_db_query(self):
        paths = ["src/api/users.py", "migrations/003.sql"]
        assert classify_diff(paths) == "api-call"

    def test_windows_path_separators_are_normalised(self):
        assert (
            classify_diff(["local-agent\\idea_board\\web.py"]) == "web-render"
        )
        assert classify_diff(["src\\api\\handler.py"]) == "api-call"

    def test_schema_py_match_is_case_insensitive(self):
        assert classify_diff(["agent/AIV_Schema.py"]) == "db-query"

    def test_schema_word_in_directory_does_not_trigger_db_query(self):
        # "schema" is in the directory name, not the filename — should fall through.
        assert classify_diff(["schemas/notes.md"]) == "tests-only"
