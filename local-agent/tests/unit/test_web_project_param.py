"""Tests for get_project_param() — TK-815.

Verifies the reusable ?project extraction utility in idea_board/web.py:
- absent param → 'technomancer'
- empty / canonical aliases → 'technomancer'
- valid non-default names → returned as-is (lowercase)
- invalid characters → ValueError
"""

from __future__ import annotations

import pytest

from idea_board.web import app, get_project_param


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestGetProjectParam:
    """Unit tests for get_project_param() inside a Flask request context."""

    def _call(self, query_string: str = "") -> str:
        with app.test_request_context(f"/?{query_string}"):
            return get_project_param()

    def _call_raises(self, query_string: str) -> Exception:
        with app.test_request_context(f"/?{query_string}"):
            with pytest.raises(ValueError):
                get_project_param()

    # --- default cases ---

    def test_no_param_returns_technomancer(self):
        assert self._call() == "technomancer"

    def test_empty_string_returns_technomancer(self):
        assert self._call("project=") == "technomancer"

    def test_alias_technomancer_returns_technomancer(self):
        assert self._call("project=technomancer") == "technomancer"

    def test_alias_primary_returns_technomancer(self):
        assert self._call("project=primary") == "technomancer"

    def test_uppercase_technomancer_returns_technomancer(self):
        assert self._call("project=TECHNOMANCER") == "technomancer"

    # --- valid non-default names ---

    def test_40acres_returns_40acres(self):
        assert self._call("project=40acres") == "40acres"

    def test_hyphenated_name(self):
        assert self._call("project=my-project") == "my-project"

    def test_underscored_name(self):
        assert self._call("project=my_project") == "my_project"

    def test_uppercase_normalized_to_lowercase(self):
        assert self._call("project=MyProject") == "myproject"

    def test_whitespace_stripped(self):
        with app.test_request_context("/?project=%20my-project%20"):
            assert get_project_param() == "my-project"

    # --- invalid names ---

    def test_invalid_chars_raise_value_error(self):
        self._call_raises("project=invalid!name")

    def test_space_only_returns_technomancer(self):
        # Stripped to "" → falls into default alias bucket
        with app.test_request_context("/?project=%20"):
            assert get_project_param() == "technomancer"

    def test_slashes_raise_value_error(self):
        self._call_raises("project=path/traversal")

    def test_dots_raise_value_error(self):
        self._call_raises("project=../../etc")

    # --- custom default ---

    def test_custom_default_returned_when_no_param(self):
        with app.test_request_context("/"):
            assert get_project_param(default="custom") == "custom"

    def test_custom_default_returned_for_technomancer_alias(self):
        with app.test_request_context("/?project=technomancer"):
            assert get_project_param(default="fallback") == "fallback"
