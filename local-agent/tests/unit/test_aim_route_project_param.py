"""Frontend integration tests for ?project param on /aim and /aim/dashboard routes.

Verifies that:
- Both routes accept and embed the ?project query param server-side.
- Default/omitted/technomancer project renders primary content.
- Non-default project (e.g. 40acres) is embedded in the JS initial state.
"""

from __future__ import annotations

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /aim — default project
# ---------------------------------------------------------------------------


class TestAimDefaultProject:
    def test_returns_200_no_param(self, client):
        assert client.get("/aim").status_code == 200

    def test_returns_200_empty_param(self, client):
        assert client.get("/aim?project=").status_code == 200

    def test_returns_200_technomancer_alias(self, client):
        assert client.get("/aim?project=technomancer").status_code == 200

    def test_returns_200_primary_alias(self, client):
        assert client.get("/aim?project=primary").status_code == 200

    def test_default_embeds_empty_string_as_initial_project(self, client):
        body = client.get("/aim").get_data(as_text=True)
        # Server injects "" (empty string) for the default project so the JS
        # falls through to the URL-param fallback or 'technomancer'.
        assert 'let currentProject = "" ||' in body

    def test_technomancer_alias_embeds_empty_string(self, client):
        body = client.get("/aim?project=technomancer").get_data(as_text=True)
        assert 'let currentProject = "" ||' in body

    def test_default_title_has_no_suffix(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert "<title>AIM Dashboard</title>" in body


# ---------------------------------------------------------------------------
# /aim — non-default project
# ---------------------------------------------------------------------------


class TestAimProjectParam:
    def test_returns_200_with_project(self, client):
        assert client.get("/aim?project=40acres").status_code == 200

    def test_project_embedded_in_js_initial_state(self, client):
        body = client.get("/aim?project=40acres").get_data(as_text=True)
        # Server injects "40acres" so the JS sets currentProject without
        # waiting for the async fetchAimProjects() to resolve.
        assert 'let currentProject = "40acres" ||' in body

    def test_project_name_in_page_title(self, client):
        body = client.get("/aim?project=40acres").get_data(as_text=True)
        assert "AIM Dashboard — 40acres" in body

    def test_project_not_in_title_for_default(self, client):
        body = client.get("/aim").get_data(as_text=True)
        assert "AIM Dashboard —" not in body

    def test_content_type_is_html(self, client):
        resp = client.get("/aim?project=40acres")
        assert "text/html" in resp.content_type

    def test_still_has_status_widget(self, client):
        body = client.get("/aim?project=40acres").get_data(as_text=True)
        assert 'id="aim-status-widget"' in body

    def test_withproject_helper_present(self, client):
        body = client.get("/aim?project=40acres").get_data(as_text=True)
        assert "function withProject(" in body


# ---------------------------------------------------------------------------
# /aim/dashboard — default project
# ---------------------------------------------------------------------------


class TestDashboardDefaultProject:
    def test_returns_200_no_param(self, client):
        assert client.get("/aim/dashboard").status_code == 200

    def test_returns_200_technomancer_alias(self, client):
        assert client.get("/aim/dashboard?project=technomancer").status_code == 200

    def test_default_embeds_empty_string_as_initial_project(self, client):
        body = client.get("/aim/dashboard").get_data(as_text=True)
        assert 'let currentProject = "" ||' in body

    def test_placeholder_not_in_default_response(self, client):
        body = client.get("/aim/dashboard").get_data(as_text=True)
        assert "__PROJECT_JSON__" not in body


# ---------------------------------------------------------------------------
# /aim/dashboard — non-default project
# ---------------------------------------------------------------------------


class TestDashboardProjectParam:
    def test_returns_200_with_project(self, client):
        assert client.get("/aim/dashboard?project=40acres").status_code == 200

    def test_project_embedded_in_js_initial_state(self, client):
        body = client.get("/aim/dashboard?project=40acres").get_data(as_text=True)
        assert 'let currentProject = "40acres" ||' in body

    def test_placeholder_replaced_in_response(self, client):
        body = client.get("/aim/dashboard?project=40acres").get_data(as_text=True)
        assert "__PROJECT_JSON__" not in body

    def test_content_type_is_html(self, client):
        resp = client.get("/aim/dashboard?project=40acres")
        assert "text/html" in resp.content_type

    def test_still_has_backlog_reference(self, client):
        body = client.get("/aim/dashboard?project=40acres").get_data(as_text=True)
        assert "/api/aim/backlog" in body

    def test_withproject_helper_present(self, client):
        body = client.get("/aim/dashboard?project=40acres").get_data(as_text=True)
        assert "function withProject(" in body

    def test_different_projects_produce_different_initial_state(self, client):
        body_40acres = client.get("/aim/dashboard?project=40acres").get_data(as_text=True)
        body_default = client.get("/aim/dashboard").get_data(as_text=True)
        assert body_40acres != body_default
