"""Tests for the project selector dropdown on /aim and /aim/dashboard."""

from __future__ import annotations

import pytest

from idea_board.web import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def aim_body(client):
    return client.get("/aim").get_data(as_text=True)


@pytest.fixture
def dashboard_body(client):
    return client.get("/aim/dashboard").get_data(as_text=True)


# ---------------------------------------------------------------------------
# /aim page
# ---------------------------------------------------------------------------


class TestAimDropdownPresent:
    def test_dropdown_container_rendered(self, aim_body):
        assert 'id="project-selector-bar"' in aim_body

    def test_dropdown_select_rendered(self, aim_body):
        assert 'id="project-select"' in aim_body

    def test_dropdown_has_default_option(self, aim_body):
        assert 'value="technomancer"' in aim_body

    def test_dropdown_has_label(self, aim_body):
        assert 'for="project-select"' in aim_body

    def test_dropdown_has_aria_label(self, aim_body):
        assert 'aria-label="Select AIM project"' in aim_body


class TestAimDropdownJS:
    def test_fetches_projects_endpoint(self, aim_body):
        assert "/api/aim/projects" in aim_body

    def test_populates_options_dynamically(self, aim_body):
        assert "fetchAimProjects" in aim_body

    def test_uses_url_param_for_initial_selection(self, aim_body):
        assert "_aimUrlParams" in aim_body
        assert "get('project')" in aim_body

    def test_change_handler_updates_current_project(self, aim_body):
        assert "currentProject = this.value" in aim_body

    def test_change_handler_uses_history_replace_state(self, aim_body):
        assert "history.replaceState" in aim_body

    def test_change_handler_triggers_poll(self, aim_body):
        assert "poll();" in aim_body

    def test_with_project_helper_defined(self, aim_body):
        assert "function withProject(" in aim_body

    def test_poll_uses_with_project(self, aim_body):
        assert "withProject('/api/aim/status')" in aim_body

    def test_technomancer_removes_param(self, aim_body):
        assert "url.searchParams.delete('project')" in aim_body

    def test_other_project_sets_param(self, aim_body):
        assert "url.searchParams.set('project', currentProject)" in aim_body


class TestAimDropdownSSENotDuplicated:
    def test_sse_source_opened_once(self, aim_body):
        """EventSource must be created exactly once — not inside the change handler."""
        assert aim_body.count("new EventSource(") == 1

    def test_sse_source_not_inside_change_handler(self, aim_body):
        """The SSE stream is global; it must not be closed/reopened on project switch."""
        change_handler_start = aim_body.find(
            "addEventListener('change', function()"
        )
        change_handler_end = aim_body.find("});", change_handler_start)
        handler_slice = aim_body[change_handler_start:change_handler_end]
        assert "EventSource" not in handler_slice


# ---------------------------------------------------------------------------
# /aim/dashboard page
# ---------------------------------------------------------------------------


class TestDashboardDropdownPresent:
    def test_dropdown_container_rendered(self, dashboard_body):
        assert 'id="project-selector-bar"' in dashboard_body

    def test_dropdown_select_rendered(self, dashboard_body):
        assert 'id="project-select"' in dashboard_body

    def test_dropdown_has_default_option(self, dashboard_body):
        assert 'value="technomancer"' in dashboard_body

    def test_dropdown_has_label(self, dashboard_body):
        assert 'for="project-select"' in dashboard_body

    def test_dropdown_has_aria_label(self, dashboard_body):
        assert 'aria-label="Select AIM project"' in dashboard_body

    def test_dropdown_inside_page_header(self, dashboard_body):
        header_start = dashboard_body.find('class="page-header"')
        # page-header ends before the first <section element
        section_start = dashboard_body.find("<section", header_start)
        assert header_start != -1 and section_start != -1
        assert 'id="project-selector-bar"' in dashboard_body[header_start:section_start]


class TestDashboardDropdownJS:
    def test_fetches_projects_endpoint(self, dashboard_body):
        assert "/api/aim/projects" in dashboard_body

    def test_populates_options_dynamically(self, dashboard_body):
        assert "fetchDashProjects" in dashboard_body

    def test_uses_url_param_for_initial_selection(self, dashboard_body):
        assert "_dashUrlParams" in dashboard_body
        assert "get('project')" in dashboard_body

    def test_change_handler_updates_current_project(self, dashboard_body):
        assert "currentProject = this.value" in dashboard_body

    def test_change_handler_uses_history_replace_state(self, dashboard_body):
        assert "history.replaceState" in dashboard_body

    def test_change_handler_triggers_fetchbacklog(self, dashboard_body):
        change_start = dashboard_body.find("addEventListener('change', function()")
        change_end = dashboard_body.find("});", change_start)
        handler = dashboard_body[change_start:change_end]
        assert "fetchBacklog()" in handler

    def test_change_handler_triggers_fetchmetrics(self, dashboard_body):
        change_start = dashboard_body.find("addEventListener('change', function()")
        change_end = dashboard_body.find("});", change_start)
        handler = dashboard_body[change_start:change_end]
        assert "fetchMetrics(currentHours)" in handler

    def test_with_project_helper_defined(self, dashboard_body):
        assert "function withProject(" in dashboard_body

    def test_fetchbacklog_uses_with_project(self, dashboard_body):
        assert "withProject('/api/aim/backlog')" in dashboard_body

    def test_fetchmetrics_uses_with_project(self, dashboard_body):
        assert "withProject('/api/aim/metrics" in dashboard_body

    def test_technomancer_removes_param(self, dashboard_body):
        assert "url.searchParams.delete('project')" in dashboard_body

    def test_other_project_sets_param(self, dashboard_body):
        assert "url.searchParams.set('project', currentProject)" in dashboard_body


class TestDropdownCSS:
    def test_aim_dropdown_css_defined(self, aim_body):
        assert "#project-selector-bar" in aim_body
        assert "#project-select" in aim_body

    def test_dashboard_dropdown_css_defined(self, dashboard_body):
        assert "#project-selector-bar" in dashboard_body
        assert "#project-select" in dashboard_body
