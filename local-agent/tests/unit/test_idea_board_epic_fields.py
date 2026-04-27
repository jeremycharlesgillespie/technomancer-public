"""Tests for the /api/jira/<key>/execute routing entry point.

The model-level set/get_execution_order, set_epic_context, and
Idea-dataclass tests that used to live here were removed in PR 6 of
the Jira-only refactor along with ``idea_board/models.py``. The
remaining test class pins the routing contract: epics route through
``execute_epic``, stories/tasks route through ``execute_idea``.
"""

from unittest.mock import MagicMock, patch

import pytest

from board.types import Idea
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestApiExecuteRouting:
    """POST /api/jira/<id>/execute — epics go through the Executor Manager
    (execute_epic), stories/tasks go directly through execute_idea.

    This is the entry point the Execute button hits; it's the glue that ties
    the execution_order / epic_context fields and the sequential story loop
    into one behavior.
    """

    def _fake_state(self):
        state = MagicMock()
        state.pid = 12345
        return state

    def test_execute_on_epic_calls_execute_epic(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", state="approved",
        )
        with patch("idea_board.web.get_idea", return_value=epic), \
             patch("idea_board.executor.execute_epic", return_value=self._fake_state()) as mock_ep, \
             patch("idea_board.executor.execute_idea") as mock_ei:
            resp = client.post("/api/jira/idea-001/execute")
            assert resp.status_code == 200
            assert resp.get_json() == {
                "status": "executing", "key": "idea-001", "pid": 12345,
            }
            mock_ep.assert_called_once_with("idea-001")
            mock_ei.assert_not_called()

    def test_execute_on_story_calls_execute_idea(self, client):
        story = Idea(
            id="idea-002", title="Story", description="D",
            idea_type="story", state="approved",
        )
        with patch("idea_board.web.get_idea", return_value=story), \
             patch("idea_board.executor.execute_epic") as mock_ep, \
             patch("idea_board.executor.execute_idea", return_value=self._fake_state()) as mock_ei:
            resp = client.post("/api/jira/idea-002/execute")
            assert resp.status_code == 200
            mock_ei.assert_called_once_with("idea-002")
            mock_ep.assert_not_called()

    def test_execute_on_task_calls_execute_idea(self, client):
        task = Idea(
            id="idea-003", title="Task", description="D",
            idea_type="task", state="approved",
        )
        with patch("idea_board.web.get_idea", return_value=task), \
             patch("idea_board.executor.execute_epic") as mock_ep, \
             patch("idea_board.executor.execute_idea", return_value=self._fake_state()) as mock_ei:
            resp = client.post("/api/jira/idea-003/execute")
            assert resp.status_code == 200
            mock_ei.assert_called_once_with("idea-003")
            mock_ep.assert_not_called()

    def test_execute_missing_idea_returns_404(self, client):
        with patch("idea_board.web.get_idea", return_value=None):
            resp = client.post("/api/jira/idea-999/execute")
            assert resp.status_code == 404

    def test_execute_epic_returning_none_is_500(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", state="approved",
        )
        with patch("idea_board.web.get_idea", return_value=epic), \
             patch("idea_board.executor.execute_epic", return_value=None):
            resp = client.post("/api/jira/idea-001/execute")
            assert resp.status_code == 500
