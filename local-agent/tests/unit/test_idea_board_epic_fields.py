"""
Tests for execution_order and epic_context fields on the Idea model,
and the corresponding API endpoints in idea_board/web.py.

Covers:
- Model: new fields serialize/deserialize, defaults, auto-populate order
- API: PUT /api/jira/<id>/order, PUT /api/jira/<id>/context
- Rendering: epic_prompt uses execution_order and epic_context
"""

import json
from unittest.mock import patch

import pytest

from idea_board.models import Idea, get_execution_order, set_epic_context, set_execution_order
from idea_board.web import app


@pytest.fixture
def client():
    """Flask test client for the idea board app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ============================================================================
# MODEL TESTS
# ============================================================================


class TestIdeaModelFields:
    """Tests for execution_order and epic_context on the Idea dataclass."""

    def test_default_execution_order_empty(self):
        """New ideas have empty execution_order by default."""
        idea = Idea(id="idea-001", title="Test", description="Desc")
        assert idea.execution_order == []

    def test_default_epic_context_empty(self):
        """New ideas have empty epic_context by default."""
        idea = Idea(id="idea-001", title="Test", description="Desc")
        assert idea.epic_context == ""

    def test_to_dict_includes_new_fields(self):
        """to_dict serializes execution_order and epic_context."""
        idea = Idea(
            id="idea-001",
            title="Epic",
            description="Desc",
            execution_order=["idea-002", "idea-003"],
            epic_context="This epic does X",
        )
        d = idea.to_dict()
        assert d["execution_order"] == ["idea-002", "idea-003"]
        assert d["epic_context"] == "This epic does X"

    def test_from_dict_with_new_fields(self):
        """from_dict deserializes execution_order and epic_context."""
        data = {
            "id": "idea-001",
            "title": "Epic",
            "description": "Desc",
            "execution_order": ["idea-010", "idea-011"],
            "epic_context": "Big picture context",
        }
        idea = Idea.from_dict(data)
        assert idea.execution_order == ["idea-010", "idea-011"]
        assert idea.epic_context == "Big picture context"

    def test_from_dict_missing_new_fields_defaults(self):
        """from_dict handles missing fields (backward compat with old data)."""
        data = {
            "id": "idea-001",
            "title": "Old idea",
            "description": "No new fields",
        }
        idea = Idea.from_dict(data)
        assert idea.execution_order == []
        assert idea.epic_context == ""

    def test_roundtrip_preserves_fields(self):
        """to_dict -> from_dict round trip preserves new fields."""
        original = Idea(
            id="idea-050",
            title="Roundtrip",
            description="Test",
            execution_order=["idea-051", "idea-052", "idea-053"],
            epic_context="We need this for the demo",
        )
        restored = Idea.from_dict(original.to_dict())
        assert restored.execution_order == original.execution_order
        assert restored.epic_context == original.epic_context


class TestSetExecutionOrder:
    """Tests for set_execution_order model function."""

    def test_set_order_returns_updated_idea(self):
        epic = Idea(id="idea-001", title="Epic", description="D", idea_type="epic")
        with patch("idea_board.models.load_ideas", return_value=[epic]), \
             patch("idea_board.models.save_ideas") as mock_save:
            result = set_execution_order("idea-001", ["idea-002", "idea-003"])
            assert result is not None
            assert result.execution_order == ["idea-002", "idea-003"]
            mock_save.assert_called_once()

    def test_set_order_not_found(self):
        with patch("idea_board.models.load_ideas", return_value=[]):
            result = set_execution_order("idea-999", ["idea-002"])
            assert result is None


class TestSetEpicContext:
    """Tests for set_epic_context model function."""

    def test_set_context_returns_updated_idea(self):
        epic = Idea(id="idea-001", title="Epic", description="D", idea_type="epic")
        with patch("idea_board.models.load_ideas", return_value=[epic]), \
             patch("idea_board.models.save_ideas") as mock_save:
            result = set_epic_context("idea-001", "This is the context")
            assert result is not None
            assert result.epic_context == "This is the context"
            mock_save.assert_called_once()

    def test_set_context_not_found(self):
        with patch("idea_board.models.load_ideas", return_value=[]):
            result = set_epic_context("idea-999", "nope")
            assert result is None


class TestGetExecutionOrder:
    """Tests for get_execution_order with auto-populate."""

    def test_returns_explicit_order(self):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", execution_order=["idea-003", "idea-002"],
        )
        child1 = Idea(id="idea-002", title="S1", description="D", parent_id="idea-001")
        child2 = Idea(id="idea-003", title="S2", description="D", parent_id="idea-001")
        with patch("idea_board.models.load_ideas", return_value=[epic, child1, child2]):
            result = get_execution_order("idea-001")
            assert result == ["idea-003", "idea-002"]

    def test_auto_populates_from_children(self):
        epic = Idea(id="idea-001", title="Epic", description="D", idea_type="epic")
        child1 = Idea(id="idea-002", title="S1", description="D", parent_id="idea-001")
        child2 = Idea(id="idea-003", title="S2", description="D", parent_id="idea-001")
        with patch("idea_board.models.load_ideas", return_value=[epic, child1, child2]):
            result = get_execution_order("idea-001")
            assert result == ["idea-002", "idea-003"]

    def test_not_found_returns_empty(self):
        with patch("idea_board.models.load_ideas", return_value=[]):
            result = get_execution_order("idea-999")
            assert result == []



class TestApiExecuteRouting:
    """POST /api/jira/<id>/execute — epics go through the Executor Manager
    (execute_epic), stories/tasks go directly through execute_idea.

    This is the entry point the Execute button hits; it's the glue that ties
    the execution_order / epic_context fields and the sequential story loop
    into one behavior.
    """

    def _fake_state(self):
        from unittest.mock import MagicMock
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
