"""
Tests for execution_order and epic_context fields on the Idea model,
and the corresponding API endpoints in idea_board/web.py.

Covers:
- Model: new fields serialize/deserialize, defaults, auto-populate order
- API: PUT /api/ideas/<id>/order, PUT /api/ideas/<id>/context
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


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================


class TestApiSetOrder:
    """Tests for PUT /api/ideas/<id>/order."""

    def test_set_order_success(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", execution_order=["idea-002", "idea-003"],
        )
        with patch("idea_board.web.set_execution_order", return_value=epic):
            resp = client.put(
                "/api/ideas/idea-001/order",
                data=json.dumps({"order": ["idea-002", "idea-003"]}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["execution_order"] == ["idea-002", "idea-003"]

    def test_set_order_not_found(self, client):
        with patch("idea_board.web.set_execution_order", return_value=None):
            resp = client.put(
                "/api/ideas/idea-999/order",
                data=json.dumps({"order": ["idea-001"]}),
                content_type="application/json",
            )
            assert resp.status_code == 404

    def test_set_order_invalid_body(self, client):
        resp = client.put(
            "/api/ideas/idea-001/order",
            data=json.dumps({"order": "not-a-list"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_set_order_missing_body(self, client):
        resp = client.put(
            "/api/ideas/idea-001/order",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestApiSetContext:
    """Tests for PUT /api/ideas/<id>/context."""

    def test_set_context_success(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", epic_context="New context",
        )
        with patch("idea_board.web.set_epic_context", return_value=epic):
            resp = client.put(
                "/api/ideas/idea-001/context",
                data=json.dumps({"context": "New context"}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["epic_context"] == "New context"

    def test_set_context_not_found(self, client):
        with patch("idea_board.web.set_epic_context", return_value=None):
            resp = client.put(
                "/api/ideas/idea-999/context",
                data=json.dumps({"context": "whatever"}),
                content_type="application/json",
            )
            assert resp.status_code == 404

    def test_set_context_empty_string_allowed(self, client):
        """Clearing epic context with empty string should work."""
        epic = Idea(id="idea-001", title="Epic", description="D", epic_context="")
        with patch("idea_board.web.set_epic_context", return_value=epic):
            resp = client.put(
                "/api/ideas/idea-001/context",
                data=json.dumps({"context": ""}),
                content_type="application/json",
            )
            assert resp.status_code == 200


class TestEpicPromptUsesOrder:
    """Tests that epic_prompt endpoint respects execution_order and epic_context."""

    def test_epic_prompt_includes_context(self, client):
        epic = Idea(
            id="idea-001", title="Big Feature", description="Build it",
            idea_type="epic", epic_context="This is the big picture",
        )
        story = Idea(
            id="idea-002", title="Story 1", description="Do thing",
            parent_id="idea-001", state="proposed",
        )
        with patch("idea_board.web.get_idea", return_value=epic), \
             patch("idea_board.web.load_ideas", return_value=[epic, story]), \
             patch("idea_board.web.get_execution_order", return_value=["idea-002"]):
            resp = client.get("/api/ideas/idea-001/epic_prompt")
            assert resp.status_code == 200
            prompt = resp.get_json()["prompt"]
            assert "This is the big picture" in prompt
            assert "Epic Context" in prompt

    def test_epic_prompt_respects_order(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic",
            execution_order=["idea-003", "idea-002"],
        )
        story1 = Idea(
            id="idea-002", title="Second", description="Do second",
            parent_id="idea-001", state="proposed",
        )
        story2 = Idea(
            id="idea-003", title="First", description="Do first",
            parent_id="idea-001", state="proposed",
        )
        with patch("idea_board.web.get_idea", return_value=epic), \
             patch("idea_board.web.load_ideas", return_value=[epic, story1, story2]), \
             patch("idea_board.web.get_execution_order", return_value=["idea-003", "idea-002"]):
            resp = client.get("/api/ideas/idea-001/epic_prompt")
            prompt = resp.get_json()["prompt"]
            # "First" story should appear before "Second" in the prompt
            assert prompt.index("First") < prompt.index("Second")

    def test_epic_prompt_no_context_omits_section(self, client):
        epic = Idea(
            id="idea-001", title="Epic", description="D",
            idea_type="epic", epic_context="",
        )
        story = Idea(
            id="idea-002", title="Story", description="Do",
            parent_id="idea-001", state="proposed",
        )
        with patch("idea_board.web.get_idea", return_value=epic), \
             patch("idea_board.web.load_ideas", return_value=[epic, story]), \
             patch("idea_board.web.get_execution_order", return_value=["idea-002"]):
            resp = client.get("/api/ideas/idea-001/epic_prompt")
            prompt = resp.get_json()["prompt"]
            assert "Epic Context" not in prompt
