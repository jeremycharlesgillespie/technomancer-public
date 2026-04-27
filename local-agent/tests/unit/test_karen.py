"""
Tests for KAREN — Kinetic Aggression Routing Enhancement Network.

Tests cover: complaint model, persistence, idea generation, auto-resolve lifecycle,
and Flask API endpoints.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idea_board.karen import (
    Complaint,
    add_complaint,
    dismiss_complaint,
    generate_ideas_from_complaint,
    load_complaints,
    resolve_complaint_for_idea,
    save_complaints,
    _next_complaint_id,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def patched_karen(tmp_path, monkeypatch):
    """Patch KAREN to use temp directory for complaints.json."""
    import idea_board.karen as karen_module

    monkeypatch.setattr(karen_module, "COMPLAINTS_FILE", tmp_path / "complaints.json")
    return tmp_path


@pytest.fixture
def patched_karen_and_provider(tmp_path, monkeypatch):
    """Patch KAREN's complaint storage and the board provider used by KAREN.

    KAREN writes generated ideas through ``board.get_provider().add(...)``.
    Tests mock the provider so they don't depend on idea-board local
    storage (deleted in PR 6) or on a live Jira backend.

    Returns a tuple ``(tmp_path, fake_provider)`` so tests can inspect
    the captured ``add(...)`` calls.
    """
    import idea_board.karen as karen_module

    monkeypatch.setattr(karen_module, "COMPLAINTS_FILE", tmp_path / "complaints.json")

    fake_provider = MagicMock()
    next_id = {"n": 0}

    def _fake_add(title, description, source="generic", category="feature", **_kwargs):
        next_id["n"] += 1
        item = MagicMock()
        item.id = f"TK-{next_id['n']:03d}"
        item.title = title
        item.description = description
        item.source = source
        item.category = category
        return item

    fake_provider.add.side_effect = _fake_add
    fake_provider.add_comment.return_value = None
    fake_provider.load_all.return_value = []

    monkeypatch.setattr("board.get_provider", lambda: fake_provider)
    monkeypatch.setattr("board.factory.get_provider", lambda: fake_provider)
    return tmp_path, fake_provider


# ============================================================================
# COMPLAINT MODEL
# ============================================================================


class TestComplaintModel:
    """Tests for Complaint dataclass."""

    def test_complaint_creation(self):
        """Complaint gets auto-timestamp and pending state."""
        c = Complaint(id="complaint-001", text="It's too slow")
        assert c.id == "complaint-001"
        assert c.text == "It's too slow"
        assert c.author == "web"
        assert c.state == "pending"
        assert c.timestamp  # Auto-set
        assert c.generated_idea_ids == []

    def test_to_dict_from_dict_roundtrip(self):
        """Serialize and deserialize a Complaint."""
        c = Complaint(
            id="complaint-002",
            text="Search is broken",
            author="testuser",
            generated_idea_ids=["idea-001", "idea-002"],
        )
        d = c.to_dict()
        restored = Complaint.from_dict(d)
        assert restored.id == c.id
        assert restored.text == c.text
        assert restored.author == c.author
        assert restored.generated_idea_ids == c.generated_idea_ids

    def test_next_complaint_id_empty(self):
        """First ID is complaint-001."""
        assert _next_complaint_id([]) == "complaint-001"

    def test_next_complaint_id_increments(self):
        """IDs increment correctly."""
        complaints = [
            Complaint(id="complaint-001", text="a"),
            Complaint(id="complaint-002", text="b"),
        ]
        assert _next_complaint_id(complaints) == "complaint-003"


# ============================================================================
# PERSISTENCE
# ============================================================================


class TestComplaintPersistence:
    """Tests for load/save complaints."""

    def test_load_empty(self, patched_karen):
        """Returns empty list when file doesn't exist."""
        assert load_complaints() == []

    def test_save_and_load(self, patched_karen):
        """Round-trip save/load of complaints."""
        complaints = [
            Complaint(id="complaint-001", text="Too slow"),
            Complaint(id="complaint-002", text="Ugly UI"),
        ]
        save_complaints(complaints)
        loaded = load_complaints()
        assert len(loaded) == 2
        assert loaded[0].text == "Too slow"
        assert loaded[1].text == "Ugly UI"

    def test_add_complaint(self, patched_karen):
        """add_complaint creates and persists a complaint."""
        c = add_complaint(text="News is stale", author="testuser")
        assert c.id == "complaint-001"
        assert c.state == "pending"
        assert c.author == "testuser"

        loaded = load_complaints()
        assert len(loaded) == 1

    def test_dismiss_complaint(self, patched_karen):
        """dismiss_complaint sets state to dismissed."""
        add_complaint(text="Something", author="web")
        result = dismiss_complaint("complaint-001")
        assert result is not None
        assert result.state == "dismissed"

        loaded = load_complaints()
        assert loaded[0].state == "dismissed"

    def test_dismiss_nonexistent(self, patched_karen):
        """Dismissing nonexistent complaint returns None."""
        assert dismiss_complaint("complaint-999") is None


# ============================================================================
# AUTO-RESOLVE LIFECYCLE
# ============================================================================


class TestAutoResolve:
    """Tests for complaint auto-resolution when ideas are accepted."""

    def test_resolve_complaint_for_idea(self, patched_karen):
        """resolve_complaint_for_idea sets complaint to resolved."""
        c = add_complaint(text="Fix the thing", author="web")
        # Manually set as processed with a linked idea
        complaints = load_complaints()
        complaints[0].state = "processed"
        complaints[0].generated_idea_ids = ["idea-042"]
        save_complaints(complaints)

        resolve_complaint_for_idea("idea-042")

        loaded = load_complaints()
        assert loaded[0].state == "resolved"

    def test_resolve_ignores_already_dismissed(self, patched_karen):
        """Already dismissed complaints are not re-resolved."""
        c = add_complaint(text="Whatever", author="web")
        complaints = load_complaints()
        complaints[0].state = "dismissed"
        complaints[0].generated_idea_ids = ["idea-042"]
        save_complaints(complaints)

        resolve_complaint_for_idea("idea-042")

        loaded = load_complaints()
        assert loaded[0].state == "dismissed"  # Unchanged

    def test_resolve_no_match(self, patched_karen):
        """No error when no complaint links to the idea."""
        add_complaint(text="Unrelated", author="web")
        resolve_complaint_for_idea("idea-999")  # No match, no crash

        loaded = load_complaints()
        assert loaded[0].state == "pending"  # Unchanged


# ============================================================================
# IDEA GENERATION
# ============================================================================


@pytest.mark.usefixtures("mock_dedup_llm")
class TestIdeaGeneration:
    """Tests for LLM-based idea generation from complaints."""

    @staticmethod
    def _mock_ollama_response(content: str):
        """Helper to create a mock Ollama client that returns given content."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": content}}
        return mock_client

    @patch("ollama.Client")
    def test_generate_ideas_from_complaint(self, mock_client_cls, patched_karen_and_provider):
        """Mock Ollama and provider; verify provider.add is called with source='karen'."""
        _tmp, fake_provider = patched_karen_and_provider

        mock_client_cls.return_value = self._mock_ollama_response(
            json.dumps([{
                "title": "Speed up search indexing",
                "description": "WHAT: Optimize search\nWHY: It's slow",
                "category": "performance",
            }])
        )

        complaint = add_complaint(text="Search is too slow", author="testuser")
        idea_ids = generate_ideas_from_complaint(complaint)

        assert len(idea_ids) == 1
        assert fake_provider.add.call_count == 1
        add_kwargs = fake_provider.add.call_args.kwargs
        assert add_kwargs["title"] == "Speed up search indexing"
        assert add_kwargs["source"] == "karen"
        assert add_kwargs["category"] == "performance"

    @patch("ollama.Client")
    def test_complaint_links_to_ideas(self, mock_client_cls, patched_karen_and_provider):
        """Complaint's generated_idea_ids is populated after generation."""
        _tmp, fake_provider = patched_karen_and_provider

        mock_client_cls.return_value = self._mock_ollama_response(
            json.dumps([
                {"title": "Idea A", "description": "Desc A", "category": "ux"},
                {"title": "Idea B", "description": "Desc B", "category": "feature"},
            ])
        )

        complaint = add_complaint(text="Everything is bad", author="web")
        idea_ids = generate_ideas_from_complaint(complaint)

        assert len(idea_ids) == 2
        assert fake_provider.add.call_count == 2
        loaded = load_complaints()
        assert loaded[0].state == "processed"
        assert loaded[0].generated_idea_ids == idea_ids

    @patch("ollama.Client")
    def test_handles_empty_response(self, mock_client_cls, patched_karen_and_provider):
        """No ideas generated when LLM returns bad JSON."""
        _tmp, fake_provider = patched_karen_and_provider

        mock_client_cls.return_value = self._mock_ollama_response(
            "I have no suggestions."
        )

        complaint = add_complaint(text="Meh", author="web")
        idea_ids = generate_ideas_from_complaint(complaint)

        assert idea_ids == []
        assert fake_provider.add.call_count == 0

    @patch("ollama.Client")
    def test_handles_llm_error(self, mock_client_cls, patched_karen_and_provider):
        """Returns empty list when Ollama call fails."""
        _tmp, fake_provider = patched_karen_and_provider

        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_client

        complaint = add_complaint(text="Help", author="web")
        idea_ids = generate_ideas_from_complaint(complaint)

        assert idea_ids == []
        assert fake_provider.add.call_count == 0
