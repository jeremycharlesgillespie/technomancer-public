"""Tests for the profiler module — request timing and performance summaries."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.profiler import RequestProfile, RequestTimer, get_performance_summary, get_recent_profiles


class TestRequestProfile:
    """Test RequestProfile dataclass."""

    def test_creates_with_defaults(self):
        p = RequestProfile(user="test", message="hello", message_length=5)
        assert p.user == "test"
        assert p.message == "hello"
        assert p.total == 0.0
        assert p.llm_calls == []
        assert p.tool_executions == []
        assert p.claude_escalated is False

    def test_to_dict_contains_all_fields(self):
        p = RequestProfile(user="test", message="hello", message_length=5)
        d = p.to_dict()
        assert d["user"] == "test"
        assert d["message"] == "hello"
        assert "llm_calls" in d
        assert "tool_executions" in d
        assert "total_seconds" in d or "total" in d
        assert "classification" in d or "context_tier" in d

    def test_to_dict_serializes_llm_calls(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        p.llm_calls.append(MagicMock(
            turn=1, duration=1.5, input_chars=100, output_chars=50,
            num_ctx=8192, tool_calls=["web_search"],
        ))
        d = p.to_dict()
        assert len(d["llm_calls"]) == 1


class TestRequestTimer:
    """Test RequestTimer context manager and recording."""

    def test_phase_records_timing(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        timer = RequestTimer(p)
        with timer.phase("context_build"):
            time.sleep(0.01)
        assert p.context_build >= 0.01

    def test_record_llm_call(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        timer = RequestTimer(p)
        timer.record_llm_call(
            turn=1, duration=2.5, input_chars=1000,
            output_chars=500, num_ctx=8192, tool_calls=["read_file"],
        )
        assert len(p.llm_calls) == 1
        assert p.llm_calls[0].duration == 2.5
        assert p.llm_calls[0].tool_calls == ["read_file"]

    def test_record_tool(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        timer = RequestTimer(p)
        timer.record_tool("web_search", 1.2, 5000, truncated=True)
        assert len(p.tool_executions) == 1
        assert p.tool_executions[0].name == "web_search"
        assert p.tool_executions[0].truncated is True

    def test_finish_sets_total(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        timer = RequestTimer(p)
        time.sleep(0.01)
        timer.finish()
        assert p.total >= 0.01
        assert p.timestamp != ""

    def test_save_writes_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.profiler.PROFILING_DIR", tmp_path)
        monkeypatch.setattr("agent.profiler.REQUESTS_LOG", tmp_path / "requests.jsonl")

        p = RequestProfile(user="test", message="hello", message_length=5)
        timer = RequestTimer(p)
        timer.finish()
        timer.save()

        log_file = tmp_path / "requests.jsonl"
        assert log_file.exists()
        data = json.loads(log_file.read_text().strip())
        assert data["user"] == "test"

    def test_multiple_phases(self):
        p = RequestProfile(user="test", message="hi", message_length=2)
        timer = RequestTimer(p)
        with timer.phase("context_build"):
            time.sleep(0.01)
        with timer.phase("pre_classification"):
            time.sleep(0.01)
        assert p.context_build >= 0.01
        assert p.pre_classification >= 0.01


class TestGetRecentProfiles:
    """Test profile retrieval."""

    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.profiler.REQUESTS_LOG", tmp_path / "nonexistent.jsonl")
        result = get_recent_profiles()
        assert result == []

    def test_reads_profiles(self, tmp_path, monkeypatch):
        log_file = tmp_path / "requests.jsonl"
        profiles = [
            {"user": "alice", "total": 1.0, "message": "q1"},
            {"user": "bob", "total": 2.0, "message": "q2"},
        ]
        log_file.write_text("\n".join(json.dumps(p) for p in profiles))
        monkeypatch.setattr("agent.profiler.REQUESTS_LOG", log_file)

        result = get_recent_profiles(n=10)
        assert len(result) == 2

    def test_respects_limit(self, tmp_path, monkeypatch):
        log_file = tmp_path / "requests.jsonl"
        profiles = [{"user": f"user{i}", "total": float(i)} for i in range(10)]
        log_file.write_text("\n".join(json.dumps(p) for p in profiles))
        monkeypatch.setattr("agent.profiler.REQUESTS_LOG", log_file)

        result = get_recent_profiles(n=3)
        assert len(result) == 3


class TestGetPerformanceSummary:
    """Test performance summary generation."""

    def test_empty_summary(self, monkeypatch):
        monkeypatch.setattr("agent.profiler.get_recent_profiles", lambda n=50: [])
        result = get_performance_summary()
        assert "No profiling data" in result

    def test_summary_with_data(self, monkeypatch):
        profiles = [
            {
                "user": "test",
                "message": "hello world",
                "total_seconds": 3.5,
                "llm_calls": [{"duration": 2.0, "input_chars": 100, "output_chars": 50}],
                "tool_executions": [],
                "llm_summary": {"total_calls": 1, "total_llm_seconds": 2.0, "total_tool_seconds": 0, "avg_call_seconds": 2.0},
                "classification": {"question_type": "factual", "context_injected_chars": 2000, "num_ctx_used": 8192},
            },
        ]
        monkeypatch.setattr("agent.profiler.get_recent_profiles", lambda n=50: profiles)
        result = get_performance_summary()
        assert "Performance Summary" in result or "profiling" in result.lower()
