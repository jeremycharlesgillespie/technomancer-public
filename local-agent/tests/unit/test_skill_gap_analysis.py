"""Tests for the skill_gap_analysis module — gap-to-learning recommendation pipeline."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.skill_gap_analysis import (
    classify_category,
    init_db,
    analyze_skill_gaps,
    get_skill_gap_report,
    get_progress_summary,
    mark_recommendation_completed,
    get_skill_gap_tools,
    _find_matching_topics,
    _find_matching_articles,
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirect SQLite to a temp directory for every test."""
    monkeypatch.setattr("agent.skill_gap_analysis.DB_DIR", tmp_path)
    monkeypatch.setattr("agent.skill_gap_analysis.DB_PATH", tmp_path / "skill_gaps.db")
    # Clear per-thread connection cache
    import agent.skill_gap_analysis as mod
    if hasattr(mod._local, "conn"):
        del mod._local.conn


# Reusable mock data for gap clusters
MOCK_GAP = {
    "query": "What are python decorators?",
    "gap_type": "uncertainty",
    "timestamp": "2026-04-01 10:00",
    "response_snippet": "...",
    "status": "open",
}

MOCK_CLUSTER = {
    "label": "python + decorators",
    "domain": "programming",
    "count": 3,
    "gap_types": {"uncertainty": 3},
    "queries": ["What are python decorators?"],
    "first_seen": "2026-04-01",
    "last_seen": "2026-04-09",
}


def _patch_gap_sources(mock_gaps=None, mock_clusters=None):
    """Return a dict of patches for gap data sources."""
    return {
        "parse": patch("agent.gap_frequency._parse_all_gaps", return_value=mock_gaps or []),
        "scan": patch("agent.gap_frequency._scan_conversations_for_gaps", return_value=[]),
        "cluster": patch("agent.gap_frequency.cluster_gaps", return_value=mock_clusters or []),
        "articles": patch("agent.github_pages.list_article_files", return_value=[]),
    }


class TestInitDb:
    def test_creates_tables(self, tmp_path):
        init_db()
        conn = sqlite3.connect(str(tmp_path / "skill_gaps.db"))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "skill_gaps" in tables
        assert "recommendations" in tables

    def test_idempotent(self, tmp_path):
        init_db()
        init_db()  # second call should not fail


class TestClassifyCategory:
    def test_python_keywords(self):
        cat = classify_category("python decorators", ["how do python decorators work"])
        assert cat == "python"

    def test_oracle_keywords(self):
        cat = classify_category("oracle execution plan", ["explain plan for query"])
        assert cat == "oracle"

    def test_system_design_keywords(self):
        cat = classify_category("caching redis", ["cache invalidation strategies"])
        assert cat == "system_design"

    def test_fallback_to_best_practices(self):
        cat = classify_category("xyz zzz", ["completely unknown topic aaa bbb"])
        assert cat == "best_practices"


class TestFindMatchingTopics:
    def test_finds_python_topics(self):
        keywords = {"python", "decorators", "advanced"}
        topics = _find_matching_topics(keywords, "python", limit=3)
        assert len(topics) >= 1
        assert any("decorator" in t.lower() for t in topics)

    def test_returns_empty_for_no_match(self):
        keywords = {"xyznonexistent"}
        topics = _find_matching_topics(keywords, "python", limit=3)
        assert topics == []

    def test_respects_limit(self):
        keywords = {"python"}
        topics = _find_matching_topics(keywords, "python", limit=1)
        assert len(topics) <= 1


class TestFindMatchingArticles:
    def test_with_no_articles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.skill_gap_analysis.settings",
            MagicMock(llm_memory_path=tmp_path),
        )
        with patch("agent.github_pages.list_article_files", return_value=[]):
            results = _find_matching_articles({"python"})
        assert results == []

    def test_matches_vault_article(self, tmp_path, monkeypatch):
        learning_dir = tmp_path / "Learning"
        learning_dir.mkdir()
        (learning_dir / "001-python-decorators-advanced.md").write_text("content")

        monkeypatch.setattr(
            "agent.skill_gap_analysis.settings",
            MagicMock(llm_memory_path=tmp_path),
        )
        with patch("agent.github_pages.list_article_files", return_value=[]):
            results = _find_matching_articles({"python", "decorators"})
        assert len(results) >= 1
        assert results[0]["type"] == "vault_article"
        assert "python" in results[0]["title"].lower()


class TestAnalyzeSkillGaps:
    def test_empty_gaps_returns_empty(self):
        patches = _patch_gap_sources()
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            result = analyze_skill_gaps(days=7)
        assert result == []

    def test_creates_gap_entries(self):
        patches = _patch_gap_sources(
            mock_gaps=[MOCK_GAP],
            mock_clusters=[MOCK_CLUSTER],
        )
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            result = analyze_skill_gaps(days=30)
        assert len(result) == 1
        assert result[0]["label"] == "python + decorators"
        assert result[0]["category"] == "python"
        assert result[0]["gap_count"] == 3

    def test_upserts_on_second_run(self):
        cluster = dict(MOCK_CLUSTER)  # copy
        patches = _patch_gap_sources(mock_gaps=[MOCK_GAP], mock_clusters=[cluster])
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            result1 = analyze_skill_gaps()

        cluster2 = dict(MOCK_CLUSTER)
        cluster2["count"] = 5
        patches2 = _patch_gap_sources(mock_gaps=[MOCK_GAP], mock_clusters=[cluster2])
        with patches2["parse"], patches2["scan"], patches2["cluster"], patches2["articles"]:
            result2 = analyze_skill_gaps()

        assert result1[0]["gap_id"] == result2[0]["gap_id"]  # same row updated
        assert result2[0]["gap_count"] == 5


class TestMarkRecommendationCompleted:
    def test_marks_complete(self):
        patches = _patch_gap_sources(mock_gaps=[MOCK_GAP], mock_clusters=[MOCK_CLUSTER])
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            analyze_skill_gaps()

        import agent.skill_gap_analysis as mod
        conn = mod._get_conn()
        rec = conn.execute("SELECT id FROM recommendations LIMIT 1").fetchone()
        if rec:
            result = mark_recommendation_completed(rec["id"])
            assert "completed" in result.lower()

    def test_nonexistent_id(self):
        init_db()
        result = mark_recommendation_completed(9999)
        assert "not found" in result.lower()


class TestGetSkillGapReport:
    def test_empty_report(self):
        report = get_skill_gap_report()
        assert "No skill gaps" in report

    def test_report_with_data(self):
        patches = _patch_gap_sources(mock_gaps=[MOCK_GAP], mock_clusters=[MOCK_CLUSTER])
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            analyze_skill_gaps()
        report = get_skill_gap_report()
        assert "Skill Gap Analysis" in report
        assert "python" in report.lower()
        assert "Progress:" in report


class TestGetProgressSummary:
    def test_empty(self):
        summary = get_progress_summary()
        assert summary["total_gaps"] == 0
        assert summary["open_gaps"] == 0

    def test_with_data(self):
        patches = _patch_gap_sources(mock_gaps=[MOCK_GAP], mock_clusters=[MOCK_CLUSTER])
        with patches["parse"], patches["scan"], patches["cluster"], patches["articles"]:
            analyze_skill_gaps()
        summary = get_progress_summary()
        assert summary["total_gaps"] >= 1
        assert summary["open_gaps"] >= 1


class TestGetSkillGapTools:
    def test_returns_three_tools(self):
        tools = get_skill_gap_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"skill_gap_analyze", "skill_gap_report", "skill_gap_complete"}

    def test_report_tool_runs(self):
        tools = get_skill_gap_tools()
        report_tool = next(t for t in tools if t.name == "skill_gap_report")
        result = report_tool.function()
        assert isinstance(result, str)
