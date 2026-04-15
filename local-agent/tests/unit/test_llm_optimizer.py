"""Tests for the llm_optimizer module — caching, complexity scoring, context decay, analytics."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from agent.llm_optimizer import (
    cache_lookup,
    cache_store,
    compute_context_decay,
    get_cache_stats,
    get_llm_optimizer_tools,
    get_model_for_complexity,
    get_usage_dashboard,
    init_cache_db,
    rank_context_by_relevance,
    score_query_complexity,
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.llm_optimizer.DATA_DIR", tmp_path)
    monkeypatch.setattr("agent.llm_optimizer.CACHE_DB_PATH", tmp_path / "response_cache.db")
    import agent.llm_optimizer as mod
    if hasattr(mod._local, "cache_conn"):
        try:
            mod._local.cache_conn.close()
        except Exception:
            pass
        del mod._local.cache_conn
    init_cache_db()


class TestResponseCache:
    def test_store_and_lookup(self):
        cache_store("What is Python?", "Python is a programming language.", "ollama", "qwen", 100)
        result = cache_lookup("What is Python?")
        assert result is not None
        assert "programming language" in result

    def test_case_insensitive(self):
        cache_store("WHAT IS PYTHON?", "Python is a language.")
        result = cache_lookup("what is python?")
        assert result is not None

    def test_whitespace_normalized(self):
        cache_store("what  is   python?", "Python is a high-level programming language.")
        result = cache_lookup("what is python?")
        assert result is not None

    def test_expired_not_returned(self):
        cache_store("old query", "old response")
        # Manually set created_at to 2 hours ago
        import agent.llm_optimizer as mod
        conn = mod._get_cache_conn()
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        conn.execute("UPDATE response_cache SET created_at = ?", (old_time,))
        conn.commit()

        result = cache_lookup("old query", max_age_hours=1)
        assert result is None

    def test_miss_returns_none(self):
        assert cache_lookup("nonexistent query") is None

    def test_hit_count_incremented(self):
        cache_store("test query", "This is a sufficiently long test response for caching purposes.")
        cache_lookup("test query")
        cache_lookup("test query")
        stats = get_cache_stats()
        assert stats["total_hits"] == 2

    def test_empty_response_not_cached(self):
        cache_store("q", "short")  # < 20 chars
        assert cache_lookup("q") is None

    def test_cache_stats(self):
        cache_store("query one", "response one is long enough to cache")
        cache_store("query two", "response two is long enough to cache")
        stats = get_cache_stats()
        assert stats["cached_queries"] == 2


class TestQueryComplexity:
    def test_simple_factual(self):
        result = score_query_complexity("What is spaghetti?")
        assert result["complexity"] == "simple"
        assert result["score"] < 0.3

    def test_complex_analytical(self):
        result = score_query_complexity(
            "Explain the differences between microservices and monolithic "
            "architecture and evaluate which one is better for a startup "
            "with limited resources and a need for rapid iteration"
        )
        assert result["complexity"] == "complex"
        assert result["score"] >= 0.6

    def test_moderate_query(self):
        result = score_query_complexity("How to implement a decorator in Python?")
        assert result["complexity"] in ("moderate", "complex")

    def test_very_short_query(self):
        result = score_query_complexity("hi")
        assert result["complexity"] == "simple"

    def test_code_query_gets_higher_score(self):
        result = score_query_complexity("Debug this function that throws an error")
        assert result["score"] > 0.2
        assert "code" in result["reasoning"]


class TestContextDecay:
    def test_fresh_context(self):
        assert compute_context_decay(0) == 1.0

    def test_one_half_life(self):
        decay = compute_context_decay(60, half_life_minutes=60)
        assert abs(decay - 0.5) < 0.01

    def test_two_half_lives(self):
        decay = compute_context_decay(120, half_life_minutes=60)
        assert abs(decay - 0.25) < 0.01

    def test_very_old_context(self):
        decay = compute_context_decay(24 * 60, half_life_minutes=60)
        assert decay < 0.001

    def test_rank_items(self):
        now = datetime.now()
        items = [
            {"text": "old", "timestamp": (now - timedelta(hours=3)).isoformat()},
            {"text": "recent", "timestamp": (now - timedelta(minutes=10)).isoformat()},
            {"text": "medium", "timestamp": (now - timedelta(hours=1)).isoformat()},
        ]
        ranked = rank_context_by_relevance(items, min_relevance=0.01)
        assert ranked[0]["text"] == "recent"
        assert ranked[-1]["text"] == "old"

    def test_filters_low_relevance(self):
        now = datetime.now()
        items = [
            {"text": "ancient", "timestamp": (now - timedelta(days=7)).isoformat()},
        ]
        ranked = rank_context_by_relevance(items, min_relevance=0.1)
        assert len(ranked) == 0  # too old


class TestUsageDashboard:
    def test_returns_string(self):
        dashboard = get_usage_dashboard(hours=1)
        assert isinstance(dashboard, str)
        assert "LLM Usage Analytics" in dashboard


class TestGetTools:
    def test_returns_tools(self):
        tools = get_llm_optimizer_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "llm_usage_dashboard" in names
        assert "query_complexity" in names


class TestModelRouting:
    """TK-390: strategic model selection for Ollama calls."""

    def test_simple_uses_fast_model_when_configured(self):
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", "qwen2.5:3b"):
            assert get_model_for_complexity("simple") == "qwen2.5:3b"

    def test_simple_falls_back_when_fast_unset(self):
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", ""):
            assert get_model_for_complexity("simple") == "qwen3.5:27b"

    def test_simple_falls_back_when_fast_whitespace(self):
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", "   "):
            assert get_model_for_complexity("simple") == "qwen3.5:27b"

    def test_moderate_always_uses_default(self):
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", "qwen2.5:3b"):
            assert get_model_for_complexity("moderate") == "qwen3.5:27b"

    def test_complex_always_uses_default(self):
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", "qwen2.5:3b"):
            assert get_model_for_complexity("complex") == "qwen3.5:27b"

    def test_unknown_complexity_uses_default(self):
        """Unknown values don't route to the fast model — safer default."""
        from agent.config import settings
        with patch.object(settings, "ollama_model", "qwen3.5:27b"), \
             patch.object(settings, "ollama_fast_model", "qwen2.5:3b"):
            assert get_model_for_complexity("garbage-tier") == "qwen3.5:27b"
