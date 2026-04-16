"""Tests for aim.evergreen — idle-cycle evergreen work generator."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aim.evergreen import (
    EVERGREEN_TOPICS,
    _build_perf_prompt,
    _build_topic_prompt,
    _parse_story,
    generate_evergreen_story,
)


# ---------------------------------------------------------------------------
# _parse_story
# ---------------------------------------------------------------------------


class TestParseStory:
    def test_valid_story(self):
        raw = json.dumps({"title": "Add metrics", "description": "Do the thing"})
        assert _parse_story(raw) == {"title": "Add metrics", "description": "Do the thing"}

    def test_wrapped_in_prose(self):
        raw = (
            'Here is the story: {"title": "X", "description": "Y"} '
            "Hope it helps."
        )
        assert _parse_story(raw) == {"title": "X", "description": "Y"}

    def test_handles_nested_braces_in_description(self):
        raw = (
            '{"title": "T", "description": "use {json: 1} shape everywhere"}'
        )
        out = _parse_story(raw)
        assert out is not None
        assert "{json: 1}" in out["description"]

    def test_missing_title_returns_none(self):
        raw = '{"description": "no title"}'
        assert _parse_story(raw) is None

    def test_missing_description_returns_none(self):
        raw = '{"title": "no desc"}'
        assert _parse_story(raw) is None

    def test_empty_returns_none(self):
        assert _parse_story("") is None

    def test_no_json_returns_none(self):
        assert _parse_story("just text, nothing to parse") is None

    def test_truncates_long_title(self):
        long = "x" * 400
        raw = json.dumps({"title": long, "description": "ok"})
        out = _parse_story(raw)
        assert out is not None
        assert len(out["title"]) == 200


# ---------------------------------------------------------------------------
# prompt builders
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def test_perf_prompt_lists_hotspots(self):
        hotspots = [
            {"name": "agent.core.foo", "call_count": 1000,
             "p95_seconds": 1.5, "total_seconds": 120.0},
            {"name": "idea_board.executor.bar", "call_count": 200,
             "p95_seconds": 0.8, "total_seconds": 40.0},
        ]
        prompt = _build_perf_prompt(hotspots)
        assert "agent.core.foo" in prompt
        assert "p95=1.50s" in prompt
        assert "ONE file, ONE function" in prompt
        assert "JSON" in prompt

    def test_topic_prompt_lists_modules(self):
        modules = ["agent/foo.py", "aim/bar.py", "board/baz.py"]
        prompt = _build_topic_prompt("test_coverage", "add tests", modules)
        assert "test_coverage" in prompt
        assert "add tests" in prompt
        for m in modules:
            assert f"- {m}" in prompt
        assert "ONE module" in prompt


# ---------------------------------------------------------------------------
# EVERGREEN_TOPICS structure
# ---------------------------------------------------------------------------


class TestTopicsStructure:
    def test_all_topics_use_auto_approve_categories(self):
        # Categories that auto-approve per settings.aim_auto_approve_categories
        auto_approve = {"quality", "performance", "test"}
        for topic, category, focus in EVERGREEN_TOPICS:
            assert category in auto_approve, (
                f"topic {topic!r} uses category {category!r} which "
                f"is NOT in auto-approve set — stories would get stuck "
                f"behind pending-approval."
            )

    def test_no_empty_topics(self):
        for topic, category, focus in EVERGREEN_TOPICS:
            assert topic
            assert category
            assert len(focus) > 40, f"focus too short for {topic}"


# ---------------------------------------------------------------------------
# generate_evergreen_story — full flow with injected runners
# ---------------------------------------------------------------------------


class TestGenerateEvergreenStory:
    def _fake_story_json(self, title="Foo", desc="Bar"):
        return json.dumps({"title": title, "description": desc})

    def test_perf_path_when_hotspots_available(self):
        with patch("aim.evergreen.random.random", return_value=0.1):
            story = generate_evergreen_story(
                claude_runner=lambda _p, timeout: self._fake_story_json(
                    title="Speed up foo", desc="Cache result"
                ),
                perf_reader=lambda: [
                    {"name": "mod.foo", "call_count": 100,
                     "p95_seconds": 2.0, "total_seconds": 200.0}
                ],
                module_sampler=lambda: ["agent/foo.py"],
            )
        assert story is not None
        assert story["category"] == "performance"
        assert story["topic"] == "performance"
        assert story["title"] == "Speed up foo"

    def test_topic_path_when_no_hotspots(self):
        story = generate_evergreen_story(
            claude_runner=lambda _p, timeout: self._fake_story_json(),
            perf_reader=lambda: [],
            module_sampler=lambda: ["agent/foo.py", "aim/bar.py"],
        )
        assert story is not None
        assert story["category"] in {"quality", "test", "performance"}
        assert story["topic"] in {t[0] for t in EVERGREEN_TOPICS}

    def test_topic_path_when_random_above_threshold(self):
        # Even with hotspots, if random > 0.5, we take the topic path.
        with patch("aim.evergreen.random.random", return_value=0.99):
            story = generate_evergreen_story(
                claude_runner=lambda _p, timeout: self._fake_story_json(),
                perf_reader=lambda: [{
                    "name": "x", "call_count": 10,
                    "p95_seconds": 1.0, "total_seconds": 10.0,
                }],
                module_sampler=lambda: ["agent/foo.py"],
            )
        assert story is not None
        assert story["topic"] != "performance"

    def test_returns_none_when_llm_returns_none(self):
        story = generate_evergreen_story(
            claude_runner=lambda _p, timeout: None,
            perf_reader=lambda: [],
            module_sampler=lambda: ["agent/foo.py"],
        )
        assert story is None

    def test_returns_none_when_llm_returns_garbage(self):
        story = generate_evergreen_story(
            claude_runner=lambda _p, timeout: "not json, just prose",
            perf_reader=lambda: [],
            module_sampler=lambda: ["agent/foo.py"],
        )
        assert story is None

    def test_returns_none_when_no_modules(self):
        story = generate_evergreen_story(
            claude_runner=lambda _p, timeout: self._fake_story_json(),
            perf_reader=lambda: [],
            module_sampler=lambda: [],
        )
        assert story is None
