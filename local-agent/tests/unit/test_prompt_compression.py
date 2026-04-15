"""Tests for agent/prompt_compression.py — message size estimation and compression."""

from agent.prompt_compression import (
    DEFAULT_COMPRESSION_THRESHOLD_CHARS,
    MIN_COMPRESS_MSG_CHARS,
    compress_messages,
    estimate_size_chars,
    is_cacheable_task,
)


class TestEstimateSizeChars:
    def test_empty_list(self):
        assert estimate_size_chars([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert estimate_size_chars(msgs) == 5

    def test_sums_across_messages(self):
        msgs = [
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "defg"},
            {"role": "tool", "content": "hi"},
        ]
        assert estimate_size_chars(msgs) == 3 + 4 + 2

    def test_ignores_non_string_content(self):
        # Vision payloads use list content; the estimator should not crash.
        msgs = [
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "user", "content": "text"},
        ]
        assert estimate_size_chars(msgs) == 4

    def test_missing_content_key(self):
        msgs = [{"role": "tool"}]
        assert estimate_size_chars(msgs) == 0


class TestCompressMessages:
    def _big(self, size: int = MIN_COMPRESS_MSG_CHARS * 2) -> str:
        return "x" * size

    def test_under_threshold_returns_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        out, saved = compress_messages(msgs, max_chars=10_000)
        assert out is msgs
        assert saved == 0

    def test_empty_list(self):
        out, saved = compress_messages([], max_chars=10)
        assert out == []
        assert saved == 0

    def test_system_prompt_preserved(self):
        system = {"role": "system", "content": "SYSTEM_PROMPT"}
        msgs = [system] + [
            {"role": "tool", "content": self._big()} for _ in range(10)
        ] + [{"role": "user", "content": "final"}]
        out, saved = compress_messages(msgs, max_chars=1000, keep_recent=1)
        assert saved > 0
        # System still present verbatim.
        assert any(m.get("content") == "SYSTEM_PROMPT" for m in out if m.get("role") == "system")

    def test_recent_messages_preserved_verbatim(self):
        msgs = [{"role": "system", "content": "sys"}]
        msgs.extend({"role": "tool", "content": self._big()} for _ in range(8))
        tail = [
            {"role": "assistant", "content": "recent-a"},
            {"role": "user", "content": "recent-u"},
            {"role": "tool", "content": "recent-t"},
        ]
        msgs.extend(tail)

        out, saved = compress_messages(msgs, max_chars=1000, keep_recent=3)
        assert saved > 0
        # Last three non-system messages should be exactly the tail.
        non_system = [m for m in out if m.get("role") != "system"]
        assert non_system[-3:] == tail

    def test_user_messages_not_compressed(self):
        big_user_content = self._big(size=5_000)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big_user_content},
            {"role": "tool", "content": self._big()},
            {"role": "tool", "content": self._big()},
            {"role": "assistant", "content": "done"},
        ]
        out, saved = compress_messages(msgs, max_chars=500, keep_recent=1)
        assert saved > 0
        # The big user message must still be present verbatim.
        assert any(m.get("content") == big_user_content for m in out if m.get("role") == "user")

    def test_tool_messages_collapsed(self):
        big = self._big(size=4_000)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "tool", "content": big},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "done"},
        ]
        out, saved = compress_messages(msgs, max_chars=500, keep_recent=1)
        assert saved > 0
        tool_contents = [m["content"] for m in out if m.get("role") == "tool"]
        # At least one tool message got collapsed (shorter than original, with marker).
        assert any("compressed" in c and len(c) < len(big) for c in tool_contents)

    def test_small_messages_not_collapsed(self):
        # Each old message is below MIN_COMPRESS_MSG_CHARS — should pass through
        # unchanged even when the total crosses the threshold.
        small_content = "a" * (MIN_COMPRESS_MSG_CHARS - 1)
        msgs = [{"role": "system", "content": "sys"}]
        msgs.extend({"role": "tool", "content": small_content} for _ in range(300))
        msgs.append({"role": "user", "content": "recent"})

        out, saved = compress_messages(msgs, max_chars=1000, keep_recent=1)
        # Nothing is over the per-message threshold so nothing gets collapsed.
        assert saved == 0

    def test_keep_recent_zero(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": self._big()},
            {"role": "tool", "content": self._big()},
        ]
        out, saved = compress_messages(msgs, max_chars=500, keep_recent=0)
        assert saved > 0
        # Every tool in output should be collapsed (no carve-out).
        for m in out:
            if m.get("role") == "tool":
                assert "compressed" in m["content"]

    def test_original_messages_not_mutated(self):
        original_content = self._big()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": original_content},
            {"role": "user", "content": "recent"},
        ]
        out, _ = compress_messages(msgs, max_chars=500, keep_recent=1)
        # The original dict must not have been modified.
        assert msgs[1]["content"] == original_content
        assert out is not msgs

    def test_default_threshold_constant_is_reasonable(self):
        # Guard against accidentally lowering the threshold — it's tied to the
        # observed latency cliff around 50K tokens.
        assert DEFAULT_COMPRESSION_THRESHOLD_CHARS >= 100_000


class TestIsCacheableTask:
    def test_short_string_is_cacheable(self):
        assert is_cacheable_task("hi") is True

    def test_empty_is_not_cacheable(self):
        assert is_cacheable_task("") is False

    def test_whitespace_only_is_not_cacheable(self):
        assert is_cacheable_task("   \n  ") is False

    def test_long_string_is_not_cacheable(self):
        assert is_cacheable_task("x" * 5000) is False

    def test_custom_limit(self):
        assert is_cacheable_task("hello", max_chars=4) is False
        assert is_cacheable_task("hi", max_chars=4) is True

    def test_non_string_is_not_cacheable(self):
        assert is_cacheable_task(None) is False  # type: ignore[arg-type]
        assert is_cacheable_task(123) is False  # type: ignore[arg-type]
