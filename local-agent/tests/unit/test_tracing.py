"""Tests for agent.tracing — ULID generation, ContextVar binding, validation."""

from __future__ import annotations

import asyncio
import re
import time
from contextvars import Token

import pytest

import agent.tracing as tracing
from agent.tracing import (
    DEFAULT_TRACE_ID,
    get_trace_id,
    is_valid_trace_id,
    new_trace_id,
    set_trace_id,
    with_trace_id,
)


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(rf"^[{_CROCKFORD}]{{26}}$")


# ---------------------------------------------------------------------------
# new_trace_id — length, alphabet, uniqueness, sortability
# ---------------------------------------------------------------------------

class TestNewTraceId:
    def test_length_is_26(self):
        assert len(new_trace_id()) == 26

    def test_alphabet_is_crockford_base32(self):
        for _ in range(50):
            tid = new_trace_id()
            assert _ULID_RE.match(tid), f"{tid!r} is not valid Crockford base32"

    def test_excludes_ambiguous_letters(self):
        # Crockford base32 omits I, L, O, U (and always uppercase).
        forbidden = set("ILOUilou")
        for _ in range(200):
            tid = new_trace_id()
            assert not (set(tid) & forbidden), (
                f"{tid!r} contains forbidden letter"
            )

    def test_values_are_unique(self):
        # 80 bits of entropy — collisions should be astronomically unlikely.
        ids = {new_trace_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_timestamp_prefix_is_monotonic(self):
        a = new_trace_id()
        time.sleep(0.005)
        b = new_trace_id()
        # Lexicographic order on the timestamp prefix tracks wall-clock time.
        assert a[:10] <= b[:10]
        # Different millisecond → different prefix.
        assert a[:10] != b[:10]

    def test_returns_str(self):
        assert isinstance(new_trace_id(), str)


# ---------------------------------------------------------------------------
# is_valid_trace_id — validator
# ---------------------------------------------------------------------------

class TestIsValidTraceId:
    def test_accepts_generated_ids(self):
        for _ in range(20):
            assert is_valid_trace_id(new_trace_id())

    def test_rejects_wrong_length(self):
        assert not is_valid_trace_id("")
        assert not is_valid_trace_id("A" * 25)
        assert not is_valid_trace_id("A" * 27)
        assert not is_valid_trace_id("A")

    def test_rejects_non_string(self):
        assert not is_valid_trace_id(None)
        assert not is_valid_trace_id(123)
        assert not is_valid_trace_id(b"A" * 26)
        assert not is_valid_trace_id(["A"] * 26)
        assert not is_valid_trace_id(object())

    def test_rejects_forbidden_letters(self):
        # Each of I, L, O, U must cause rejection when it appears anywhere.
        for bad in "ILOU":
            sample = bad + ("A" * 25)
            assert not is_valid_trace_id(sample), f"{sample!r} was accepted"

    def test_rejects_lowercase(self):
        tid = new_trace_id().lower()
        assert not is_valid_trace_id(tid)

    def test_rejects_whitespace_and_punctuation(self):
        assert not is_valid_trace_id(" " * 26)
        assert not is_valid_trace_id("-" * 26)
        tid = new_trace_id()
        assert not is_valid_trace_id(tid[:-1] + " ")

    def test_rejects_sentinel(self):
        assert not is_valid_trace_id(DEFAULT_TRACE_ID)


# ---------------------------------------------------------------------------
# set_trace_id / get_trace_id — explicit ContextVar binding
# ---------------------------------------------------------------------------

class TestSetGetTraceId:
    def test_default_when_unset(self):
        assert get_trace_id() == DEFAULT_TRACE_ID

    def test_set_updates_current(self):
        tid = new_trace_id()
        token = set_trace_id(tid)
        try:
            assert get_trace_id() == tid
        finally:
            tracing._trace_id_var.reset(token)
        assert get_trace_id() == DEFAULT_TRACE_ID

    def test_set_returns_token(self):
        token = set_trace_id("anything")
        try:
            assert isinstance(token, Token)
        finally:
            tracing._trace_id_var.reset(token)

    def test_set_none_uses_default(self):
        token = set_trace_id(None)
        try:
            assert get_trace_id() == DEFAULT_TRACE_ID
        finally:
            tracing._trace_id_var.reset(token)

    def test_set_empty_string_uses_default(self):
        token = set_trace_id("")
        try:
            assert get_trace_id() == DEFAULT_TRACE_ID
        finally:
            tracing._trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# with_trace_id — context manager with auto-reset
# ---------------------------------------------------------------------------

class TestWithTraceId:
    def test_binds_for_block_and_resets_on_exit(self):
        assert get_trace_id() == DEFAULT_TRACE_ID
        with with_trace_id("fixed-trace"):
            assert get_trace_id() == "fixed-trace"
        assert get_trace_id() == DEFAULT_TRACE_ID

    def test_yields_bound_value(self):
        with with_trace_id("abc123") as tid:
            assert tid == "abc123"
            assert get_trace_id() == tid

    def test_auto_generates_when_value_is_none(self):
        with with_trace_id() as tid:
            assert is_valid_trace_id(tid)
            assert get_trace_id() == tid
        assert get_trace_id() == DEFAULT_TRACE_ID

    def test_auto_generates_when_value_is_empty(self):
        with with_trace_id("") as tid:
            assert is_valid_trace_id(tid)

    def test_nested_stacks_restore_outer(self):
        with with_trace_id("outer"):
            assert get_trace_id() == "outer"
            with with_trace_id("inner"):
                assert get_trace_id() == "inner"
            assert get_trace_id() == "outer"
        assert get_trace_id() == DEFAULT_TRACE_ID

    def test_resets_even_when_body_raises(self):
        assert get_trace_id() == DEFAULT_TRACE_ID
        with pytest.raises(RuntimeError):
            with with_trace_id("boom"):
                assert get_trace_id() == "boom"
                raise RuntimeError("oops")
        assert get_trace_id() == DEFAULT_TRACE_ID


# ---------------------------------------------------------------------------
# asyncio — ContextVar isolation across concurrent tasks
# ---------------------------------------------------------------------------

class TestAsyncioIsolation:
    def test_each_task_has_its_own_trace_id(self):
        """Concurrent tasks must not see one another's trace_id."""

        async def bind_and_yield(name: str, seen: dict) -> None:
            with with_trace_id(name):
                # Yield control several times so tasks interleave.
                for _ in range(3):
                    await asyncio.sleep(0)
                seen[name] = get_trace_id()

        async def main() -> dict:
            seen: dict = {}
            await asyncio.gather(
                bind_and_yield("trace-A", seen),
                bind_and_yield("trace-B", seen),
                bind_and_yield("trace-C", seen),
            )
            return seen

        result = asyncio.run(main())
        assert result == {
            "trace-A": "trace-A",
            "trace-B": "trace-B",
            "trace-C": "trace-C",
        }

    def test_child_task_does_not_leak_into_parent(self):
        """A trace_id set inside a spawned Task must not overwrite the parent's."""

        async def child() -> None:
            with with_trace_id("child-trace"):
                await asyncio.sleep(0)

        async def main() -> str:
            with with_trace_id("parent-trace"):
                task = asyncio.create_task(child())
                await task
                return get_trace_id()

        assert asyncio.run(main()) == "parent-trace"

    def test_parent_trace_id_propagates_into_new_task(self):
        """A task spawned inside a bound scope inherits the parent's trace_id."""

        async def child() -> str:
            # No with_trace_id here — should inherit whatever the parent bound.
            await asyncio.sleep(0)
            return get_trace_id()

        async def main() -> str:
            with with_trace_id("parent-only"):
                return await asyncio.create_task(child())

        assert asyncio.run(main()) == "parent-only"


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_default_sentinel_is_dash(self):
        assert DEFAULT_TRACE_ID == "-"

    def test_contextvar_name_is_trace_id(self):
        assert tracing._trace_id_var.name == "trace_id"

    def test_contextvar_default_matches_sentinel(self):
        # A fresh read with no binding returns the module sentinel.
        assert tracing._trace_id_var.get() == DEFAULT_TRACE_ID
