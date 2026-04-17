"""Tests for agent.discord_delivery — splitter, retry policy, spool, replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import discord_delivery
from agent.discord_delivery import (
    DEFAULT_SOFT_LIMIT,
    DISCORD_HARD_LIMIT,
    DiscordDelivery,
    SpooledMessage,
    replay_spool,
    spool_append,
    spool_load,
    split_for_discord,
)


# ---------------------------------------------------------------------------
# Fake exception type — emulates discord.HTTPException without pulling it in.
# ---------------------------------------------------------------------------


class FakeHTTPException(Exception):
    """Matches the duck-typed interface discord_delivery expects."""

    def __init__(
        self,
        status: int,
        message: str = "",
        *,
        retry_after: float | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message or f"HTTP {status}")
        self.status = status
        if retry_after is not None:
            self.retry_after = retry_after
        if response_headers is not None:
            resp = MagicMock()
            resp.headers = response_headers
            resp.status = status
            self.response = resp


# =============================================================================
# Splitter tests
# =============================================================================


class TestSplitForDiscord:
    def test_empty_returns_empty_list(self):
        assert split_for_discord("") == []

    def test_short_returns_single_chunk(self):
        assert split_for_discord("hello") == ["hello"]

    def test_respects_soft_limit(self):
        text = "a" * 3000
        chunks = split_for_discord(text, soft_limit=1000)
        assert all(len(c) <= DISCORD_HARD_LIMIT for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_paragraph_boundary(self):
        para1 = "one " * 200  # ~800 chars
        para2 = "two " * 200
        text = para1 + "\n\n" + para2
        chunks = split_for_discord(text, soft_limit=900)
        assert len(chunks) == 2
        assert chunks[0].strip().startswith("one")
        assert chunks[1].strip().startswith("two")

    def test_keeps_fenced_block_intact(self):
        text = (
            "Here is some code:\n\n"
            "```python\n"
            "def foo():\n    return 1\n"
            "```\n\n"
            "And more prose after it."
        )
        chunks = split_for_discord(text, soft_limit=DEFAULT_SOFT_LIMIT)
        assert len(chunks) == 1
        assert "```python" in chunks[0]
        assert chunks[0].count("```") == 2  # opening + closing fence

    def test_rewraps_oversized_fence_preserving_lang(self):
        body = "\n".join([f"line{i}" for i in range(500)])
        text = f"```python\n{body}\n```"
        chunks = split_for_discord(text, soft_limit=1000)
        assert len(chunks) > 1
        # Every chunk must still be a valid fenced block with the language tag.
        for c in chunks:
            assert c.startswith("```python\n")
            assert c.endswith("```")
            assert len(c) <= DISCORD_HARD_LIMIT

    def test_unclosed_fence_treated_as_text(self):
        # Without a closing fence, we must not swallow everything into one block.
        text = "```\n" + ("x" * 3000)
        chunks = split_for_discord(text, soft_limit=1000)
        assert len(chunks) > 1
        assert all(len(c) <= DISCORD_HARD_LIMIT for c in chunks)

    def test_hard_split_for_pathological_single_line(self):
        # One giant line with no natural boundaries at all.
        text = "x" * 5000
        chunks = split_for_discord(text, soft_limit=1000)
        assert all(len(c) <= DISCORD_HARD_LIMIT for c in chunks)
        assert "".join(chunks) == text

    def test_roundtrip_preserves_fence_body(self):
        body = "\n".join(f"row{i} = {i}" for i in range(300))
        text = f"intro\n\n```python\n{body}\n```\n\noutro"
        chunks = split_for_discord(text, soft_limit=1000)
        # Stitched-together code body should still contain all original lines.
        combined = "\n".join(chunks)
        assert "row0 = 0" in combined
        assert "row299 = 299" in combined

    def test_preserves_line_boundaries_when_no_paragraphs(self):
        text = "\n".join(f"line {i}" for i in range(400))
        chunks = split_for_discord(text, soft_limit=500)
        # No chunk should begin or end with a partial line fragment.
        for c in chunks:
            assert not c.startswith(" line")  # shouldn't split mid-line
            assert len(c) <= DISCORD_HARD_LIMIT


# =============================================================================
# Spool tests
# =============================================================================


class TestSpool:
    def test_append_and_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "spool.jsonl"
        msg1 = SpooledMessage(channel_id=42, content="hello")
        msg2 = SpooledMessage(channel_id=99, content="world", reply_to=7)
        spool_append(msg1, path)
        spool_append(msg2, path)
        loaded = spool_load(path)
        assert len(loaded) == 2
        assert loaded[0].channel_id == 42
        assert loaded[0].content == "hello"
        assert loaded[1].reply_to == 7
        # Load should clear the spool for exactly-once replay.
        assert not path.exists()

    def test_load_missing_returns_empty(self, tmp_path: Path):
        assert spool_load(tmp_path / "nope.jsonl") == []

    def test_corrupt_file_is_quarantined(self, tmp_path: Path):
        path = tmp_path / "spool.jsonl"
        path.write_text("not json\nalso not json\n", encoding="utf-8")
        loaded = spool_load(path)
        assert loaded == []
        # Original is gone, corrupt copy exists.
        assert not path.exists()
        assert (tmp_path / "spool.jsonl.corrupt").exists()

    def test_partial_corruption_drops_bad_lines(self, tmp_path: Path):
        path = tmp_path / "spool.jsonl"
        good = SpooledMessage(channel_id=1, content="ok").to_json()
        path.write_text(good + "\n{garbage\n", encoding="utf-8")
        loaded = spool_load(path)
        assert len(loaded) == 1
        assert loaded[0].content == "ok"

    def test_append_ioerror_does_not_raise(self, tmp_path: Path, caplog):
        # Point the spool at a path where mkdir will fail.
        bad_path = tmp_path / "file_as_dir" / "spool.jsonl"
        (tmp_path / "file_as_dir").write_text("I am a file", encoding="utf-8")
        msg = SpooledMessage(channel_id=1, content="x")
        # Must not raise — caller shouldn't crash because spool failed.
        spool_append(msg, bad_path)


# =============================================================================
# Delivery engine tests
# =============================================================================


def _make_send_func(failures: list[Any] | None = None, call_log: list[tuple] | None = None):
    """Build a fake send function that raises *failures* in order, then succeeds.

    *call_log* receives ``(channel_id, content, reply_to)`` tuples for every
    invocation — both failed and successful — so tests can assert the retry
    loop actually retried and didn't silently skip.
    """
    failures = list(failures or [])
    log = call_log if call_log is not None else []

    async def send(channel_id: int, content: str, reply_to: int | None) -> None:
        log.append((channel_id, content, reply_to))
        if failures:
            raise failures.pop(0)

    return send


@pytest.fixture(autouse=True)
def _reset_outbound_limiter():
    """The outbound rate limiter is a process-level singleton; reset it so
    one test's sends don't throttle another's."""
    with patch.object(discord_delivery, "get_outbound_limiter") as factory:
        limiter = MagicMock()

        async def acquire():
            return None

        limiter.acquire = acquire
        factory.return_value = limiter
        yield


@pytest.mark.asyncio
async def test_send_delivers_short_message():
    log: list[tuple] = []
    send = _make_send_func(call_log=log)
    delivery = DiscordDelivery(send)
    n = await delivery.send(123, "hello world")
    await delivery.shutdown()
    assert n == 1
    assert log == [(123, "hello world", None)]


@pytest.mark.asyncio
async def test_send_splits_long_message(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(call_log=log)
    delivery = DiscordDelivery(send, soft_limit=100, spool_path=tmp_path / "spool.jsonl")
    long = "\n\n".join(["para " * 30 for _ in range(5)])
    n = await delivery.send(1, long)
    await delivery.shutdown()
    assert n >= 2
    assert len(log) == n
    # All chunks addressed the same channel, in order.
    assert all(entry[0] == 1 for entry in log)


@pytest.mark.asyncio
async def test_empty_content_is_noop():
    log: list[tuple] = []
    delivery = DiscordDelivery(_make_send_func(call_log=log))
    assert await delivery.send(1, "") == 0
    await delivery.shutdown()
    assert log == []


@pytest.mark.asyncio
async def test_retry_on_429_then_succeeds(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(429, retry_after=0.01)],
        call_log=log,
    )
    delivery = DiscordDelivery(send, spool_path=tmp_path / "spool.jsonl")
    await delivery.send(1, "hi")
    await delivery.shutdown()
    # Two calls: one failure, one success.
    assert len(log) == 2


@pytest.mark.asyncio
async def test_retry_on_5xx_then_succeeds(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(503)],
        call_log=log,
    )
    delivery = DiscordDelivery(send, spool_path=tmp_path / "spool.jsonl")
    # Patch sleep so the backoff doesn't slow the test.
    with patch("agent.discord_delivery.asyncio.sleep", new=_noop_sleep):
        await delivery.send(1, "hi")
        await delivery.shutdown()
    assert len(log) == 2


@pytest.mark.asyncio
async def test_403_is_dropped_permanently(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(403, "channel deleted")],
        call_log=log,
    )
    spool = tmp_path / "spool.jsonl"
    delivery = DiscordDelivery(send, spool_path=spool)
    await delivery.send(1, "hi")
    await delivery.shutdown()
    assert len(log) == 1  # no retry
    assert not spool.exists()  # not spooled either


@pytest.mark.asyncio
async def test_401_triggers_alert_and_drop(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(401, "token revoked")],
        call_log=log,
    )
    spool = tmp_path / "spool.jsonl"
    delivery = DiscordDelivery(send, spool_path=spool)
    with patch("agent.discord_delivery.alerts.send_alert") as alert:
        await delivery.send(1, "hi")
        await delivery.shutdown()
    alert.assert_called_once()
    assert alert.call_args.kwargs.get("level") == "critical"
    assert not spool.exists()


@pytest.mark.asyncio
async def test_401_alert_fires_only_once(tmp_path: Path):
    """Token revocation spams 401s — but we only alert the operator once per
    delivery instance so the alerts channel doesn't drown."""
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(401), FakeHTTPException(401)],
        call_log=log,
    )
    delivery = DiscordDelivery(send, spool_path=tmp_path / "spool.jsonl")
    with patch("agent.discord_delivery.alerts.send_alert") as alert:
        await delivery.send(1, "one")
        await delivery.send(1, "two")
        await delivery.shutdown()
    assert alert.call_count == 1


@pytest.mark.asyncio
async def test_exhausted_retries_get_spooled(tmp_path: Path):
    log: list[tuple] = []
    # Always fail — engine should retry MAX_ATTEMPTS times then spool.
    send = _make_send_func(
        failures=[FakeHTTPException(500) for _ in range(10)],
        call_log=log,
    )
    spool = tmp_path / "spool.jsonl"
    delivery = DiscordDelivery(send, max_attempts=3, spool_path=spool)
    with patch("agent.discord_delivery.asyncio.sleep", new=_noop_sleep):
        await delivery.send(42, "retry me", reply_to=7)
        await delivery.shutdown()
    assert len(log) == 3
    # Message was written to disk for replay.
    assert spool.exists()
    spooled = [json.loads(line) for line in spool.read_text().splitlines() if line]
    assert len(spooled) == 1
    assert spooled[0]["channel_id"] == 42
    assert spooled[0]["content"] == "retry me"
    assert spooled[0]["reply_to"] == 7


@pytest.mark.asyncio
async def test_non_retryable_status_is_dropped(tmp_path: Path):
    """A 400 (bad request) means the payload itself is invalid — retrying
    won't help, so we drop without spooling."""
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(400, "bad payload")],
        call_log=log,
    )
    spool = tmp_path / "spool.jsonl"
    delivery = DiscordDelivery(send, spool_path=spool)
    await delivery.send(1, "hi")
    await delivery.shutdown()
    assert len(log) == 1
    assert not spool.exists()


@pytest.mark.asyncio
async def test_ordering_preserved_per_channel(tmp_path: Path):
    log: list[tuple] = []
    send = _make_send_func(call_log=log)
    delivery = DiscordDelivery(send, soft_limit=50, spool_path=tmp_path / "s.jsonl")
    # Three separate submissions to the same channel — order must be preserved
    # even though each splits into multiple chunks.
    await delivery.send(1, "first " * 20)
    await delivery.send(1, "second " * 20)
    await delivery.send(1, "third " * 20)
    await delivery.shutdown()
    firsts = [i for i, (_, c, _) in enumerate(log) if "first" in c]
    seconds = [i for i, (_, c, _) in enumerate(log) if "second" in c]
    thirds = [i for i, (_, c, _) in enumerate(log) if "third" in c]
    assert max(firsts) < min(seconds)
    assert max(seconds) < min(thirds)


@pytest.mark.asyncio
async def test_send_does_not_block_on_slow_network(tmp_path: Path):
    """The whole point: send() must return immediately even if the underlying
    network send is hanging."""

    async def slow_send(cid, content, reply):
        await asyncio.sleep(10)  # would be a test timeout if we awaited it

    delivery = DiscordDelivery(slow_send, spool_path=tmp_path / "s.jsonl")
    t0 = asyncio.get_event_loop().time()
    await delivery.send(1, "hi")
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed < 0.5  # well under the 10s sleep
    # Cancel the in-flight worker so shutdown doesn't wait forever.
    for task in delivery._workers.values():
        task.cancel()
    await asyncio.gather(*delivery._workers.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_honors_retry_after_header(tmp_path: Path):
    """When the response carries Retry-After, we should use it verbatim
    (plus jitter) rather than our own backoff schedule."""
    log: list[tuple] = []
    send = _make_send_func(
        failures=[FakeHTTPException(429, retry_after=1.5)],
        call_log=log,
    )
    delivery = DiscordDelivery(send, spool_path=tmp_path / "s.jsonl")
    sleeps: list[float] = []

    async def capture_sleep(t: float):
        sleeps.append(t)

    with patch("agent.discord_delivery.asyncio.sleep", new=capture_sleep):
        await delivery.send(1, "hi")
        await delivery.shutdown()
    # At least one recorded sleep should be >= the Retry-After value.
    assert any(s >= 1.5 for s in sleeps), sleeps


# =============================================================================
# Replay tests
# =============================================================================


@pytest.mark.asyncio
async def test_replay_spool_resubmits_messages(tmp_path: Path):
    spool = tmp_path / "spool.jsonl"
    spool_append(SpooledMessage(channel_id=1, content="one"), spool)
    spool_append(SpooledMessage(channel_id=2, content="two"), spool)

    log: list[tuple] = []
    delivery = DiscordDelivery(
        _make_send_func(call_log=log), spool_path=spool
    )
    n = await replay_spool(delivery, spool)
    await delivery.shutdown()
    assert n == 2
    assert {entry[0] for entry in log} == {1, 2}
    # Spool cleared after replay.
    assert not spool.exists()


@pytest.mark.asyncio
async def test_replay_empty_spool_is_noop(tmp_path: Path):
    delivery = DiscordDelivery(_make_send_func(), spool_path=tmp_path / "s.jsonl")
    n = await replay_spool(delivery, tmp_path / "s.jsonl")
    await delivery.shutdown()
    assert n == 0


@pytest.mark.asyncio
async def test_replay_quarantines_corrupt_spool(tmp_path: Path):
    spool = tmp_path / "spool.jsonl"
    spool.write_text("garbage only\n", encoding="utf-8")
    delivery = DiscordDelivery(_make_send_func(), spool_path=spool)
    n = await replay_spool(delivery, spool)
    await delivery.shutdown()
    assert n == 0
    assert (tmp_path / "spool.jsonl.corrupt").exists()


# =============================================================================
# Helpers
# =============================================================================


async def _noop_sleep(_: float) -> None:
    """Replacement for asyncio.sleep that returns immediately."""
    return None
