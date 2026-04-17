"""
Agent Core - Ollama-based autonomous agent with tool calling.

This agent can:
- Use tools (file I/O, memory, web, etc.)
- Maintain a knowledge graph
- Feed information back to Claude when needed
- Run with safe_update.py workflow for tested deployments
"""

import contextvars
import json
import re
import threading
import time as _time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


def extract_thinking(text: str) -> str:
    """Extract content inside <think>...</think> tags, or empty string if none."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from response, keeping only the final answer."""
    # Remove thinking blocks (can be multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =============================================================================
# TOOL RESULT STORAGE - Truncate large results, store full on disk
# =============================================================================

# Per-tool truncation thresholds (bytes). Results larger than this get stored
# and replaced with a preview in context.
TOOL_RESULT_THRESHOLDS: dict[str, int] = {
    "web_fetch": 2000,
    "web_search": 3000,
    "web_search_smart": 3500,
    "web_search_news": 3000,
    "read_file": 4000,
    "run_command": 3000,
    "web_search_claude": 4000,
}
DEFAULT_THRESHOLD = 4000  # For tools not listed above

# Try to connect to Redis; fall back to in-memory dict if unavailable.
_redis_client = None
try:
    import redis as _redis_mod

    _r = _redis_mod.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=1)
    _r.ping()
    _redis_client = _r
    print("[ToolResultStorage] Using Redis for result caching")
except Exception:
    print("[ToolResultStorage] Redis unavailable — using in-memory cache (96GB RAM, no problem)")


class ToolResultStorage:
    """Stores large tool results in memory (or Redis), keeps previews in context.

    With 96GB RAM, we keep everything in-memory for zero-latency retrieval.
    If Redis is available, results are stored there (survives bot restarts,
    shared across processes, auto-expires via TTL). Otherwise, a plain dict
    is used — fast and effective for a single-process bot.
    """

    # Redis key prefix and TTL (results expire after 2 hours)
    REDIS_PREFIX = "technomancer:tool_result:"
    REDIS_TTL = 7200  # 2 hours

    def __init__(self) -> None:
        """Initialize storage backend, preferring Redis with in-memory fallback."""
        self.redis = _redis_client
        # In-memory fallback store
        self._mem_store: dict[str, str] = {}
        # Stats for monitoring
        self.stats = {
            "total_results": 0,
            "truncated_results": 0,
            "bytes_saved": 0,
        }
        self.backend = "redis" if self.redis else "memory"
        # Guards stats increments and the in-memory store when multiple
        # tool calls are executed concurrently.
        self._lock = threading.Lock()

    def _store(self, result_id: str, content: str) -> None:
        """Store a full result in Redis or memory."""
        if self.redis:
            try:
                self.redis.setex(f"{self.REDIS_PREFIX}{result_id}", self.REDIS_TTL, content)
                return
            except Exception:
                pass  # Fall through to memory
        with self._lock:
            self._mem_store[result_id] = content

    def _retrieve(self, result_id: str) -> str | None:
        """Retrieve a stored result from Redis or memory."""
        if self.redis:
            try:
                val = self.redis.get(f"{self.REDIS_PREFIX}{result_id}")
                if val is not None:
                    return val
            except Exception:
                pass  # Fall through to memory
        with self._lock:
            return self._mem_store.get(result_id)

    def maybe_truncate(self, tool_name: str, result: str) -> str:
        """Truncate a tool result if it exceeds the threshold.

        Returns the original result if small enough, or a preview with a
        storage reference if truncated.
        """
        threshold = TOOL_RESULT_THRESHOLDS.get(tool_name, DEFAULT_THRESHOLD)

        if len(result) <= threshold:
            with self._lock:
                self.stats["total_results"] += 1
            return result

        # Store full result
        result_id = uuid.uuid4().hex[:12]
        self._store(result_id, result)

        # Build preview: first ~80% of budget for head, ~20% for tail
        head_size = int(threshold * 0.8)
        tail_size = threshold - head_size
        preview = result[:head_size]
        tail = result[-tail_size:] if tail_size > 0 else ""

        truncated = (
            f"{preview}\n\n"
            f"[... TRUNCATED — {len(result):,} chars total, showing first {head_size:,} + last {tail_size:,} ...]\n"
            f"[Full result stored as ID: {result_id} — use get_stored_result tool to retrieve]\n\n"
            f"{tail}"
        )

        with self._lock:
            self.stats["total_results"] += 1
            self.stats["truncated_results"] += 1
            self.stats["bytes_saved"] += len(result) - len(truncated)

        return truncated

    def get_full_result(self, result_id: str) -> str:
        """Retrieve a full stored result by ID."""
        content = self._retrieve(result_id)
        if content is not None:
            return content
        return f"Error: No stored result found with ID '{result_id}'"

    def cleanup_session(self) -> None:
        """Clear all stored results for this session (Redis keys and in-memory)."""
        if self.redis:
            try:
                keys = self.redis.keys(f"{self.REDIS_PREFIX}*")
                if keys:
                    self.redis.delete(*keys)
            except Exception:
                pass
        with self._lock:
            self._mem_store.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return truncation stats for this session."""
        stored_count = len(self._mem_store)
        if self.redis:
            try:
                stored_count = len(self.redis.keys(f"{self.REDIS_PREFIX}*"))
            except Exception:
                pass
        return {
            **self.stats,
            "backend": self.backend,
            "stored_results": stored_count,
            "truncation_rate": (
                f"{self.stats['truncated_results'] / self.stats['total_results'] * 100:.1f}%"
                if self.stats["total_results"] > 0
                else "0%"
            ),
        }


from .perf_monitor import record_llm_call as _record_perf
from .prompt_compression import (
    DEFAULT_COMPRESSION_THRESHOLD_CHARS,
    compress_messages,
    estimate_size_chars,
    is_cacheable_task,
)

try:
    import ollama
except ImportError:
    raise ImportError("Install ollama: pip install ollama")

from .config import settings as _settings
from .logging_config import (
    DEFAULT_REQUEST_ID as _DEFAULT_REQUEST_ID,
    request_id_var as _request_id_var,
)
from .ollama_health import ollama_call_with_retries


def _seed_request_id_if_unset() -> None:
    """Generate and set a UUID4 request id when no caller has seeded one.

    Discord's ``on_message`` handler (and other entry points) seed a
    descriptive id like ``discord-{message.id}`` before invoking the agent
    so the whole exchange shares one id. Standalone callers (news_digest,
    background tasks, tests) don't, so Agent.run() falls back to a fresh
    UUID4 — guaranteeing every run() correlates its logs and downstream
    HTTP calls under a single id.
    """
    if _request_id_var.get() == _DEFAULT_REQUEST_ID:
        _request_id_var.set(uuid.uuid4().hex)


def _build_ollama_client() -> "ollama.Client":
    """Construct the Ollama client with a configured request timeout.

    Ollama's Python client defaults to an unlimited httpx timeout, which lets
    a hung server freeze the bot indefinitely (observed p95 = 1464s). Passing
    a finite timeout ensures hangs surface as exceptions instead.
    """
    return ollama.Client(
        host=_settings.ollama_host,
        timeout=_settings.ollama_request_timeout,
    )


# Create explicit client to avoid connection issues on Windows
_ollama_client = _build_ollama_client()


# =============================================================================
# CIRCUIT BREAKER — Short-circuits Ollama calls when the server is unhealthy.
# =============================================================================
#
# Primitive only: this story introduces the type and state machine. Wiring it
# into real Ollama calls is a separate story — importing these names must not
# change runtime behavior.


class OllamaCircuitOpenError(Exception):
    """Raised when a call is blocked because the Ollama circuit breaker is open."""


@dataclass
class CircuitBreaker:
    """Failure-counting circuit breaker with closed / open / half-open states.

    States:
        closed     — calls pass through; failures accumulate in a rolling window.
        open       — calls should short-circuit; transitions to half-open after
                     ``reset_seconds`` elapse.
        half-open  — one trial call allowed; success closes the circuit, another
                     failure re-opens it.

    Callers drive the state machine with ``record_success`` / ``record_failure``
    after each attempt, and consult ``is_open`` before making a new call.
    """

    state: str = "closed"
    failure_count: int = 0
    opened_at: float = 0.0
    threshold: int = 5
    window_seconds: float = 60.0
    reset_seconds: float = 30.0
    # Private: start of the current rolling failure window. Not part of the
    # public API — failures older than window_seconds are discarded via this.
    _window_start: float = 0.0

    def is_open(self) -> bool:
        """Return True if the circuit currently blocks calls.

        Transitions ``open`` → ``half-open`` when ``reset_seconds`` have
        elapsed since the circuit opened. Returns False in the half-open state
        so a single trial call is permitted.
        """
        if self.state == "open":
            if _time.monotonic() - self.opened_at >= self.reset_seconds:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call — closes the circuit and clears failures."""
        self.state = "closed"
        self.failure_count = 0
        self.opened_at = 0.0
        self._window_start = 0.0

    def record_failure(self) -> None:
        """Record a failed call.

        In ``half-open`` any failure trips the breaker back to ``open``. In
        ``closed`` the failure count accumulates within a rolling
        ``window_seconds`` window — once ``threshold`` is reached the circuit
        opens. Failures older than the window start a fresh window.
        """
        now = _time.monotonic()
        if self.state == "half-open":
            self.state = "open"
            self.opened_at = now
            self.failure_count = self.threshold
            return
        # closed state: rolling window
        if self.failure_count == 0 or (now - self._window_start) > self.window_seconds:
            self._window_start = now
            self.failure_count = 1
        else:
            self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = "open"
            self.opened_at = now


@dataclass
class Tool:
    """A tool the agent can use."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable[..., Any]
    timeout: int | None = None  # Per-tool timeout in seconds (None = no limit)


@dataclass
class AgentConfig:
    """Agent configuration."""

    model: str = "llama3.1"
    temperature: float = 0.7
    max_turns: int = 20  # Max tool-calling turns per task
    system_prompt: str = ""
    verbose: bool = True
    # Response cache: skip the Ollama call for repeated short prompts that
    # previously completed without tool calls. Uses llm_optimizer.cache_*.
    enable_response_cache: bool = False
    response_cache_max_age_hours: int = 1
    # Prompt compression: before each Ollama turn, collapse bulky older
    # tool/assistant messages once the running context crosses this size.
    compression_threshold_chars: int = DEFAULT_COMPRESSION_THRESHOLD_CHARS
    compression_keep_recent: int = 4
    # Parallel tool execution: when the LLM emits multiple independent tool
    # calls in a single response, run them concurrently. Falls back to
    # sequential execution when dependencies are detected between calls.
    parallel_tool_execution: bool = True
    max_parallel_tools: int = 8


class Agent:
    """
    Autonomous agent powered by Ollama with tool support.

    Usage:
        agent = Agent(config)
        agent.register_tool(tool)
        result = agent.run("What files are in the current directory?")
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        """Initialize the agent with config, tool storage, and built-in tools."""
        self.config = config or AgentConfig()
        self.tools: dict[str, Tool] = {}
        self.messages: list[dict] = []
        self.turn_count = 0
        self.last_thinking: str = ""  # Raw <think> content from most recent chat()/run()

        # Tool result storage — truncates large results, stores full on disk
        self.result_storage = ToolResultStorage()

        # Profiling hook — set externally by discord_memory_bot to record timing
        self._request_timer = None

        # Set default system prompt if not provided
        if not self.config.system_prompt:
            self.config.system_prompt = self._default_system_prompt()

        # Register the built-in get_stored_result tool
        self._register_builtin_tools()

    def _default_system_prompt(self) -> str:
        """Return the default system prompt with the current timestamp."""
        return f"""You are an autonomous AI agent running locally via Ollama.
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

You have access to tools that let you interact with the system.
Use tools when needed to accomplish tasks. Be concise and efficient.

When you have completed a task or have information to share, respond directly.
If you need to perform multiple steps, do them one at a time."""

    def _register_builtin_tools(self) -> None:
        """Register built-in tools that are always available."""
        self.tools["get_stored_result"] = Tool(
            name="get_stored_result",
            description=(
                "Retrieve the full content of a previously truncated tool result. "
                "When a tool result was too large for context, a preview was shown "
                "with a result ID. Use this tool with that ID to get the full content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result_id": {
                        "type": "string",
                        "description": "The stored result ID shown in the truncation notice",
                    }
                },
                "required": ["result_id"],
            },
            function=self.result_storage.get_full_result,
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
        if self.config.verbose:
            print(f"[Agent] Registered tool: {tool.name}")

    def _get_ollama_tools(self) -> list[dict[str, Any]]:
        """Convert registered tools to Ollama's expected tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def _run_with_timeout(self, func: Callable, args: dict, timeout: int) -> str:
        """Run a tool function in a thread with a timeout.

        Returns the result string, or a fallback message if the call
        exceeds *timeout* seconds.  The worker thread is left as a daemon
        so it won't block the process if it hangs.
        """
        result_box: list[str | None] = [None]
        error_box: list[BaseException | None] = [None]

        def _worker() -> None:
            try:
                r = func(**args)
                result_box[0] = r if isinstance(r, str) else json.dumps(r, indent=2, default=str)
            except Exception as exc:
                error_box[0] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running — return a fallback
            return (
                f"⏱️ Tool timed out after {timeout}s. "
                f"The external service may be slow or unreachable. "
                f"Please try again in a moment."
            )

        if error_box[0] is not None:
            raise error_box[0]

        return result_box[0]  # type: ignore[return-value]

    def _tool_calls_have_dependencies(self, tool_calls: list[dict]) -> bool:
        """Return True if any call's arguments reference an earlier call's output.

        The LLM normally cannot reference outputs that don't exist yet within a
        single response turn, so this is nearly always False. Kept as a
        conservative guard: if a later call's arguments contain a placeholder
        pointing at an earlier tool's name or a stored-result marker, fall back
        to sequential execution.
        """
        if len(tool_calls) <= 1:
            return False

        earlier_names: list[str] = []
        for tc in tool_calls:
            func = tc.get("function", {}) or {}
            args = func.get("arguments", {}) or {}
            try:
                args_str = json.dumps(args, default=str).lower()
            except (TypeError, ValueError):
                args_str = str(args).lower()

            for earlier in earlier_names:
                lname = earlier.lower()
                if not lname:
                    continue
                patterns = (
                    f"{{{{{lname}",        # {{tool_name...
                    f"{{{lname}}}",         # {tool_name}
                    f"${lname}",            # $tool_name
                    f"output_of_{lname}",   # output_of_tool_name
                    f"<{lname}_result>",    # <tool_name_result>
                    f"result_of_{lname}",   # result_of_tool_name
                )
                if any(p in args_str for p in patterns):
                    return True
            earlier_names.append(func.get("name", ""))
        return False

    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute a batch of tool calls, returning tool-role messages in order.

        Uses a thread pool when the batch has multiple independent calls and
        ``config.parallel_tool_execution`` is enabled. Falls back to sequential
        execution for a single call, detected dependencies, or when disabled.
        The returned list matches the order of ``tool_calls`` so downstream
        message appends stay deterministic.
        """
        if not tool_calls:
            return []

        parallel = (
            self.config.parallel_tool_execution
            and len(tool_calls) > 1
            and not self._tool_calls_have_dependencies(tool_calls)
        )

        def _run_one(tc: dict) -> str:
            func = tc.get("function", {}) or {}
            name = func.get("name", "")
            args = func.get("arguments", {}) or {}
            mode = "parallel" if parallel else "sequential"
            self._log(f"Calling tool ({mode}): {name}({json.dumps(args, default=str)[:100]}...)")
            result = self._execute_tool(name, args)
            self._log(f"Tool result ({name}): {result[:200]}...")
            return result

        if parallel:
            self._log(f"Running {len(tool_calls)} tool calls in parallel")
            max_workers = min(len(tool_calls), max(1, self.config.max_parallel_tools))

            # Each worker thread needs its own context copy — a single Context
            # cannot be entered concurrently from multiple threads. Copying
            # per-call propagates the active request_id ContextVar so tool
            # functions and downstream Claude HTTP calls see the same id.
            def _run_with_ctx(tc: dict) -> str:
                return contextvars.copy_context().run(_run_one, tc)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_run_with_ctx, tool_calls))
        else:
            results = [_run_one(tc) for tc in tool_calls]

        return [{"role": "tool", "content": r} for r in results]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string.

        Large results are automatically truncated and stored to disk.
        The agent can retrieve full results via get_stored_result.
        If the tool has a timeout set, the call is wrapped in a thread
        with that deadline; a friendly fallback is returned on expiry.
        """
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"

        tool = self.tools[name]
        start = _time.perf_counter()
        try:
            # Use timeout wrapper if tool has a timeout configured
            if tool.timeout is not None:
                result = self._run_with_timeout(tool.function, arguments, tool.timeout)
            else:
                result = tool.function(**arguments)
                if not isinstance(result, str):
                    result = json.dumps(result, indent=2, default=str)

            # Truncate large results (skip for get_stored_result to avoid recursion)
            truncated = False
            if name != "get_stored_result":
                original_len = len(result)
                result = self.result_storage.maybe_truncate(name, result)
                truncated = len(result) < original_len

            # Record tool timing if profiler is attached
            duration = _time.perf_counter() - start
            if self._request_timer:
                self._request_timer.record_tool(name, duration, len(result), truncated)

            # Log to tool analytics
            from .tool_analytics import record_tool_call
            record_tool_call(name, success=True, duration_ms=round(duration * 1000, 1), result_size=len(result))

            return result
        except Exception as e:
            duration = _time.perf_counter() - start
            if self._request_timer:
                self._request_timer.record_tool(name, duration, 0)

            from .tool_analytics import record_tool_call
            record_tool_call(name, success=False, duration_ms=round(duration * 1000, 1), error=str(e)[:200])

            return f"Error executing {name}: {e}"

    def _log(self, msg: str) -> None:
        """Log a message if verbose mode is on."""
        if self.config.verbose:
            print(f"[Agent] {msg}")

    def set_temperature(self, temp: float) -> None:
        """Override temperature for the next interaction (resets after one use)."""
        self._temp_override = temp

    def _get_temperature(self) -> float:
        """Get current temperature, using and consuming any one-shot override."""
        temp = getattr(self, "_temp_override", None)
        if temp is not None:
            self._temp_override = None  # reset after use
            return temp
        return self.config.temperature

    def _estimate_ctx_size(self) -> int:
        """Estimate the context window size needed for the current messages.

        Starts at 8192 and scales up in steps based on actual message content.
        This avoids allocating 131K of KV cache for a simple "what time is it?"
        while still supporting large contexts when genuinely needed.
        """
        # Rough estimate: 4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        estimated_tokens = total_chars // 4

        # Add headroom for the response (at least 2K tokens)
        needed = estimated_tokens + 2048

        # Snap to standard sizes (avoid constant reallocation)
        if needed <= 4096:
            return 8192
        elif needed <= 8192:
            return 16384
        elif needed <= 16384:
            return 32768
        elif needed <= 32768:
            return 65536
        else:
            return 131072

    def _try_cache_lookup(self, task: str) -> str | None:
        """Look up a cached response for a single-turn task.

        Failures in the cache layer (SQLite locked, permission error, etc.)
        must never break the agent — they just skip the fast path.
        """
        try:
            from .llm_optimizer import cache_lookup

            return cache_lookup(task, max_age_hours=self.config.response_cache_max_age_hours)
        except Exception as e:
            self._log(f"cache_lookup failed: {e}")
            return None

    def _cache_store(self, task: str, response: str) -> None:
        """Store a successful response in the cache. Soft-fails on errors."""
        try:
            from .llm_optimizer import cache_store

            cache_store(task, response, endpoint="ollama", model=self.config.model)
        except Exception as e:
            self._log(f"cache_store failed: {e}")

    def _maybe_compress_messages(self) -> None:
        """Compress older bulky messages in-place if the history is large."""
        before = estimate_size_chars(self.messages)
        if before <= self.config.compression_threshold_chars:
            return
        compressed, saved = compress_messages(
            self.messages,
            max_chars=self.config.compression_threshold_chars,
            keep_recent=self.config.compression_keep_recent,
        )
        if saved > 0:
            self.messages = compressed
            self._log(
                f"Compressed messages: {before:,} → {before - saved:,} chars "
                f"(saved {saved:,})"
            )

    def run(self, task: str, context: str = "", images: list[bytes] | None = None) -> str:
        """
        Run the agent on a task.

        Args:
            task: The task or question for the agent
            context: Optional additional context
            images: Optional list of image bytes for vision models

        Returns:
            The agent's final response
        """
        # Seed the request-id ContextVar for this run if no caller has
        # already done so. Logs emitted from this point on (including from
        # tools and the Claude bridge) will carry the id.
        _seed_request_id_if_unset()

        # Initialize conversation
        self.messages = [{"role": "system", "content": self.config.system_prompt}]

        if context:
            self.messages.append({"role": "system", "content": f"Additional context:\n{context}"})

        # Build user message - add images if provided (for vision-capable models)
        user_message: dict[str, Any] = {"role": "user", "content": task}
        if images:
            user_message["images"] = images
            self._log(f"Including {len(images)} image(s) for vision analysis")

        self.messages.append(user_message)
        self.turn_count = 0

        self._log(f"Starting task: {task[:100]}...")

        # Response cache: short text-only prompts may have a fresh cached
        # answer. Only attempt when images aren't present and the task is
        # small enough to make equality hashing meaningful.
        if (
            self.config.enable_response_cache
            and not images
            and is_cacheable_task(task)
        ):
            cached = self._try_cache_lookup(task)
            if cached is not None:
                self._log("Cache hit — skipping Ollama call")
                return cached

        # Agent loop
        while self.turn_count < self.config.max_turns:
            self.turn_count += 1
            self._log(f"Turn {self.turn_count}/{self.config.max_turns}")

            # Compress older tool/assistant messages if the history is getting
            # large. Keeps the system prompt and the recent reasoning chain
            # intact — only stale tool output is collapsed.
            self._maybe_compress_messages()

            num_ctx = self._estimate_ctx_size()
            input_chars = sum(len(m.get("content", "")) for m in self.messages)
            llm_start = _time.perf_counter()
            try:
                response = ollama_call_with_retries(
                    _ollama_client.chat,
                    model=self.config.model,
                    messages=self.messages,
                    tools=self._get_ollama_tools() if self.tools else None,
                    options={"temperature": self._get_temperature(), "num_ctx": num_ctx},
                    keep_alive=-1,
                    think=True,
                )
            except Exception as e:
                llm_duration = _time.perf_counter() - llm_start
                _record_perf("ollama", llm_duration, success=False,
                             model=self.config.model, error=str(e))
                self._log(f"Ollama error: {e}")
                return f"Agent error: {e}"
            llm_duration = _time.perf_counter() - llm_start

            message = response.get("message", {})
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls") or []

            # Record to perf monitor (endpoint-level metrics)
            _record_perf(
                "ollama", llm_duration, success=True,
                model=self.config.model,
                input_tokens=input_chars // 4,  # rough estimate
                output_tokens=len(content) // 4,
            )

            # Record LLM call timing (per-request profiler)
            tool_names = [
                (tc.get("function", {}) or {}).get("name", "")
                for tc in tool_calls
            ]
            if self._request_timer:
                self._request_timer.record_llm_call(
                    turn=self.turn_count, duration=llm_duration,
                    input_chars=input_chars, output_chars=len(content),
                    num_ctx=num_ctx, tool_calls=tool_names,
                )
            self._log(f"LLM call: {llm_duration:.1f}s, ctx={num_ctx}, tools={tool_names or 'none'}")

            # Add assistant message to history
            self.messages.append(message)

            # If no tool calls, we're done
            if not tool_calls:
                self._log("Task complete (no more tool calls)")
                result = strip_thinking_tags(content)
                # Guard against empty responses (LLM put everything in <think> tags)
                final = result if result else "I processed your request but didn't generate a visible response. Could you try rephrasing?"
                # Only cache single-turn, tool-free runs — those are the ones
                # cheap-enough to treat as deterministic.
                if (
                    self.config.enable_response_cache
                    and self.turn_count == 1
                    and not images
                    and is_cacheable_task(task)
                    and result
                ):
                    self._cache_store(task, final)
                return final

            # Execute tool calls (in parallel when independent)
            self.messages.extend(self._execute_tool_calls(tool_calls))

        self._log("Max turns reached")
        return (
            strip_thinking_tags(content)
            if content
            else "Agent reached maximum turns without completing."
        )

    def chat(self, message: str) -> str:
        """
        Continue a conversation with the agent.
        Maintains history from previous interactions.
        """
        self.messages.append({"role": "user", "content": message})

        # Run one turn
        chat_start = _time.perf_counter()
        try:
            response = ollama_call_with_retries(
                _ollama_client.chat,
                model=self.config.model,
                messages=self.messages,
                tools=self._get_ollama_tools() if self.tools else None,
                options={"temperature": self._get_temperature(), "num_ctx": 131072},
                keep_alive=-1,
                think=True,
            )
        except Exception as e:
            _record_perf("ollama", _time.perf_counter() - chat_start,
                         success=False, model=self.config.model, error=str(e))
            return f"Error: {e}"
        _record_perf("ollama", _time.perf_counter() - chat_start,
                     success=True, model=self.config.model)

        response_message: dict[str, Any] = response.get("message", {})
        content = response_message.get("content", "") or ""
        tool_calls = response_message.get("tool_calls") or []

        self.messages.append(response_message)

        # Capture thinking from initial response (may be overwritten if tool calls follow)
        self.last_thinking = extract_thinking(content)

        # Handle tool calls if any (in parallel when independent)
        if tool_calls:
            self.messages.extend(self._execute_tool_calls(tool_calls))

            # Get final response after tool execution
            chat_start2 = _time.perf_counter()
            try:
                response = ollama_call_with_retries(
                    _ollama_client.chat,
                    model=self.config.model,
                    messages=self.messages,
                    options={"temperature": self._get_temperature(), "num_ctx": self._estimate_ctx_size()},
                    keep_alive=-1,
                    think=True,
                )
                _record_perf("ollama", _time.perf_counter() - chat_start2,
                             success=True, model=self.config.model)
            except Exception as e:
                _record_perf("ollama", _time.perf_counter() - chat_start2,
                             success=False, model=self.config.model, error=str(e))
                raise
            content = response.get("message", {}).get("content", "") or ""
            self.messages.append(response.get("message", {}))

        self.last_thinking = extract_thinking(content)
        return strip_thinking_tags(content)

    def get_history(self) -> list[dict]:
        """Get the conversation history."""
        return self.messages.copy()

    def clear_history(self) -> None:
        """Clear conversation history and reset turn counter."""
        self.messages = []
        self.turn_count = 0


def create_tool(
    name: str,
    description: str,
    parameters: dict,
    function: Callable,
    timeout: int | None = None,
) -> Tool:
    """Helper to create a Tool instance from individual arguments.

    Args:
        timeout: Optional per-tool timeout in seconds.  If set, the agent
                 will abort the call after this many seconds and return a
                 graceful fallback message instead of hanging.
    """
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        function=function,
        timeout=timeout,
    )
