"""
Agent Core - Ollama-based autonomous agent with tool calling.

This agent can:
- Use tools (file I/O, memory, web, etc.)
- Maintain a knowledge graph
- Feed information back to Claude when needed
- Run with safe_update.py workflow for tested deployments
"""

import json
import re
import threading
import time as _time
import uuid
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

    def _store(self, result_id: str, content: str) -> None:
        """Store a full result in Redis or memory."""
        if self.redis:
            try:
                self.redis.setex(f"{self.REDIS_PREFIX}{result_id}", self.REDIS_TTL, content)
                return
            except Exception:
                pass  # Fall through to memory
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
        return self._mem_store.get(result_id)

    def maybe_truncate(self, tool_name: str, result: str) -> str:
        """Truncate a tool result if it exceeds the threshold.

        Returns the original result if small enough, or a preview with a
        storage reference if truncated.
        """
        self.stats["total_results"] += 1
        threshold = TOOL_RESULT_THRESHOLDS.get(tool_name, DEFAULT_THRESHOLD)

        if len(result) <= threshold:
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

try:
    import ollama
except ImportError:
    raise ImportError("Install ollama: pip install ollama")

# Create explicit client to avoid connection issues on Windows
_ollama_client = ollama.Client(host="http://127.0.0.1:11434")


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

        # Agent loop
        while self.turn_count < self.config.max_turns:
            self.turn_count += 1
            self._log(f"Turn {self.turn_count}/{self.config.max_turns}")

            num_ctx = self._estimate_ctx_size()
            input_chars = sum(len(m.get("content", "")) for m in self.messages)
            llm_start = _time.perf_counter()
            try:
                response = _ollama_client.chat(
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
                return result if result else "I processed your request but didn't generate a visible response. Could you try rephrasing?"

            # Execute tool calls
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})

                self._log(f"Calling tool: {name}({json.dumps(args)[:100]}...)")

                result = self._execute_tool(name, args)

                self._log(f"Tool result: {result[:200]}...")

                # Add tool result to messages
                self.messages.append(
                    {
                        "role": "tool",
                        "content": result,
                    }
                )

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
            response = _ollama_client.chat(
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

        # Handle tool calls if any
        if tool_calls:
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})

                result = self._execute_tool(name, args)
                self.messages.append({"role": "tool", "content": result})

            # Get final response after tool execution
            chat_start2 = _time.perf_counter()
            try:
                response = _ollama_client.chat(
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
