"""Bridge that exposes MCP-server tools to the Ollama coder loop.

OllamaCoder talks to Ollama's ``/api/chat`` with native function-calling
JSON; Ollama is not MCP-aware. This bridge spans the gap:

  1. Spawns an MCP server (default: ``mcp_servers.technomancer_context``)
     as a subprocess via stdio transport.
  2. Calls ``initialize`` + ``list_tools`` over the wire.
  3. Translates each MCP tool's JSON-Schema into the Ollama tool-
     definition format and exposes them via :attr:`tool_definitions`.
  4. Proxies ``call(name, args)`` from the Ollama loop back to the MCP
     server and returns the resulting text content.

Why a bridge: keeps the MCP server fully MCP-compliant (so non-Ollama
consumers — Claude, Cursor, the MCP Inspector — work natively against
the same server) while letting the existing Ollama wire protocol stay
untouched. The model experiences the bridge tools the same way it
experiences any other Ollama tool.

Failure mode: if the subprocess won't start, or initialize fails, the
bridge logs and returns no tools. The story keeps running with the
existing six core tools (read_file/write_file/edit_file/list_files/
search_code/finish). MCP context is a *helpful augmentation*, not a
hard dependency — a transient stutter must not kill a story.

Threading: the bridge owns a dedicated background asyncio loop running
on a daemon thread. All MCP wire calls are scheduled onto that loop
via :func:`asyncio.run_coroutine_threadsafe`. The Ollama coder calls
:meth:`call` synchronously from its own thread; the bridge blocks on
the future until the MCP server replies (or times out).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# How long any single MCP wire call may take. The server is local
# stdio + static text; if it doesn't respond in 10s something is wrong
# and the bridge bails to its no-MCP fallback.
_MCP_CALL_TIMEOUT_SECONDS: float = 10.0

# How long to wait for the subprocess + initialize handshake.
_MCP_STARTUP_TIMEOUT_SECONDS: float = 10.0


class MCPBridge:
    """Adapter from MCP stdio server -> Ollama tool definitions + dispatch.

    Lifecycle::

        bridge = MCPBridge()
        if bridge.start():
            ollama_tools.extend(bridge.tool_definitions)
            ...
            text = bridge.call("aiw_purpose", {})
            ...
        bridge.stop()

    Use as a context manager (``with MCPBridge() as bridge:``) for
    automatic cleanup. ``start()`` returns True iff the subprocess
    came up and at least one tool was discovered. False means
    "fall back to core tools only" — never an exception thrown into
    the coder loop.
    """

    def __init__(
        self,
        server_module: str = "mcp_servers.technomancer_context",
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.server_module: str = server_module
        # ``log`` defaults to the module logger so a missing state
        # parameter doesn't crash. Callers (OllamaCoder) pass
        # ``state.log`` so MCP activity shows up in the run log.
        self._log: Callable[[str], None] = log or (lambda msg: logger.info(msg))

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready: threading.Event = threading.Event()

        # Held for the lifetime of the bridge — closed in stop().
        self._stack_close: Optional[Callable[[], Any]] = None
        self._session: Any = None  # mcp.ClientSession when started

        # Public: filled by start(), consumed by OllamaCoder.
        self.tool_definitions: list[dict[str, Any]] = []
        # Names of tools the MCP server exposed (for fast membership check).
        self._mcp_tool_names: set[str] = set()

        self._started: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Spawn the MCP subprocess and discover its tools.

        Returns True iff the server is up and at least one tool was
        registered. On any failure (import error, subprocess failed
        to spawn, initialize timeout, no tools discovered) this
        method logs the cause and returns False — the caller should
        fall back to running with the core tool set only.
        """
        if self._started:
            return bool(self.tool_definitions)

        try:
            self._spawn_loop_thread()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[MCPBridge] failed to start asyncio thread: {exc}")
            return False

        try:
            tools_ok: bool = self._run_coro(
                self._async_start(),
                timeout=_MCP_STARTUP_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"[MCPBridge] startup failed: {exc}")
            self._teardown_loop_thread()
            return False

        if not tools_ok or not self.tool_definitions:
            self._log("[MCPBridge] startup completed but no tools discovered")
            self.stop()
            return False

        self._started = True
        names: str = ", ".join(sorted(self._mcp_tool_names))
        self._log(
            f"[MCPBridge] started ({len(self.tool_definitions)} tools): {names}"
        )
        return True

    def has_tool(self, name: str) -> bool:
        """Return True if ``name`` is a tool exposed by the MCP server."""
        return name in self._mcp_tool_names

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Invoke an MCP tool and return its text content.

        Returns a string starting with ``"ERROR: "`` on any failure
        (unknown tool, MCP server crashed, timeout). The Ollama loop
        treats the return value as the tool result body — same shape
        as the core tools.
        """
        if not self._started or self._session is None:
            return "ERROR: MCP bridge is not running"
        if name not in self._mcp_tool_names:
            return f"ERROR: MCP tool '{name}' not registered"

        try:
            return self._run_coro(
                self._async_call_tool(name, args),
                timeout=_MCP_CALL_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"[MCPBridge] call({name}) failed: {exc}")
            return f"ERROR: MCP call failed: {exc}"

    def stop(self) -> None:
        """Tear down the subprocess and the asyncio loop thread.

        Safe to call multiple times. Safe to call from __exit__ even
        if start() never succeeded.
        """
        if self._loop is not None and self._loop.is_running():
            try:
                self._run_coro(self._async_stop(), timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[MCPBridge] stop() raised: {exc}")
        self._teardown_loop_thread()
        self._started = False
        self.tool_definitions = []
        self._mcp_tool_names = set()

    def __enter__(self) -> "MCPBridge":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals — asyncio loop on a background thread
    # ------------------------------------------------------------------

    def _spawn_loop_thread(self) -> None:
        """Start a daemon thread running a dedicated asyncio loop."""
        self._loop_ready.clear()

        def _runner() -> None:
            loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                # Drain any pending tasks so the loop exits cleanly.
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                except Exception:
                    pass
                loop.close()

        thread = threading.Thread(
            target=_runner,
            name="mcp-bridge-loop",
            daemon=True,
        )
        thread.start()
        self._loop_thread = thread
        # Wait briefly for the loop to be ready before scheduling on it.
        if not self._loop_ready.wait(timeout=5.0):
            raise RuntimeError("asyncio loop thread failed to start")

    def _teardown_loop_thread(self) -> None:
        """Stop the asyncio loop and join its thread."""
        loop = self._loop
        thread = self._loop_thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None

    def _run_coro(self, coro: Any, *, timeout: float) -> Any:
        """Submit a coroutine to the loop thread and wait synchronously."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("MCP bridge loop is not running")
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Internals — async MCP wire calls
    # ------------------------------------------------------------------

    async def _async_start(self) -> bool:
        """Spawn the subprocess, initialize, list tools, build defs."""
        # Imported lazily so a unit-test environment without the mcp
        # package can still import this module to inspect the surface
        # area (the import error is then caught by start()).
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self.server_module],
        )
        stack = AsyncExitStack()
        await stack.__aenter__()
        # On any failure below we close the stack so the subprocess
        # is reaped even if initialize() blew up.
        try:
            transport = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = transport
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            tools_result = await session.list_tools()
            self._session = session
            self._stack_close = stack.aclose

            self.tool_definitions = []
            self._mcp_tool_names = set()
            for tool in tools_result.tools:
                ollama_def = _mcp_tool_to_ollama(tool)
                if ollama_def is None:
                    continue
                self.tool_definitions.append(ollama_def)
                self._mcp_tool_names.add(tool.name)
            return bool(self.tool_definitions)
        except Exception:
            await stack.aclose()
            raise

    async def _async_call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Call ``name`` over the MCP wire and return its first text block."""
        session = self._session
        if session is None:
            return "ERROR: MCP session not initialized"
        result = await session.call_tool(name, args)
        # Result content is a list of content blocks. The technomancer
        # context server only returns text blocks; concatenate them so
        # the model sees one contiguous response.
        if not result.content:
            return ""
        chunks: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    async def _async_stop(self) -> None:
        """Close the AsyncExitStack so the subprocess is terminated."""
        close = self._stack_close
        self._stack_close = None
        self._session = None
        if close is not None:
            try:
                await close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[MCPBridge] stack close raised: %s", exc)


def _mcp_tool_to_ollama(tool: Any) -> Optional[dict[str, Any]]:
    """Translate an MCP ``Tool`` into Ollama's tool-definition shape.

    Ollama expects::

        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": {...}}}

    MCP exposes ``tool.name``, ``tool.description``, and
    ``tool.inputSchema`` (already a JSON Schema dict). We pass the
    schema through unchanged — Ollama and MCP both speak JSON Schema.
    Returns ``None`` if the tool object is missing required fields,
    in which case the caller skips it.
    """
    name = getattr(tool, "name", None)
    if not name:
        return None
    description = getattr(tool, "description", "") or ""
    schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }
