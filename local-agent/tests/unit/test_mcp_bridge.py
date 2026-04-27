"""Unit tests for the MCP bridge that adapts MCP -> Ollama tool calls.

These exercise the bridge end-to-end against the real
technomancer-context MCP server (spawned as a subprocess) — the
bridge is only ~250 lines and most of its complexity is in the
async/sync glue, which is hard to test meaningfully with mocks.
"""

from __future__ import annotations

from idea_board.mcp_bridge import MCPBridge, _mcp_tool_to_ollama


class TestBridgeStart:
    """Bridge starts, discovers tools, and stops cleanly."""

    def test_start_discovers_tools(self) -> None:
        bridge = MCPBridge()
        try:
            assert bridge.start() is True
            assert bridge._started is True
            # All six technomancer-context tools should be present.
            for name in (
                "overview",
                "aiw_purpose",
                "allowed_commands",
                "conventions",
                "scoring_rubric",
                "codebase_index",
            ):
                assert bridge.has_tool(name), f"missing: {name}"
            # tool_definitions are in Ollama shape.
            for tool_def in bridge.tool_definitions:
                assert tool_def["type"] == "function"
                assert "name" in tool_def["function"]
                assert "parameters" in tool_def["function"]
        finally:
            bridge.stop()

    def test_context_manager(self) -> None:
        with MCPBridge() as bridge:
            assert bridge._started is True
            assert len(bridge.tool_definitions) > 0
        # After exit, bridge is stopped.
        assert bridge._started is False
        assert bridge.tool_definitions == []

    def test_call_returns_tool_text(self) -> None:
        with MCPBridge() as bridge:
            text = bridge.call("aiw_purpose", {})
        assert "AIW" in text
        assert "mechanical" in text.lower()

    def test_call_with_args(self) -> None:
        with MCPBridge() as bridge:
            text = bridge.call("codebase_index", {"area": "aiv"})
        assert "aiv/" in text
        assert "scorer" in text.lower()

    def test_unknown_tool_returns_error_string(self) -> None:
        with MCPBridge() as bridge:
            text = bridge.call("not_a_tool", {})
        assert text.startswith("ERROR:")
        assert "not_a_tool" in text

    def test_call_before_start_returns_error(self) -> None:
        bridge = MCPBridge()
        # No start() — call should fail safely without raising.
        text = bridge.call("aiw_purpose", {})
        assert text.startswith("ERROR:")

    def test_failed_start_with_bogus_module(self) -> None:
        # A module that doesn't exist should produce a clean failure.
        bridge = MCPBridge(server_module="mcp_servers.does_not_exist")
        try:
            assert bridge.start() is False
            assert bridge._started is False
            assert bridge.tool_definitions == []
        finally:
            bridge.stop()

    def test_stop_is_idempotent(self) -> None:
        bridge = MCPBridge()
        # Never started — stop should not raise.
        bridge.stop()
        bridge.stop()


class TestMcpToolToOllama:
    """The schema translator preserves tool shape for Ollama."""

    def test_translates_minimal_tool(self) -> None:
        class FakeTool:
            name = "foo"
            description = "do foo"
            inputSchema = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }

        result = _mcp_tool_to_ollama(FakeTool())
        assert result is not None
        assert result["type"] == "function"
        assert result["function"]["name"] == "foo"
        assert result["function"]["description"] == "do foo"
        assert result["function"]["parameters"]["properties"] == {
            "x": {"type": "string"}
        }

    def test_returns_none_when_name_missing(self) -> None:
        class FakeTool:
            name = ""
            description = "x"
            inputSchema = {"type": "object", "properties": {}}

        assert _mcp_tool_to_ollama(FakeTool()) is None

    def test_uses_empty_object_schema_when_input_schema_missing(self) -> None:
        class FakeTool:
            name = "foo"
            description = ""
            inputSchema = None

        result = _mcp_tool_to_ollama(FakeTool())
        assert result is not None
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}
