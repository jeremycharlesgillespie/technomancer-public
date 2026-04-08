"""
Tests for agent/tools.py - File operations and command filtering.

CRITICAL: The command filtering tests are security-critical.
These tests ensure destructive commands are blocked unless explicitly allowed.
"""

from unittest.mock import MagicMock, patch

from agent.tools import (
    _is_destructive_command,
    # Command filtering functions
    _is_safe_command,
    append_file,
    get_all_tools,
    get_current_time,
    # Tool getters
    get_file_tools,
    # System tools
    get_system_info,
    get_system_tools,
    list_directory,
    # File operations
    read_file,
    run_command,
    search_files,
    write_file,
)

# =============================================================================
# COMMAND FILTERING - SECURITY CRITICAL
# =============================================================================


class TestSafeCommandPatterns:
    """Tests for _is_safe_command - verifying read-only commands are allowed."""

    def test_safe_ls(self):
        """ls is recognized as safe."""
        assert _is_safe_command("ls -la")
        assert _is_safe_command("ls")
        assert _is_safe_command("ls /home")

    def test_safe_dir(self):
        """dir (Windows) is safe."""
        assert _is_safe_command("dir")
        assert _is_safe_command("dir /b")

    def test_safe_cat(self):
        """cat is safe."""
        assert _is_safe_command("cat file.txt")
        assert _is_safe_command("cat /etc/passwd")

    def test_safe_grep(self):
        """grep is safe."""
        assert _is_safe_command("grep pattern file.txt")
        assert _is_safe_command("grep -r pattern .")

    def test_safe_git_status(self):
        """git status is safe."""
        assert _is_safe_command("git status")
        assert _is_safe_command("git status -s")

    def test_safe_git_log(self):
        """git log is safe."""
        assert _is_safe_command("git log")
        assert _is_safe_command("git log --oneline")
        assert _is_safe_command("git log -n 5")

    def test_safe_git_diff(self):
        """git diff is safe."""
        assert _is_safe_command("git diff")
        assert _is_safe_command("git diff HEAD~1")

    def test_safe_git_show(self):
        """git show is safe."""
        assert _is_safe_command("git show HEAD")

    def test_safe_git_branch(self):
        """git branch (listing) is safe."""
        assert _is_safe_command("git branch")
        assert _is_safe_command("git branch -a")

    def test_safe_pwd(self):
        """pwd is safe."""
        assert _is_safe_command("pwd")

    def test_safe_whoami(self):
        """whoami is safe."""
        assert _is_safe_command("whoami")

    def test_safe_python_version(self):
        """python --version is safe."""
        assert _is_safe_command("python --version")
        # Note: python3 not explicitly in SAFE_PATTERNS - only python\s+--version

    def test_safe_curl(self):
        """curl is safe."""
        assert _is_safe_command("curl https://example.com")

    def test_not_safe_arbitrary_command(self):
        """Arbitrary commands are not explicitly safe."""
        # These are not in SAFE_PATTERNS
        assert not _is_safe_command("some_random_command")
        assert not _is_safe_command("./run_script.sh")


class TestDestructiveCommandPatterns:
    """Tests for _is_destructive_command - SECURITY CRITICAL."""

    # File deletion
    def test_destructive_rm(self):
        """rm is destructive."""
        assert _is_destructive_command("rm file.txt")
        assert _is_destructive_command("rm -rf /")
        assert _is_destructive_command("rm -r directory")

    def test_destructive_del(self):
        """del (Windows) is destructive."""
        assert _is_destructive_command("del file.txt")
        assert _is_destructive_command("del /f file.txt")

    def test_destructive_rmdir(self):
        """rmdir is destructive."""
        assert _is_destructive_command("rmdir directory")

    # Process killing
    def test_destructive_kill(self):
        """kill command is destructive."""
        assert _is_destructive_command("kill 1234")
        assert _is_destructive_command("kill -9 1234")

    def test_destructive_taskkill(self):
        """Windows taskkill is destructive."""
        assert _is_destructive_command("taskkill /F /PID 1234")
        assert _is_destructive_command("taskkill /IM process.exe")

    def test_destructive_pkill(self):
        """pkill is destructive."""
        assert _is_destructive_command("pkill python")

    # Git destructive operations
    def test_destructive_git_push(self):
        """git push is destructive."""
        assert _is_destructive_command("git push")
        assert _is_destructive_command("git push origin main")
        assert _is_destructive_command("git push --force")

    def test_destructive_git_reset_hard(self):
        """git reset --hard is destructive."""
        assert _is_destructive_command("git reset --hard")
        assert _is_destructive_command("git reset --hard HEAD~1")

    def test_destructive_git_clean(self):
        """git clean is destructive."""
        assert _is_destructive_command("git clean -fd")
        assert _is_destructive_command("git clean -f")

    # SQL destructive operations
    def test_destructive_sql_drop_database(self):
        """SQL DROP DATABASE is destructive."""
        assert _is_destructive_command("drop database mydb")
        assert _is_destructive_command("DROP DATABASE production")

    def test_destructive_sql_drop_table(self):
        """SQL DROP TABLE is destructive."""
        assert _is_destructive_command("drop table users")
        assert _is_destructive_command("DROP TABLE customers")

    def test_destructive_sql_truncate(self):
        """SQL TRUNCATE TABLE is destructive."""
        assert _is_destructive_command("truncate table logs")

    # System control
    def test_destructive_shutdown(self):
        """shutdown is destructive."""
        assert _is_destructive_command("shutdown now")
        assert _is_destructive_command("shutdown -h now")

    def test_destructive_reboot(self):
        """reboot is destructive."""
        assert _is_destructive_command("reboot")

    # Package management
    def test_destructive_pip_uninstall(self):
        """pip uninstall is destructive."""
        assert _is_destructive_command("pip uninstall package")
        # Note: pip3 not explicitly in DESTRUCTIVE_PATTERNS - only pip\s+uninstall

    def test_destructive_npm_publish(self):
        """npm publish is destructive."""
        assert _is_destructive_command("npm publish")

    # Non-destructive commands should not match
    def test_not_destructive_safe_commands(self):
        """Safe commands are not destructive."""
        assert not _is_destructive_command("ls -la")
        assert not _is_destructive_command("cat file.txt")
        assert not _is_destructive_command("git status")
        assert not _is_destructive_command("git log")
        assert not _is_destructive_command("echo hello")


class TestRunCommand:
    """Tests for run_command function - integration of filtering."""

    def test_run_command_blocks_destructive(self):
        """run_command blocks destructive without flag."""
        result = run_command("rm -rf /important")

        assert "BLOCKED" in result
        assert "destructive" in result.lower()

    def test_run_command_allows_safe(self):
        """run_command allows safe commands."""
        # echo should work
        result = run_command("echo test")

        # Should not be blocked
        assert "BLOCKED" not in result

    @patch("subprocess.run")
    def test_run_command_allows_destructive_with_flag(self, mock_subprocess):
        """run_command allows destructive with allow_destructive=True."""
        mock_subprocess.return_value = MagicMock(stdout="deleted", stderr="", returncode=0)

        # With flag, should not be blocked (but we mock the actual execution)
        result = run_command("rm file.txt", allow_destructive=True)

        assert "BLOCKED" not in result
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_subprocess):
        """run_command respects timeout parameter."""
        import subprocess

        mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 5)

        result = run_command("sleep 100", timeout=5)

        assert "timed out" in result.lower()

    @patch("subprocess.run")
    def test_run_command_captures_output(self, mock_subprocess):
        """run_command captures stdout and stderr."""
        mock_subprocess.return_value = MagicMock(stdout="output", stderr="error", returncode=0)

        result = run_command("some_command")

        assert "output" in result


# =============================================================================
# FILE OPERATIONS
# =============================================================================


class TestReadFile:
    """Tests for read_file function."""

    def test_read_file_exists(self, tmp_path):
        """read_file returns content for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = read_file(str(test_file))

        assert result == "Hello, World!"

    def test_read_file_not_found(self, tmp_path):
        """read_file returns error for missing file."""
        result = read_file(str(tmp_path / "nonexistent.txt"))

        assert "Error" in result
        assert "not found" in result.lower()

    def test_read_file_truncates_long(self, tmp_path):
        """read_file truncates after max_lines."""
        test_file = tmp_path / "long.txt"
        lines = "\n".join([f"Line {i}" for i in range(1000)])
        test_file.write_text(lines)

        result = read_file(str(test_file), max_lines=10)

        assert "Line 0" in result
        assert "Line 9" in result
        assert "more lines" in result.lower()

    def test_read_file_directory_error(self, tmp_path):
        """read_file returns error for directory."""
        result = read_file(str(tmp_path))

        assert "Error" in result
        assert "Not a file" in result


class TestWriteFile:
    """Tests for write_file function."""

    def test_write_file_creates(self, tmp_path):
        """write_file creates new file."""
        test_file = tmp_path / "new.txt"

        result = write_file(str(test_file), "Content")

        assert "Successfully" in result
        assert test_file.exists()
        assert test_file.read_text() == "Content"

    def test_write_file_overwrites(self, tmp_path):
        """write_file overwrites existing file."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Old content")

        result = write_file(str(test_file), "New content")

        assert "Successfully" in result
        assert test_file.read_text() == "New content"

    def test_write_file_creates_dirs(self, tmp_path):
        """write_file creates parent directories."""
        test_file = tmp_path / "deep" / "nested" / "file.txt"

        result = write_file(str(test_file), "Content")

        assert "Successfully" in result
        assert test_file.exists()


class TestAppendFile:
    """Tests for append_file function."""

    def test_append_file(self, tmp_path):
        """append_file adds to existing content."""
        test_file = tmp_path / "append.txt"
        test_file.write_text("Line 1\n")

        result = append_file(str(test_file), "Line 2\n")

        assert "Successfully" in result
        assert test_file.read_text() == "Line 1\nLine 2\n"

    def test_append_file_creates(self, tmp_path):
        """append_file creates file if not exists."""
        test_file = tmp_path / "new_append.txt"

        result = append_file(str(test_file), "Content")

        assert "Successfully" in result
        assert test_file.exists()


class TestListDirectory:
    """Tests for list_directory function."""

    def test_list_directory(self, tmp_path):
        """list_directory shows files with metadata."""
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("code")
        (tmp_path / "subdir").mkdir()

        result = list_directory(str(tmp_path))

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result
        assert "DIR" in result  # Directory marker
        assert "FILE" in result  # File marker

    def test_list_directory_with_pattern(self, tmp_path):
        """list_directory respects glob pattern."""
        (tmp_path / "file.txt").write_text("text")
        (tmp_path / "file.py").write_text("python")

        result = list_directory(str(tmp_path), pattern="*.py")

        assert "file.py" in result
        assert "file.txt" not in result

    def test_list_directory_not_found(self):
        """list_directory returns error for missing directory."""
        result = list_directory("/nonexistent/path")

        assert "Error" in result
        assert "not found" in result.lower()


class TestSearchFiles:
    """Tests for search_files function."""

    def test_search_files_by_name(self, tmp_path):
        """search_files finds by glob pattern."""
        (tmp_path / "app.py").write_text("python code")
        (tmp_path / "test.py").write_text("test code")
        (tmp_path / "readme.md").write_text("docs")

        result = search_files(str(tmp_path), "*.py")

        assert "app.py" in result
        assert "test.py" in result
        assert "readme.md" not in result

    def test_search_files_by_content(self, tmp_path):
        """search_files filters by content."""
        (tmp_path / "file1.txt").write_text("Hello World")
        (tmp_path / "file2.txt").write_text("Goodbye World")
        (tmp_path / "file3.txt").write_text("Nothing here")

        result = search_files(str(tmp_path), "*.txt", content_pattern="World")

        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "file3.txt" not in result

    def test_search_files_no_matches(self, tmp_path):
        """search_files returns message when no matches."""
        (tmp_path / "file.txt").write_text("content")

        result = search_files(str(tmp_path), "*.nonexistent")

        assert "No matches" in result


# =============================================================================
# SYSTEM TOOLS
# =============================================================================


class TestGetSystemInfo:
    """Tests for get_system_info function."""

    def test_returns_json(self):
        """get_system_info returns valid JSON."""
        import json

        result = get_system_info()

        # Should be parseable JSON
        data = json.loads(result)
        assert "hostname" in data
        assert "os" in data
        assert "python_version" in data
        assert "cwd" in data


class TestGetCurrentTime:
    """Tests for get_current_time function."""

    def test_returns_json(self):
        """get_current_time returns valid JSON."""
        import json

        result = get_current_time()

        data = json.loads(result)
        assert "date" in data
        assert "time" in data
        assert "day_of_week" in data
        assert "formatted" in data


# =============================================================================
# TOOL GETTERS
# =============================================================================


class TestToolGetters:
    """Tests for tool getter functions."""

    def test_get_file_tools(self):
        """get_file_tools returns list of Tool objects."""
        tools = get_file_tools()

        assert len(tools) >= 4  # At least read, write, append, list, search
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names

    def test_get_system_tools(self):
        """get_system_tools returns list of Tool objects."""
        tools = get_system_tools()

        tool_names = [t.name for t in tools]
        assert "run_command" in tool_names
        assert "get_system_info" in tool_names
        assert "get_current_time" in tool_names

    def test_get_all_tools(self):
        """get_all_tools combines all tools."""
        all_tools = get_all_tools()
        file_tools = get_file_tools()
        system_tools = get_system_tools()

        # Should have at least file + system tools
        assert len(all_tools) >= len(file_tools) + len(system_tools)
