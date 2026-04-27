"""
Tests for agent/accountability.py - Verification tools.
"""

from unittest.mock import MagicMock

import pytest

from agent.accountability import (
    GitDirtyError,
    GitNotInstalledError,
    get_accountability_tools,
    verify_content_contains,
    verify_file_exists,
    verify_file_modified_recently,
    verify_git_clean,
    verify_memory_saved,
    verify_readme_writable,
)


class TestVerifyFileExists:
    """Tests for verify_file_exists function."""

    def test_verify_file_exists_true(self, patched_accountability):
        """verify_file_exists returns VERIFIED for existing file."""
        # Create a test file
        test_file = patched_accountability / "Permanent" / "test.md"
        test_file.write_text("content")

        result = verify_file_exists(str(test_file))

        assert "VERIFIED" in result
        assert "exists" in result.lower()

    def test_verify_file_exists_false(self, patched_accountability):
        """verify_file_exists returns NOT FOUND for missing."""
        result = verify_file_exists(str(patched_accountability / "nonexistent.md"))

        assert "NOT FOUND" in result

    def test_verify_file_exists_relative_path(self, patched_accountability):
        """verify_file_exists handles relative paths to vault."""
        # Create file in vault
        test_file = patched_accountability / "Permanent" / "memories.md"
        test_file.write_text("memories")

        # Use relative path
        result = verify_file_exists("Permanent/memories.md")

        assert "VERIFIED" in result

    def test_verify_file_exists_shows_metadata(self, patched_accountability):
        """verify_file_exists shows file size and modification time."""
        test_file = patched_accountability / "test.md"
        test_file.write_text("Some content here")

        result = verify_file_exists(str(test_file))

        assert "VERIFIED" in result
        assert "bytes" in result.lower()
        assert "modified" in result.lower()


class TestVerifyFileModifiedRecently:
    """Tests for verify_file_modified_recently function."""

    def test_verify_modified_recently_true(self, patched_accountability):
        """verify_file_modified_recently VERIFIED for fresh file."""
        test_file = patched_accountability / "fresh.md"
        test_file.write_text("just created")

        result = verify_file_modified_recently(str(test_file), minutes=5)

        assert "VERIFIED" in result

    def test_verify_modified_recently_not_found(self, patched_accountability):
        """verify_file_modified_recently NOT FOUND for missing file."""
        result = verify_file_modified_recently(
            str(patched_accountability / "nonexistent.md"), minutes=5
        )

        assert "NOT FOUND" in result

    def test_verify_modified_shows_time(self, patched_accountability):
        """verify_file_modified_recently shows modification time."""
        test_file = patched_accountability / "recent.md"
        test_file.write_text("content")

        result = verify_file_modified_recently(str(test_file), minutes=5)

        assert "VERIFIED" in result
        # Should mention how recently it was modified
        assert "seconds" in result.lower() or "modified" in result.lower()


class TestVerifyContentContains:
    """Tests for verify_content_contains function."""

    def test_verify_content_contains_true(self, patched_accountability):
        """verify_content_contains VERIFIED when text found."""
        test_file = patched_accountability / "content.md"
        test_file.write_text("The quick brown fox jumps over the lazy dog")

        result = verify_content_contains(str(test_file), "brown fox")

        assert "VERIFIED" in result
        assert "contains" in result.lower()

    def test_verify_content_contains_false(self, patched_accountability):
        """verify_content_contains NOT FOUND when missing."""
        test_file = patched_accountability / "content.md"
        test_file.write_text("Hello World")

        result = verify_content_contains(str(test_file), "Python")

        assert "NOT FOUND" in result
        assert "does not contain" in result.lower()

    def test_verify_content_file_not_found(self, patched_accountability):
        """verify_content_contains NOT FOUND for missing file."""
        result = verify_content_contains(str(patched_accountability / "nonexistent.md"), "anything")

        assert "NOT FOUND" in result

    def test_verify_content_truncates_long_text(self, patched_accountability):
        """verify_content_contains truncates long search text in output."""
        test_file = patched_accountability / "long.md"
        long_text = "A" * 100
        test_file.write_text(long_text)

        result = verify_content_contains(str(test_file), long_text)

        assert "VERIFIED" in result
        # Should truncate to 50 chars with ...
        assert "..." in result


class TestVerifyMemorySaved:
    """Tests for verify_memory_saved function."""

    def test_verify_memory_saved_exists(self, patched_accountability):
        """verify_memory_saved checks permanent memory files."""
        # Create a memory file
        memory_file = patched_accountability / "Permanent" / "test_category.md"
        memory_file.write_text("Some saved memory")

        result = verify_memory_saved("test_category")

        assert "VERIFIED" in result or "EXISTS" in result

    def test_verify_memory_saved_not_found(self, patched_accountability):
        """verify_memory_saved NOT FOUND for missing category."""
        result = verify_memory_saved("nonexistent_category")

        assert "NOT FOUND" in result

    def test_verify_memory_saved_shows_size(self, patched_accountability):
        """verify_memory_saved shows file size."""
        memory_file = patched_accountability / "Permanent" / "with_size.md"
        memory_file.write_text("Memory content here")

        result = verify_memory_saved("with_size")

        # Should mention size
        assert "bytes" in result.lower()


class TestVerifyGitClean:
    """Tests for verify_git_clean function."""

    def test_verify_git_clean_clean_repo(self, patched_accountability):
        """verify_git_clean returns VERIFIED for clean repository."""
        # Mock git status to return empty output (clean)
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            # First subprocess.run is the availability probe (TK-1119),
            # second is the rev-parse with check=True, third is the
            # status check.  All succeed for a clean repo.
            mock_check = MagicMock()
            mock_check.returncode = 0
            mock_check.stdout = ""
            mock_check.stderr = ""

            mock_rev_parse = MagicMock()
            mock_rev_parse.returncode = 0
            mock_rev_parse.stdout = ""

            mock_status = MagicMock()
            mock_status.returncode = 0
            mock_status.stdout = ""

            mock_run.side_effect = [mock_check, mock_rev_parse, mock_status]

            result = verify_git_clean()

            assert "VERIFIED" in result
            assert "clean" in result.lower()

    def test_verify_git_clean_dirty_repo(self, patched_accountability):
        """verify_git_clean raises GitDirtyError for repository with changes."""
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            mock_check = MagicMock()
            mock_check.returncode = 0
            mock_check.stdout = ""
            mock_check.stderr = ""

            mock_rev_parse = MagicMock()
            mock_rev_parse.returncode = 0
            mock_rev_parse.stdout = ""

            # Mock git status to return output (dirty working directory)
            mock_status = MagicMock()
            mock_status.returncode = 0
            mock_status.stdout = " M file1.py\n D file2.md\n"

            mock_run.side_effect = [mock_check, mock_rev_parse, mock_status]

            # Test that GitDirtyError is raised
            with pytest.raises(GitDirtyError):
                verify_git_clean()

    def test_verify_git_clean_no_git(self, patched_accountability):
        """verify_git_clean returns NOT FOUND when not in a git repository."""
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            # First call: availability probe with check=False — non-zero
            # but stderr does NOT mention "not found", so we don't raise
            # GitNotInstalledError and fall through to the second call.
            mock_check = MagicMock()
            mock_check.returncode = 1
            mock_check.stdout = ""
            mock_check.stderr = "fatal: not a git repository (or any of the parent directories)"

            # Second call: rev-parse with check=True raises CalledProcessError
            # whose stderr triggers the "NOT FOUND" branch.
            err = subprocess.CalledProcessError(
                returncode=128,
                cmd=["git", "rev-parse", "--git-dir"],
                stderr="fatal: not a git repository (or any of the parent directories)",
            )
            mock_run.side_effect = [mock_check, err]

            result = verify_git_clean()

            assert "NOT FOUND" in result
            assert "git" in result.lower()

    def test_verify_git_clean_file_not_found_error(self, patched_accountability):
        """verify_git_clean handles FileNotFoundError when git command not found."""
        import subprocess
        from unittest.mock import patch
        
        with patch('subprocess.run') as mock_run:
            # Mock subprocess.run to raise FileNotFoundError
            mock_run.side_effect = FileNotFoundError("git command not found")
            
            # Test that GitNotInstalledError is raised
            with pytest.raises(GitNotInstalledError):
                verify_git_clean()

    def test_verify_git_clean_raises_git_not_installed_error(self, patched_accountability):
        """verify_git_clean raises GitNotInstalledError when git is not in PATH."""
        import subprocess
        from unittest.mock import patch
        
        with patch('subprocess.run') as mock_run:
            # Mock subprocess.run to raise FileNotFoundError (git not found)
            mock_run.side_effect = FileNotFoundError("git command not found")
            
            # Test that GitNotInstalledError is raised
            with pytest.raises(GitNotInstalledError):
                verify_git_clean()

    def test_verify_git_clean_not_in_repo(self, patched_accountability):
        """verify_git_clean returns NOT FOUND when not in a git repository."""
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            mock_check = MagicMock()
            mock_check.returncode = 1
            mock_check.stdout = ""
            mock_check.stderr = "fatal: not a git repository (or any of the parent directories)"

            err = subprocess.CalledProcessError(
                returncode=128,
                cmd=["git", "rev-parse", "--git-dir"],
                stderr="fatal: not a git repository (or any of the parent directories)",
            )
            mock_run.side_effect = [mock_check, err]

            result = verify_git_clean()

            assert "NOT FOUND" in result
            assert "repository" in result.lower()

    def test_verify_git_clean_non_zero_exit_code(self, patched_accountability):
        """verify_git_clean raises GitNotInstalledError when git is not in PATH."""
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            # Availability probe returns non-zero AND stderr mentions
            # "not found" → triggers the explicit GitNotInstalledError
            # raise before any subsequent subprocess call.
            mock_check = MagicMock()
            mock_check.returncode = 1
            mock_check.stdout = ""
            mock_check.stderr = "git: not found"

            mock_run.side_effect = [mock_check]

            # Test that GitNotInstalledError is raised
            with pytest.raises(GitNotInstalledError):
                verify_git_clean()

    def test_verify_git_clean_dirty_raises_error(self, patched_accountability):
        """verify_git_clean raises GitDirtyError for repository with uncommitted changes."""
        import subprocess
        from unittest.mock import patch

        with patch('subprocess.run') as mock_run:
            mock_check = MagicMock()
            mock_check.returncode = 0
            mock_check.stdout = ""
            mock_check.stderr = ""

            mock_rev_parse = MagicMock()
            mock_rev_parse.returncode = 0
            mock_rev_parse.stdout = ""

            # Mock git status to return output (dirty working directory)
            mock_status = MagicMock()
            mock_status.returncode = 0
            mock_status.stdout = " M file1.py\n D file2.md\n"

            mock_run.side_effect = [mock_check, mock_rev_parse, mock_status]

            # Test that GitDirtyError is raised
            with pytest.raises(GitDirtyError):
                verify_git_clean()


class TestGetAccountabilityTools:
    """Tests for get_accountability_tools function."""

    def test_returns_tool_list(self):
        """get_accountability_tools returns list of Tool objects."""
        tools = get_accountability_tools()

        assert len(tools) >= 5
        tool_names = [t.name for t in tools]
        assert "verify_file_exists" in tool_names
        assert "verify_file_modified" in tool_names
        assert "verify_content" in tool_names
        assert "verify_memory_saved" in tool_names
        assert "verify_readme_writable" in tool_names

    def test_tools_have_descriptions(self):
        """All tools have descriptions."""
        tools = get_accountability_tools()

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10

    def test_tools_have_parameters(self):
        """All tools have parameter schemas."""
        tools = get_accountability_tools()

        for tool in tools:
            assert "type" in tool.parameters
            assert "properties" in tool.parameters


class TestVerifyReadmeWritable:
    """Tests for verify_readme_writable function."""

    def test_verify_readme_writable_exists_and_writable(self, tmp_path, monkeypatch):
        """verify_readme_writable VERIFIED when README.md exists and is writable."""
        # Create a temporary README.md at the project root
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# Test\n")

        # Monkeypatch the project root path used by verify_readme_writable
        from agent import accountability
        monkeypatch.setattr(accountability, "__file__", str(tmp_path / "agent" / "accountability.py"))

        result = verify_readme_writable()

        assert "VERIFIED" in result
        assert "README.md" in result
        assert "writable" in result.lower()

    def test_verify_readme_writable_not_found(self, tmp_path, monkeypatch):
        """verify_readme_writable NOT FOUND when README.md does not exist."""
        # Monkeypatch the project root path used by verify_readme_writable
        from agent import accountability
        monkeypatch.setattr(accountability, "__file__", str(tmp_path / "agent" / "accountability.py"))

        result = verify_readme_writable()

        assert "NOT FOUND" in result
        assert "README.md" in result

    def test_verify_readme_writable_permission_denied(self, tmp_path, monkeypatch):
        """verify_readme_writable ERROR when README.md exists but is not writable."""
        # Create a temporary README.md at the project root
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# Test\n")

        # Monkeypatch the project root path used by verify_readme_writable
        from agent import accountability
        monkeypatch.setattr(accountability, "__file__", str(tmp_path / "agent" / "accountability.py"))

        # Make the file read-only
        readme_path.chmod(0o444)

        result = verify_readme_writable()

        assert "ERROR" in result
        assert "permission" in result.lower()

    def test_verify_readme_writable_preserves_content(self, tmp_path, monkeypatch):
        """verify_readme_writable preserves original content after test."""
        # Create a temporary README.md with original content at the project root
        original_content = "# Technomancer\n\nThis is the original README.\n"
        readme_path = tmp_path / "README.md"
        readme_path.write_text(original_content)

        # Monkeypatch the project root path used by verify_readme_writable
        from agent import accountability
        monkeypatch.setattr(accountability, "__file__", str(tmp_path / "agent" / "accountability.py"))

        result = verify_readme_writable()

        # Verify the function succeeded
        assert "VERIFIED" in result

        # Verify the original content is preserved
        final_content = readme_path.read_text()
        assert original_content in final_content
        assert "[TK-1217] Canary check - writable" not in final_content
