"""Tests for the GitHub Pages deployment module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from agent import github_pages


class TestEnsureDocsStructure:
    """Tests for ensure_docs_structure function."""

    def test_creates_directories(self, tmp_path):
        """Test that directories are created."""
        with patch.object(github_pages, "DOCS_PATH", tmp_path / "docs"):
            with patch.object(github_pages, "LEARNING_PATH", tmp_path / "docs" / "learning"):
                github_pages.ensure_docs_structure()

                assert (tmp_path / "docs").exists()
                assert (tmp_path / "docs" / "learning").exists()

    def test_creates_nojekyll(self, tmp_path):
        """Test that .nojekyll file is created."""
        docs_path = tmp_path / "docs"
        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", docs_path / "learning"):
                github_pages.ensure_docs_structure()

                assert (docs_path / ".nojekyll").exists()

    def test_creates_root_index(self, tmp_path):
        """Test that root index.html is created."""
        docs_path = tmp_path / "docs"
        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", docs_path / "learning"):
                github_pages.ensure_docs_structure()

                index_path = docs_path / "index.html"
                assert index_path.exists()
                content = index_path.read_text()
                assert "learning/index.html" in content


class TestSaveArticleHtml:
    """Tests for save_article_html function."""

    def test_saves_html_file(self, tmp_path):
        """Test that HTML file is saved."""
        docs_path = tmp_path / "docs"
        learning_path = docs_path / "learning"

        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", learning_path):
                file_path, url = github_pages.save_article_html(
                    topic="Test Topic",
                    category="python",
                    content="# Test\n\nSome content",
                    date=datetime(2026, 3, 15),
                )

                assert file_path.exists()
                assert file_path.suffix == ".html"
                assert "2026-03-15" in file_path.name
                assert "python" in file_path.name

    def test_returns_correct_url(self, tmp_path):
        """Test that correct GitHub Pages URL is returned."""
        docs_path = tmp_path / "docs"
        learning_path = docs_path / "learning"

        test_url = "https://test.github.io/technomancer"
        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", learning_path):
                with patch.object(github_pages, "GITHUB_PAGES_URL", test_url):
                    _, url = github_pages.save_article_html(
                        topic="Test Topic",
                        category="python",
                        content="Content",
                        date=datetime(2026, 3, 15),
                    )

                    assert url.startswith(f"{test_url}/learning/")
                    assert ".html" in url

    def test_regenerates_index(self, tmp_path):
        """Test that index is regenerated after saving."""
        docs_path = tmp_path / "docs"
        learning_path = docs_path / "learning"

        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", learning_path):
                github_pages.save_article_html(
                    topic="Test Topic",
                    category="python",
                    content="Content",
                )

                index_path = learning_path / "index.html"
                assert index_path.exists()
                content = index_path.read_text()
                assert "Test Topic" in content


class TestListArticleFiles:
    """Tests for list_article_files function."""

    def test_empty_directory(self, tmp_path):
        """Test with empty learning directory."""
        learning_path = tmp_path / "learning"
        learning_path.mkdir()

        with patch.object(github_pages, "LEARNING_PATH", learning_path):
            articles = github_pages.list_article_files()
            assert articles == []

    def test_lists_articles(self, tmp_path):
        """Test listing existing articles."""
        learning_path = tmp_path / "learning"
        learning_path.mkdir()

        # Create test files
        (learning_path / "2026-03-15_python_test-topic.html").write_text("<html></html>")
        (learning_path / "2026-03-14_oracle_sql-tips.html").write_text("<html></html>")
        (learning_path / "index.html").write_text("<html></html>")  # Should be excluded

        with patch.object(github_pages, "LEARNING_PATH", learning_path):
            articles = github_pages.list_article_files()

            assert len(articles) == 2
            # Should be sorted by date descending
            assert articles[0]["date"] == "March 15, 2026"
            assert articles[1]["date"] == "March 14, 2026"

    def test_excludes_index(self, tmp_path):
        """Test that index.html is excluded from listing."""
        learning_path = tmp_path / "learning"
        learning_path.mkdir()

        (learning_path / "index.html").write_text("<html></html>")

        with patch.object(github_pages, "LEARNING_PATH", learning_path):
            articles = github_pages.list_article_files()
            assert len(articles) == 0


class TestRegenerateIndex:
    """Tests for regenerate_index function."""

    def test_creates_index(self, tmp_path):
        """Test that index.html is created."""
        docs_path = tmp_path / "docs"
        learning_path = docs_path / "learning"

        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", learning_path):
                index_path = github_pages.regenerate_index()

                assert index_path.exists()
                assert "index.html" in str(index_path)

    def test_includes_articles(self, tmp_path):
        """Test that index includes article links."""
        docs_path = tmp_path / "docs"
        learning_path = docs_path / "learning"
        learning_path.mkdir(parents=True)

        # Create a test article
        (learning_path / "2026-03-15_python_decorators.html").write_text("<html></html>")

        with patch.object(github_pages, "DOCS_PATH", docs_path):
            with patch.object(github_pages, "LEARNING_PATH", learning_path):
                index_path = github_pages.regenerate_index()

                content = index_path.read_text()
                assert "2026-03-15_python_decorators.html" in content
                assert "Python" in content


class TestDeployToGithub:
    """Tests for deploy_to_github function."""

    def test_calls_git_commands(self, tmp_path):
        """Test that git commands are called."""
        with patch.object(github_pages, "PROJECT_ROOT", tmp_path):
            with patch("subprocess.run") as mock_run:
                # Mock successful git operations
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                github_pages.deploy_to_github("Test commit")

                # Verify git commands were called
                calls = mock_run.call_args_list
                assert any("add" in str(call) for call in calls)

    def test_returns_false_on_git_failure(self, tmp_path):
        """Test that False is returned on git failure."""
        with patch.object(github_pages, "PROJECT_ROOT", tmp_path):
            with patch("subprocess.run") as mock_run:
                # Mock failed git add
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

                result = github_pages.deploy_to_github("Test commit")

                assert result is False

    def test_handles_nothing_to_commit(self, tmp_path):
        """Test handling of 'nothing to commit' scenario."""
        with patch.object(github_pages, "PROJECT_ROOT", tmp_path):
            with patch("subprocess.run") as mock_run:
                # First call (git add) succeeds
                # Second call (git status) returns empty (no changes)
                mock_run.side_effect = [
                    MagicMock(returncode=0, stdout="", stderr=""),  # git add
                    MagicMock(returncode=0, stdout="", stderr=""),  # git status (empty)
                ]

                result = github_pages.deploy_to_github("Test commit")

                assert result is True


class TestGetArticleUrl:
    """Tests for get_article_url function."""

    def test_returns_correct_url(self):
        """Test that correct URL is returned."""
        test_url = "https://test.github.io/technomancer"
        with patch.object(github_pages, "GITHUB_PAGES_URL", test_url):
            url = github_pages.get_article_url("2026-03-15_python_test.html")
            assert url == f"{test_url}/learning/2026-03-15_python_test.html"
