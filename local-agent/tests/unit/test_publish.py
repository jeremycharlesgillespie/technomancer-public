"""Tests for publish.py commit-message filtering.

The public repo's commit history should surface real feature work, not
auto-generated stats/deploy/merge commits. These tests lock in the
filter that picks the first meaningful commit as the title.
"""

from unittest.mock import patch

import publish


class TestIsGenericCommit:
    """Cover every pattern the filter is supposed to drop."""

    def test_readme_stats_is_generic(self):
        assert publish._is_generic_commit("Update README with latest stats [auto]") is True

    def test_readme_stats_without_auto_suffix_is_generic(self):
        assert publish._is_generic_commit("Update README with latest stats") is True

    def test_timestamped_merge_is_generic(self):
        msg = "Merge branch '2026-04-15-143022-fix-bug' - automated safe_update"
        assert publish._is_generic_commit(msg) is True

    def test_auto_suffix_is_generic(self):
        assert publish._is_generic_commit("Bump deps [auto]") is True

    def test_tk_merge_is_generic(self):
        msg = "[TK-409] Merge branch '2026-04-15-221820-TK-409' - executor auto-deploy"
        assert publish._is_generic_commit(msg) is True

    def test_tk_deploy_stats_is_generic(self):
        assert publish._is_generic_commit("[TK-410] Deploy + update stats") is True

    def test_real_feature_commit_is_not_generic(self):
        msg = "[TK-409] Include story ID in auto-deploy commit messages"
        assert publish._is_generic_commit(msg) is False

    def test_plain_feature_commit_is_not_generic(self):
        assert publish._is_generic_commit("Fix executor merge losing work") is False

    def test_empty_string_is_not_generic(self):
        assert publish._is_generic_commit("") is False


class TestFilterMeaningfulCommits:
    """Filter should strip all generic commits while preserving order."""

    def test_filters_out_stats_and_merges_keeps_real_work(self):
        commits = [
            "Update README with latest stats [auto]",
            "[TK-401] Merge branch '2026-04-15-TK-401' - executor auto-deploy",
            "[TK-401] Exclude vetoed Jira issues",
        ]
        assert publish._filter_meaningful_commits(commits) == [
            "[TK-401] Exclude vetoed Jira issues"
        ]

    def test_preserves_order_of_multiple_real_commits(self):
        commits = [
            "Update README with latest stats [auto]",
            "[TK-500] Newer real work",
            "[TK-500] Deploy + update stats",
            "[TK-499] Older real work",
        ]
        assert publish._filter_meaningful_commits(commits) == [
            "[TK-500] Newer real work",
            "[TK-499] Older real work",
        ]

    def test_returns_empty_when_all_generic(self):
        commits = [
            "Update README with latest stats [auto]",
            "[TK-410] Deploy + update stats",
            "Merge branch '2026-04-15-foo' - automated safe_update",
        ]
        assert publish._filter_meaningful_commits(commits) == []

    def test_empty_input_returns_empty(self):
        assert publish._filter_meaningful_commits([]) == []


class TestBuildPublishMessage:
    """End-to-end behavior of the commit-message builder."""

    @patch("publish._get_recent_private_commits")
    def test_title_is_real_work_not_stats_or_merge(self, mock_commits):
        # Scenario from the acceptance criteria in TK-411.
        mock_commits.return_value = [
            "Update README with latest stats [auto]",
            "[TK-401] Merge branch '2026-04-15-TK-401' - executor auto-deploy",
            "[TK-401] Exclude vetoed Jira issues",
        ]
        msg = publish._build_publish_message(copied=10, deleted=0)
        # Title line
        assert msg.split("\n", 1)[0] == "[TK-401] Exclude vetoed Jira issues"

    @patch("publish._get_recent_private_commits")
    def test_body_bullets_exclude_generic_commits(self, mock_commits):
        mock_commits.return_value = [
            "Update README with latest stats [auto]",
            "[TK-500] Real feature A",
            "[TK-500] Deploy + update stats",
            "[TK-499] Real feature B",
        ]
        msg = publish._build_publish_message(copied=5, deleted=0)
        assert "Update README with latest stats" not in msg
        assert "Deploy + update stats" not in msg
        assert "- [TK-500] Real feature A" in msg
        assert "- [TK-499] Real feature B" in msg

    @patch("publish._get_recent_private_commits")
    def test_falls_back_when_all_filtered(self, mock_commits):
        mock_commits.return_value = [
            "Update README with latest stats [auto]",
            "[TK-410] Deploy + update stats",
        ]
        msg = publish._build_publish_message(copied=7, deleted=0)
        assert msg.startswith("Update (")
        assert "7 files synced" in msg

    @patch("publish._get_recent_private_commits")
    def test_falls_back_when_no_commits(self, mock_commits):
        mock_commits.return_value = []
        msg = publish._build_publish_message(copied=3, deleted=0)
        assert msg.startswith("Update (")
        assert "3 files synced" in msg

    @patch("publish._get_recent_private_commits")
    def test_single_real_commit_returns_just_title(self, mock_commits):
        mock_commits.return_value = [
            "Update README with latest stats [auto]",
            "[TK-123] Add retry logic",
        ]
        assert publish._build_publish_message(copied=1, deleted=0) == "[TK-123] Add retry logic"
