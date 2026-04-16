"""Tests for generate_readme.py idempotent-write behavior."""

from unittest.mock import patch

import generate_readme


class TestWriteIfChanged:
    """Cover the skip-when-equal branch that keeps the tree clean."""

    def test_writes_when_file_missing(self, tmp_path):
        path = tmp_path / "README.md"
        assert generate_readme.write_if_changed(path, "hello") is True
        assert path.read_text(encoding="utf-8") == "hello"

    def test_skips_when_content_matches(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text("same content", encoding="utf-8")
        original_mtime = path.stat().st_mtime_ns

        assert generate_readme.write_if_changed(path, "same content") is False
        # File must not be touched — mtime stays identical so git sees no change.
        assert path.stat().st_mtime_ns == original_mtime
        assert path.read_text(encoding="utf-8") == "same content"

    def test_writes_when_content_differs(self, tmp_path):
        path = tmp_path / "README.md"
        path.write_text("old", encoding="utf-8")
        assert generate_readme.write_if_changed(path, "new") is True
        assert path.read_text(encoding="utf-8") == "new"


class TestGeneratedReadmeHasNoTimestamp:
    """Per-run timestamps in the body defeat the idempotency check."""

    def test_body_has_no_last_updated_line(self):
        test_stats = {"test_count": 42, "coverage_pct": 80.0}
        code_stats = {
            "module_count": 10,
            "total_lines": 1234,
            "test_file_count": 5,
            "categories": {"Core": ["core"]},
        }
        readme = generate_readme.generate_readme(test_stats, code_stats)
        assert "Last updated" not in readme

    def test_two_calls_produce_identical_output(self):
        """Stable output between calls — required for the skip check."""
        test_stats = {"test_count": 42, "coverage_pct": 80.0}
        code_stats = {
            "module_count": 10,
            "total_lines": 1234,
            "test_file_count": 5,
            "categories": {"Core": ["core"]},
        }
        first = generate_readme.generate_readme(test_stats, code_stats)
        second = generate_readme.generate_readme(test_stats, code_stats)
        assert first == second


class TestMainIdempotent:
    """End-to-end: two consecutive main() calls must leave the file untouched.

    This is the real acceptance criterion from the epic — the working tree
    must stay clean between runs when no stats have changed, so the next
    worker's 'clean main' safety check doesn't trip.
    """

    def test_second_main_call_does_not_touch_files(self, tmp_path):
        stats_test = {"test_count": 100, "coverage_pct": 75.0}
        stats_code = {
            "module_count": 20,
            "total_lines": 5000,
            "test_file_count": 15,
            "categories": {"Core": ["core"]},
        }

        local_readme = tmp_path / "README.md"
        root_readme = tmp_path / "parent_README.md"

        with patch.object(generate_readme, "README_PATH", local_readme), \
             patch.object(generate_readme, "ROOT_README_PATH", root_readme), \
             patch.object(generate_readme, "collect_test_stats", return_value=stats_test), \
             patch.object(generate_readme, "collect_code_stats", return_value=stats_code), \
             patch.object(generate_readme.sys, "argv", ["generate_readme.py"]):
            generate_readme.main()

            assert local_readme.exists()
            assert root_readme.exists()
            first_local_mtime = local_readme.stat().st_mtime_ns
            first_root_mtime = root_readme.stat().st_mtime_ns

            generate_readme.main()

            # If main() re-wrote either file, mtime would change — which is
            # exactly what git diff detects and what dirties the tree.
            assert local_readme.stat().st_mtime_ns == first_local_mtime
            assert root_readme.stat().st_mtime_ns == first_root_mtime
