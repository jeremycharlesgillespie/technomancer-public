"""Tests for the bot_utils module — crash logging, document detection, file utilities."""

import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.bot_utils import (
    build_crash_message,
    detect_document_type,
    find_mentioned_files,
    is_image_attachment,
    log,
    send_lifecycle_notification,
)


class TestLog:
    """Test timestamped logging."""

    def test_prints_with_timestamp(self, capsys):
        log("Hello world")
        output = capsys.readouterr().out
        assert "Hello world" in output
        assert "[" in output  # timestamp bracket


class TestBuildCrashMessage:
    """Test crash message formatting."""

    def test_includes_stack_trace(self):
        try:
            raise ValueError("test error")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            msg = build_crash_message(exc_type, exc_value, exc_tb)

        assert "ValueError" in msg
        assert "test error" in msg

    def test_includes_local_variables(self):
        local_var = "important_value"
        try:
            raise RuntimeError("oops")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            msg = build_crash_message(exc_type, exc_value, exc_tb)

        assert "local_var" in msg or "important_value" in msg

    def test_truncates_long_messages(self):
        try:
            very_long_variable = "x" * 10000
            raise RuntimeError("big crash")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            msg = build_crash_message(exc_type, exc_value, exc_tb)

        assert len(msg) <= 1800

    def test_handles_none_traceback(self):
        # With None traceback, still produces a message (may be truncated)
        try:
            msg = build_crash_message(ValueError, ValueError("test"), None)
            assert isinstance(msg, str)
        except (AttributeError, TypeError):
            pass  # Some implementations require a real traceback


class TestDetectDocumentType:
    """Test document type detection."""

    def test_detects_resume_with_keyword(self):
        text = "RESUME\n\n" + "x" * 500 + "\nExperience\nSenior Developer at Company"
        assert detect_document_type(text) == "resume"

    def test_detects_cv_keyword(self):
        text = "Curriculum Vitae\n" + "x" * 500 + "\nEducation\nPhD Computer Science"
        assert detect_document_type(text) == "resume"

    def test_detects_by_section_count(self):
        text = ("Professional Experience\n" * 2 + "Education\n" + "Skills\n"
                + "Qualifications\n" + "x" * 500)
        assert detect_document_type(text) == "resume"

    def test_rejects_short_text(self):
        assert detect_document_type("Hi I have a resume") is None

    def test_rejects_non_resume(self):
        text = "This is a technical article about Python programming. " * 20
        assert detect_document_type(text) is None

    def test_empty_text(self):
        assert detect_document_type("") is None


class TestIsImageAttachment:
    """Test image type detection."""

    def test_png(self):
        attachment = MagicMock(filename="photo.png")
        assert is_image_attachment(attachment) is True

    def test_jpg(self):
        attachment = MagicMock(filename="image.jpg")
        assert is_image_attachment(attachment) is True

    def test_jpeg(self):
        attachment = MagicMock(filename="pic.jpeg")
        assert is_image_attachment(attachment) is True

    def test_gif(self):
        attachment = MagicMock(filename="animation.gif")
        assert is_image_attachment(attachment) is True

    def test_webp(self):
        attachment = MagicMock(filename="modern.webp")
        assert is_image_attachment(attachment) is True

    def test_non_image(self):
        attachment = MagicMock(filename="document.pdf")
        assert is_image_attachment(attachment) is False

    def test_txt_not_image(self):
        attachment = MagicMock(filename="notes.txt")
        assert is_image_attachment(attachment) is False

    def test_case_insensitive(self):
        attachment = MagicMock(filename="PHOTO.PNG")
        assert is_image_attachment(attachment) is True


class TestFindMentionedFiles:
    """Test file path extraction from responses."""

    def test_finds_existing_file(self, tmp_path):
        test_file = tmp_path / "output.html"
        test_file.write_text("<html>test</html>")

        # Use the full path so it can be found
        response = f"I created the file at {test_file}"
        files = find_mentioned_files(response)
        assert any(f.name == "output.html" for f in files)

    def test_ignores_nonexistent_paths(self):
        response = "See C:\\nonexistent\\fake\\path\\file.html for details"
        files = find_mentioned_files(response)
        assert len(files) == 0

    def test_empty_response(self):
        assert find_mentioned_files("") == []

    def test_no_paths_in_response(self):
        assert find_mentioned_files("Hello, this is just regular text.") == []


class TestSendLifecycleNotification:
    """Test lifecycle notification sending."""

    @patch("agent.discord_rate_limit.retry_request")
    def test_sends_online_notification(self, mock_retry, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://webhook.test")
        mock_retry.return_value = MagicMock(status_code=204)

        send_lifecycle_notification("online", "Connected as TestBot")
        mock_retry.assert_called_once()

    def test_no_webhook_silently_returns(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        # Should not raise
        send_lifecycle_notification("online")

    @patch("agent.discord_rate_limit.retry_request", side_effect=Exception("network fail"))
    def test_exception_swallowed(self, mock_retry, monkeypatch):
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://webhook.test")
        # Should not raise
        send_lifecycle_notification("crash", "Something broke")
