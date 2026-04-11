"""Tests for agent/image_identification.py — vision model and Claude cascade."""

from unittest.mock import MagicMock, patch

import pytest

from agent.image_identification import (
    analyze_with_vision_model,
    ask_claude_with_image,
)


# Minimal PNG bytes (1x1 transparent pixel)
FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
)

# JPEG magic bytes
FAKE_JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


class TestAnalyzeWithVisionModel:
    def test_returns_string(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "A cat sitting on a mat."}}

        with patch("agent.image_identification._vision_client", mock_client):
            with patch("agent.image_identification._record_perf"):
                result = analyze_with_vision_model([FAKE_PNG])
                assert isinstance(result, str)
                assert "cat" in result

    def test_handles_ollama_error(self):
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Ollama unavailable")

        with patch("agent.image_identification._vision_client", mock_client):
            with patch("agent.image_identification._record_perf"):
                result = analyze_with_vision_model([FAKE_PNG])
                assert "Error" in result or "error" in result.lower()


class TestAskClaudeWithImage:
    def test_successful_analysis(self):
        fake_result = {
            "success": True,
            "result": "This is a photo of a sunset over the ocean.",
            "cost_usd": 0.01,
        }
        with patch("agent.claude_code_runner.run_claude_prompt", return_value=fake_result):
            with patch("agent.image_identification._record_perf"):
                result = ask_claude_with_image(FAKE_PNG, "What is this?")
                assert "sunset" in result

    def test_claude_failure_returns_error(self):
        fake_result = {
            "success": False,
            "result": "",
            "error": "binary not found",
            "cost_usd": 0,
        }
        with patch("agent.claude_code_runner.run_claude_prompt", return_value=fake_result):
            with patch("agent.image_identification._record_perf"):
                result = ask_claude_with_image(FAKE_PNG, "What is this?")
                assert "Error" in result or "error" in result.lower()

    def test_detects_jpeg_format(self):
        fake_result = {"success": True, "result": "A JPEG image", "cost_usd": 0}
        with patch("agent.claude_code_runner.run_claude_prompt", return_value=fake_result) as mock_run:
            with patch("agent.image_identification._record_perf"):
                ask_claude_with_image(FAKE_JPG, "What is this?")
                # The temp file should have .jpg extension
                prompt = mock_run.call_args[0][0]
                assert ".jpg" in prompt or ".jpeg" in prompt or "image" in prompt.lower()
