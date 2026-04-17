"""Tests for agent/ollama_warmup.py — startup model preloading."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from agent import ollama_warmup
from agent.ollama_warmup import warmup_models


def _run(coro):
    """Run an async coroutine to completion in a sync test."""
    return asyncio.run(coro)


class TestWarmupModels:
    def test_calls_generate_once_per_model(self):
        """One ``_ollama_client.generate`` call per configured model."""
        mock_client = MagicMock()
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models(["model-a", "model-b"]))

        assert mock_client.generate.call_count == 2
        models_called = [
            kwargs.get("model") or args[0]
            for args, kwargs in (
                (c.args, c.kwargs) for c in mock_client.generate.call_args_list
            )
        ]
        assert models_called == ["model-a", "model-b"]

    def test_generate_uses_one_token_budget(self):
        """Warmup must use num_predict=1 — no need to burn VRAM on real output."""
        mock_client = MagicMock()
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models(["model-a"]))

        _, kwargs = mock_client.generate.call_args
        assert kwargs["model"] == "model-a"
        assert kwargs["options"] == {"num_predict": 1}

    def test_exceptions_are_caught_not_raised(self):
        """A failing model must not break the warmup or propagate."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("model not pulled")
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            # Must not raise
            _run(warmup_models(["missing-model"]))

        assert mock_client.generate.call_count == 1

    def test_one_model_failing_does_not_stop_others(self):
        """Independent per-model failure: a broken model can't poison the list."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = [
            RuntimeError("boom"),
            MagicMock(),  # second call succeeds
        ]
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models(["broken", "works"]))

        # Both models attempted despite the first failure.
        assert mock_client.generate.call_count == 2

    def test_exception_is_logged_as_warning(self, caplog):
        """Failures should surface in logs so operators know warmup didn't fire."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("model not pulled")
        with caplog.at_level(logging.WARNING, logger="agent.ollama_warmup"):
            with patch.object(ollama_warmup, "_ollama_client", mock_client):
                _run(warmup_models(["missing"]))

        assert any(
            "warmup failed" in rec.message.lower() and "missing" in rec.message
            for rec in caplog.records
        )

    def test_deduplicates_model_names(self):
        """Same model listed twice (e.g. chat == vision) should warm once."""
        mock_client = MagicMock()
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models(["same", "same"]))

        assert mock_client.generate.call_count == 1

    def test_skips_empty_strings(self):
        """Empty / None-ish names (e.g. unset ollama_fast_model) are skipped."""
        mock_client = MagicMock()
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models(["", "real-model", ""]))

        assert mock_client.generate.call_count == 1
        _, kwargs = mock_client.generate.call_args
        assert kwargs["model"] == "real-model"

    def test_empty_list_is_noop(self):
        """Warmup with no models should not raise or call anything."""
        mock_client = MagicMock()
        with patch.object(ollama_warmup, "_ollama_client", mock_client):
            _run(warmup_models([]))

        assert mock_client.generate.call_count == 0

    def test_success_logged_with_timing(self, caplog):
        """Successful warmup logs should include the model name."""
        mock_client = MagicMock()
        with caplog.at_level(logging.INFO, logger="agent.ollama_warmup"):
            with patch.object(ollama_warmup, "_ollama_client", mock_client):
                _run(warmup_models(["fast-model"]))

        assert any(
            "fast-model" in rec.message and "ready" in rec.message.lower()
            for rec in caplog.records
        )
