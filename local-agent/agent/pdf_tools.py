"""
PDF Tools — REMOVED 2026-04-17.

The bespoke PyPDF2-based extractor + summarizer was retired. Every PDF
path now goes through ``claude -p`` directly — Claude Code reads the
attached PDF and produces extracted text or summary inline, which
removes:

- The PyPDF2 deprecation warning (library is in maintenance mode).
- The test_pdf_tools_extended.test_no_parameters_returns_error flake
  that blocked 4+ TK stories on 2026-04-17.
- Duplicate extraction logic that rarely matched Claude's output
  quality anyway.

This module stub stays only so stale imports don't break. Every symbol
is a no-op that raises a clear error explaining the new path.

If you landed here looking for PDF handling, use::

    from .claude_code_runner import run_claude_prompt
    result = run_claude_prompt(
        f"Extract the text from this PDF: {path_or_url}",
        timeout=120, max_turns=3,
    )
"""
from __future__ import annotations


class PdfToolsRemoved(RuntimeError):
    """Raised when someone calls a retired pdf_tools function."""


def extract_text_from_pdf(*args, **kwargs):  # noqa: ARG001 — retired shim
    raise PdfToolsRemoved(
        "extract_text_from_pdf was removed. Use claude -p via "
        "agent.claude_code_runner.run_claude_prompt(...) for PDF extraction."
    )


def summarize_pdf(*args, **kwargs):  # noqa: ARG001 — retired shim
    raise PdfToolsRemoved(
        "summarize_pdf was removed. Use claude -p via "
        "agent.claude_code_runner.run_claude_prompt(...) for PDF summarization."
    )


def get_pdf_tools() -> list:
    """Return an empty tool list — PDF handling is no longer a registered tool."""
    return []
