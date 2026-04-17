"""Tests for the pdf_tools module — RETIRED 2026-04-17.

The pdf_tools module was gutted (PDF handling moved to claude -p
directly). Every test in this file is skipped; the file is preserved
only so the test-discovery path doesn't break and the history of the
now-removed assertions stays readable.

The flaky test_no_parameters_returns_error that blocked 4+ TK stories
on 2026-04-17 lived here — removing the entire test surface made the
flake moot.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="pdf_tools retired 2026-04-17 — PDF handling moved to claude -p"
)

import io  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from agent.pdf_tools import (  # noqa: E402
    extract_text_from_pdf,
    get_pdf_tools,
    summarize_pdf,
)


class TestExtractTextFromPdf:
    """Test PDF text extraction."""

    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    def test_no_parameters_returns_error(self):
        result = extract_text_from_pdf()
        assert "Error" in result
        assert "Must provide" in result

    @patch("PyPDF2.PdfReader")
    def test_extracts_from_bytes(self, mock_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(data=b"%PDF-fake")
        assert "Page 1 content" in result

    @patch("PyPDF2.PdfReader")
    def test_multiple_pages(self, mock_reader_cls):
        pages = []
        for i in range(3):
            p = MagicMock()
            p.extract_text.return_value = f"Page {i+1}"
            pages.append(p)
        mock_reader = MagicMock()
        mock_reader.pages = pages
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(data=b"%PDF-fake")
        assert "Page 1" in result
        assert "Page 2" in result
        assert "Page 3" in result

    @patch("PyPDF2.PdfReader")
    def test_empty_pdf_returns_error(self, mock_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(data=b"%PDF-fake")
        assert "Error" in result or "Could not extract" in result

    def test_nonexistent_file(self):
        result = extract_text_from_pdf(file_path="/nonexistent/file.pdf")
        assert "Error" in result

    def test_corrupt_pdf(self):
        result = extract_text_from_pdf(data=b"not a real pdf at all")
        assert "Error" in result

    @patch("requests.get")
    @patch("PyPDF2.PdfReader")
    def test_url_download(self, mock_reader_cls, mock_get):
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-fake"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Downloaded content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(url="https://example.com/doc.pdf")
        assert "Downloaded content" in result


class TestSummarizePdf:
    """Test PDF summarization prompt generation."""

    @patch("agent.pdf_tools.extract_text_from_pdf")
    def test_concise_summary(self, mock_extract):
        mock_extract.return_value = "This is a long document about Python. " * 20
        result = summarize_pdf(data=b"fake", summary_type="concise")
        assert "concise summary" in result.lower() or "summarize" in result.lower()

    @patch("agent.pdf_tools.extract_text_from_pdf")
    def test_bullet_points(self, mock_extract):
        mock_extract.return_value = "Content about machine learning. " * 20
        result = summarize_pdf(data=b"fake", summary_type="bullet_points")
        assert "bullet" in result.lower()

    @patch("agent.pdf_tools.extract_text_from_pdf")
    def test_detailed_summary(self, mock_extract):
        mock_extract.return_value = "Detailed research paper content. " * 20
        result = summarize_pdf(data=b"fake", summary_type="detailed")
        assert "detailed" in result.lower()

    @patch("agent.pdf_tools.extract_text_from_pdf")
    def test_extraction_error_passed_through(self, mock_extract):
        mock_extract.return_value = "Error: Could not read PDF"
        result = summarize_pdf(data=b"fake")
        assert "Error" in result

    @patch("agent.pdf_tools.extract_text_from_pdf")
    def test_short_text_rejected(self, mock_extract):
        mock_extract.return_value = "Short"
        result = summarize_pdf(data=b"fake")
        assert "too short" in result.lower() or "Error" in result


class TestGetPdfTools:
    """Test tool registration."""

    def test_returns_tools(self):
        tools = get_pdf_tools()
        assert len(tools) >= 2
        names = {t.name for t in tools}
        assert "summarize_pdf" in names or "extract_pdf_text" in names
