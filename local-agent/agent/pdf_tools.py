"""
PDF Tools - Extract and summarize PDF files.

Supports:
- Local PDF files by path
- PDF URLs (downloads and processes)
- Summarization via LLM
"""

import io
from pathlib import Path

import requests


def extract_text_from_pdf(file_path: str = None, url: str = None, data: bytes = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to local PDF file
        url: URL to download PDF from
        data: Raw PDF bytes (from Discord attachment)

    Returns:
        Extracted text content or error message
    """
    try:
        from PyPDF2 import PdfReader

        # Determine source
        if data is not None:
            # Already have bytes
            reader = PdfReader(io.BytesIO(data))
        elif url is not None:
            # Download from URL
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
        elif file_path is not None:
            # Read from local file
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: File not found: {file_path}"
            if not p.is_file():
                return f"Error: Not a file: {file_path}"
            reader = PdfReader(str(p))
        else:
            return "Error: Must provide file_path, url, or data"

        # Extract text from all pages
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            return "Error: Could not extract text from PDF (may be scan/image-based)"

        return full_text

    except ImportError:
        return "Error: PyPDF2 not installed. Install with: pip install PyPDF2"
    except requests.exceptions.Timeout:
        return "Error: Download timed out"
    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {e}"
    except Exception as e:
        return f"Error processing PDF: {e}"


def summarize_pdf(
    file_path: str = None,
    url: str = None,
    data: bytes = None,
    summary_type: str = "concise",
    max_length: int = 500,
) -> str:
    """
    Extract and summarize a PDF file.

    Args:
        file_path: Path to local PDF file
        url: URL to download PDF from
        data: Raw PDF bytes
        summary_type: "concise", "detailed", or "bullet_points"
        max_length: Maximum characters in summary (for concise mode)

    Returns:
        Summary of the PDF content
    """
    # Extract text first
    text = extract_text_from_pdf(file_path=file_path, url=url, data=data)

    if text.startswith("Error:"):
        return text

    if len(text) < 100:
        return f"PDF too short to summarize meaningfully.\n\nContent:\n{text}"

    # Prepare summary prompt based on type
    if summary_type == "bullet_points":
        prompt = f"""Please summarize the following PDF content as bullet points. Focus on key takeaways, main arguments, and important details. Keep it scannable and organized:

{text[:15000]}

Provide a clear, well-structured bullet point summary:"""

    elif summary_type == "detailed":
        prompt = f"""Please provide a detailed summary of the following PDF content. Include:
- Main topic/subject
- Key points and arguments
- Important findings or conclusions
- Any notable data, statistics, or examples

Content:
{text[:20000]}

Detailed summary:"""

    else:  # concise (default)
        prompt = f"""Please provide a concise summary of the following PDF content in about {max_length} characters. Capture the main idea and key points:

{text[:15000]}

Summary:"""

    return prompt


def get_pdf_page_count(file_path: str = None, url: str = None, data: bytes = None) -> str:
    """
    Get the number of pages in a PDF.

    Args:
        file_path: Path to local PDF file
        url: URL to download PDF from
        data: Raw PDF bytes

    Returns:
        Number of pages or error message
    """
    try:
        from PyPDF2 import PdfReader

        if data is not None:
            reader = PdfReader(io.BytesIO(data))
        elif url is not None:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
        elif file_path is not None:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: File not found: {file_path}"
            reader = PdfReader(str(p))
        else:
            return "Error: Must provide file_path, url, or data"

        return f"PDF has {len(reader.pages)} page(s)"

    except ImportError:
        return "Error: PyPDF2 not installed"
    except Exception as e:
        return f"Error: {e}"


def get_pdf_metadata(file_path: str = None, url: str = None, data: bytes = None) -> str:
    """
    Get metadata from a PDF file.

    Args:
        file_path: Path to local PDF file
        url: URL to download PDF from
        data: Raw PDF bytes

    Returns:
        PDF metadata as formatted string
    """
    try:
        from PyPDF2 import PdfReader

        if data is not None:
            reader = PdfReader(io.BytesIO(data))
        elif url is not None:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
        elif file_path is not None:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: File not found: {file_path}"
            reader = PdfReader(str(p))
        else:
            return "Error: Must provide file_path, url, or data"

        metadata = reader.metadata
        result = ["PDF Metadata:"]

        if metadata:
            for key in [
                "title",
                "author",
                "subject",
                "creator",
                "producer",
                "creationDate",
                "modDate",
            ]:
                value = getattr(metadata, key, None)
                if value:
                    result.append(f"  {key}: {value}")
        else:
            result.append("  (No metadata found)")

        result.append(f"  Pages: {len(reader.pages)}")

        return "\n".join(result)

    except ImportError:
        return "Error: PyPDF2 not installed"
    except Exception as e:
        return f"Error: {e}"


# Tool registry for the agent
from .core import Tool, create_tool  # noqa: E402


def get_pdf_tools() -> list[Tool]:
    """Get PDF tools for the agent."""
    return [
        create_tool(
            name="summarize_pdf",
            description=(
                "Summarize a PDF file. Can process local files, URLs, or uploaded attachments. "
                "Supports concise summaries, detailed analysis, and bullet point formats. "
                "Use this when users want to quickly understand PDF content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to local PDF file"},
                    "url": {
                        "type": "string",
                        "description": "URL to download PDF from (e.g., https://example.com/doc.pdf)",
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["concise", "detailed", "bullet_points"],
                        "description": "Type of summary: concise (default), detailed, or bullet_points",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters for concise summaries (default 500)",
                    },
                },
                "required": [],
            },
            function=summarize_pdf,
        ),
        create_tool(
            name="extract_pdf_text",
            description=(
                "Extract all text content from a PDF file. Returns the raw extracted text. "
                "Use this when you need to read or analyze the full PDF content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to local PDF file"},
                    "url": {"type": "string", "description": "URL to download PDF from"},
                },
                "required": [],
            },
            function=extract_text_from_pdf,
        ),
        create_tool(
            name="get_pdf_info",
            description=(
                "Get information about a PDF file including page count and metadata. "
                "Useful for quickly checking what's in a PDF before processing it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to local PDF file"},
                    "url": {"type": "string", "description": "URL to download PDF from"},
                },
                "required": [],
            },
            function=get_pdf_metadata,
        ),
    ]
