"""Tests for the HTML generator module."""

from agent.html_generator import (
    extract_body_content,
    extract_title,
    generate_article_html,
    generate_filename,
    generate_index_html,
    markdown_to_html,
    normalize_html,
    slugify,
)


class TestMarkdownToHtml:
    """Tests for markdown_to_html conversion."""

    def test_basic_paragraph(self):
        """Test basic text becomes a paragraph."""
        result = markdown_to_html("Hello world")
        assert "<p>Hello world</p>" in result

    def test_headers(self):
        """Test header conversion."""
        result = markdown_to_html("# H1\n\n## H2\n\n### H3")
        assert "<h1>H1</h1>" in result
        assert "<h2>H2</h2>" in result
        assert "<h3>H3</h3>" in result

    def test_bold_and_italic(self):
        """Test bold and italic formatting."""
        result = markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result

    def test_code_block_with_language(self):
        """Test code blocks with language hint."""
        md = "```python\nprint('hello')\n```"
        result = markdown_to_html(md)
        assert '<code class="language-python">' in result
        assert "print(&#x27;hello&#x27;)" in result  # HTML escaped

    def test_code_block_without_language(self):
        """Test code blocks without language hint."""
        md = "```\nsome code\n```"
        result = markdown_to_html(md)
        assert "<pre><code>" in result
        assert "some code" in result

    def test_inline_code(self):
        """Test inline code formatting."""
        result = markdown_to_html("Use `print()` function")
        assert "<code>print()</code>" in result

    def test_links(self):
        """Test link conversion."""
        result = markdown_to_html("[Google](https://google.com)")
        assert '<a href="https://google.com">Google</a>' in result

    def test_unordered_list(self):
        """Test unordered list conversion."""
        md = "- Item 1\n- Item 2\n- Item 3"
        result = markdown_to_html(md)
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result

    def test_ordered_list(self):
        """Test ordered list conversion."""
        md = "1. First\n2. Second\n3. Third"
        result = markdown_to_html(md)
        assert "<ol>" in result
        assert "<li>First</li>" in result

    def test_blockquote(self):
        """Test blockquote conversion."""
        result = markdown_to_html("> This is a quote")
        assert "<blockquote><p>This is a quote</p></blockquote>" in result

    def test_horizontal_rule(self):
        """Test horizontal rule conversion."""
        result = markdown_to_html("---")
        assert "<hr>" in result

    def test_html_escaping(self):
        """Test that HTML entities are escaped."""
        result = markdown_to_html("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result


class TestGenerateArticleHtml:
    """Tests for generate_article_html."""

    def test_basic_structure(self):
        """Test that generated HTML has required structure."""
        html = generate_article_html(
            topic="Test Topic",
            category="Python",
            date="March 15, 2026",
            content="Some content here",
        )
        assert "<!DOCTYPE html>" in html
        assert "<html lang=" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_includes_title(self):
        """Test that title is included."""
        html = generate_article_html(
            topic="My Topic",
            category="Python",
            date="March 15, 2026",
            content="Content",
        )
        assert "<title>My Topic - Technomancer Learning</title>" in html
        assert "<h1>My Topic</h1>" in html

    def test_includes_metadata(self):
        """Test that category and date are included."""
        html = generate_article_html(
            topic="Topic",
            category="System Design",
            date="March 15, 2026",
            content="Content",
        )
        assert "System Design" in html
        assert "March 15, 2026" in html

    def test_includes_highlight_js(self):
        """Test that Highlight.js is included."""
        html = generate_article_html(
            topic="Topic",
            category="Python",
            date="March 15, 2026",
            content="```python\ncode\n```",
        )
        assert "highlight.js" in html
        assert "hljs.highlightAll()" in html

    def test_includes_dark_mode_css(self):
        """Test that dark mode CSS is included."""
        html = generate_article_html(
            topic="Topic",
            category="Python",
            date="March 15, 2026",
            content="Content",
        )
        assert "prefers-color-scheme: dark" in html

    def test_includes_back_link(self):
        """Test that back to index link is included."""
        html = generate_article_html(
            topic="Topic",
            category="Python",
            date="March 15, 2026",
            content="Content",
        )
        assert 'href="index.html"' in html
        assert "Back to articles" in html

    def test_escapes_html_in_title(self):
        """Test that HTML is escaped in title."""
        html = generate_article_html(
            topic="<script>bad</script>",
            category="Python",
            date="March 15, 2026",
            content="Content",
        )
        assert "&lt;script&gt;" in html


class TestGenerateIndexHtml:
    """Tests for generate_index_html."""

    def test_empty_list(self):
        """Test index with no articles."""
        html = generate_index_html([])
        assert "No articles yet" in html

    def test_with_articles(self):
        """Test index with articles."""
        articles = [
            {
                "filename": "2026-03-15_python_test.html",
                "topic": "Test Topic",
                "category": "Python",
                "date": "March 15, 2026",
            },
            {
                "filename": "2026-03-14_oracle_sql.html",
                "topic": "SQL Tips",
                "category": "Oracle",
                "date": "March 14, 2026",
            },
        ]
        html = generate_index_html(articles)
        assert "Test Topic" in html
        assert "SQL Tips" in html
        assert 'href="2026-03-15_python_test.html"' in html
        assert "Python" in html
        assert "Oracle" in html


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_text(self):
        """Test basic text slugification."""
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        """Test removal of special characters."""
        # Apostrophes and colons are removed, leaving adjacent letters
        assert slugify("Python's GIL: explained!") == "pythons-gil-explained"

    def test_multiple_spaces(self):
        """Test multiple spaces become single hyphen."""
        assert slugify("too   many   spaces") == "too-many-spaces"

    def test_leading_trailing_hyphens(self):
        """Test leading/trailing hyphens are stripped."""
        assert slugify("--hello--") == "hello"


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        from datetime import datetime

        date = datetime(2026, 3, 15)
        filename = generate_filename("Python Decorators", "python", date)
        assert filename == "2026-03-15_python_python-decorators.html"

    def test_long_topic_truncated(self):
        """Test that long topics are truncated."""
        from datetime import datetime

        date = datetime(2026, 3, 15)
        long_topic = "A" * 100
        filename = generate_filename(long_topic, "python", date)
        # Slug should be truncated to 50 chars
        assert len(filename.split("_")[2].replace(".html", "")) <= 50


# =============================================================================
# HTML NORMALIZATION TESTS
# =============================================================================


class TestExtractBodyContent:
    """Tests for extracting body content from HTML."""

    def test_extracts_body(self):
        html = "<html><head><style>.x{}</style></head><body><h1>Hello</h1><p>World</p></body></html>"
        result = extract_body_content(html)
        assert "<h1>Hello</h1>" in result
        assert "<p>World</p>" in result

    def test_strips_style_tags(self):
        html = "<body><style>.bad { color: red; }</style><p>Content</p></body>"
        result = extract_body_content(html)
        assert "<style>" not in result
        assert "Content" in result

    def test_strips_script_tags(self):
        html = "<body><p>Text</p><script>alert('x')</script></body>"
        result = extract_body_content(html)
        assert "<script>" not in result
        assert "Text" in result

    def test_strips_inline_styles(self):
        html = '<body><div style="background: purple; color: white;">Content</div></body>'
        result = extract_body_content(html)
        assert 'style="' not in result
        assert "Content" in result

    def test_handles_no_body_tag(self):
        """If there's no <body>, return the content as-is (it's a fragment)."""
        fragment = "<h1>Just a heading</h1><p>Some text</p>"
        result = extract_body_content(fragment)
        assert "Just a heading" in result
        assert "Some text" in result


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extracts_from_title_tag(self):
        html = "<html><head><title>My Page</title></head><body></body></html>"
        assert extract_title(html) == "My Page"

    def test_extracts_from_h1(self):
        html = "<html><body><h1>Japan Travel Guide</h1></body></html>"
        assert extract_title(html) == "Japan Travel Guide"

    def test_strips_site_name_suffix(self):
        html = "<title>Guide - My Site</title>"
        assert extract_title(html) == "Guide"

    def test_fallback_default(self):
        html = "<html><body><p>No title here</p></body></html>"
        assert extract_title(html) == "Technomancer"


class TestNormalizeHtml:
    """Tests for the full HTML normalization pipeline."""

    def test_replaces_inline_css(self):
        """Inline styles should be stripped and replaced with standard CSS."""
        raw = """<html><head><style>
        body { background: purple; font-family: Comic Sans; }
        .card { border: 3px solid gold; }
        </style></head>
        <body>
        <h1>My Guide</h1>
        <div class="card" style="background: pink;">Content here</div>
        </body></html>"""

        result = normalize_html(raw)

        # Should NOT contain the LLM's custom CSS
        assert "Comic Sans" not in result
        assert "purple" not in result
        assert "gold" not in result
        assert 'style="background: pink;"' not in result

        # SHOULD contain our standard CSS variables
        assert "--bg:" in result
        assert "--text:" in result
        assert "prefers-color-scheme: dark" in result

        # Content preserved
        assert "Content here" in result

    def test_preserves_body_content(self):
        """All visible content from the body should survive normalization."""
        raw = """<html><body>
        <h1>Title</h1>
        <h2>Section 1</h2>
        <p>Paragraph one</p>
        <h2>Section 2</h2>
        <ul><li>Item A</li><li>Item B</li></ul>
        </body></html>"""

        result = normalize_html(raw)
        assert "Title" in result
        assert "Section 1" in result
        assert "Paragraph one" in result
        assert "Section 2" in result
        assert "Item A" in result
        assert "Item B" in result

    def test_adds_highlight_js(self):
        """Normalized output should include syntax highlighting scripts."""
        raw = "<html><body><p>Hello</p></body></html>"
        result = normalize_html(raw)
        assert "highlight.min.js" in result
        assert "hljs.highlightAll()" in result

    def test_skips_already_normalized(self):
        """Pages already using our template should pass through unchanged."""
        from agent.html_generator import ARTICLE_CSS

        already_good = f"""<!DOCTYPE html>
<html><head><style>{ARTICLE_CSS}</style></head>
<body><article><header><h1>Test</h1>
<p class="meta">Already normalized</p></header>
<main><p>Content</p></main></article></body></html>"""

        result = normalize_html(already_good)
        # Should be returned as-is since it's already normalized
        assert result == already_good

    def test_extracts_title_into_header(self):
        """The page title should appear in the normalized header."""
        raw = "<html><head><title>Japan Events Guide</title></head><body><p>Events here</p></body></html>"
        result = normalize_html(raw)
        assert "Japan Events Guide" in result

    def test_consistent_output_structure(self):
        """Normalized HTML should always have article > header + main structure."""
        raw = "<html><body><div><p>Random content</p></div></body></html>"
        result = normalize_html(raw)
        assert "<article>" in result
        assert "<header>" in result
        assert "<main>" in result
        assert "</article>" in result

    def test_handles_multiple_style_blocks(self):
        """Multiple <style> blocks (common in LLM drift) should all be stripped."""
        raw = """<html><body>
        <style>.theme-a { color: red; }</style>
        <h2>Part 1</h2>
        <p>Content A</p>
        <style>.theme-b { color: blue; }</style>
        <h2>Part 2</h2>
        <p>Content B</p>
        </body></html>"""

        result = normalize_html(raw)
        # Both style blocks removed
        assert "theme-a" not in result
        assert "theme-b" not in result
        # Content preserved
        assert "Content A" in result
        assert "Content B" in result
