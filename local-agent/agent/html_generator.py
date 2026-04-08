"""
HTML article generator for GitHub Pages.

Converts markdown learning articles to mobile-responsive HTML with:
- Clean typography
- Syntax highlighting (Highlight.js)
- Dark mode support
- No external dependencies
"""

import html
import re
from datetime import datetime

# Embedded CSS for mobile-responsive, dark-mode-aware styling
ARTICLE_CSS = """
:root {
    --bg: #ffffff;
    --text: #1a1a1a;
    --text-muted: #666666;
    --code-bg: #f5f5f5;
    --code-border: #e0e0e0;
    --link: #0066cc;
    --heading: #111111;
    --blockquote-border: #ddd;
    --blockquote-bg: #f9f9f9;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1a1a1a;
        --text: #e0e0e0;
        --text-muted: #999999;
        --code-bg: #2d2d2d;
        --code-border: #404040;
        --link: #66b3ff;
        --heading: #ffffff;
        --blockquote-border: #444;
        --blockquote-bg: #252525;
    }
}

* {
    box-sizing: border-box;
}

html {
    font-size: 18px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    -webkit-font-smoothing: antialiased;
}

header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--code-border);
}

header a {
    font-size: 0.9rem;
    color: var(--text-muted);
    text-decoration: none;
}

header a:hover {
    color: var(--link);
}

h1 {
    font-size: 2rem;
    line-height: 1.2;
    color: var(--heading);
    margin: 1rem 0 0.5rem 0;
}

h2 {
    font-size: 1.5rem;
    line-height: 1.3;
    color: var(--heading);
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    font-size: 1.25rem;
    line-height: 1.4;
    color: var(--heading);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

.meta {
    color: var(--text-muted);
    font-size: 0.9rem;
}

a {
    color: var(--link);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

p {
    margin: 1rem 0;
}

ul, ol {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

li {
    margin: 0.5rem 0;
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--code-border);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    margin: 1.5rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

code {
    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
    font-size: 0.9em;
}

:not(pre) > code {
    background: var(--code-bg);
    padding: 0.2em 0.4em;
    border-radius: 4px;
}

blockquote {
    border-left: 4px solid var(--blockquote-border);
    background: var(--blockquote-bg);
    margin: 1.5rem 0;
    padding: 1rem 1.5rem;
    font-style: italic;
}

blockquote p {
    margin: 0;
}

hr {
    border: none;
    border-top: 1px solid var(--code-border);
    margin: 2rem 0;
}

strong {
    font-weight: 600;
}

/* Index page specific styles */
.article-list {
    list-style: none;
    padding: 0;
}

.article-list li {
    padding: 1rem 0;
    border-bottom: 1px solid var(--code-border);
}

.article-list li:last-child {
    border-bottom: none;
}

.article-list a {
    font-size: 1.1rem;
    font-weight: 500;
}

.article-list .date {
    display: block;
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

.article-list .category {
    display: inline-block;
    background: var(--code-bg);
    padding: 0.2em 0.6em;
    border-radius: 4px;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-left: 0.5rem;
}
"""

# HTML template for individual articles
ARTICLE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Technomancer Learning</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" media="(prefers-color-scheme: light)">
    <style>
{css}
    </style>
</head>
<body>
    <article>
        <header>
            <a href="index.html">&larr; Back to articles</a>
            <h1>{title}</h1>
            <p class="meta">{category} &bull; {date}</p>
        </header>
        <main>
{content}
        </main>
    </article>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>
</html>
"""

# HTML template for index page
INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Articles - Technomancer</title>
    <style>
{css}
    </style>
</head>
<body>
    <header>
        <h1>Developer Learning</h1>
        <p class="meta">Daily insights to level up your skills</p>
    </header>
    <main>
        <ul class="article-list">
{articles}
        </ul>
    </main>
</body>
</html>
"""


def markdown_to_html(content: str) -> str:
    """
    Convert markdown content to HTML.

    Handles:
    - Headers (h1-h3)
    - Code blocks with language hints
    - Inline code
    - Bold and italic
    - Links
    - Unordered and ordered lists
    - Blockquotes
    - Horizontal rules
    - Paragraphs

    Args:
        content: Markdown formatted string

    Returns:
        HTML formatted string
    """
    result = content

    # Code blocks first - extract and preserve them
    # Match ```language\ncode\n```
    code_blocks: list[str] = []

    def save_code_block(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2)
        code = html.escape(code)
        if lang:
            block = f'<pre><code class="language-{lang}">{code}</code></pre>'
        else:
            block = f"<pre><code>{code}</code></pre>"
        code_blocks.append(block)
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    result = re.sub(r"```(\w*)\n(.*?)```", save_code_block, result, flags=re.DOTALL)

    # Handle blockquotes before HTML escaping (since > gets escaped)
    blockquotes: list[str] = []

    def save_blockquote(match: re.Match) -> str:
        quote_text = html.escape(match.group(1))
        block = f"<blockquote><p>{quote_text}</p></blockquote>"
        blockquotes.append(block)
        return f"__BLOCKQUOTE_{len(blockquotes) - 1}__"

    result = re.sub(r"^> (.+)$", save_blockquote, result, flags=re.MULTILINE)

    # Now escape HTML in the rest of the content
    result = html.escape(result)

    # Restore code blocks and blockquotes
    for i, block in enumerate(code_blocks):
        result = result.replace(f"__CODE_BLOCK_{i}__", block)
    for i, block in enumerate(blockquotes):
        result = result.replace(f"__BLOCKQUOTE_{i}__", block)

    # Inline code (backticks)
    result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)

    # Headers (must check longer patterns first)
    result = re.sub(r"^### (.+)$", r"<h3>\1</h3>", result, flags=re.MULTILINE)
    result = re.sub(r"^## (.+)$", r"<h2>\1</h2>", result, flags=re.MULTILINE)
    result = re.sub(r"^# (.+)$", r"<h1>\1</h1>", result, flags=re.MULTILINE)

    # Bold and italic
    result = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", result)
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"\*(.+?)\*", r"<em>\1</em>", result)

    # Links [text](url)
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', result)

    # Horizontal rules
    result = re.sub(r"^---+$", r"<hr>", result, flags=re.MULTILINE)

    # Lists - unordered
    def process_unordered_list(match: re.Match) -> str:
        items = match.group(0)
        list_items = re.findall(r"^[-*] (.+)$", items, re.MULTILINE)
        html_items = "\n".join(f"<li>{item}</li>" for item in list_items)
        return f"<ul>\n{html_items}\n</ul>"

    result = re.sub(r"(^[-*] .+$\n?)+", process_unordered_list, result, flags=re.MULTILINE)

    # Lists - ordered
    def process_ordered_list(match: re.Match) -> str:
        items = match.group(0)
        list_items = re.findall(r"^\d+\. (.+)$", items, re.MULTILINE)
        html_items = "\n".join(f"<li>{item}</li>" for item in list_items)
        return f"<ol>\n{html_items}\n</ol>"

    result = re.sub(r"(^\d+\. .+$\n?)+", process_ordered_list, result, flags=re.MULTILINE)

    # Paragraphs - wrap loose text in <p> tags
    # Split by double newlines, wrap non-tag content
    paragraphs = re.split(r"\n\n+", result)
    processed = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Don't wrap if already a block element
        if re.match(r"^<(h[1-6]|ul|ol|pre|blockquote|hr)", p):
            processed.append(p)
        else:
            # Replace single newlines with <br> within paragraphs
            p = re.sub(r"\n", "<br>\n", p)
            processed.append(f"<p>{p}</p>")

    return "\n\n".join(processed)


def generate_article_html(topic: str, category: str, date: str, content: str) -> str:
    """
    Generate a complete HTML page for a learning article.

    Args:
        topic: Article title/topic
        category: Category (e.g., "Python", "System Design")
        date: Date string (e.g., "March 15, 2026")
        content: Markdown content of the article

    Returns:
        Complete HTML document as string
    """
    html_content = markdown_to_html(content)

    return ARTICLE_TEMPLATE.format(
        title=html.escape(topic),
        category=html.escape(category),
        date=html.escape(date),
        content=html_content,
        css=ARTICLE_CSS,
    )


def generate_index_html(articles: list[dict]) -> str:
    """
    Generate an index page listing all learning articles.

    Args:
        articles: List of dicts with keys: topic, category, date, filename
                  Should be sorted newest first

    Returns:
        Complete HTML document as string
    """
    items = []
    for article in articles:
        item = f"""            <li>
                <a href="{html.escape(article['filename'])}">{html.escape(article['topic'])}</a>
                <span class="category">{html.escape(article['category'])}</span>
                <span class="date">{html.escape(article['date'])}</span>
            </li>"""
        items.append(item)

    articles_html = "\n".join(items) if items else "<li>No articles yet</li>"

    return INDEX_TEMPLATE.format(css=ARTICLE_CSS, articles=articles_html)


def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.

    Args:
        text: Input text

    Returns:
        Lowercase slug with hyphens
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


def generate_filename(topic: str, category: str, date: datetime | None = None) -> str:
    """
    Generate a filename for a learning article HTML file.

    Args:
        topic: Article topic
        category: Article category
        date: Date for the article (defaults to now)

    Returns:
        Filename like "2026-03-15_python_decorators-in-depth.html"
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y-%m-%d")
    category_slug = slugify(category)
    topic_slug = slugify(topic)[:50]  # Limit length

    return f"{date_str}_{category_slug}_{topic_slug}.html"


# =============================================================================
# HTML NORMALIZATION — Re-wrap bot-generated HTML in consistent styling
# =============================================================================

# Template for normalized pages — uses our standard CSS but no "Back to articles" link
NORMALIZED_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" media="(prefers-color-scheme: light)">
    <style>
{css}
    </style>
</head>
<body>
    <article>
        <header>
            <h1>{title}</h1>
            <p class="meta">{meta}</p>
        </header>
        <main>
{content}
        </main>
    </article>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>
</html>"""


def extract_body_content(html_str: str) -> str:
    """Extract the inner content of <body> from an HTML document.

    Strips <script> and <style> tags from the extracted content so only
    the visible markup remains. If no <body> tag is found, returns the
    input unchanged (it may already be a fragment).
    """
    # Try to find <body>...</body>
    body_match = re.search(
        r"<body[^>]*>(.*)</body>", html_str, re.DOTALL | re.IGNORECASE
    )
    if body_match:
        content = body_match.group(1)
    else:
        content = html_str

    # Strip <script> blocks
    content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Strip <style> blocks (our template provides consistent CSS)
    content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Strip inline style attributes
    content = re.sub(r'\s+style="[^"]*"', "", content)
    content = re.sub(r"\s+style='[^']*'", "", content)

    return content.strip()


def extract_title(html_str: str) -> str:
    """Extract the <title> or first <h1> from HTML as a page title."""
    # Try <title>
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_str, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # Remove " - Site Name" suffixes
        title = re.sub(r"\s*[-|].*$", "", title)
        if title:
            return html.unescape(title)

    # Try first <h1>
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_str, re.IGNORECASE | re.DOTALL)
    if h1_match:
        # Strip any HTML tags inside the h1
        h1_text = re.sub(r"<[^>]+>", "", h1_match.group(1))
        return html.unescape(h1_text.strip())

    return "Technomancer"


def normalize_html(html_str: str) -> str:
    """Normalize a bot-generated HTML page for consistent rendering.

    Extracts the body content, strips all inline styles and embedded CSS,
    and re-wraps everything in our standard template with:
    - Consistent dark/light mode CSS
    - Mobile-responsive layout
    - Syntax highlighting
    - Clean typography

    This prevents the LLM from generating HTML that drifts in style
    midway through the document.

    Args:
        html_str: Raw HTML string (full document or fragment)

    Returns:
        Normalized HTML document using the standard template
    """
    title = extract_title(html_str)
    body = extract_body_content(html_str)

    # If the body already has our template wrapper (article > header + main),
    # it was already normalized — skip to avoid double-wrapping
    if "<article>" in body and '<p class="meta">' in body:
        return html_str

    # Generate a meta line from the current date
    meta = datetime.now().strftime("Generated %B %d, %Y at %I:%M %p")

    return NORMALIZED_TEMPLATE.format(
        title=html.escape(title),
        meta=meta,
        content=body,
        css=ARTICLE_CSS,
    )
