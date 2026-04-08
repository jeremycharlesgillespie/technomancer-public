"""
Itinerary Command - Routes travel/event requests directly to Claude API
for fast, high-quality HTML generation.

Bypasses the local LLM entirely. Claude receives a pre-built prompt with
formatting instructions and returns structured markdown that gets rendered
through our normalized HTML template and deployed to GitHub Pages.

Usage in Discord: itinerary <your request>
Example: itinerary Japan April 4-23, cherry blossoms, food, anime, temples
"""

import time
from datetime import datetime

from .claude_bridge import ClaudeBridge
from .config import settings
from .html_generator import ARTICLE_CSS, markdown_to_html, normalize_html
from .perf_monitor import record_llm_call as _record_perf

# Pre-built system prompt that tells Claude exactly how to format the output
ITINERARY_SYSTEM_PROMPT = """You are an expert travel planner creating a detailed, day-by-day itinerary.

OUTPUT FORMAT: Write your response in clean markdown. Use this structure:

# [Trip Title]

## Overview
Brief 2-3 sentence overview of the trip.

## Day-by-Day Itinerary

### Day 1 — [Date] ([Day of Week])
**Theme:** [Day's theme]

#### Morning
- **[Activity Name]** — [Location]
  - [Description, 1-2 sentences]
  - ⏰ [Hours] | 💰 [Cost if applicable]
  - 🔗 [URL if you know it]

#### Afternoon
[Same format]

#### Evening
[Same format]

---

[Repeat for each day]

## Practical Tips
- [Tip 1]
- [Tip 2]

## Packing List
- [Item 1]
- [Item 2]

RULES:
- Include specific venue names, addresses/neighborhoods, and opening hours
- Include estimated costs in local currency
- Mix popular tourist spots with hidden gems and local favorites
- Include food recommendations for each day (breakfast, lunch, dinner spots)
- Note any holidays, festivals, or seasonal events during the dates
- Include transportation tips between locations
- Be specific — real place names, not generic suggestions
- Aim for 3-5 activities per day with realistic timing
- Include rest/downtime — don't over-schedule"""


ITINERARY_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}

/* Itinerary-specific enhancements */
h3 {{
    border-bottom: 2px solid var(--link);
    padding-bottom: 0.5rem;
}}

h4 {{
    color: var(--link);
    font-size: 1.1rem;
    margin-top: 1.5rem;
}}

hr {{
    margin: 2.5rem 0;
}}
    </style>
</head>
<body>
    <article>
        <header>
            <h1>{title}</h1>
            <p class="meta">Generated {date} &bull; Powered by Claude</p>
        </header>
        <main>
{content}
        </main>
        <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--code-border); color: var(--text-muted); font-size: 0.85rem;">
            <p>Generated in {duration}s by Technomancer &bull; Verify details before traveling</p>
        </footer>
    </article>
</body>
</html>"""


async def generate_itinerary(request: str) -> tuple[str, str | None, float]:
    """Generate a travel itinerary via Claude API.

    Args:
        request: The user's travel request (e.g. "Japan April 4-23, food, anime")

    Returns:
        Tuple of (markdown_response, html_content_or_none, duration_seconds)
    """
    import asyncio

    start = time.time()

    bridge = ClaudeBridge(
        mode="api",
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        timeout=180,
    )

    if not bridge.client:
        return "Error: Claude API not configured. Set ANTHROPIC_API_KEY in .env", None, 0

    # Inject today's date so Claude doesn't guess
    today = datetime.now()
    date_info = f"TODAY IS: {today.strftime('%A, %B %d, %Y')}"

    # Build the full prompt
    prompt = f"""{ITINERARY_SYSTEM_PROMPT}

{date_info}
IMPORTANT: Use Python-style date math for day-of-week. Today is {today.strftime('%A')}. Count forward from there. Do NOT guess days of week — calculate them.

USER REQUEST:
{request}

Generate the complete itinerary now. Be thorough and specific."""

    try:
        # Call Claude API directly for maximum control
        response = await asyncio.to_thread(
            bridge.client.messages.create,
            model=bridge.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_output = response.content[0].text
        duration = time.time() - start

        # Strip markdown code fences if Claude wrapped its HTML in ```html ... ```
        stripped = raw_output.strip()
        if stripped.startswith("```"):
            # Remove opening fence (```html or ```)
            first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
            stripped = stripped[first_newline + 1:]
            # Remove closing fence
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
            raw_output = stripped

        # Detect if Claude returned HTML instead of markdown
        is_html = raw_output.strip().startswith(("<!DOCTYPE", "<html", "<HTML", "<head", "<body"))

        if is_html:
            # Claude returned a full HTML page — extract body and use our template
            from .html_generator import extract_body_content, extract_title

            title = extract_title(raw_output)
            html_body = extract_body_content(raw_output)
        else:
            # Claude returned markdown as instructed — convert to HTML
            title = "Travel Itinerary"
            for line in raw_output.split("\n"):
                line = line.strip()
                if line.startswith("# ") and not line.startswith("## "):
                    title = line[2:].strip()
                    break
            html_body = markdown_to_html(raw_output)

        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        html_page = ITINERARY_HTML_TEMPLATE.format(
            title=title,
            date=date_str,
            content=html_body,
            css=ARTICLE_CSS,
            duration=f"{duration:.1f}",
        )

        # Calculate token usage for cost tracking
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Sonnet pricing: $3/M input, $15/M output
        cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        print(
            f"[Itinerary] Claude: {input_tokens} in / {output_tokens} out "
            f"(${cost:.4f}) in {duration:.1f}s"
        )

        _record_perf(
            "claude_api", duration, success=True,
            model=bridge.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return raw_output, html_page, duration

    except Exception as e:
        duration = time.time() - start
        _record_perf(
            "claude_api", duration, success=False,
            model=bridge.model, error=str(e)[:200],
        )
        return f"Error generating itinerary: {e}", None, duration
