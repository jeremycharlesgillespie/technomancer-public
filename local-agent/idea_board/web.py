"""
Idea Board Web Dashboard — Flask app for viewing, voting, and executing ideas.

Runs on port 8322, binds to 0.0.0.0 for Tailscale/LAN access.
Starts as a daemon thread from the Discord bot process.

Routes:
    GET  /                        — HTML dashboard
    GET  /api/ideas               — JSON list (filterable: ?state=proposed)
    POST /api/ideas/<id>/vote     — Vote on an idea
    POST /api/ideas/<id>/comment  — Add a comment
    POST /api/ideas/<id>/execute  — Trigger Claude Code execution
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from agent.config import settings

from .models import (
    add_comment,
    delete_idea,
    get_idea,
    load_ideas,
    mark_done,
    mark_executing,
    mark_failed,
    vote,
)

logger = logging.getLogger(__name__)

BOARD_PORT: int = 8322

app = Flask(__name__)


# ============================================================================
# Discord notifications for idea lifecycle events
# ============================================================================

_BRIDGE_TOKEN_FILE = Path(__file__).parent.parent / ".bridge_token"


def _send_to_discord(message: str) -> None:
    """Send a message to the llm_chat Discord channel via the bridge API."""
    try:
        import requests as _req

        if not _BRIDGE_TOKEN_FILE.exists():
            logger.debug("Bridge token file not found, skipping Discord notification")
            return
        token = _BRIDGE_TOKEN_FILE.read_text(encoding="utf-8").strip()
        _req.post(
            "http://127.0.0.1:8321/api/send",
            headers={"X-Bridge-Token": token, "Content-Type": "application/json"},
            json={"message": message},
            timeout=5,
        )
    except Exception:
        logger.debug("Could not send Discord notification via bridge", exc_info=True)


def _notify_idea_complete(idea_id: str, title: str) -> None:
    """Send a Discord notification when an idea is marked done."""
    _send_to_discord(f"✅ **Idea Completed** — **{idea_id}**: {title}")


def _notify_idea_failed(idea_id: str, title: str, error_text: str) -> None:
    """Send a Discord notification when an idea execution fails."""
    snippet = error_text[:200] if len(error_text) > 200 else error_text
    _send_to_discord(f"❌ **Idea Failed** — **{idea_id}**: {title}\n{snippet}")


# ============================================================================
# LLM CONVERSATION FOR IDEAS
# ============================================================================

IDEA_DISCUSSION_PROMPT = """You are an eager, thoughtful software engineer discussing an improvement idea
with your manager ({owner_name}). You originally proposed this idea. Now {owner_name} is
giving you feedback and asking questions about it.

YOUR IDEA:
Title: {title}
Description: {description}
Category: {category}

CONVERSATION SO FAR:
{conversation}

RULES:
- Be direct, specific, and technical — {owner_name} is a senior software engineer
- If they ask you to explain, give concrete technical details
- If they push back, consider their point honestly — maybe the idea needs refinement
- If they're interested, suggest next steps or implementation approach
- If you realize the idea is bad based on their feedback, say so honestly
- Keep responses concise (2-4 sentences) — this is a chat, not an essay
- Reference specific files, functions, or patterns from the Technomancer codebase when relevant
- You're enthusiastic but not pushy — respect your manager's judgment"""


def _generate_idea_reply(idea: "Idea") -> str | None:
    """Generate an LLM reply to the latest comment on an idea.

    Creates an isolated Agent instance (never touches the main bot)
    and sends it the full idea context + conversation history.

    Args:
        idea: The Idea with updated comments

    Returns:
        The LLM's reply text, or None on failure
    """
    # Build conversation history
    conv_lines = []
    for c in idea.comments:
        role = f"{settings.owner_name} (manager)" if c.author == "owner" else "You (engineer)"
        conv_lines.append(f"{role}: {c.text}")
    conversation = "\n".join(conv_lines)

    prompt = IDEA_DISCUSSION_PROMPT.format(
        owner_name=settings.owner_name,
        title=idea.title,
        description=idea.description,
        category=idea.category,
        conversation=conversation,
    )

    try:
        import ollama

        client = ollama.Client(host="http://127.0.0.1:11434")
        response = client.chat(
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_ctx": 8192},
        )
        content = response.get("message", {}).get("content", "") or ""

        # Strip thinking tags if present
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return content if content else None
    except Exception as e:
        logger.error(f"[IdeaBoard] LLM reply error: {e}")
        return None


# ============================================================================
# HTML DASHBOARD
# ============================================================================

DASHBOARD_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --orange: #ff9800;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.6; }
h1 { margin-bottom: 1rem; color: var(--accent); }
.stats { color: var(--muted); margin-bottom: 2rem; font-size: 0.9rem; }
.section-title { font-size: 1.2rem; color: var(--accent); margin: 2rem 0 1rem;
                 border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
.card { background: var(--surface); border-radius: 8px; padding: 1.2rem;
        margin-bottom: 1rem; border-left: 4px solid var(--border); }
.card.proposed { border-left-color: var(--accent); }
.card.approved { border-left-color: var(--green); }
.card.vetoed { border-left-color: var(--red); opacity: 0.6; }
.card.executing { border-left-color: var(--orange); }
.card.done { border-left-color: var(--green); opacity: 0.7; }
.card.failed { border-left-color: var(--red); opacity: 0.7; }
.card.refining { border-left-color: var(--orange); }
.card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
.card-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; }
.card-desc { margin-bottom: 0.8rem; font-size: 0.95rem; }
.desc-section { margin-bottom: 0.4rem; line-height: 1.5; }
.desc-label { font-weight: 600; color: var(--accent); }
.desc-text { margin-bottom: 0.4rem; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 0.75rem; margin-right: 4px; }
.badge-cat { background: #333; color: var(--accent); }
.badge-src { background: #333; color: var(--muted); }
.badge-state { background: #333; font-weight: 600; }
.badge-state.proposed { color: var(--accent); }
.badge-state.approved { color: var(--green); }
.badge-state.vetoed { color: var(--red); }
.badge-state.executing { color: var(--orange); }
.badge-state.done { color: var(--green); }
.badge-state.failed { color: var(--red); }
.actions { margin-top: 0.8rem; display: flex; gap: 8px; flex-wrap: wrap; }
.btn { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
       font-size: 0.85rem; font-weight: 500; }
.btn-approve { background: var(--green); color: white; }
.btn-veto { background: var(--red); color: white; }
.btn-execute { background: var(--orange); color: white; }
.btn:hover { opacity: 0.85; }
.comments { margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid var(--border);
            max-height: 400px; overflow-y: auto; }
.comment { font-size: 0.9rem; margin-bottom: 0.6rem; padding: 8px 12px;
           border-radius: 6px; max-width: 85%; }
.comment.owner { background: #1a3a5c; margin-left: auto; }
.comment.llm { background: #2d2d2d; }
.comment-author { font-weight: 600; font-size: 0.75rem; margin-bottom: 2px; }
.comment-author.owner { color: var(--accent); }
.comment-author.llm { color: var(--green); }
.comment-text { line-height: 1.5; }
.comment-time { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }
.comment-form { display: flex; gap: 8px; margin-top: 0.8rem; }
.comment-input { flex: 1; padding: 8px 12px; background: var(--bg); border: 1px solid var(--border);
                 border-radius: 6px; color: var(--text); font-size: 0.9rem; }
.comment-input:focus { outline: none; border-color: var(--accent); }
.btn-comment { background: var(--accent); color: white; }
.thinking { color: var(--muted); font-style: italic; padding: 8px 12px;
            animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
@keyframes fadeIn { from { opacity: 0; transform: translateX(-50%) translateY(20px); }
                    to { opacity: 1; transform: translateX(-50%) translateY(0); } }
.exec-log { background: #111; border: 1px solid var(--border); border-radius: 6px;
            padding: 10px; margin-top: 0.8rem; max-height: 300px; overflow-y: auto;
            font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.8rem;
            line-height: 1.4; white-space: pre-wrap; color: #ccc; }
.exec-log .line { margin: 0; }
.exec-header { display: flex; justify-content: space-between; align-items: center;
               margin-top: 0.8rem; }
.exec-status { font-size: 0.85rem; }
.btn-cancel { background: var(--red); color: white; font-size: 0.8rem; padding: 4px 12px; }
.btn-done { background: var(--green); color: white; }
.btn-delete { background: transparent; color: var(--muted); border: 1px solid var(--border);
              font-size: 0.8rem; }
.badge-type { font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-epic { background: #4a1a6b; color: #c084fc; }
.badge-story { background: #1a3a5c; color: var(--accent); }
.badge-task { background: #333; color: var(--muted); }
.epic-group { border: 2px solid #4a1a6b; border-radius: 10px; padding: 0.5rem;
              margin-bottom: 1.2rem; background: rgba(74,26,107,0.08); }
.epic-group > .card.epic-card { border-left-width: 6px; margin-bottom: 0.5rem; }
.epic-children { margin-left: 1.5rem; border-left: 2px dashed #4a1a6b;
                 padding-left: 0.75rem; }
.epic-children > .card.child-card { opacity: 0.95; font-size: 0.95em; }
.btn-epic-copy { background: #7c3aed; color: white; font-weight: 600; }
.archive-section { margin-top: 2rem; border-top: 2px solid var(--border); padding-top: 1rem; }
.archive-toggle { cursor: pointer; color: var(--muted); font-size: 1.1rem; padding: 0.8rem 0;
                  user-select: none; list-style: none; }
.archive-toggle::-webkit-details-marker { display: none; }
.archive-toggle::before { content: '\\25B6  '; font-size: 0.8rem; }
details[open] > .archive-toggle::before { content: '\\25BC  '; }
.archive-toggle:hover { color: var(--accent); }
"""


def _format_description(raw: str) -> str:
    """Format a structured idea description into HTML sections.

    Detects WHAT:, WHY:, HOW:, BENEFITS:, COST:, UNLOCKS: headers and renders
    each as a labeled section.  Works whether headers are on separate lines or
    inline (all on one line).  Falls back to plain text for descriptions
    without any recognised section headers.
    """
    import re

    section_labels = {
        "WHAT": "What",
        "WHY": "Why",
        "HOW": "How",
        "BENEFITS": "Benefits",
        "COST": "Cost",
        "UNLOCKS": "Unlocks",
    }

    # Check if description has structured sections (inline or newline-separated)
    header_keys = "|".join(section_labels)
    has_sections = bool(re.search(rf"(?:^|\b)(?:{header_keys}):", raw))

    if not has_sections:
        return html.escape(raw)

    # Split on section headers — works for both newline-separated and inline
    pattern = rf"((?:{header_keys}):)"
    splits = re.split(pattern, raw.strip())

    parts = []
    # First element might be text before any header
    if splits[0].strip():
        parts.append(f'<div class="desc-text">{html.escape(splits[0].strip())}</div>')

    # Process header + content pairs
    i = 1
    while i < len(splits) - 1:
        header_key = splits[i].rstrip(":")
        content = splits[i + 1].strip()
        label = section_labels.get(header_key, header_key)
        parts.append(
            f'<div class="desc-section">'
            f'<span class="desc-label">{html.escape(label)}:</span> '
            f'{html.escape(content)}'
            f'</div>'
        )
        i += 2

    return "\n".join(parts)


def _render_idea_card(idea: dict[str, Any], is_child: bool = False) -> str:
    """Render a single idea as an HTML card."""
    eid = html.escape(idea["id"])
    title = html.escape(idea["title"])
    desc = _format_description(idea["description"])
    state = idea["state"]
    category = html.escape(idea.get("category", ""))
    source = html.escape(idea.get("source", ""))
    idea_type = idea.get("idea_type", "story")
    parent_id = idea.get("parent_id", "")
    created = idea.get("created", "")[:10]
    claude_vote = idea.get("votes", {}).get("claude") or "—"
    owner_vote = idea.get("votes", {}).get("owner") or "—"
    owner_display = html.escape(settings.owner_name)

    # Comments section — chat-style conversation
    comments_html = ""
    for c in idea.get("comments", []):
        author = html.escape(c["author"])
        text = html.escape(c["text"])
        ts = c.get("timestamp", "")[:16]
        label = owner_display if author == "owner" else "LLM"
        comments_html += (
            f'<div class="comment {author}">'
            f'<div class="comment-author {author}">{label}</div>'
            f'<div class="comment-text">{text}</div>'
            f'<div class="comment-time">{ts}</div>'
            f'</div>\n'
        )

    # Action buttons
    actions = ""
    approve_btn = f'<button class="btn btn-approve" onclick="doVote(\'{eid}\',\'approve\')">Approve</button>'
    veto_btn = f'<button class="btn btn-veto" onclick="doVote(\'{eid}\',\'veto\')">Veto</button>'
    execute_btn = f'<button class="btn btn-execute" onclick="doExecute(\'{eid}\')">Copy for Claude Code</button>'
    epic_btn = f'<button class="btn btn-epic-copy" onclick="doExecuteEpic(\'{eid}\')">Copy Epic for Claude Code</button>'
    done_btn = f'<button class="btn btn-done" onclick="doMarkDone(\'{eid}\')">Mark Done</button>'
    delete_btn = f'<button class="btn btn-delete" onclick="doDelete(\'{eid}\')">Delete</button>'
    copy_btn = epic_btn if idea_type == "epic" else execute_btn
    if state in ("proposed", "refining"):
        actions = f'<div class="actions">{approve_btn} {veto_btn} {copy_btn} {delete_btn}</div>'
    elif state == "approved":
        actions = f'<div class="actions">{copy_btn} {done_btn} {delete_btn}</div>'
    elif state == "failed":
        actions = f'<div class="actions">{copy_btn} {done_btn} {delete_btn}</div>'
    elif state == "done":
        actions = f'<div class="actions">{delete_btn}</div>'
    elif state == "vetoed":
        actions = f'<div class="actions">{delete_btn}</div>'
    elif state == "executing":
        actions = f"""
        <div class="exec-header">
            <span class="exec-status thinking">Claude Code is working...</span>
            <button class="btn btn-cancel" onclick="doCancel('{eid}')">Cancel</button>
        </div>
        <div class="exec-log" id="log-{eid}">Loading execution log...</div>"""

    # Show execution log for done/failed (collapsed)
    exec_log_section = ""
    if state in ("done", "failed") and idea.get("execution_log"):
        log_text = html.escape(idea["execution_log"][-2000:])
        exec_log_section = f"""
        <details style="margin-top: 0.8rem;">
            <summary style="cursor:pointer;color:var(--muted);font-size:0.85rem">Execution Log</summary>
            <div class="exec-log">{log_text}</div>
        </details>"""

    type_badge = f'<span class="badge badge-type badge-{idea_type}">{idea_type}</span>'
    parent_link = ""
    if parent_id:
        safe_pid = html.escape(parent_id)
        parent_link = f' &bull; <a href="#" onclick="document.querySelector(\'[data-idea=\\x27{safe_pid}\\x27]\')?.scrollIntoView({{behavior:\\x27smooth\\x27}});return false" style="color:var(--accent)">parent: {safe_pid}</a>'

    child_class = " child-card" if is_child else ""
    epic_class = " epic-card" if idea_type == "epic" else ""

    return f"""
    <div class="card {state}{epic_class}{child_class}" data-idea="{eid}">
        <div class="card-title">{eid}: {title}</div>
        <div class="card-meta">
            {type_badge}
            <span class="badge badge-state {state}">{state}</span>
            <span class="badge badge-cat">{category}</span>
            <span class="badge badge-src">{source}</span>
            &bull; {created} &bull; Claude: {claude_vote} &bull; {owner_display}: {owner_vote}{parent_link}
        </div>
        <div class="card-desc">{desc}</div>
        {actions}
        {exec_log_section}
        <div class="comments">
            {comments_html}
            <form class="comment-form" onsubmit="doComment(event,'{eid}')">
                <input class="comment-input" name="text" placeholder="Add a comment..." autocomplete="off">
                <button class="btn btn-comment" type="submit">Send</button>
            </form>
        </div>
    </div>"""


def _render_dashboard(ideas: list[dict[str, Any]]) -> str:
    """Render the full HTML dashboard page."""
    owner_name_js = html.escape(settings.owner_name, quote=True)
    # Build lookup for parent→children
    by_id: dict[str, dict] = {i["id"]: i for i in ideas}
    children_of: dict[str, list[dict]] = {}
    for idea in ideas:
        pid = idea.get("parent_id")
        if pid:
            children_of.setdefault(pid, []).append(idea)

    # Group top-level ideas by state (exclude children — they render under parents)
    groups: dict[str, list[dict]] = {}
    for idea in ideas:
        if idea.get("parent_id") and idea["parent_id"] in by_id:
            continue  # will be rendered under its parent
        state = idea["state"]
        groups.setdefault(state, []).append(idea)

    # Render active sections (visible by default)
    active_states = ["proposed", "refining", "approved", "executing"]
    archived_states = ["done", "failed", "vetoed"]

    def _render_state_section(state: str, items: list[dict]) -> str:
        html = f'<h2 class="section-title">{state.upper()} ({len(items)})</h2>\n'
        for idea in items:
            idea_type = idea.get("idea_type", "story")
            kids = children_of.get(idea["id"], [])
            if idea_type == "epic" or kids:
                html += '<div class="epic-group">\n'
                html += _render_idea_card(idea)
                if kids:
                    html += '<div class="epic-children">\n'
                    for child in kids:
                        html += _render_idea_card(child, is_child=True)
                    html += '</div>\n'
                html += '</div>\n'
            else:
                html += _render_idea_card(idea)
        return html

    sections_html = ""
    for state in active_states:
        items = groups.get(state, [])
        if items:
            sections_html += _render_state_section(state, items)

    # Render archived sections (collapsed by default)
    archived_count = sum(len(groups.get(s, [])) for s in archived_states)
    archived_html = ""
    for state in archived_states:
        items = groups.get(state, [])
        if items:
            archived_html += _render_state_section(state, items)

    if archived_html:
        sections_html += (
            f'<details class="archive-section">'
            f'<summary class="archive-toggle">'
            f'Archived ({archived_count} completed/vetoed/failed)'
            f'</summary>\n'
            f'{archived_html}'
            f'</details>\n'
        )

    total = len(ideas)
    proposed = len(groups.get("proposed", []))
    now = datetime.now().strftime("%I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ideas - Technomancer Hub</title>
    <style>{DASHBOARD_CSS}
.nav {{ margin-bottom: 1.5rem; display: flex; gap: 12px; flex-wrap: wrap; }}
.nav a {{ color: var(--accent); text-decoration: none; padding: 6px 14px;
         border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }}
.nav a:hover, .nav a.active {{ background: var(--accent); color: #000; }}
</style>
</head>
<body>
    <h1>Technomancer Hub</h1>
    <div class="nav">
        <a href="/">Hub</a>
        <a href="/ideas" class="active">Ideas</a>
        <a href="/news">News Config</a>
        <a href="/karen">KAREN</a>
    </div>
    <p class="stats" id="stats">{total} ideas total &bull; {proposed} pending review &bull; Last refresh: {now}</p>

    <div id="evolve-panel" class="card" style="border-left-color: var(--accent); margin-bottom: 1.5rem;">
        <div class="card-title">Evolve Status</div>
        <div id="evolve-content" style="color: var(--muted);">Loading...</div>
    </div>

    {sections_html if sections_html else '<p style="color:var(--muted)">No ideas yet. They will start appearing hourly.</p>'}

    <script>
    const ownerName = '{owner_name_js}';
    async function doVote(id, v) {{
        const card = document.querySelector(`[data-idea="${{id}}"]`) || event.target.closest('.card');
        const btn = event.target;
        btn.disabled = true;
        btn.textContent = '...';
        await fetch(`/api/ideas/${{id}}/vote`, {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{voter: 'owner', vote: v}})
        }});
        // Update card state visually
        if (card) {{
            card.className = 'card ' + (v === 'approve' ? 'approved' : 'vetoed');
            const stateEl = card.querySelector('.badge-state');
            if (stateEl) {{ stateEl.textContent = v === 'approve' ? 'approved' : 'vetoed'; stateEl.className = 'badge badge-state ' + (v === 'approve' ? 'approved' : 'vetoed'); }}
            const actions = card.querySelector('.actions');
            if (v === 'veto' && actions) actions.remove();
        }}
    }}
    async function doExecute(id) {{
        const btn = event.target;
        btn.disabled = true;
        btn.textContent = 'Copying...';

        // Fetch the formatted prompt
        const resp = await fetch(`/api/ideas/${{id}}/prompt`);
        const data = await resp.json();
        const prompt = data.prompt;

        // Copy to clipboard
        let copied = false;
        try {{
            await navigator.clipboard.writeText(prompt);
            copied = true;
        }} catch(e) {{
            // Fallback for non-HTTPS or older browsers
            const ta = document.createElement('textarea');
            ta.value = prompt;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            copied = document.execCommand('copy');
            document.body.removeChild(ta);
        }}

        // Show toast notification
        showToast(copied
            ? 'Instructions copied to clipboard! Paste into Claude Code.'
            : 'Could not copy — open Claude Code and use the prompt below.');

        // Update button
        btn.textContent = copied ? 'Copied to clipboard' : 'Copy failed';
        btn.style.background = copied ? 'var(--green)' : 'var(--red)';

        // Mark as approved
        await fetch(`/api/ideas/${{id}}/vote`, {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{voter: 'owner', vote: 'approve'}})
        }});

        // Reset button after 5 seconds
        setTimeout(() => {{
            btn.textContent = 'Copy for Claude Code';
            btn.style.background = '';
            btn.disabled = false;
        }}, 5000);
    }}

    async function doExecuteEpic(id) {{
        const btn = event.target;
        btn.disabled = true;
        btn.textContent = 'Copying epic...';

        const resp = await fetch(`/api/ideas/${{id}}/epic_prompt`);
        const data = await resp.json();
        const prompt = data.prompt;

        let copied = false;
        try {{
            await navigator.clipboard.writeText(prompt);
            copied = true;
        }} catch(e) {{
            const ta = document.createElement('textarea');
            ta.value = prompt;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            copied = document.execCommand('copy');
            document.body.removeChild(ta);
        }}

        showToast(copied
            ? 'Epic instructions copied! Paste into Claude Code to implement all stories.'
            : 'Could not copy — try manually.');

        btn.textContent = copied ? 'Epic copied!' : 'Copy failed';
        btn.style.background = copied ? 'var(--green)' : 'var(--red)';

        // Approve all child stories
        const cards = document.querySelectorAll('[data-idea]');
        for (const card of cards) {{
            const parentLink = card.querySelector('a[onclick*="' + id + '"]');
            if (parentLink) {{
                const childId = card.dataset.idea;
                await fetch(`/api/ideas/${{childId}}/vote`, {{
                    method: 'POST', headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{voter: 'owner', vote: 'approve'}})
                }});
            }}
        }}
        // Approve the epic itself
        await fetch(`/api/ideas/${{id}}/vote`, {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{voter: 'owner', vote: 'approve'}})
        }});

        setTimeout(() => {{
            btn.textContent = 'Copy Epic for Claude Code';
            btn.style.background = '';
            btn.disabled = false;
        }}, 5000);
    }}

    async function doMarkDone(id) {{
        await fetch(`/api/ideas/${{id}}/done`, {{method: 'POST'}});
        showToast('Marked as done.');
        const card = document.querySelector(`[data-idea="${{id}}"]`);
        if (card) {{
            card.className = 'card done';
            const stateEl = card.querySelector('.badge-state');
            if (stateEl) {{ stateEl.textContent = 'done'; stateEl.className = 'badge badge-state done'; }}
            const actions = card.querySelector('.actions');
            if (actions) {{ actions.innerHTML = '<button class="btn btn-delete" onclick="doDelete(\\'' + id + '\\')">Delete</button>'; }}
        }}
    }}

    async function doDelete(id) {{
        if (!confirm('Permanently delete this idea?')) return;
        await fetch(`/api/ideas/${{id}}`, {{method: 'DELETE'}});
        showToast('Idea deleted.');
        const card = document.querySelector(`[data-idea="${{id}}"]`);
        if (card) card.remove();
    }}

    function showToast(msg) {{
        // Remove existing toast
        const old = document.getElementById('toast');
        if (old) old.remove();

        const toast = document.createElement('div');
        toast.id = 'toast';
        toast.textContent = msg;
        toast.style.cssText = `
            position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
            background: var(--surface); color: var(--text); padding: 14px 28px;
            border-radius: 8px; border: 1px solid var(--accent); font-size: 0.95rem;
            z-index: 9999; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            animation: fadeIn 0.3s ease;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }}

    async function doCancel(id) {{
        if (!confirm('Cancel this execution?')) return;
        await fetch(`/api/ideas/${{id}}/cancel`, {{method: 'POST'}});
    }}

    function startLogPolling(id, card) {{
        // Poll execution log every 3 seconds
        const poll = setInterval(async () => {{
            try {{
                // Check idea state
                const ideaResp = await fetch(`/api/ideas/${{id}}`);
                const idea = await ideaResp.json();

                // Get live log
                const logResp = await fetch(`/api/ideas/${{id}}/log`);
                const logData = await logResp.json();

                const logEl = document.getElementById(`log-${{id}}`);
                if (logEl && logData.lines.length > 0) {{
                    logEl.textContent = logData.lines.slice(-50).join('\\n');
                    logEl.scrollTop = logEl.scrollHeight;
                }}

                // Update elapsed time
                const statusEl = card.querySelector('.exec-status');
                if (statusEl && logData.is_alive) {{
                    statusEl.textContent = `Claude Code is working... (${{Math.round(logData.elapsed)}}s)`;
                }}

                // Check for completion
                if (idea.state === 'done') {{
                    clearInterval(poll);
                    const stateEl = card.querySelector('.badge-state');
                    if (stateEl) {{ stateEl.textContent = 'done'; stateEl.className = 'badge badge-state done'; }}
                    card.className = 'card done';
                    if (statusEl) {{ statusEl.textContent = 'Done'; statusEl.className = 'exec-status'; statusEl.style.color = 'var(--green)'; }}
                    const cancelBtn = card.querySelector('.btn-cancel');
                    if (cancelBtn) cancelBtn.remove();
                }} else if (idea.state === 'failed') {{
                    clearInterval(poll);
                    const stateEl = card.querySelector('.badge-state');
                    if (stateEl) {{ stateEl.textContent = 'failed'; stateEl.className = 'badge badge-state failed'; }}
                    card.className = 'card failed';
                    if (statusEl) {{ statusEl.textContent = 'Failed'; statusEl.className = 'exec-status'; statusEl.style.color = 'var(--red)'; }}
                    const cancelBtn = card.querySelector('.btn-cancel');
                    if (cancelBtn) cancelBtn.remove();
                }}
            }} catch(e) {{}}
        }}, 3000);
    }}

    // Auto-start log polling for any cards already in executing state
    document.querySelectorAll('.card.executing').forEach(card => {{
        const id = card.dataset.idea;
        if (id) startLogPolling(id, card);
    }})
    async function doComment(e, id) {{
        e.preventDefault();
        const input = e.target.text;
        const text = input.value.trim();
        if (!text) return;
        input.value = '';

        // Show the message immediately
        const commentsDiv = e.target.closest('.comments');
        commentsDiv.insertAdjacentHTML('beforeend',
            `<div class="comment owner">` +
            `<div class="comment-author owner">${ownerName}</div>` +
            `<div class="comment-text">${{text}}</div>` +
            `</div>`
        );

        // Show thinking indicator
        commentsDiv.insertAdjacentHTML('beforeend',
            `<div class="thinking" id="thinking-${{id}}">LLM is thinking...</div>`
        );

        // Send to API (LLM reply happens in background)
        await fetch(`/api/ideas/${{id}}/comment`, {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{author: 'owner', text: text}})
        }});

        // Poll for the LLM reply (check every 2 seconds for up to 60 seconds)
        let attempts = 0;
        const poll = setInterval(async () => {{
            attempts++;
            const resp = await fetch(`/api/ideas/${{id}}`);
            const idea = await resp.json();
            const comments = idea.comments || [];
            const lastComment = comments[comments.length - 1];
            if (lastComment && lastComment.author === 'llm') {{
                clearInterval(poll);
                const el = document.getElementById(`thinking-${{id}}`);
                if (el) el.remove();
                commentsDiv.insertAdjacentHTML('beforeend',
                    `<div class="comment llm">` +
                    `<div class="comment-author llm">LLM</div>` +
                    `<div class="comment-text">${{lastComment.text}}</div>` +
                    `</div>`
                );
            }}
            if (attempts >= 30) {{
                clearInterval(poll);
                const el = document.getElementById(`thinking-${{id}}`);
                if (el) el.textContent = 'LLM did not respond.';
            }}
        }}, 2000);
    }}

    // Auto-scroll comment threads to bottom on page load
    document.querySelectorAll('.comments').forEach(el => {{
        el.scrollTop = el.scrollHeight;
    }});

    // Background poll for new ideas (updates stats bar, no page reload)
    let lastIdeaCount = {total};
    setInterval(async () => {{
        try {{
            const resp = await fetch('/api/ideas');
            const ideas = await resp.json();
            if (ideas.length > lastIdeaCount) {{
                const diff = ideas.length - lastIdeaCount;
                const statsEl = document.getElementById('stats');
                if (statsEl) {{
                    statsEl.innerHTML = `${{ideas.length}} ideas total &bull; `
                        + `<strong style="color:var(--green)">${{diff}} new!</strong> `
                        + `&bull; <a href="/" style="color:var(--accent)">Refresh to see</a>`;
                }}
                lastIdeaCount = ideas.length;
            }}
        }} catch(e) {{}}
    }}, 30000);

    // Evolve status polling
    async function updateEvolveStatus() {{
        try {{
            const resp = await fetch('/api/evolve/status');
            const data = await resp.json();
            const el = document.getElementById('evolve-content');
            if (!el) return;

            if (data.running) {{
                const pct = data.tests_total > 0
                    ? Math.round((data.tests_completed / data.tests_total) * 100) : 0;
                const filled = Math.round(pct / 5);
                const bar = '\u2588'.repeat(filled) + '\u2591'.repeat(20 - filled);

                let logHtml = '';
                const recentLog = (data.log || []).slice(-8);
                for (const line of recentLog) {{
                    const color = line.includes('FAIL') ? 'var(--red)'
                        : line.includes('PASS') ? 'var(--green)' : 'var(--muted)';
                    logHtml += `<div style="color:${{color}};font-size:0.8rem;font-family:monospace">${{line}}</div>`;
                }}

                el.innerHTML = `
                    <div style="margin-bottom:0.5rem">
                        <span class="thinking">Running</span> &bull;
                        Phase: <strong>${{data.phase}}</strong> &bull;
                        ${{data.tests_completed}}/${{data.tests_total}} tests &bull;
                        Avg: ${{data.avg_score || 0}}/10
                    </div>
                    <div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;color:var(--accent)">
                        ${{bar}} ${{pct}}%
                    </div>
                    ${{logHtml}}
                `;
                document.getElementById('evolve-panel').style.borderLeftColor = 'var(--orange)';
            }} else if (data.phase === 'complete' || data.last_run) {{
                const lastRun = data.last_run || data.started || 'unknown';
                const ts = lastRun.includes('T') ? lastRun.split('T')[1]?.slice(0,5) || lastRun : lastRun;
                el.innerHTML = `
                    Last run: ${{ts}} &bull;
                    ${{data.tests_total || '?'}} tests &bull;
                    Avg: ${{data.avg_score || '?'}}/10 &bull;
                    ${{data.tests_passed || '?'}} passed
                `;
                document.getElementById('evolve-panel').style.borderLeftColor = 'var(--green)';
            }} else {{
                el.textContent = 'Never run. Type "evolve" in Discord to start.';
            }}
        }} catch(e) {{}}
    }}

    updateEvolveStatus();
    setInterval(updateEvolveStatus, 5000);
    </script>
</body>
</html>"""


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/ideas")
def dashboard() -> str:
    """Serve the HTML idea board dashboard."""
    ideas = [i.to_dict() for i in load_ideas()]
    return _render_dashboard(ideas)


@app.route("/")
def hub() -> str:
    """Serve the central Technomancer hub page."""
    return _render_hub()


@app.route("/api/ideas")
def api_ideas() -> tuple:
    """GET /api/ideas — list all ideas, optionally filtered by state."""
    ideas = load_ideas()
    state_filter = request.args.get("state")
    if state_filter:
        ideas = [i for i in ideas if i.state == state_filter]
    return jsonify([i.to_dict() for i in ideas])


@app.route("/api/ideas/<idea_id>")
def api_idea(idea_id: str) -> tuple:
    """GET /api/ideas/<id> — get a single idea by ID."""
    idea = get_idea(idea_id)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/vote", methods=["POST"])
def api_vote(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/vote — record a vote."""
    data = request.get_json(silent=True) or {}
    voter = data.get("voter", "")
    vote_value = data.get("vote", "")
    if voter not in ("owner", "claude") or not vote_value:
        return jsonify({"error": "Need voter (owner|claude) and vote"}), 400

    idea = vote(idea_id, voter, vote_value)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/comment", methods=["POST"])
def api_comment(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/comment — add a comment and get LLM reply.

    When the owner posts a comment, the LLM reads the full idea context
    and conversation history, then replies as an eager employee
    discussing the idea with their manager.
    """
    data = request.get_json(silent=True) or {}
    author = data.get("author", "owner")
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Need text"}), 400

    idea = add_comment(idea_id, author, text)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404

    # Notify Discord when Claude reports a failure
    if author == "claude" and "execution failed" in text.lower():
        _notify_idea_failed(idea_id, idea.title, text)

    # If the owner posted, trigger an LLM reply in a background thread
    if author == "owner":
        import threading

        def _llm_reply() -> None:
            try:
                reply = _generate_idea_reply(idea)
                if reply:
                    add_comment(idea_id, "llm", reply)
            except Exception as e:
                logger.error(f"[IdeaBoard] LLM reply failed: {e}")

        threading.Thread(target=_llm_reply, daemon=True).start()

    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/done", methods=["POST"])
def api_done(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/done — manually mark an idea as implemented."""
    idea = mark_done(idea_id, f"Manually marked as done by {settings.owner_name}.")
    if not idea:
        return jsonify({"error": "Idea not found"}), 404

    _notify_idea_complete(idea_id, idea.title)
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>", methods=["DELETE"])
def api_delete(idea_id: str) -> tuple:
    """DELETE /api/ideas/<id> — permanently remove an idea from the board."""
    if delete_idea(idea_id):
        return jsonify({"status": "deleted", "idea_id": idea_id})
    return jsonify({"error": "Idea not found"}), 404


@app.route("/api/ideas/<idea_id>/type", methods=["POST"])
def api_set_type(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/type — change an idea's type (epic/story/task)."""
    data = request.get_json(silent=True) or {}
    new_type = data.get("idea_type", "").strip().lower()
    if new_type not in ("epic", "story", "task"):
        return jsonify({"error": "idea_type must be epic, story, or task"}), 400

    ideas = load_ideas()
    for idea in ideas:
        if idea.id == idea_id:
            idea.idea_type = new_type
            save_ideas(ideas)
            return jsonify(idea.to_dict())
    return jsonify({"error": "Idea not found"}), 404


@app.route("/api/ideas/<idea_id>/add_story", methods=["POST"])
def api_add_story(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/add_story — create a new story under an epic."""
    parent = get_idea(idea_id)
    if not parent:
        return jsonify({"error": "Parent idea not found"}), 404

    data = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()
    description = data.get("description", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    idea = add_idea(
        title=title,
        description=description or f"Story under epic: {parent.title}",
        source=parent.source,
        category=parent.category,
        idea_type="story",
        parent_id=idea_id,
    )
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/prompt")
def api_prompt(idea_id: str) -> tuple:
    """GET /api/ideas/<id>/prompt — build a ready-to-paste prompt for Claude Code.

    Formats the idea title, description, and full discussion thread into
    a prompt that can be pasted directly into an interactive Claude Code
    session (VS Code or claude.ai/code).
    """
    idea = get_idea(idea_id)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404

    # Build discussion context
    discussion = ""
    if idea.comments:
        discussion = "\n\nDiscussion (what was decided):\n"
        for c in idea.comments:
            label = f"{settings.owner_name} (manager)" if c.author == "owner" else "LLM (engineer)"
            discussion += f"- {label}: {c.text}\n"

    # Build epic context if this story belongs to an epic
    epic_context = ""
    if idea.parent_id:
        parent = get_idea(idea.parent_id)
        if parent:
            sibling_ideas = load_ideas()
            siblings = [i for i in sibling_ideas if i.parent_id == idea.parent_id and i.id != idea.id]
            sibling_info = ""
            for s in siblings:
                done_marker = " [DONE]" if s.state == "done" else ""
                sibling_info += f"  - {s.id}: {s.title}{done_marker}\n"
            epic_context = (
                f"\n## Parent Epic: {parent.title}\n"
                f"**Epic Description:** {parent.description}\n"
                f"**Other stories in this epic:**\n{sibling_info}\n"
                f"This story is part of a larger initiative. Ensure your implementation "
                f"integrates with the other stories and contributes to the epic's full lifecycle goal.\n"
            )

    # Build sibling context if this IS an epic
    children_context = ""
    if idea.idea_type == "epic":
        all_ideas = load_ideas()
        kids = [i for i in all_ideas if i.parent_id == idea.id]
        if kids:
            children_context = "\n**Stories in this epic:**\n"
            for k in kids:
                done_marker = " [DONE]" if k.state == "done" else ""
                children_context += f"  - {k.id}: {k.title}{done_marker}\n"
            children_context += "\n"

    type_label = f"[{idea.idea_type.upper()}] " if idea.idea_type != "story" else ""

    prompt = (
        f"Implement this improvement for the Technomancer project.\n\n"
        f"## {type_label}Idea: {idea.title}\n\n"
        f"**Description:** {idea.description}\n"
        f"**Category:** {idea.category}\n"
        f"{epic_context}{children_context}"
        f"{discussion}\n"
        f"## Requirements\n"
        f"- Follow the safe_update.py git workflow (branch, code, validate, commit, merge)\n"
        f"- Run `python validate.py startup` before committing\n"
        f"- Verify `python bot_service.py status` shows Bot running: True after deploy\n"
        f"- Read CLAUDE.md for project conventions\n\n"
        f"## After completion — update the idea board\n"
        f"When you are DONE and everything is deployed and verified, run:\n"
        f'```bash\ncurl -X POST http://localhost:8322/api/ideas/{idea.id}/done\n```\n\n'
        f"If you CANNOT complete this task or it fails, run:\n"
        f'```bash\ncurl -X POST http://localhost:8322/api/ideas/{idea.id}/comment '
        f'-H "Content-Type: application/json" '
        f"-d '{{\"author\": \"claude\", \"text\": \"Execution failed: <describe what went wrong>\"}}'\n```\n"
    )

    return jsonify({"idea_id": idea_id, "prompt": prompt})


@app.route("/api/ideas/<idea_id>/epic_prompt")
def api_epic_prompt(idea_id: str) -> tuple:
    """GET /api/ideas/<id>/epic_prompt — build a prompt for implementing an entire epic.

    Generates a master prompt that includes the epic context and every child
    story in order, instructing Claude Code to implement them sequentially
    with a full safe_update cycle for each.
    """
    epic = get_idea(idea_id)
    if not epic:
        return jsonify({"error": "Idea not found"}), 404

    all_ideas = load_ideas()
    stories = [i for i in all_ideas if i.parent_id == idea_id and i.state != "done"]
    done_stories = [i for i in all_ideas if i.parent_id == idea_id and i.state == "done"]

    if not stories and not done_stories:
        # Not an epic or no children — fall back to single prompt
        return api_prompt(idea_id)

    # Build the story sections
    story_sections = ""
    for idx, story in enumerate(stories, 1):
        discussion = ""
        if story.comments:
            discussion = "Discussion:\n"
            for c in story.comments:
                label = settings.owner_name if c.author == "owner" else "LLM"
                discussion += f"  - {label}: {c.text}\n"

        story_sections += (
            f"\n{'=' * 70}\n"
            f"## Story {idx}/{len(stories)}: {story.title}\n"
            f"**ID:** {story.id}\n"
            f"**Category:** {story.category}\n\n"
            f"**Description:** {story.description}\n"
            f"{discussion}\n"
            f"**After completing this story**, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/done\n"
            f"```\n"
            f"If this story fails, run:\n"
            f"```bash\n"
            f"curl -X POST http://localhost:8322/api/ideas/{story.id}/comment "
            f"-H \"Content-Type: application/json\" "
            f"-d '{{\"author\": \"claude\", \"text\": \"Execution failed: <describe what went wrong>\"}}'\n"
            f"```\n"
            f"Then move to the next story.\n"
        )

    # Done stories context
    done_context = ""
    if done_stories:
        done_context = "\n## Already Completed Stories\n"
        for d in done_stories:
            done_context += f"- {d.id}: {d.title} [DONE]\n"
        done_context += "\nThese are already implemented. Build on them, don't duplicate them.\n"

    prompt = (
        f"# EPIC: {epic.title}\n\n"
        f"You are implementing an entire epic for the Technomancer project.\n"
        f"This epic has **{len(stories)} stories** to implement sequentially.\n\n"
        f"## Epic Description\n"
        f"{epic.description}\n"
        f"{done_context}\n"
        f"## Implementation Process\n\n"
        f"For EACH story below, follow this exact cycle:\n"
        f"1. Read CLAUDE.md for project conventions\n"
        f"2. Run `python safe_update.py <short-name>` to create a branch\n"
        f"3. Implement the story (code, tests)\n"
        f"4. Run `python validate.py startup` before committing\n"
        f"5. Commit and run `python safe_update.py continue` to test, merge, restart\n"
        f"6. Verify `python bot_service.py status` shows Bot running: True\n"
        f"7. Mark the story done with the curl command provided\n"
        f"8. Move to the next story\n\n"
        f"IMPORTANT:\n"
        f"- Each story gets its OWN safe_update branch and commit\n"
        f"- Do NOT batch multiple stories into one branch\n"
        f"- If a story fails, log it and move to the next one\n"
        f"- Each story should build on what the previous stories created\n"
        f"- When ALL stories are complete, mark the epic done:\n"
        f"```bash\n"
        f"curl -X POST http://localhost:8322/api/ideas/{epic.id}/done\n"
        f"```\n"
        f"\n# Stories to Implement\n"
        f"{story_sections}"
    )

    return jsonify({"idea_id": idea_id, "prompt": prompt})


@app.route("/api/ideas/<idea_id>/execute", methods=["POST"])
def api_execute(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/execute — trigger Claude Code to implement this idea.

    Uses the executor module for live streaming, PID tracking,
    auto-timeout recovery, and cancel support.
    """
    from .executor import execute_idea

    state = execute_idea(idea_id)
    if not state:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify({"status": "executing", "idea_id": idea_id, "pid": state.pid})


@app.route("/api/ideas/<idea_id>/cancel", methods=["POST"])
def api_cancel(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/cancel — cancel a running execution."""
    from .executor import cancel_execution

    if cancel_execution(idea_id):
        return jsonify({"status": "cancelling", "idea_id": idea_id})
    return jsonify({"error": "Not currently executing"}), 400


@app.route("/api/ideas/<idea_id>/log")
def api_log(idea_id: str) -> tuple:
    """GET /api/ideas/<id>/log — get the live execution log.

    Returns the current stdout buffer from the running Claude Code
    process. Poll this every few seconds for live updates.
    """
    from .executor import get_execution

    state = get_execution(idea_id)
    if state:
        return jsonify({
            "idea_id": idea_id,
            "pid": state.pid,
            "elapsed": round(state.elapsed, 1),
            "is_alive": state.is_alive,
            "lines": state.log_lines,
            "line_count": len(state.log_lines),
        })

    # Not actively executing — return stored log from idea
    idea = get_idea(idea_id)
    if idea and idea.execution_log:
        return jsonify({
            "idea_id": idea_id,
            "pid": None,
            "elapsed": 0,
            "is_alive": False,
            "lines": idea.execution_log.split("\n"),
            "line_count": len(idea.execution_log.split("\n")),
        })

    return jsonify({"idea_id": idea_id, "lines": [], "line_count": 0})


@app.route("/api/evolve/status")
def api_evolve_status() -> tuple:
    """GET /api/evolve/status — get evolve cycle progress.

    Reads .evolve_status.json for live progress, falls back to
    .auto_improve_report.json for last-run summary.
    """
    import os
    from pathlib import Path as _Path

    status_file = _Path(__file__).parent.parent / ".evolve_status.json"
    report_file = _Path(__file__).parent.parent / ".auto_improve_report.json"

    # Check live status first
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            # Verify PID is still alive if marked as running
            if data.get("running") and data.get("pid"):
                try:
                    os.kill(data["pid"], 0)
                except (OSError, ProcessLookupError):
                    data["running"] = False
                    data["phase"] = "crashed"
                    data["progress"] = "Process died unexpectedly"
            return jsonify(data)
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to last report
    if report_file.exists():
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
            scores = [r.get("score", 0) for r in report.get("results", []) if "score" in r]
            return jsonify({
                "running": False,
                "phase": "complete",
                "last_run": report.get("timestamp", "unknown"),
                "tests_total": len(scores),
                "tests_passed": sum(1 for s in scores if s >= 7),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "log": [],
            })
        except (json.JSONDecodeError, OSError):
            pass

    return jsonify({"running": False, "phase": "never_run", "log": []})


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

# Import here to avoid circular — IDEAS_DIR used in _execute
from .models import IDEAS_DIR


HUB_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --orange: #ff9800;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.6; }
h1 { margin-bottom: 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px; }
.card { background: var(--surface); border-radius: 8px; padding: 1.5rem;
        border-left: 4px solid var(--accent); text-decoration: none;
        color: var(--text); transition: transform 0.15s, box-shadow 0.15s; display: block; }
.card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
.card h2 { font-size: 1.1rem; margin-bottom: 0.4rem; color: var(--accent); }
.card p { font-size: 0.9rem; color: var(--muted); }
.card.green { border-left-color: var(--green); }
.card.orange { border-left-color: var(--orange); }
.card.external { border-left-color: var(--muted); }
.card .badge { font-size: 0.7rem; background: #333; padding: 2px 8px;
               border-radius: 4px; color: var(--muted); margin-top: 0.5rem;
               display: inline-block; }
.evolve-panel { background: var(--surface); border-radius: 8px; padding: 1.5rem;
                border-left: 4px solid var(--muted); margin-top: 1.5rem; }
.evolve-panel h2 { font-size: 1.1rem; margin-bottom: 0.6rem; color: var(--accent); }
.evolve-panel .status-line { font-size: 0.9rem; color: var(--muted); }
.evolve-panel .log-line { font-size: 0.8rem; font-family: monospace; margin-top: 2px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.thinking { animation: pulse 1.5s infinite; color: var(--orange); font-weight: bold; }
"""


def _render_hub() -> str:
    """Render the central hub page with links to all services."""
    ideas = load_ideas()
    proposed = len([i for i in ideas if i.state == "proposed"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technomancer Hub</title>
    <style>{HUB_CSS}</style>
</head>
<body>
    <h1>Technomancer Hub</h1>
    <p class="subtitle">Central control panel for all Technomancer services</p>

    <div class="grid">
        <a href="/ideas" class="card green">
            <h2>Idea Board</h2>
            <p>View, vote, and discuss improvement ideas.</p>
            <span class="badge">{proposed} pending review</span>
        </a>
        <a href="/news" class="card orange">
            <h2>News Config</h2>
            <p>Manage RSS feeds, topic preferences, and digest schedule.</p>
        </a>
        <a href="/karen" class="card" style="border-left: 4px solid #e94560;">
            <h2>K.A.R.E.N.</h2>
            <p>Submit complaints. They get turned into improvement ideas.</p>
        </a>
        <a href="http://localhost:9090" target="_blank" class="card external">
            <h2>Prometheus</h2>
            <p>Metrics and monitoring dashboard.</p>
            <span class="badge">:9090</span>
        </a>
        <a href="http://localhost:3000" target="_blank" class="card external">
            <h2>Grafana</h2>
            <p>Visualization and alerting dashboards.</p>
            <span class="badge">:3000</span>
        </a>
        <a href="http://localhost:8321" target="_blank" class="card external">
            <h2>Discord Bridge API</h2>
            <p>REST API for sending messages through the Discord bot.</p>
            <span class="badge">:8321</span>
        </a>
    </div>

    <div class="evolve-panel" id="evolve-panel">
        <h2>Evolve Status</h2>
        <div id="evolve-content" class="status-line">Loading...</div>
    </div>

    <script>
    async function updateEvolveStatus() {{
        try {{
            const resp = await fetch('/api/evolve/status');
            const data = await resp.json();
            const el = document.getElementById('evolve-content');
            const panel = document.getElementById('evolve-panel');
            if (!el) return;

            if (data.running) {{
                const pct = data.tests_total > 0
                    ? Math.round((data.tests_completed / data.tests_total) * 100) : 0;
                const filled = Math.round(pct / 5);
                const bar = '\u2588'.repeat(filled) + '\u2591'.repeat(20 - filled);

                let logHtml = '';
                for (const line of (data.log || []).slice(-6)) {{
                    const color = line.includes('FAIL') ? 'var(--red)'
                        : line.includes('PASS') ? 'var(--green)' : 'var(--muted)';
                    logHtml += '<div class="log-line" style="color:' + color + '">' + line + '</div>';
                }}

                el.innerHTML = '<div style="margin-bottom:0.5rem">'
                    + '<span class="thinking">Running</span> &bull; '
                    + 'Phase: <strong>' + data.phase + '</strong> &bull; '
                    + (data.tests_completed || 0) + '/' + (data.tests_total || 0) + ' tests &bull; '
                    + 'Avg: ' + (data.avg_score || 0) + '/10'
                    + '</div>'
                    + '<div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;color:var(--accent)">'
                    + bar + ' ' + pct + '%</div>'
                    + logHtml;
                panel.style.borderLeftColor = 'var(--orange)';
            }} else if (data.phase === 'complete' || data.last_run) {{
                const lastRun = data.last_run || data.started || 'unknown';
                const ts = lastRun.includes('T') ? lastRun.split('T')[1]?.slice(0,5) || lastRun : lastRun;
                el.innerHTML = 'Last run: ' + ts + ' &bull; '
                    + (data.tests_total || '?') + ' tests &bull; '
                    + 'Avg: ' + (data.avg_score || '?') + '/10 &bull; '
                    + (data.tests_passed || '?') + ' passed';
                panel.style.borderLeftColor = 'var(--green)';
            }} else {{
                el.textContent = 'Never run. Type "evolve" in Discord to start.';
            }}
        }} catch(e) {{}}
    }}
    updateEvolveStatus();
    setInterval(updateEvolveStatus, 5000);
    </script>
</body>
</html>"""


def start_idea_board() -> None:
    """Start the Flask hub (idea board + news config) in a daemon thread.

    Binds to 0.0.0.0 so it's accessible over Tailscale/LAN.
    """
    # Register blueprints
    from .karen import karen_bp
    from .news_config import news_bp
    app.register_blueprint(news_bp)
    app.register_blueprint(karen_bp)

    def _run() -> None:
        app.run(host="0.0.0.0", port=BOARD_PORT, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="idea-board")
    thread.start()
    logger.info(f"Technomancer Hub running on http://0.0.0.0:{BOARD_PORT}")
