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
    GET  /api/ideas/<id>/log/stream — SSE stream for live execution log
    GET  /api/errors              — JSON list of recent crashes from crash_log.md
    GET  /errors                  — HTML crash log viewer with collapsible stack traces
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import time

from flask import Flask, Response, jsonify, request

from agent.config import settings

from .executor import get_execution
from .models import (
    Idea,
    add_comment,
    add_idea,
    delete_idea,
    get_execution_order,
    get_idea,
    load_ideas,
    mark_done,
    mark_executing,
    mark_failed,
    save_ideas,
    set_epic_context,
    set_execution_order,
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
.btn-run { background: #2196F3; color: white; font-weight: 600; }
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
.drag-child.drag-over { border-top: 2px solid #7c3aed; }
.drag-handle { display: inline-block; vertical-align: middle; }
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
    copy_btn = f'<button class="btn btn-execute" onclick="doExecute(\'{eid}\')">Copy for Claude Code</button>'
    epic_copy_btn = f'<button class="btn btn-epic-copy" onclick="doExecuteEpic(\'{eid}\')">Copy Epic for Claude Code</button>'
    run_btn = f'<button class="btn btn-run" onclick="doRunExecute(\'{eid}\')">Execute</button>'
    done_btn = f'<button class="btn btn-done" onclick="doMarkDone(\'{eid}\')">Mark Done</button>'
    delete_btn = f'<button class="btn btn-delete" onclick="doDelete(\'{eid}\')">Delete</button>'
    clipboard_btn = epic_copy_btn if idea_type == "epic" else copy_btn
    if state in ("proposed", "refining"):
        actions = f'<div class="actions">{approve_btn} {veto_btn} {run_btn} {clipboard_btn} {delete_btn}</div>'
    elif state == "approved":
        actions = f'<div class="actions">{run_btn} {clipboard_btn} {done_btn} {delete_btn}</div>'
    elif state == "failed":
        actions = f'<div class="actions">{run_btn} {clipboard_btn} {done_btn} {delete_btn}</div>'
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

    # Epic context section (only for epics with context set)
    epic_context_html = ""
    if idea_type == "epic" and idea.get("epic_context"):
        ctx_text = html.escape(idea["epic_context"])
        epic_context_html = f"""
        <div class="epic-context" style="margin: 0.8rem 0; padding: 0.6rem; background: rgba(74,26,107,0.15); border-radius: 6px; border-left: 3px solid #7c3aed;">
            <div style="font-size: 0.8rem; color: #c084fc; font-weight: 600; margin-bottom: 4px;">Epic Context</div>
            <div style="font-size: 0.9rem; white-space: pre-wrap;">{ctx_text}</div>
        </div>
        <div style="margin-top: 4px;">
            <button class="btn" style="font-size:0.75rem;padding:3px 8px;background:transparent;color:var(--muted);border:1px solid var(--border)" onclick="editEpicContext('{eid}')">Edit Context</button>
        </div>"""
    elif idea_type == "epic":
        epic_context_html = f"""
        <div style="margin: 0.5rem 0;">
            <button class="btn" style="font-size:0.75rem;padding:3px 8px;background:transparent;color:var(--muted);border:1px solid var(--border)" onclick="editEpicContext('{eid}')">Add Epic Context</button>
        </div>"""

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
        {epic_context_html}
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

    def _order_children(parent: dict, kids: list[dict]) -> list[dict]:
        """Sort children by execution_order if set, else by creation order."""
        order = parent.get("execution_order", [])
        if not order:
            return kids
        kid_by_id = {k["id"]: k for k in kids}
        ordered = [kid_by_id[oid] for oid in order if oid in kid_by_id]
        # Append any children not in execution_order at the end
        ordered_ids = set(order)
        for k in kids:
            if k["id"] not in ordered_ids:
                ordered.append(k)
        return ordered

    def _render_state_section(state: str, items: list[dict]) -> str:
        html_out = f'<h2 class="section-title">{state.upper()} ({len(items)})</h2>\n'
        for idea in items:
            idea_type = idea.get("idea_type", "story")
            kids = children_of.get(idea["id"], [])
            if idea_type == "epic" or kids:
                html_out += '<div class="epic-group">\n'
                html_out += _render_idea_card(idea)
                if kids:
                    ordered_kids = _order_children(idea, kids)
                    epic_id = html.escape(idea["id"])
                    html_out += f'<div class="epic-children" id="children-{epic_id}" data-epic="{epic_id}">\n'
                    for idx, child in enumerate(ordered_kids):
                        child_id = html.escape(child["id"])
                        html_out += f'<div class="drag-child" draggable="true" data-child-id="{child_id}" data-order="{idx}">\n'
                        html_out += f'<span class="drag-handle" style="cursor:grab;color:var(--muted);margin-right:6px;font-size:0.9rem" title="Drag to reorder">&#9776;</span>'
                        html_out += _render_idea_card(child, is_child=True)
                        html_out += '</div>\n'
                    html_out += '</div>\n'
                html_out += '</div>\n'
            else:
                html_out += _render_idea_card(idea)
        return html_out

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
        <a href="/analytics">Analytics</a>
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

    async function doRunExecute(id) {{
        if (!confirm('Run Claude Code autonomously on this idea? It will use safe_update workflow.')) return;
        const btn = event.target;
        btn.disabled = true;
        btn.textContent = 'Starting...';
        try {{
            const resp = await fetch(`/api/ideas/${{id}}/execute`, {{method: 'POST'}});
            const data = await resp.json();
            if (resp.ok) {{
                showToast('Claude Code execution started! Watch the log below.');
                btn.textContent = 'Running...';
                btn.style.background = 'var(--green)';
                const card = document.querySelector(`[data-idea="${{id}}"]`);
                if (card) startLogStream(id, card);
            }} else {{
                showToast(data.error || 'Failed to start execution');
                btn.textContent = 'Execute';
                btn.style.background = '';
                btn.disabled = false;
            }}
        }} catch(e) {{
            showToast('Error: ' + e.message);
            btn.textContent = 'Execute';
            btn.style.background = '';
            btn.disabled = false;
        }}
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

    function updateCardState(card, statusEl, ideaState) {{
        const stateEl = card.querySelector('.badge-state');
        const cancelBtn = card.querySelector('.btn-cancel');
        if (ideaState === 'done') {{
            if (stateEl) {{ stateEl.textContent = 'done'; stateEl.className = 'badge badge-state done'; }}
            card.className = 'card done';
            if (statusEl) {{ statusEl.textContent = 'Done'; statusEl.className = 'exec-status'; statusEl.style.color = 'var(--green)'; }}
        }} else if (ideaState === 'failed') {{
            if (stateEl) {{ stateEl.textContent = 'failed'; stateEl.className = 'badge badge-state failed'; }}
            card.className = 'card failed';
            if (statusEl) {{ statusEl.textContent = 'Failed'; statusEl.className = 'exec-status'; statusEl.style.color = 'var(--red)'; }}
        }}
        if (cancelBtn && (ideaState === 'done' || ideaState === 'failed')) cancelBtn.remove();
    }}

    function startLogPoll(id, card, initialLines) {{
        const logEl = document.getElementById(`log-${{id}}`);
        const statusEl = card.querySelector('.exec-status');
        let allLines = initialLines || [];
        if (statusEl) statusEl.textContent = 'Claude Code is working... (polling)';

        const iv = setInterval(async () => {{
            try {{
                const resp = await fetch(`/api/ideas/${{id}}/log`);
                if (!resp.ok) return;
                const data = await resp.json();
                allLines = data.lines || [];
                if (logEl && allLines.length > 0) {{
                    logEl.textContent = allLines.slice(-50).join('\\n');
                    logEl.scrollTop = logEl.scrollHeight;
                }}
                if (statusEl && data.is_alive) {{
                    statusEl.textContent = `Claude Code is working... (${{Math.round(data.elapsed)}}s)`;
                }}
                if (!data.is_alive || data.idea_state === 'done' || data.idea_state === 'failed') {{
                    clearInterval(iv);
                    updateCardState(card, statusEl, data.idea_state);
                }}
            }} catch (_) {{
                // Network error during poll — keep trying
            }}
        }}, 3000);
    }}

    function startLogStream(id, card) {{
        const logEl = document.getElementById(`log-${{id}}`);
        const statusEl = card.querySelector('.exec-status');
        let allLines = [];
        let sseOpened = false;
        const src = new EventSource(`/api/ideas/${{id}}/log/stream`);

        src.addEventListener('log', (e) => {{
            sseOpened = true;
            const data = JSON.parse(e.data);
            allLines = allLines.concat(data.lines);
            if (logEl && allLines.length > 0) {{
                logEl.textContent = allLines.slice(-50).join('\\n');
                logEl.scrollTop = logEl.scrollHeight;
            }}
        }});

        src.addEventListener('state', (e) => {{
            sseOpened = true;
            const data = JSON.parse(e.data);
            if (statusEl && data.is_alive) {{
                statusEl.textContent = `Claude Code is working... (${{Math.round(data.elapsed)}}s)`;
            }}
        }});

        src.addEventListener('done', (e) => {{
            src.close();
            const data = JSON.parse(e.data);
            updateCardState(card, statusEl, data.idea_state);
        }});

        src.onerror = () => {{
            src.close();
            // SSE failed — fall back to polling
            if (!sseOpened) {{
                // Never connected successfully — start polling from scratch
                startLogPoll(id, card, []);
            }} else {{
                // Had partial data — continue from where SSE left off
                startLogPoll(id, card, allLines);
            }}
        }};
    }}

    // Auto-start log streaming for any cards already in executing state
    document.querySelectorAll('.card.executing').forEach(card => {{
        const id = card.dataset.idea;
        if (id) startLogStream(id, card);
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
            `<div class="comment-author owner">${{ownerName}}</div>` +
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

    // --- Epic context editing ---
    async function editEpicContext(epicId) {{
        const idea = await (await fetch(`/api/ideas/${{epicId}}`)).json();
        const current = idea.epic_context || '';
        const newCtx = prompt('Epic Context (big-picture narrative for this epic):', current);
        if (newCtx === null) return;  // cancelled
        await fetch(`/api/ideas/${{epicId}}/context`, {{
            method: 'PUT', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{context: newCtx}})
        }});
        location.reload();
    }}

    // --- Drag-to-reorder stories ---
    let dragSrcEl = null;
    document.querySelectorAll('.drag-child').forEach(item => {{
        item.addEventListener('dragstart', function(e) {{
            dragSrcEl = this;
            this.style.opacity = '0.4';
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', this.dataset.childId);
        }});
        item.addEventListener('dragend', function() {{
            this.style.opacity = '1';
            document.querySelectorAll('.drag-child').forEach(el => el.classList.remove('drag-over'));
        }});
        item.addEventListener('dragover', function(e) {{
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            this.classList.add('drag-over');
        }});
        item.addEventListener('dragleave', function() {{
            this.classList.remove('drag-over');
        }});
        item.addEventListener('drop', async function(e) {{
            e.preventDefault();
            this.classList.remove('drag-over');
            if (dragSrcEl === this) return;
            const container = this.parentElement;
            const epicId = container.dataset.epic;
            // Reorder DOM
            const children = [...container.querySelectorAll('.drag-child')];
            const fromIdx = children.indexOf(dragSrcEl);
            const toIdx = children.indexOf(this);
            if (fromIdx < toIdx) {{
                container.insertBefore(dragSrcEl, this.nextSibling);
            }} else {{
                container.insertBefore(dragSrcEl, this);
            }}
            // Build new order from DOM
            const newOrder = [...container.querySelectorAll('.drag-child')].map(el => el.dataset.childId);
            // Save to API
            await fetch(`/api/ideas/${{epicId}}/order`, {{
                method: 'PUT', headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{order: newOrder}})
            }});
            showToast('Story order updated');
        }});
    }});
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


@app.route("/analytics")
def analytics() -> str:
    """Serve the engagement analytics dashboard."""
    return _render_analytics()


@app.route("/api/analytics")
def api_analytics() -> tuple:
    """GET /api/analytics — engagement data as JSON."""
    days = int(request.args.get("days", 7))
    from agent.engagement_analytics import get_command_stats, get_daily_activity, get_top_users, get_underused_commands
    return jsonify({
        "days": days,
        "commands": get_command_stats(days),
        "daily_activity": get_daily_activity(days),
        "top_users": get_top_users(days),
        "underused": get_underused_commands(days),
    })


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


@app.route("/api/ideas/<idea_id>/order", methods=["PUT"])
def api_set_order(idea_id: str) -> tuple:
    """PUT /api/ideas/<id>/order — set execution order for an epic's stories."""
    data = request.get_json(silent=True) or {}
    order = data.get("order")
    if not isinstance(order, list):
        return jsonify({"error": "order must be a list of story IDs"}), 400

    idea = set_execution_order(idea_id, order)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/context", methods=["PUT"])
def api_set_context(idea_id: str) -> tuple:
    """PUT /api/ideas/<id>/context — set epic context narrative."""
    data = request.get_json(silent=True) or {}
    context = data.get("context", "")
    if not isinstance(context, str):
        return jsonify({"error": "context must be a string"}), 400

    idea = set_epic_context(idea_id, context)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
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
    all_children = [i for i in all_ideas if i.parent_id == idea_id]

    if not all_children:
        # Not an epic or no children — fall back to single prompt
        return api_prompt(idea_id)

    # Order children by execution_order (auto-populated if empty)
    order = get_execution_order(idea_id)
    child_by_id = {i.id: i for i in all_children}
    ordered_children = [child_by_id[oid] for oid in order if oid in child_by_id]
    # Append any not in order
    ordered_ids = set(order)
    for c in all_children:
        if c.id not in ordered_ids:
            ordered_children.append(c)

    stories = [i for i in ordered_children if i.state != "done"]
    done_stories = [i for i in ordered_children if i.state == "done"]

    if not stories and not done_stories:
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

    # Include epic context if set
    epic_ctx_section = ""
    if epic.epic_context:
        epic_ctx_section = (
            f"\n## Epic Context\n"
            f"{epic.epic_context}\n"
        )

    prompt = (
        f"# EPIC: {epic.title}\n\n"
        f"You are implementing an entire epic for the Technomancer project.\n"
        f"This epic has **{len(stories)} stories** to implement sequentially.\n\n"
        f"## Epic Description\n"
        f"{epic.description}\n"
        f"{epic_ctx_section}"
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

    For epics with child stories, uses execute_epic() to run stories
    sequentially. For stories/tasks, uses execute_idea() directly.
    """
    idea = get_idea(idea_id)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404

    if idea.idea_type == "epic":
        from .executor import execute_epic

        state = execute_epic(idea_id)
    else:
        from .executor import execute_idea

        state = execute_idea(idea_id)

    if not state:
        return jsonify({"error": "Failed to start execution"}), 500
    return jsonify({"status": "executing", "idea_id": idea_id, "pid": state.pid})


@app.route("/execute/<idea_id>")
def execute_page(idea_id: str):
    """GET /execute/<id> — clickable execute page (linked from Jira).

    Shows idea details and a one-tap Execute button. After execution
    starts, redirects to the live log stream.
    """
    idea = get_idea(idea_id)
    title = idea.title if idea else idea_id
    state = idea.state if idea else "unknown"
    idea_type = idea.idea_type if idea else "story"
    title_color = "#9b59b6" if idea_type == "epic" else "#2ecc71"

    # Detect stale "executing" (no executor thread alive)
    if state == "executing":
        live = get_execution(idea_id)
        if not live:
            state = "interrupted"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Execute {idea_id}</title>
    <style>
        body {{ font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 2rem; }}
        .card {{ max-width: 600px; margin: 2rem auto; background: #16213e; border-radius: 12px; padding: 2rem; border-left: 4px solid #0f3460; }}
        h1 {{ color: {title_color}; font-size: 1.4rem; margin-top: 0; }}
        .idea-id {{ color: #0f3460; font-size: 0.9rem; }}
        .state {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; background: #0f3460; color: #e0e0e0; }}
        .state.done {{ background: #1b5e20; }}
        .state.executing {{ background: #e65100; }}
        .state.failed {{ background: #b71c1c; }}
        .state.interrupted {{ background: #6a1b9a; }}
        button {{ background: #D97706; color: white; border: none; padding: 14px 32px; border-radius: 8px; font-size: 1.1rem; cursor: pointer; width: 100%; margin-top: 1.5rem; }}
        button:hover {{ background: #b45309; }}
        button:disabled {{ background: #555; cursor: not-allowed; }}
        .log {{ margin-top: 1.5rem; background: #0a0a1a; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto; display: none; }}
        .log-line {{ margin: 2px 0; }}
        .status {{ margin-top: 1rem; font-size: 0.9rem; color: #aaa; }}
    </style>
</head>
<body>
    <div class="card">
        <span class="idea-id">{idea_id}</span>
        <span class="state {state}" id="state-badge">{state}</span>
        <h1>{title}</h1>
        <button id="exec-btn" onclick="doExecute()">Execute with Claude Code</button>
        <div class="status" id="status"></div>
        <div class="log" id="log"></div>
    </div>
    <script>
    async function doExecute() {{
        const btn = document.getElementById('exec-btn');
        const status = document.getElementById('status');
        const log = document.getElementById('log');
        btn.disabled = true;
        btn.textContent = 'Starting...';
        status.textContent = 'Triggering executor...';

        try {{
            const resp = await fetch('/api/ideas/{idea_id}/execute', {{method: 'POST'}});
            const data = await resp.json();
            if (resp.ok) {{
                btn.textContent = 'Running...';
                btn.style.background = '#e65100';
                status.textContent = 'Execution started. Streaming log...';
                log.style.display = 'block';
                setBadge('executing', 'executing');
                streamLog();
            }} else {{
                btn.textContent = 'Failed';
                btn.style.background = '#b71c1c';
                status.textContent = data.error || 'Execution failed to start';
                setBadge('failed', 'failed');
            }}
        }} catch(e) {{
            btn.textContent = 'Error';
            status.textContent = e.message;
        }}
    }}

    function setBadge(text, cls) {{
        const badge = document.getElementById('state-badge');
        badge.textContent = text;
        badge.className = 'state ' + cls;
    }}

    function streamLog() {{
        const log = document.getElementById('log');
        const status = document.getElementById('status');
        const btn = document.getElementById('exec-btn');
        const es = new EventSource('/api/ideas/{idea_id}/log/stream');

        es.addEventListener('log', function(e) {{
            try {{
                const data = JSON.parse(e.data);
                const lines = data.lines || [];
                for (const line of lines) {{
                    const div = document.createElement('div');
                    div.className = 'log-line';
                    div.textContent = line;
                    log.appendChild(div);
                }}
            }} catch(err) {{
                // Fallback: treat as plain text
                const div = document.createElement('div');
                div.className = 'log-line';
                div.textContent = e.data;
                log.appendChild(div);
            }}
            log.scrollTop = log.scrollHeight;
        }});

        es.addEventListener('state', function(e) {{
            try {{
                const data = JSON.parse(e.data);
                const st = data.idea_state || 'unknown';
                const elapsed = data.elapsed ? Math.round(data.elapsed) + 's' : '';
                status.textContent = 'Status: ' + st + (elapsed ? ' (' + elapsed + ')' : '');
                setBadge(st, st);
                if (st === 'done') {{
                    btn.textContent = 'Done';
                    btn.style.background = '#1b5e20';
                    es.close();
                }} else if (st === 'failed') {{
                    btn.textContent = 'Failed';
                    btn.style.background = '#b71c1c';
                    es.close();
                }}
            }} catch(err) {{}}
        }});

        es.addEventListener('done', function(e) {{
            try {{
                const data = JSON.parse(e.data);
                const st = data.idea_state || 'done';
                setBadge(st, st);
                if (st === 'done') {{
                    btn.textContent = 'Done';
                    btn.style.background = '#1b5e20';
                }} else {{
                    btn.textContent = st;
                    btn.style.background = '#b71c1c';
                }}
                status.textContent = 'Execution complete: ' + st;
            }} catch(err) {{}}
            es.close();
        }});

        es.onerror = function() {{
            es.close();
            // If no log lines were received, the execution is stale (lost on restart)
            if (log.children.length === 0) {{
                status.textContent = 'Execution was interrupted (bot restarted). Click to re-execute.';
                btn.textContent = 'Re-execute with Claude Code';
                btn.style.background = '#D97706';
                btn.disabled = false;
                setBadge('interrupted', 'failed');
            }} else {{
                status.textContent = 'Log stream ended';
            }}
        }};
    }}

    // Auto-detect on page load
    if ('{state}' === 'executing') {{
        const btn = document.getElementById('exec-btn');
        const log = document.getElementById('log');
        btn.textContent = 'Running...';
        btn.style.background = '#e65100';
        btn.disabled = true;
        log.style.display = 'block';
        document.getElementById('status').textContent = 'Execution in progress. Streaming log...';
        streamLog();
    }} else if ('{state}' === 'interrupted') {{
        document.getElementById('status').textContent = 'Execution was interrupted (bot restarted). Click to re-execute.';
    }}
    </script>
</body>
</html>"""


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
    state = get_execution(idea_id)
    idea = get_idea(idea_id)
    idea_state = idea.state if idea else "unknown"
    if state:
        return jsonify({
            "idea_id": idea_id,
            "pid": state.pid,
            "elapsed": round(state.elapsed, 1),
            "is_alive": state.is_alive,
            "lines": state.log_lines,
            "line_count": len(state.log_lines),
            "idea_state": idea_state,
        })

    # Not actively executing — return stored log from idea
    if idea and idea.execution_log:
        return jsonify({
            "idea_id": idea_id,
            "pid": None,
            "elapsed": 0,
            "is_alive": False,
            "lines": idea.execution_log.split("\n"),
            "line_count": len(idea.execution_log.split("\n")),
            "idea_state": idea_state,
        })

    return jsonify({
        "idea_id": idea_id,
        "lines": [],
        "line_count": 0,
        "idea_state": idea_state,
    })


@app.route("/api/ideas/<idea_id>/log/stream")
def api_log_stream(idea_id: str) -> Response:
    """GET /api/ideas/<id>/log/stream — SSE stream for live execution log.

    Replaces polling of /api/ideas/<id>/log with a single persistent
    connection.  Sends three event types:

    - ``log``   : new log lines (JSON list of strings)
    - ``state`` : execution metadata (elapsed, is_alive, idea state)
    - ``done``  : final event when execution finishes (includes idea state)

    The stream closes itself once the execution is no longer alive and all
    lines have been flushed, or after a short idle period if the idea is
    not currently executing at all (returns stored log in one shot).
    """

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        state = get_execution(idea_id)

        # ---- Not actively executing: send stored log and close ----
        if not state:
            idea = get_idea(idea_id)
            lines = idea.execution_log.split("\n") if idea and idea.execution_log else []

            # Detect stale "executing" state (executor thread lost on restart)
            actual_state = idea.state if idea else "unknown"
            if actual_state == "executing":
                actual_state = "interrupted"

            if lines:
                yield _sse("log", {"lines": lines})
            yield _sse("done", {
                "idea_state": actual_state,
                "is_alive": False,
            })
            return

        # ---- Live execution: stream incremental updates ----
        sent = 0
        while True:
            current_lines = state.log_lines
            new_count = len(current_lines)

            # Send any new lines since last push
            if new_count > sent:
                yield _sse("log", {"lines": current_lines[sent:]})
                sent = new_count

            # Send state update
            alive = state.is_alive
            idea = get_idea(idea_id)
            idea_state = idea.state if idea else "unknown"
            yield _sse("state", {
                "elapsed": round(state.elapsed, 1),
                "is_alive": alive,
                "idea_state": idea_state,
            })

            # Execution finished — flush remaining lines and close
            if not alive or idea_state in ("done", "failed"):
                final_lines = state.log_lines
                if len(final_lines) > sent:
                    yield _sse("log", {"lines": final_lines[sent:]})
                yield _sse("done", {
                    "idea_state": idea_state,
                    "is_alive": False,
                })
                return

            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/health")
def api_health() -> tuple:
    """GET /api/health — aggregated health status of all services.

    Checks: Discord bot process, Ollama API, Discord Bridge, Idea Board (self).
    Returns JSON with per-service status, uptime, and last error.
    """
    import urllib.request

    services: list[dict] = []

    # 1. Discord Bot — read service_state.json + check PID
    bot_status: dict[str, Any] = {"name": "Discord Bot", "id": "bot"}
    state_file = Path(__file__).resolve().parent.parent / "service_state.json"
    pid_file = Path(__file__).resolve().parent.parent / "bot.pid"
    try:
        pid_alive = False
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=5,
            )
            pid_alive = str(pid) in result.stdout

        if state_file.exists():
            state = json.loads(state_file.read_text())
        else:
            state = {}

        bot_status["healthy"] = pid_alive
        bot_status["status"] = "running" if pid_alive else "stopped"
        bot_status["last_error"] = state.get("last_error")
        bot_status["restarts"] = state.get("total_restarts", 0)
        bot_status["consecutive_failures"] = state.get("consecutive_failures", 0)
    except Exception as e:
        bot_status["healthy"] = False
        bot_status["status"] = "error"
        bot_status["last_error"] = str(e)
    services.append(bot_status)

    # 2. Ollama — check /api/tags
    ollama_status: dict[str, Any] = {"name": "Ollama", "id": "ollama"}
    try:
        with urllib.request.urlopen(f"{settings.ollama_host}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "?") for m in data.get("models", [])]
            ollama_status["healthy"] = True
            ollama_status["status"] = f"{len(models)} models loaded"
            ollama_status["models"] = models
    except Exception as e:
        ollama_status["healthy"] = False
        ollama_status["status"] = "unreachable"
        ollama_status["last_error"] = str(e)[:200]
    services.append(ollama_status)

    # 3. Discord Bridge — check /api/health on port 8321
    bridge_status: dict[str, Any] = {"name": "Discord Bridge", "id": "bridge"}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8321/api/health", timeout=3) as resp:
            data = json.loads(resp.read())
            secs = int(data.get("uptime", 0))
            bridge_status["healthy"] = data.get("status") == "ok"
            bridge_status["status"] = "connected"
            bridge_status["uptime_seconds"] = secs
    except Exception as e:
        bridge_status["healthy"] = False
        bridge_status["status"] = "unreachable"
        bridge_status["last_error"] = str(e)[:200]
    services.append(bridge_status)

    # 4. Idea Board — self-check (if we're responding, we're healthy)
    board_status: dict[str, Any] = {
        "name": "Idea Board",
        "id": "idea_board",
        "healthy": True,
        "status": "running",
    }
    try:
        ideas = load_ideas()
        board_status["idea_count"] = len(ideas)
    except Exception as e:
        board_status["healthy"] = False
        board_status["status"] = "error"
        board_status["last_error"] = str(e)[:200]
    services.append(board_status)

    all_healthy = all(s["healthy"] for s in services)
    return jsonify({
        "overall": "healthy" if all_healthy else "degraded",
        "services": services,
        "checked_at": datetime.now().isoformat(),
    })


# ============================================================================
# CRASH LOG / ERRORS
# ============================================================================


def _parse_crash_log() -> list[dict]:
    """Parse crash_log.md into structured entries.

    Each entry starts with '# Bot Crash Report'. Returns entries
    newest-first, limited to 10.
    """
    crash_file = settings.vault_path / "LLM Memory" / "Permanent" / "crash_log.md"
    if not crash_file.exists():
        return []
    try:
        content = crash_file.read_text(encoding="utf-8")
    except Exception:
        return []

    if not content.strip():
        return []

    # Split into individual crash reports
    parts = content.split("# Bot Crash Report")
    entries: list[dict] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        entry: dict[str, Any] = {}

        # Extract timestamp
        for line in part.splitlines():
            if line.startswith("**Timestamp:**"):
                entry["timestamp"] = line.replace("**Timestamp:**", "").strip()
            elif line.startswith("**Exception Type:**"):
                entry["exception_type"] = line.replace("**Exception Type:**", "").strip()
            elif line.startswith("**Exception Message:**"):
                entry["exception_message"] = line.replace("**Exception Message:**", "").strip()

        if not entry.get("timestamp"):
            continue

        # Extract stack trace (between ```python and ``` in "## Full Stack Trace")
        stack_trace = ""
        in_trace_section = False
        in_code_block = False
        trace_lines: list[str] = []
        for line in part.splitlines():
            if "## Full Stack Trace" in line:
                in_trace_section = True
                continue
            if in_trace_section and line.strip() == "```python":
                in_code_block = True
                continue
            if in_trace_section and in_code_block and line.strip() == "```":
                in_code_block = False
                in_trace_section = False
                continue
            if in_code_block and in_trace_section:
                trace_lines.append(line)
        stack_trace = "\n".join(trace_lines)
        entry["stack_trace"] = stack_trace

        # Extract local variables section
        vars_section = ""
        vars_start = part.find("## Local Variables by Frame")
        if vars_start != -1:
            vars_section = part[vars_start:]
        entry["local_variables"] = vars_section

        # Build a short summary (first meaningful line of traceback)
        summary = ""
        for tl in reversed(trace_lines):
            tl_stripped = tl.strip()
            if tl_stripped and not tl_stripped.startswith("Traceback") and not tl_stripped.startswith("File"):
                summary = tl_stripped
                break
        entry["summary"] = summary or entry.get("exception_message", "Unknown error")

        entries.append(entry)

    # Return newest first, max 10
    entries.reverse()
    return entries[:10]


@app.route("/api/errors")
def api_errors() -> tuple:
    """GET /api/errors — last 10 crash/error entries from crash_log.md."""
    entries = _parse_crash_log()
    return jsonify({"errors": entries, "count": len(entries)})


@app.route("/errors")
def errors_page() -> str:
    """Serve the crash log / errors viewer page."""
    return _render_errors()


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
# ACTION ENDPOINTS (restart, cleanup, github-sync)
# ============================================================================


@app.route("/api/actions/restart", methods=["POST"])
def api_action_restart() -> tuple:
    """POST /api/actions/restart — restart the Discord bot via bot_service.py start."""

    def _restart() -> dict:
        base = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "bot_service.py", "start"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(base),
        )
        return {
            "success": result.returncode == 0,
            "output": (result.stdout + result.stderr).strip()[-500:],
        }

    try:
        result = _restart()
        status_code = 200 if result["success"] else 500
        return jsonify({"action": "restart", **result}), status_code
    except subprocess.TimeoutExpired:
        return jsonify({"action": "restart", "success": False, "output": "Timed out after 60s"}), 504
    except Exception as e:
        return jsonify({"action": "restart", "success": False, "output": str(e)}), 500


@app.route("/api/actions/cleanup", methods=["POST"])
def api_action_cleanup() -> tuple:
    """POST /api/actions/cleanup — run cleanup.py (kill orphans + restart)."""

    def _cleanup() -> dict:
        base = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "cleanup.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(base),
        )
        return {
            "success": result.returncode == 0,
            "output": (result.stdout + result.stderr).strip()[-1000:],
        }

    try:
        result = _cleanup()
        status_code = 200 if result["success"] else 500
        return jsonify({"action": "cleanup", **result}), status_code
    except subprocess.TimeoutExpired:
        return jsonify({"action": "cleanup", "success": False, "output": "Timed out after 120s"}), 504
    except Exception as e:
        return jsonify({"action": "cleanup", "success": False, "output": str(e)}), 500


@app.route("/api/actions/github-sync", methods=["POST"])
def api_action_github_sync() -> tuple:
    """POST /api/actions/github-sync — trigger sync_all_projects()."""
    try:
        from agent.project_tracker import sync_all_projects

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(sync_all_projects())
        finally:
            loop.close()

        ok = sum(1 for v in results.values() if v)
        return jsonify({
            "action": "github-sync",
            "success": True,
            "output": f"Synced {ok}/{len(results)} projects",
            "details": results,
        })
    except Exception as e:
        return jsonify({"action": "github-sync", "success": False, "output": str(e)}), 500


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
.health-panel { margin-bottom: 2rem; }
.health-panel h2 { font-size: 1.1rem; margin-bottom: 0.8rem; color: var(--accent); }
.health-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.health-card { background: var(--surface); border-radius: 8px; padding: 1rem 1.2rem;
               border-left: 4px solid var(--muted); transition: border-color 0.3s; }
.health-card.up { border-left-color: var(--green); }
.health-card.down { border-left-color: var(--red); }
.health-card .svc-name { font-size: 0.95rem; font-weight: 600; display: flex;
                         align-items: center; gap: 8px; margin-bottom: 4px; }
.health-card .dot { width: 10px; height: 10px; border-radius: 50%;
                    display: inline-block; flex-shrink: 0; }
.health-card .dot.up { background: var(--green); box-shadow: 0 0 6px var(--green); }
.health-card .dot.down { background: var(--red); box-shadow: 0 0 6px var(--red); }
.health-card .dot.unknown { background: var(--muted); }
.health-card .svc-detail { font-size: 0.8rem; color: var(--muted); }
.health-card .svc-error { font-size: 0.75rem; color: var(--red); margin-top: 4px;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 600px) {
    .health-grid { grid-template-columns: 1fr; }
    .actions-grid { grid-template-columns: 1fr !important; }
}
.actions-panel { margin-bottom: 2rem; }
.actions-panel h2 { font-size: 1.1rem; margin-bottom: 0.8rem; color: var(--accent); }
.actions-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.action-btn { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
              padding: 1rem 1.2rem; cursor: pointer; text-align: left;
              transition: border-color 0.2s, transform 0.15s; color: var(--text); }
.action-btn:hover { border-color: var(--accent); transform: translateY(-1px); }
.action-btn:active { transform: translateY(0); }
.action-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.action-btn .action-name { font-size: 0.95rem; font-weight: 600; display: flex;
                           align-items: center; gap: 8px; margin-bottom: 4px; }
.action-btn .action-desc { font-size: 0.8rem; color: var(--muted); }
.action-btn.running .action-name::after { content: ''; width: 14px; height: 14px;
    border: 2px solid var(--accent); border-top-color: transparent;
    border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; }
.action-btn.success { border-color: var(--green); }
.action-btn.error { border-color: var(--red); }
@keyframes spin { to { transform: rotate(360deg); } }
.action-output { font-size: 0.75rem; color: var(--muted); margin-top: 6px;
                 font-family: monospace; white-space: pre-wrap; max-height: 80px;
                 overflow-y: auto; }
.hub-toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
             background: var(--surface); color: var(--text); padding: 14px 28px;
             border-radius: 8px; font-size: 0.95rem; z-index: 9999;
             box-shadow: 0 4px 20px rgba(0,0,0,0.5); animation: fadeIn 0.3s ease; }
.hub-toast.success { border: 1px solid var(--green); }
.hub-toast.error { border: 1px solid var(--red); }
@keyframes fadeIn { from { opacity: 0; transform: translateX(-50%) translateY(20px); }
                    to { opacity: 1; transform: translateX(-50%) translateY(0); } }
"""


def _render_analytics() -> str:
    """Render the engagement analytics dashboard."""
    from agent.engagement_analytics import get_command_stats, get_daily_activity, get_top_users, get_underused_commands
    from agent.discord_errors import get_gateway_trend, get_feedback_summary

    days = 14
    cmd_stats = get_command_stats(days)
    daily = get_daily_activity(days)
    top_users = get_top_users(days, limit=10)
    underused = get_underused_commands(days)

    total_cmds = sum(r["cnt"] for r in cmd_stats)
    total_msgs = sum(d["messages"] for d in daily) if daily else 0

    # Command usage table
    cmd_rows = ""
    for r in cmd_stats[:20]:
        pct = round(r["cnt"] / total_cmds * 100, 1) if total_cmds else 0
        bar_width = min(pct * 3, 100)
        cmd_rows += f"""<tr>
            <td><code>{html.escape(r['command'])}</code></td>
            <td>{r['cnt']}</td>
            <td>{r['unique_users']}</td>
            <td><div style="background:var(--accent);height:8px;width:{bar_width}%;border-radius:4px"></div> {pct}%</td>
        </tr>"""

    # Daily activity for chart (simple text-based)
    daily_rows = ""
    max_msgs = max((d["messages"] for d in daily), default=1)
    for d in daily[-14:]:
        bar_width = min(d["messages"] / max_msgs * 100, 100) if max_msgs else 0
        daily_rows += f"""<tr>
            <td>{d['day']}</td>
            <td>{d['messages']}</td>
            <td>{d['commands']}</td>
            <td>{d['users']}</td>
            <td><div style="background:var(--green);height:8px;width:{bar_width}%;border-radius:4px"></div></td>
        </tr>"""

    # User leaderboard
    user_rows = ""
    for u in top_users:
        user_rows += f"""<tr>
            <td>{html.escape(u['user_name'])}</td>
            <td>{u['messages']}</td>
            <td>{u['commands']}</td>
        </tr>"""

    # Underused features
    underused_html = ""
    if underused:
        underused_html = "<h2>Underused Features (0 invocations)</h2><p>" + ", ".join(
            f"<code>{html.escape(c)}</code>" for c in underused
        ) + "</p>"

    # Gateway health
    try:
        gw = get_gateway_trend(24)
        health = gw["current_health"]
        gw_html = f"""<h2>Gateway Health</h2>
        <p>Score: <strong>{health['health_score']}/100</strong> ({health['prediction']})
        &bull; Disconnects (24h): {gw['disconnects']}
        &bull; Resumes: {gw['resumes']}</p>"""
        if gw["latency"]:
            gw_html += f"<p>Latency: avg {gw['latency']['avg']}ms, p95 {gw['latency']['p95']}ms</p>"
    except Exception:
        gw_html = ""

    # Feedback
    try:
        fb = get_feedback_summary(days)
        fb_html = f"""<h2>Response Satisfaction</h2>
        <p>Total feedback: {fb.get('total', 0)} &bull; Satisfaction: {fb.get('satisfaction_rate', 0)}%</p>"""
    except Exception:
        fb_html = ""

    now = datetime.now().strftime("%I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics - Technomancer Hub</title>
    <style>{DASHBOARD_CSS}
.nav {{ margin-bottom: 1.5rem; display: flex; gap: 12px; flex-wrap: wrap; }}
.nav a {{ color: var(--accent); text-decoration: none; padding: 6px 14px;
         border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }}
.nav a:hover, .nav a.active {{ background: var(--accent); color: #000; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1rem 0; }}
.stat-card {{ background: var(--surface); padding: 1rem; border-radius: 8px; text-align: center; }}
.stat-card .number {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
.stat-card .label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; }}
</style></head>
<body>
    <h1>Engagement Analytics</h1>
    <div class="nav">
        <a href="/">Hub</a>
        <a href="/ideas">Ideas</a>
        <a href="/news">News Config</a>
        <a href="/karen">KAREN</a>
        <a href="/analytics" class="active">Analytics</a>
    </div>

    <div class="stats-grid">
        <div class="stat-card"><div class="number">{total_cmds}</div><div class="label">Commands ({days}d)</div></div>
        <div class="stat-card"><div class="number">{total_msgs}</div><div class="label">Messages ({days}d)</div></div>
        <div class="stat-card"><div class="number">{len(cmd_stats)}</div><div class="label">Unique Commands</div></div>
        <div class="stat-card"><div class="number">{len(underused)}</div><div class="label">Unused Features</div></div>
    </div>

    <h2>Command Usage</h2>
    <table>
        <tr><th>Command</th><th>Count</th><th>Users</th><th>Share</th></tr>
        {cmd_rows if cmd_rows else '<tr><td colspan="4" style="color:var(--muted)">No command usage data yet. Commands will be tracked as they are used.</td></tr>'}
    </table>

    {underused_html}

    <h2>Daily Activity</h2>
    <table>
        <tr><th>Date</th><th>Messages</th><th>Commands</th><th>Users</th><th>Volume</th></tr>
        {daily_rows if daily_rows else '<tr><td colspan="5" style="color:var(--muted)">No activity data yet.</td></tr>'}
    </table>

    <h2>Top Users</h2>
    <table>
        <tr><th>User</th><th>Messages</th><th>Commands</th></tr>
        {user_rows if user_rows else '<tr><td colspan="3" style="color:var(--muted)">No user data yet.</td></tr>'}
    </table>

    {gw_html}
    {fb_html}

    <p style="color:var(--muted);font-size:0.8rem;margin-top:2rem">Last refresh: {now} &bull; Data period: {days} days</p>
</body></html>"""


ERRORS_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --orange: #ff9800;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.6; }
h1 { margin-bottom: 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; }
a { color: var(--accent); }
.error-card { background: var(--surface); border-radius: 8px; padding: 1rem 1.2rem;
              border-left: 4px solid var(--red); margin-bottom: 12px; }
.error-header { cursor: pointer; display: flex; align-items: flex-start;
                justify-content: space-between; gap: 12px; }
.error-header:hover { opacity: 0.85; }
.error-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
.error-type { font-weight: 600; color: var(--red); font-size: 0.95rem; }
.error-summary { font-size: 0.85rem; color: var(--text); margin-top: 4px;
                 font-family: 'Cascadia Code', 'Fira Code', monospace; }
.error-toggle { flex-shrink: 0; font-size: 1.2rem; color: var(--muted);
                transition: transform 0.2s; user-select: none; }
.error-toggle.open { transform: rotate(90deg); }
.error-detail { display: none; margin-top: 12px; border-top: 1px solid var(--border);
                padding-top: 12px; }
.error-detail.open { display: block; }
.error-detail pre { background: #1e1e1e; border-radius: 6px; padding: 12px;
                    overflow-x: auto; font-size: 0.8rem; line-height: 1.5;
                    font-family: 'Cascadia Code', 'Fira Code', monospace;
                    -webkit-overflow-scrolling: touch; white-space: pre;
                    max-height: 500px; overflow-y: auto; }
.error-detail h3 { font-size: 0.9rem; color: var(--accent); margin: 12px 0 6px; }
.empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); }
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
@media (max-width: 600px) {
    body { padding: 12px; }
    .error-detail pre { font-size: 0.7rem; padding: 8px; }
}
"""


def _render_errors() -> str:
    """Render the crash log / errors viewer page."""
    entries = _parse_crash_log()
    now = datetime.now().strftime("%H:%M")

    if not entries:
        cards_html = """<div class="empty-state">
            <div class="icon">&#10003;</div>
            <p>No crash reports found.</p>
            <p style="font-size:0.85rem;margin-top:0.5rem">crash_log.md is empty or doesn't exist.</p>
        </div>"""
    else:
        card_parts = []
        for i, entry in enumerate(entries):
            ts = html.escape(entry.get("timestamp", "Unknown"))
            exc_type = html.escape(entry.get("exception_type", "Unknown"))
            summary = html.escape(entry.get("summary", ""))
            stack_trace = html.escape(entry.get("stack_trace", ""))
            local_vars = html.escape(entry.get("local_variables", ""))

            card_parts.append(f"""<div class="error-card">
    <div class="error-header" onclick="toggleError({i})">
        <div>
            <div class="error-meta">{ts}</div>
            <div class="error-type">{exc_type}</div>
            <div class="error-summary">{summary}</div>
        </div>
        <span class="error-toggle" id="toggle-{i}">&#9654;</span>
    </div>
    <div class="error-detail" id="detail-{i}">
        <h3>Stack Trace</h3>
        <pre>{stack_trace}</pre>
        {"<h3>Local Variables</h3><pre>" + local_vars + "</pre>" if local_vars else ""}
    </div>
</div>""")
        cards_html = "\n".join(card_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Errors &amp; Crashes</title>
    <style>{ERRORS_CSS}</style>
</head>
<body>
    <h1>Errors &amp; Crashes</h1>
    <p class="subtitle"><a href="/">&larr; Hub</a> &middot; {len(entries)} recent error(s) &middot; <a href="/api/errors">API: /api/errors</a></p>

    {cards_html}

    <script>
    function toggleError(i) {{
        const detail = document.getElementById('detail-' + i);
        const toggle = document.getElementById('toggle-' + i);
        const isOpen = detail.classList.contains('open');
        detail.classList.toggle('open');
        toggle.classList.toggle('open');
    }}
    </script>
    <p style="color:var(--muted);font-size:0.8rem;margin-top:2rem">Generated at {now}</p>
</body>
</html>"""


def _render_hub() -> str:
    """Render the central hub page with links to all services."""
    ideas = load_ideas()
    total = len(ideas)
    done = len([i for i in ideas if i.state == "done"])
    proposed = len([i for i in ideas if i.state == "proposed"])
    executing = len([i for i in ideas if i.state == "executing"])
    completion_pct = round(done / total * 100) if total else 0
    generated_at = datetime.now().strftime("%H:%M")

    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        git_hash = "unknown"

    # Codebase stats for footer
    agent_dir = Path(__file__).resolve().parent.parent / "agent"
    py_files = list(agent_dir.glob("*.py"))
    total_lines = sum(f.read_text(encoding="utf-8", errors="ignore").count("\n") for f in py_files)

    # Bot uptime from Discord Bridge health endpoint
    uptime_str = ""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8321/api/health", timeout=2) as resp:
            import json as _json
            data = _json.loads(resp.read())
            secs = int(data.get("uptime", 0))
            if secs > 0:
                days, rem = divmod(secs, 86400)
                hours, rem = divmod(rem, 3600)
                mins, _ = divmod(rem, 60)
                parts = []
                if days:
                    parts.append(f"{days}d")
                if hours:
                    parts.append(f"{hours}h")
                parts.append(f"{mins}m")
                uptime_str = f" &middot; uptime: {''.join(parts)}"
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technomancer Hub ({len(ideas)} ideas)</title>
    <style>{HUB_CSS}</style>
</head>
<body>
    <h1>Technomancer Hub</h1>
    <p class="subtitle">Central control panel for all Technomancer services · <a href="/api/ideas" style="color: var(--muted); font-size: 0.85em;">API: /api/ideas</a></p>

    <div class="health-panel">
        <h2>Service Health <a href="/errors" style="font-size:0.7rem;font-weight:normal;color:var(--muted);margin-left:8px">View errors &rarr;</a></h2>
        <div class="health-grid" id="health-grid">
            <div class="health-card" id="health-bot">
                <div class="svc-name"><span class="dot unknown"></span> Discord Bot</div>
                <div class="svc-detail">Checking...</div>
            </div>
            <div class="health-card" id="health-ollama">
                <div class="svc-name"><span class="dot unknown"></span> Ollama</div>
                <div class="svc-detail">Checking...</div>
            </div>
            <div class="health-card" id="health-bridge">
                <div class="svc-name"><span class="dot unknown"></span> Discord Bridge</div>
                <div class="svc-detail">Checking...</div>
            </div>
            <div class="health-card" id="health-idea_board">
                <div class="svc-name"><span class="dot unknown"></span> Idea Board</div>
                <div class="svc-detail">Checking...</div>
            </div>
        </div>
    </div>

    <div class="actions-panel">
        <h2>Quick Actions</h2>
        <div class="actions-grid">
            <button class="action-btn" id="action-restart" onclick="runAction('restart', true)">
                <div class="action-name">Restart Bot</div>
                <div class="action-desc">Stop and restart the Discord bot process</div>
            </button>
            <button class="action-btn" id="action-cleanup" onclick="runAction('cleanup')">
                <div class="action-name">Run Cleanup</div>
                <div class="action-desc">Kill orphaned processes, restart services</div>
            </button>
            <button class="action-btn" id="action-github-sync" onclick="runAction('github-sync')">
                <div class="action-name">Sync GitHub</div>
                <div class="action-desc">Pull latest data for all tracked projects</div>
            </button>
        </div>
    </div>

    <div class="grid">
        <a href="/ideas" class="card green">
            <h2>Idea Board</h2>
            <p>View, vote, and discuss improvement ideas.</p>
            <span class="badge">{done}/{total} done ({completion_pct}%) &middot; {proposed} pending &middot; {executing} running</span>
        </a>
        <a href="/news" class="card orange">
            <h2>News Config</h2>
            <p>Manage RSS feeds, topic preferences, and digest schedule.</p>
        </a>
        <a href="/karen" class="card" style="border-left: 4px solid #e94560;">
            <h2>K.A.R.E.N.</h2>
            <p>Submit complaints. They get turned into improvement ideas.</p>
        </a>
        <a href="/analytics" class="card" style="border-left: 4px solid #5865F2;">
            <h2>Analytics</h2>
            <p>Command usage, engagement trends, and feature adoption.</p>
        </a>
        <a href="/errors" class="card" style="border-left: 4px solid var(--red);">
            <h2>Errors &amp; Crashes</h2>
            <p>Recent crash reports with stack traces.</p>
        </a>
        <a href="http://{settings.server_host}:9090" target="_blank" class="card external">
            <h2>Prometheus</h2>
            <p>Metrics and monitoring dashboard.</p>
            <span class="badge">:9090</span>
        </a>
        <a href="http://{settings.server_host}:3000" target="_blank" class="card external">
            <h2>Grafana</h2>
            <p>Visualization and alerting dashboards.</p>
            <span class="badge">:3000</span>
        </a>
        <a href="http://{settings.server_host}:8321" target="_blank" class="card external">
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
    function formatUptime(secs) {{
        if (!secs || secs <= 0) return '';
        const d = Math.floor(secs / 86400);
        const h = Math.floor((secs % 86400) / 3600);
        const m = Math.floor((secs % 3600) / 60);
        let parts = [];
        if (d) parts.push(d + 'd');
        if (h) parts.push(h + 'h');
        parts.push(m + 'm');
        return parts.join(' ');
    }}

    async function updateHealth() {{
        try {{
            const resp = await fetch('/api/health');
            const data = await resp.json();
            for (const svc of data.services) {{
                const card = document.getElementById('health-' + svc.id);
                if (!card) continue;
                const dot = card.querySelector('.dot');
                const detail = card.querySelector('.svc-detail');

                card.className = 'health-card ' + (svc.healthy ? 'up' : 'down');
                dot.className = 'dot ' + (svc.healthy ? 'up' : 'down');

                let info = svc.status;
                if (svc.uptime_seconds) info += ' \u00b7 ' + formatUptime(svc.uptime_seconds);
                if (svc.restarts) info += ' \u00b7 ' + svc.restarts + ' restarts';
                if (svc.idea_count !== undefined) info += ' \u00b7 ' + svc.idea_count + ' ideas';
                detail.textContent = info;

                let errEl = card.querySelector('.svc-error');
                if (svc.last_error && !svc.healthy) {{
                    if (!errEl) {{
                        errEl = document.createElement('div');
                        errEl.className = 'svc-error';
                        card.appendChild(errEl);
                    }}
                    errEl.textContent = svc.last_error;
                    errEl.title = svc.last_error;
                }} else if (errEl) {{
                    errEl.remove();
                }}
            }}
        }} catch(e) {{
            document.querySelectorAll('.health-card .dot').forEach(d => d.className = 'dot unknown');
        }}
    }}
    updateHealth();
    setInterval(updateHealth, 10000);
    </script>

    <script>
    function hubToast(msg, type) {{
        const old = document.getElementById('hub-toast');
        if (old) old.remove();
        const t = document.createElement('div');
        t.id = 'hub-toast';
        t.className = 'hub-toast ' + (type || '');
        t.textContent = msg;
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 4000);
    }}

    async function runAction(action, needsConfirm) {{
        if (needsConfirm && !confirm('Restart the Discord bot? This will briefly disconnect it.')) return;

        const btn = document.getElementById('action-' + action);
        if (!btn || btn.disabled) return;

        btn.disabled = true;
        btn.classList.add('running');
        btn.classList.remove('success', 'error');

        // Remove any previous output
        const oldOutput = btn.querySelector('.action-output');
        if (oldOutput) oldOutput.remove();

        try {{
            const resp = await fetch('/api/actions/' + action, {{method: 'POST'}});
            const data = await resp.json();

            btn.classList.remove('running');
            btn.classList.add(data.success ? 'success' : 'error');

            hubToast(
                data.success ? action + ' completed successfully' : action + ' failed: ' + (data.output || 'unknown error'),
                data.success ? 'success' : 'error'
            );

            if (data.output) {{
                const out = document.createElement('div');
                out.className = 'action-output';
                out.textContent = data.output;
                btn.appendChild(out);
            }}

            // Refresh health after restart/cleanup
            if (data.success && (action === 'restart' || action === 'cleanup')) {{
                setTimeout(updateHealth, 3000);
            }}
        }} catch(e) {{
            btn.classList.remove('running');
            btn.classList.add('error');
            hubToast('Network error: ' + e.message, 'error');
        }}

        // Re-enable after 3 seconds
        setTimeout(() => {{
            btn.disabled = false;
            btn.classList.remove('success', 'error');
        }}, 5000);
    }}
    </script>

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
    <p style="color:var(--muted);font-size:0.8rem;margin-top:2rem">{len(py_files)} modules &middot; {total_lines:,} lines of code &middot; Page generated at {generated_at} &middot; v: {git_hash}{uptime_str}</p>
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
