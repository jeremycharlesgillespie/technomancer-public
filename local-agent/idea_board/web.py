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
    POST /api/jira/create         — Create a Jira story/epic via BoardProvider
    GET  /api/jira/dlq             — Recent Jira-sync dead-letter queue entries
    GET  /api/perf/functions      — Top-N per-function perf stats (time/calls/variance)
    GET  /api/claude_vault/stats  — Process-wide claude_vault prompt-cache stats
    GET  /api/embeddings/stats    — Embedding store totals, stale/orphan counts, last sweep
    GET  /api/executor/run/<id>/tools — Per-tool telemetry rows for an executor run
    GET  /api/executor/run/<run_id>/status — State snapshot for a single run
    GET  /api/executor/runs       — Last 100 executor runs (cost, duration, status, error)
    POST /api/executor/run/<id>/kill — SIGTERM→SIGKILL a runaway executor run
    GET  /executor-runs           — HTML dashboard with sortable table + totals
    GET  /api/memory/integrity    — Memory compaction health (backup counts, last verify, age)
    GET  /api/metrics             — Observability snapshot + flat SQLite counters as JSON
    GET  /metrics                 — Prometheus exposition of the same snapshot
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Any

import time

import requests as _requests_lib

from flask import Flask, Response, jsonify, request

from agent import fn_profiler, metrics
from agent.config import settings
from agent.run_context import with_run_context

from .executor import EXECUTION_LOGS_DIR, get_execution
from .models import Idea, save_ideas

from board import get_provider as _get_board_provider
from idea_board.jira_sync import is_jira_configured, _api as _jira_api
from idea_board.jira_sync_dlq import get_jira_dlq_entries

from aim import event_log as aim_event_log, jira_reader as aim_jira_reader, state as aim_state


def load_ideas():
    return _get_board_provider().load_all()


def get_idea(idea_id):
    return _get_board_provider().get(idea_id)


def add_idea(title, description, source="llm_analysis", category="feature",
             idea_type="story", parent_id=None):
    return _get_board_provider().add(
        title=title, description=description, source=source,
        category=category, idea_type=idea_type, parent_id=parent_id,
    )


def vote(idea_id, voter, value):
    return _get_board_provider().vote(idea_id, voter, value)


def add_comment(idea_id, author, text):
    return _get_board_provider().add_comment(idea_id, author, text)


def mark_executing(idea_id):
    return _get_board_provider().mark_executing(idea_id)


def mark_done(idea_id, execution_log):
    return _get_board_provider().mark_done(idea_id, execution_log)


def mark_failed(idea_id, error):
    return _get_board_provider().mark_failed(idea_id, error)


def delete_idea(idea_id):
    return _get_board_provider().delete(idea_id)


def set_execution_order(idea_id, order):
    return _get_board_provider().set_execution_order(idea_id, order)


def get_execution_order(idea_id):
    return _get_board_provider().get_execution_order(idea_id)


def set_epic_context(idea_id, context):
    return _get_board_provider().set_epic_context(idea_id, context)

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


@app.route("/api/analytics/unused")
def api_analytics_unused() -> tuple:
    """GET /api/analytics/unused — detailed list of commands with zero invocations."""
    days = int(request.args.get("days", 14))
    from agent.engagement_analytics import get_unused_commands_detailed
    return jsonify({
        "days": days,
        "definition": f"Commands with zero invocations in the last {days} days",
        "commands": get_unused_commands_detailed(days),
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

    A run-context is bound around the whole handler so that pre-executor
    log lines (validation, branch setup inside execute_idea before the
    background thread spawns) carry the correlation id for this run.
    """
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{idea_id or 'unknown'}"
    with with_run_context(run_id=run_id, idea_key=idea_id):
        idea = get_idea(idea_id)
        if not idea:
            logger.info("[Executor] Unknown idea %s — nothing to execute", idea_id)
            return jsonify({"error": "Idea not found"}), 404

        logger.info(
            "[Executor] Starting run for %s (type=%s)", idea_id, idea.idea_type
        )
        if idea.idea_type == "epic":
            from .executor import execute_epic

            state = execute_epic(idea_id)
        else:
            from .executor import execute_idea

            state = execute_idea(idea_id)

        if not state:
            logger.warning("[Executor] Failed to start execution for %s", idea_id)
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
        .idea-id {{ color: #ffffff; font-size: 0.9rem; font-weight: bold; }}
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

    def _final_state_from_sentinel(done_path: Path) -> str:
        """Read the idea's terminal state from the ``.done`` sentinel.

        Falls back to the board provider (and ultimately ``"unknown"``) when
        the sentinel is missing or unreadable — keeps the SSE ``done`` event
        well-formed even under filesystem hiccups.
        """
        try:
            if done_path.exists():
                content = done_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except OSError:
            pass
        idea = get_idea(idea_id)
        return idea.state if idea else "unknown"

    def _tail_disk_log(log_path: Path, done_path: Path):
        """Tail ``log_path`` line-by-line; close when ``done_path`` appears.

        Cross-process: works even when the execution runs in a separate
        worker process whose ``_active`` dict this server can't see.
        """
        poll_interval = 0.5
        # Hard safety nets — in production the .done sentinel always lands
        # eventually; these only trigger on truly wedged runs so a wedged
        # SSE connection doesn't hold a gunicorn worker open forever.
        max_idle_seconds = 900
        max_total_seconds = 3600

        start = time.time()
        last_progress = start

        with open(log_path, encoding="utf-8") as fh:
            while True:
                line = fh.readline()
                if line:
                    yield _sse("log", {"lines": [line.rstrip("\n")]})
                    last_progress = time.time()
                    continue

                # EOF — check the sentinel before sleeping so a completed
                # run closes as fast as possible.
                if done_path.exists():
                    yield _sse("done", {
                        "idea_state": _final_state_from_sentinel(done_path),
                        "is_alive": False,
                    })
                    return

                now = time.time()
                if now - start > max_total_seconds or now - last_progress > max_idle_seconds:
                    break
                time.sleep(poll_interval)

        # Timeout / idle fallback — still emit a terminal event.
        yield _sse("done", {
            "idea_state": _final_state_from_sentinel(done_path),
            "is_alive": False,
        })

    def generate():
        log_path = EXECUTION_LOGS_DIR / f"{idea_id}.log"
        done_path = EXECUTION_LOGS_DIR / f"{idea_id}.done"

        # ---- Disk log present: cross-process tail (preferred path) ----
        if log_path.exists():
            yield from _tail_disk_log(log_path, done_path)
            return

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

        # ---- Live execution, no disk log yet: stream from memory ----
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


_LIVE_LOG_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Live Log — {item_id}</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family: ui-monospace, Menlo, monospace;
         margin:0; padding:1rem; line-height:1.4; }}
  header {{ display:flex; justify-content:space-between; align-items:baseline;
           border-bottom:1px solid #30363d; padding-bottom:0.5rem; margin-bottom:0.75rem; }}
  header h1 {{ font-size:1.1rem; margin:0; color:#58a6ff; }}
  header .status {{ font-size:0.85rem; color:#8b949e; }}
  header .status.alive {{ color:#3fb950; }}
  header .status.done  {{ color:#58a6ff; }}
  header .status.failed {{ color:#f85149; }}
  #log {{ white-space:pre-wrap; word-break:break-word; font-size:0.8rem; }}
  .line {{ padding:0.05rem 0; border-left:2px solid transparent; padding-left:0.5rem; }}
  .line:hover {{ background:#161b22; border-left-color:#30363d; }}
</style>
</head>
<body>
  <header>
    <h1>Live Log — {item_id}</h1>
    <span class="status" id="status">connecting…</span>
  </header>
  <div id="log"></div>
<script>
  const logEl = document.getElementById('log');
  const statusEl = document.getElementById('status');
  const src = new EventSource('/api/ideas/{item_id}/log/stream');

  function appendLines(lines) {{
    const near = window.innerHeight + window.scrollY + 100 >= document.body.offsetHeight;
    for (const line of lines) {{
      const div = document.createElement('div');
      div.className = 'line';
      div.textContent = line;
      logEl.appendChild(div);
    }}
    if (near) window.scrollTo(0, document.body.scrollHeight);
  }}

  src.addEventListener('log', e => appendLines(JSON.parse(e.data).lines || []));
  src.addEventListener('state', e => {{
    const d = JSON.parse(e.data);
    statusEl.textContent = d.is_alive ? 'running (' + (d.elapsed|0) + 's)' : (d.idea_state || 'idle');
    statusEl.className = 'status ' + (d.is_alive ? 'alive' : (d.idea_state === 'done' ? 'done' : (d.idea_state === 'failed' ? 'failed' : '')));
  }});
  src.addEventListener('done', e => {{
    const d = JSON.parse(e.data);
    statusEl.textContent = d.idea_state || 'done';
    statusEl.className = 'status ' + (d.idea_state === 'done' ? 'done' : (d.idea_state === 'failed' ? 'failed' : ''));
    src.close();
  }});
  src.onerror = () => {{ statusEl.textContent = 'disconnected'; statusEl.className = 'status'; }};
</script>
</body>
</html>
"""


@app.route("/live/<item_id>")
def live_log_viewer(item_id: str) -> Response:
    """Live "look over the shoulder" log viewer for an executing item.

    Works for both local idea IDs (idea-XXX) and Jira keys (TK-XXX).
    Streams from the existing /api/ideas/<id>/log/stream SSE endpoint.
    """
    html = _LIVE_LOG_HTML.format(item_id=item_id)
    return Response(html, mimetype="text/html")


@app.route("/api/health")
def api_health() -> tuple:
    """GET /api/health — aggregated health for external monitors.

    Shape:
        {status: 'healthy'|'degraded'|'unhealthy',
         checks: {bot, ollama, jira, executor, disk},
         timestamp}

    Each check carries ``{ok, latency_ms, detail, ...}``. Required checks
    (bot) flip the overall status to ``unhealthy``; optional checks flip
    it to ``degraded``. HTTP 200 when healthy or degraded, 503 when
    unhealthy. Results are cached for ``health.CACHE_TTL_SECONDS``.
    """
    from . import health as _health

    payload = _health.run_checks()
    status_code = 503 if payload["status"] == "unhealthy" else 200
    return jsonify(payload), status_code


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


# ============================================================================
# AIM STATUS SNAPSHOT
# ============================================================================


@app.route("/api/aim/status")
def api_aim_status() -> tuple:
    """GET /api/aim/status — point-in-time snapshot of AIM and Worker state.

    Reads the shared state file for manager/worker fields and scans the
    recent event log for the last few ``decision_made`` events. Intended
    for the /aim dashboard status widget and external health monitoring.
    """
    state = aim_state.load_state()
    worker = state.worker

    events = aim_event_log.read_events(limit=500)
    decisions = [e for e in events if e.get("type") == "decision_made"]
    last_decisions = list(reversed(decisions[-3:]))

    return jsonify({
        "worker": {
            "pid": worker.pid,
            "status": worker.status,
            "current_idea_id": worker.current_idea_id,
            "started_at": worker.started_at,
            "last_heartbeat": worker.last_heartbeat,
            "last_observation": worker.last_observation,
            "consecutive_failures": worker.consecutive_failures,
        },
        "manager_pid": state.manager_pid,
        "manager_started_at": state.manager_started_at,
        "cycle_count": state.cycle_count,
        "current_idea_id": worker.current_idea_id,
        "last_cycle": state.last_cycle,
        "last_completion": state.last_completion,
        "completions_today": state.completions_today,
        "last_error": state.last_error,
        "last_decisions": last_decisions,
        "snapshot_at": datetime.now().isoformat(timespec="seconds"),
    })


@app.route("/api/aim/metrics")
def api_aim_metrics() -> tuple:
    """GET /api/aim/metrics?hours=N — time-bucketed completion and failure counts.

    Reads ``execution_completed`` and ``execution_failed`` events from the
    AIM event log, buckets them into per-hour slots over the trailing
    ``hours`` window, and returns timestamps/completions/failures arrays
    plus a totals summary. ``hours`` defaults to 24 and is clamped to
    [1, 168]. Feeds the AI Dev Team Dashboard throughput and reliability
    charts.
    """
    try:
        hours = int(request.args.get("hours", "24"))
    except (TypeError, ValueError):
        hours = 24
    hours = max(1, min(hours, 168))

    # Window: N per-hour buckets ending with the current (partial) hour.
    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    buckets = [current_hour - timedelta(hours=hours - 1 - i) for i in range(hours)]
    bucket_index = {b: i for i, b in enumerate(buckets)}
    window_start = buckets[0]

    completions = [0] * hours
    failures = [0] * hours

    # Pull a generous slice — events.jsonl rotates at ~5MB so this bounds
    # the scan without dropping any events in the window.
    events = aim_event_log.read_events(limit=50000)

    for event in events:
        etype = event.get("type")
        if etype not in ("execution_completed", "execution_failed"):
            continue
        ts_raw = event.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except (TypeError, ValueError):
            continue
        hour_floor = ts.replace(minute=0, second=0, microsecond=0)
        idx = bucket_index.get(hour_floor)
        if idx is None or hour_floor < window_start:
            continue
        if etype == "execution_completed":
            completions[idx] += 1
        else:
            failures[idx] += 1

    total_completions = sum(completions)
    total_failures = sum(failures)
    denom = total_completions + total_failures
    completion_rate = (total_completions / denom) if denom > 0 else 0.0

    return jsonify({
        "hours": hours,
        "timestamps": [b.isoformat(timespec="seconds") for b in buckets],
        "completions": completions,
        "failures": failures,
        "totals": {
            "completions": total_completions,
            "failures": total_failures,
            "completion_rate": completion_rate,
        },
    })


@app.route("/api/aim/backlog")
def api_aim_backlog() -> tuple:
    """GET /api/aim/backlog — live Jira snapshot for the dashboard top bands.

    Returns counts per status column, the key/title of the single currently
    In Progress item (or null if none), and today-only done/failed counts
    based on ``resolutiondate >= startOfDay()``. Feeds the AI Dev Team
    Dashboard's current-work and backlog-counts bands in one round-trip.
    """
    counts = aim_jira_reader.count_issues_by_status()

    in_progress: dict[str, str] | None = None
    today_done = 0
    today_failed = 0

    if is_jira_configured():
        try:
            resp = _jira_api(
                "post",
                "/search/jql",
                json={
                    "jql": (
                        f'project = {settings.jira_project_key} '
                        f'AND status = "In Progress"'
                    ),
                    "maxResults": 1,
                    "fields": ["summary"],
                },
            )
            if resp.status_code == 200:
                issues = resp.json().get("issues", [])
                if issues:
                    in_progress = {
                        "key": issues[0]["key"],
                        "title": issues[0]["fields"].get("summary", ""),
                    }
        except Exception as exc:
            logger.warning(
                "[AimBacklog] in_progress lookup failed: %s", exc
            )

        try:
            resp = _jira_api(
                "post",
                "/search/jql",
                json={
                    "jql": (
                        f'project = {settings.jira_project_key} '
                        f'AND resolutiondate >= startOfDay() '
                        f'AND status in ("Done", "Failed")'
                    ),
                    "maxResults": 100,
                    "fields": ["status"],
                },
            )
            if resp.status_code == 200:
                for issue in resp.json().get("issues", []):
                    status_name = issue["fields"]["status"]["name"]
                    if status_name == "Done":
                        today_done += 1
                    elif status_name == "Failed":
                        today_failed += 1
        except Exception as exc:
            logger.warning(
                "[AimBacklog] today counts lookup failed: %s", exc
            )

    return jsonify({
        "counts": counts,
        "in_progress": in_progress,
        "today": {
            "done": today_done,
            "failed": today_failed,
        },
    })


# Repo paths for /api/aim/commits — mirror publish.py layout.
# web.py lives at technomancer/local-agent/idea_board/web.py, so the
# technomancer root is three parents up.
_PRIVATE_REPO_PATH = Path(__file__).parent.parent.parent
_PUBLIC_REPO_PATH = (
    Path(__file__).parent.parent.parent.parent / "technomancer-public"
)


def _read_git_commits(repo_path: Path, limit: int) -> list[dict[str, str]]:
    """Return the last ``limit`` commits in ``repo_path`` as dicts.

    Uses NUL-delimited ``--pretty=format`` so commit subjects with special
    characters parse cleanly. Raises ``RuntimeError`` when git fails so
    the caller can map the failure onto a 500 response.
    """
    result = subprocess.run(
        [
            "git",
            "log",
            f"-{limit}",
            "--pretty=format:%h%x00%s%x00%aN%x00%aI",
        ],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or "").strip() or "git log failed"
        )

    commits: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\x00")
        if len(parts) != 4:
            continue
        sha, subject, author, timestamp = parts
        commits.append({
            "sha": sha,
            "subject": subject,
            "author": author,
            "timestamp": timestamp,
        })
    return commits


@app.route("/api/aim/commits")
def api_aim_commits() -> tuple:
    """GET /api/aim/commits?limit=N&repo=private|public|both — recent commits.

    Shells out to ``git log`` on the selected repo(s) and returns a JSON
    list of commits with short SHA, subject, author, and ISO timestamp.
    When ``repo=both`` (default), returns ``{"private": [...], "public":
    [...]}``. ``limit`` defaults to 10 and is clamped to ``[1, 50]``. On
    git error, returns 500 with ``{"error": "..."}``.
    """
    try:
        limit = int(request.args.get("limit", "10"))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 50))

    repo = (request.args.get("repo") or "both").lower()
    if repo not in ("private", "public", "both"):
        repo = "both"

    try:
        if repo == "private":
            return jsonify(_read_git_commits(_PRIVATE_REPO_PATH, limit))
        if repo == "public":
            return jsonify(_read_git_commits(_PUBLIC_REPO_PATH, limit))
        return jsonify({
            "private": _read_git_commits(_PRIVATE_REPO_PATH, limit),
            "public": _read_git_commits(_PUBLIC_REPO_PATH, limit),
        })
    except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/aim/events/stream")
def api_aim_events_stream() -> Response:
    """GET /api/aim/events/stream — SSE stream of AIM/Worker events.

    Tails ``aim/events.jsonl`` and emits each newly appended line as an
    ``event``-named SSE frame whose data payload is the raw JSON event.
    Mirrors the polling pattern used by ``/api/ideas/<id>/log/stream``:
    open the file, seek to the end, and poll for new lines roughly once
    per second.
    """

    def generate():
        log_path = aim_event_log.LOG_FILE
        # Wait briefly for the log to exist — a fresh install may not have
        # one yet.  Yield a heartbeat comment so the client doesn't time
        # out while waiting.
        waited = 0.0
        while not log_path.exists() and waited < 5.0:
            yield ": waiting for event log\n\n"
            time.sleep(1)
            waited += 1.0

        if not log_path.exists():
            yield ": event log not found\n\n"
            return

        with log_path.open("r", encoding="utf-8") as f:
            f.seek(0, 2)  # Seek to end — only stream newly appended lines
            while True:
                line = f.readline()
                if not line:
                    # Heartbeat comment keeps the connection warm without
                    # emitting a spurious event.
                    yield ": keep-alive\n\n"
                    time.sleep(1)
                    continue

                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # Skip malformed lines silently
                except json.JSONDecodeError:
                    continue

                yield f"event: event\ndata: {line}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# PER-FUNCTION PERFORMANCE METRICS
# ============================================================================


@app.route("/api/perf/functions")
def api_perf_functions() -> Response:
    """GET /api/perf/functions?limit=N — top-N functions ranked three ways.

    Reads the ``fn_stats`` SQLite table via ``fn_profiler.snapshot_from_db``
    and returns three ranked lists: ``by_total_time``, ``by_call_count``,
    and ``by_variance`` (population stddev of ``last_n_durations``).  Each
    item has ``name``, ``call_count``, ``total_seconds``, ``p50_seconds``,
    ``p95_seconds``, ``stddev_seconds``.  ``limit`` defaults to 20 and is
    clamped to ``[1, 100]``.  Functions with ``call_count == 0`` are
    omitted from every list.
    """
    try:
        limit = int(request.args.get("limit", "20"))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))

    stats = fn_profiler.snapshot_from_db()

    items: list[dict[str, Any]] = []
    for name, s in stats.items():
        if s.call_count <= 0:
            continue
        durations = list(s.last_n_durations)
        stddev = pstdev(durations) if len(durations) >= 2 else 0.0
        items.append({
            "name": name,
            "call_count": s.call_count,
            "total_seconds": s.total_seconds,
            "p50_seconds": s.p50(),
            "p95_seconds": s.p95(),
            "stddev_seconds": stddev,
        })

    by_total_time = sorted(
        items, key=lambda x: x["total_seconds"], reverse=True
    )[:limit]
    by_call_count = sorted(
        items, key=lambda x: x["call_count"], reverse=True
    )[:limit]
    by_variance = sorted(
        items, key=lambda x: x["stddev_seconds"], reverse=True
    )[:limit]

    return jsonify({
        "by_total_time": by_total_time,
        "by_call_count": by_call_count,
        "by_variance": by_variance,
        "meta": {
            "collected_since": fn_profiler.get_collected_since(),
            "limit": limit,
        },
    })


@app.route("/api/claude_vault/stats")
def api_claude_vault_stats() -> Response:
    """GET /api/claude_vault/stats — process-wide prompt-cache stats.

    Returns JSON with cumulative token counts and cache hit rate from every
    claude_vault Anthropic call since the bot started. Lets regressions in
    prompt caching (CLAUDE.md claims ~84% savings) be spotted within minutes
    after model, prompt, or TTL changes.
    """
    from agent.claude_vault import get_cache_stats

    return jsonify(get_cache_stats())


@app.route("/api/embeddings/stats")
def api_embeddings_stats() -> Response:
    """GET /api/embeddings/stats — embedding store lifecycle visibility.

    Surfaces total row count, per-source breakdown, current stale count,
    orphan count and timestamp from the last sweep. Feeds the dashboard's
    staleness widget and lets us notice when the semantic index has drifted
    from live source content without opening the SQLite file by hand.
    """
    from agent import embedding_store

    return jsonify(embedding_store.get_stats())


@app.route("/api/executor/run/<int:run_id>/tools")
def api_executor_run_tools(run_id: int) -> Response:
    """GET /api/executor/run/<id>/tools — per-tool telemetry for one run.

    Returns ``{"run_id": N, "tool_calls": [...]}`` with one entry per tool
    invocation recorded during the executor run, sorted by ``started_at``.
    When a run has no recorded tool calls the list is empty and the status
    is still 200 — callers can distinguish "no tools" from "unknown run" by
    checking whether a run row exists via /api/aim/status or logs.
    """
    from agent import executor_runs_db

    return jsonify({
        "run_id": run_id,
        "tool_calls": executor_runs_db.get_tool_calls(run_id),
    })


@app.route("/api/executor/run/<run_id>/status")
def api_executor_run_status(run_id: str) -> Response:
    """GET /api/executor/run/<run_id>/status — state snapshot for one run.

    Cheap single-row lookup so external monitors (Discord /status command,
    SLO alerts, dashboards) can poll a run without loading the full runs
    list or subscribing to the streaming log.

    ``run_id`` is the artifact-style sortable id (``YYYYMMDD-HHMMSS-<key>``),
    not the integer primary key — the other per-run endpoints use the int id
    for compatibility reasons, but external callers only ever have the
    sortable string on hand.

    Status codes:
        * 200 — run found; body is ``{run_id, idea_id, status, started_at,
          ended_at, exit_code, pid}``
        * 404 — ``{"error": "not_found"}`` for an unknown run_id
        * 500 — ``{"error": "db_error"}`` on any DB failure
    """
    from agent import executor_runs_db

    try:
        row = executor_runs_db.get_run_by_run_id(run_id)
    except Exception:
        logger.exception("api_executor_run_status: db lookup failed for %s", run_id)
        return jsonify({"error": "db_error"}), 500

    if row is None:
        return jsonify({"error": "not_found"}), 404

    return jsonify({
        "run_id": row.get("run_id"),
        "idea_id": row.get("jira_key"),
        "status": row.get("status"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "exit_code": row.get("exit_code"),
        "pid": row.get("pid"),
    })


@app.route("/api/executor/runs")
def api_executor_runs() -> Response:
    """GET /api/executor/runs — last 100 executor runs, newest first.

    Returns a JSON array of run summaries used by the /executor-runs
    dashboard and any external cost/duration monitoring. Each row includes
    ``id, jira_key, started_at, duration_ms, cost_usd, status`` and the
    ``error_message`` from the first failed tool call for that run (or
    ``null`` when the run had no tool-call failures).
    """
    from agent import executor_runs_db

    rows = executor_runs_db.get_recent(limit=100)
    # Attach error_message from the first failed tool_call for each run so
    # the UI can show a human-readable reason next to failed runs without a
    # second round-trip.
    conn = executor_runs_db._get_conn()
    for row in rows:
        row["error_message"] = None
        if row.get("status") and str(row["status"]).lower() in {
            "failed", "error", "crashed"
        }:
            fail = conn.execute(
                "SELECT error_message FROM executor_tool_calls "
                "WHERE run_id = ? AND ok = 0 AND error_message IS NOT NULL "
                "ORDER BY started_at ASC, id ASC LIMIT 1",
                (int(row["id"]),),
            ).fetchone()
            if fail is not None:
                row["error_message"] = fail["error_message"]
    return jsonify(rows)


@app.route("/api/executor/run/<int:run_id>/kill", methods=["POST"])
def api_executor_run_kill(run_id: int) -> Response:
    """POST /api/executor/run/<id>/kill — stop a runaway executor run.

    Looks up the run's recorded PID, sends SIGTERM, waits 10 seconds, and
    escalates to SIGKILL if the process is still alive. The row is then
    marked ``status='killed'`` with ``killed_at`` and an optional
    ``kill_reason`` (POST body: ``{"reason": "..."}``).

    Status codes:
        * 404 — no run row with this id
        * 409 — run is already terminal (body includes current status)
        * 202 — kill initiated; the actual SIGTERM/SIGKILL dance runs on a
          background thread so operators don't wait for the full 10-second
          grace period on the HTTP round-trip
    """
    from agent import executor_runs_db

    body = request.get_json(silent=True) or {}
    reason = None
    if isinstance(body, dict):
        raw_reason = body.get("reason")
        if isinstance(raw_reason, str) and raw_reason.strip():
            reason = raw_reason.strip()

    row = executor_runs_db.get_run(run_id)
    if row is None:
        return jsonify({"error": f"executor run {run_id} not found"}), 404

    status = (row.get("status") or "").lower()
    if status in executor_runs_db.TERMINAL_RUN_STATUSES:
        return jsonify({
            "error": f"executor run {run_id} is already terminal",
            "status": row.get("status"),
        }), 409

    # Push the signal + wait + DB update to a daemon thread so the HTTP
    # response is immediate regardless of how long SIGTERM takes to settle.
    threading.Thread(
        target=_run_kill_background,
        args=(int(run_id), reason),
        daemon=True,
        name=f"kill-run-{run_id}",
    ).start()

    return jsonify({
        "status": "killing",
        "run_id": int(run_id),
        "pid": row.get("pid"),
    }), 202


def _run_kill_background(run_id: int, reason: str | None) -> None:
    """Thread target — invoke :func:`executor_runs_db.kill_run` and log errors.

    Exceptions are swallowed; operators see the outcome via the DB row (status
    stays at ``running`` if the kill failed) and the structured log line.
    """
    from agent import executor_runs_db

    try:
        result = executor_runs_db.kill_run(int(run_id), reason=reason)
        logger.info(
            "kill-run-background: %s",
            {"run_id": run_id, **result},
        )
    except executor_runs_db.RunNotFoundError:
        logger.warning("kill-run-background: run %s disappeared", run_id)
    except executor_runs_db.RunAlreadyTerminalError as exc:
        logger.info(
            "kill-run-background: run %s already terminal (%s)",
            run_id, exc.status,
        )
    except Exception:
        logger.exception(
            "kill-run-background: unexpected failure for run %s", run_id,
        )


@app.route("/executor-runs")
def executor_runs_page() -> str:
    """GET /executor-runs — HTML dashboard for executor run cost/duration.

    Renders a mobile-responsive sortable table of the last 100 runs with
    totals (sum cost, avg duration, success rate) at the top. Loads rows
    from /api/executor/runs via client-side fetch so the table refreshes
    without a page reload.
    """
    return _render_executor_runs()


# ---------------------------------------------------------------------------
# Memory integrity — /api/memory/integrity
# ---------------------------------------------------------------------------
# Dashboard widget for memory compaction health. The vault is not under git,
# so a bad compaction can silently drop context. This endpoint reports whether
# compaction is running (via backup dir presence), when it last ran, and the
# pass/fail result of the latest post-compaction verify.

# Match either `**Compaction Verify:** PASSED` or `**Compaction Verify:** FAILED`
# anywhere in the crash log (the marker from the compaction-verify story is
# written alongside crash reports so a single file holds both signals).
_COMPACTION_VERIFY_RE = re.compile(
    r"\*\*Compaction Verify:\*\*\s*(PASSED|FAILED)",
    re.IGNORECASE,
)


def _parse_snapshot_timestamp(name: str) -> datetime | None:
    """Parse a backup dir name of the form ``YYYYMMDD-HHMMSS`` (optionally
    suffixed by ``-N`` for same-second collisions) into a datetime."""
    base = "-".join(name.split("-")[:2])
    try:
        return datetime.strptime(base, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _read_last_compaction_verify(crash_log: Path) -> bool | None:
    """Return True/False for the most recent compaction-verify marker in
    crash_log.md, or None if no marker is present or the file is missing."""
    if not crash_log.exists():
        return None
    try:
        content = crash_log.read_text(encoding="utf-8")
    except OSError:
        return None
    matches = _COMPACTION_VERIFY_RE.findall(content)
    if not matches:
        return None
    return matches[-1].upper() == "PASSED"


def _collect_memory_integrity(backups_root: Path, crash_log: Path) -> dict[str, Any]:
    """Build the payload returned by /api/memory/integrity.

    Extracted so tests can drive the logic with a stubbed directory tree
    without spinning up the Flask test client.
    """
    payload: dict[str, Any] = {
        "last_compaction_at": None,
        "last_verify_passed": _read_last_compaction_verify(crash_log),
        "backup_count_hourly": 0,
        "backup_count_daily": 0,
        "newest_backup_age_seconds": None,
    }

    if not backups_root.is_dir():
        return payload

    snapshots = sorted(
        (d for d in backups_root.iterdir() if d.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not snapshots:
        return payload

    payload["backup_count_hourly"] = sum(
        1 for d in snapshots if (d / "hourly.md").exists()
    )
    payload["backup_count_daily"] = sum(
        1 for d in snapshots if (d / "daily.md").exists()
    )

    newest = snapshots[0]
    parsed = _parse_snapshot_timestamp(newest.name)
    if parsed is not None:
        payload["last_compaction_at"] = parsed.isoformat()
    else:
        try:
            payload["last_compaction_at"] = datetime.fromtimestamp(
                newest.stat().st_mtime
            ).isoformat()
        except OSError:
            payload["last_compaction_at"] = None

    try:
        age = time.time() - newest.stat().st_mtime
        payload["newest_backup_age_seconds"] = max(0, int(age))
    except OSError:
        payload["newest_backup_age_seconds"] = None

    return payload


@app.route("/api/memory/integrity")
def api_memory_integrity() -> Response:
    """GET /api/memory/integrity — memory vault compaction health snapshot.

    JSON shape::

        {
          "last_compaction_at":       ISO timestamp of newest snapshot or null,
          "last_verify_passed":       true | false | null,
          "backup_count_hourly":      int,
          "backup_count_daily":       int,
          "newest_backup_age_seconds": int | null,
        }

    Snapshots live under ``<vault>/Backups/memory/<YYYYMMDD-HHMMSS>/`` and are
    created by ``MemorySystem._snapshot_before_compaction``. The verify field
    is read from ``<vault>/LLM Memory/Permanent/crash_log.md``.
    """
    backups_root = settings.vault_path / "Backups" / "memory"
    crash_log = settings.vault_path / "LLM Memory" / "Permanent" / "crash_log.md"
    return jsonify(_collect_memory_integrity(backups_root, crash_log))


# ---------------------------------------------------------------------------
# Flat counter metrics — scrape-friendly operational totals.
# ---------------------------------------------------------------------------
# These paths are module-level so tests can monkeypatch them at temp SQLite
# files without touching the production data directory.

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DATA_DIR: Path = _REPO_ROOT / "data"
_EXECUTOR_RUNS_DB: Path = _DATA_DIR / "executor_runs.db"
_CRASH_TRIAGE_DB: Path = _DATA_DIR / "crash_triage_seen.db"
_JIRA_SYNC_DLQ_DB: Path = _DATA_DIR / "jira_sync_dlq.db"
_EMBEDDINGS_DB: Path = _DATA_DIR / "embeddings.db"
_SERVICE_STATE_FILE: Path = _REPO_ROOT / "service_state.json"


def _scalar_count(db_path: Path, sql: str, params: tuple = ()) -> int:
    """Run a COUNT query against ``db_path`` and return the result as int.

    Returns 0 if the database file is missing, the target table does not
    exist yet, or the query fails — callers want a number, not an error.
    """
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.OperationalError:
        return 0
    try:
        try:
            row = conn.execute(sql, params).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def _executor_runs_totals_by_status() -> dict[str, int]:
    """Return ``{status: count}`` for every row in ``executor_runs``.

    A NULL status column bucket is surfaced as ``"unknown"`` rather than
    dropped, so the totals always sum to the true row count.
    """
    if not _EXECUTOR_RUNS_DB.exists():
        return {}
    try:
        conn = sqlite3.connect(
            f"file:{_EXECUTOR_RUNS_DB}?mode=ro", uri=True, timeout=5
        )
    except sqlite3.OperationalError:
        return {}
    try:
        try:
            rows = conn.execute(
                "SELECT COALESCE(status, 'unknown') AS status, COUNT(*) "
                "FROM executor_runs GROUP BY COALESCE(status, 'unknown')"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {str(status): int(count) for status, count in rows}
    finally:
        conn.close()


def _bot_uptime_seconds() -> int | None:
    """Seconds since the bot was last started.

    Reads ``bot_started_at`` from ``service_state.json`` (written by
    ``bot_service.start_bot``). Returns ``None`` if the file is missing,
    unreadable, or has no start timestamp — callers render that as a JSON
    null rather than pretending the bot is running.
    """
    if not _SERVICE_STATE_FILE.exists():
        return None
    try:
        state = json.loads(_SERVICE_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = state.get("bot_started_at")
    if not raw:
        return None
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return max(0, int((datetime.now() - started).total_seconds()))


def _collect_metrics() -> dict[str, Any]:
    """Aggregate flat operational counters from SQLite tables + service state.

    Returns a dict with:
      * ``executor_runs_total``          — ``{status: count}`` across all rows
      * ``executor_runs_last_24h``       — rows in ``executor_runs`` with
        ``started_at`` within the past 24 hours
      * ``crash_triage_stories_last_24h`` — rows in ``crash_triage_seen``
        with ``first_seen`` within the past 24 hours
      * ``jira_sync_dlq_depth``          — total rows in ``jira_sync_dlq``
      * ``embedding_store_rows``         — total rows in ``embeddings``
      * ``bot_uptime_seconds``           — int or ``None`` if unknown
    """
    cutoff_iso = (datetime.now() - timedelta(hours=24)).isoformat(
        sep=" ", timespec="seconds"
    )
    # crash_triage writes ISO 8601 with a "T" separator to ``first_seen``;
    # SQLite's ``datetime()`` normalizes both forms so comparisons work.
    return {
        "executor_runs_total": _executor_runs_totals_by_status(),
        "executor_runs_last_24h": _scalar_count(
            _EXECUTOR_RUNS_DB,
            "SELECT COUNT(*) FROM executor_runs "
            "WHERE started_at IS NOT NULL "
            "AND datetime(started_at) >= datetime(?)",
            (cutoff_iso,),
        ),
        "crash_triage_stories_last_24h": _scalar_count(
            _CRASH_TRIAGE_DB,
            "SELECT COUNT(*) FROM crash_triage_seen "
            "WHERE first_seen IS NOT NULL "
            "AND datetime(first_seen) >= datetime(?)",
            (cutoff_iso,),
        ),
        "jira_sync_dlq_depth": _scalar_count(
            _JIRA_SYNC_DLQ_DB, "SELECT COUNT(*) FROM jira_sync_dlq"
        ),
        "embedding_store_rows": _scalar_count(
            _EMBEDDINGS_DB, "SELECT COUNT(*) FROM embeddings"
        ),
        "bot_uptime_seconds": _bot_uptime_seconds(),
    }


@app.route("/api/metrics")
def api_metrics() -> Response:
    """GET /api/metrics — unified observability snapshot as JSON.

    Combines the cached observability snapshot from ``agent.metrics`` (30-
    second TTL) with a set of flat operational counters read live from the
    supporting SQLite tables via :func:`_collect_metrics`. Safe to poll from
    external alerting pipelines without re-hitting Jira.
    """
    payload = dict(metrics.get_snapshot())
    payload.update(_collect_metrics())
    return jsonify(payload)


@app.route("/metrics")
def prometheus_metrics() -> Response:
    """GET /metrics — Prometheus exposition of the same snapshot.

    Renders ``agent.metrics.render_prometheus`` as plain text with the
    standard Prometheus content type so a scraper can consume it directly.
    Served from the same 30-second cache as ``/api/metrics``.
    """
    return Response(
        metrics.render_prometheus(),
        mimetype="text/plain; version=0.0.4",
    )


@app.route("/aim")
def aim_dashboard() -> str:
    """Serve the AIM dashboard HTML page.

    Renders a header widget that polls /api/aim/status every 5 seconds
    and displays worker PID, status (color-coded), current assignment,
    cycle count, and last decision summary.
    """
    return _render_aim_dashboard()


# ---------------------------------------------------------------------------
# AI Dev Team Dashboard — /aim/dashboard
# ---------------------------------------------------------------------------
# Single-page health view that polls /api/aim/backlog, /api/aim/commits, and
# /api/aim/metrics and renders four bands: current work, backlog counts, two
# Chart.js charts, and two commit columns.  See the parent epic's Decision 6
# for the locked-in layout and Decision 5 for polling intervals.

_AIM_TEAM_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Dev Team Dashboard</title>
<style>
:root {
  --bg: #1a1a1a; --surface: #252525; --surface-2: #2d2d2d; --text: #e0e0e0;
  --muted: #888; --accent: #66b3ff; --green: #4caf50; --red: #f44336;
  --yellow: #ffb74d; --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { color: var(--accent); font-size: 1.4rem; }
h2 { font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
     letter-spacing: 0.08em; margin-bottom: 0.5rem; }
h3 { font-size: 0.9rem; color: var(--accent); margin-bottom: 0.5rem; }

.page-header {
  display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;
  padding-bottom: 0.75rem; border-bottom: 1px solid var(--border);
}
.page-header .subtitle { color: var(--muted); font-size: 0.85rem; }

#window-selector { display: flex; gap: 0.25rem; }
#window-selector button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  padding: 0.3rem 0.7rem; font-size: 0.8rem; border-radius: 6px; cursor: pointer;
  font-family: inherit;
}
#window-selector button:hover { background: var(--surface-2); }
#window-selector button.active {
  background: var(--accent); color: #0a1a2a; border-color: var(--accent);
  font-weight: 600;
}

section { margin-bottom: 1.25rem; }

/* Band 1: Current work */
#current-work {
  background: var(--surface); border-radius: 10px; padding: 1rem 1.2rem;
  border-left: 4px solid var(--muted);
}
#current-work.active { border-left-color: var(--yellow); }
#current-work-body {
  font-size: 1rem;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  word-break: break-word;
}
#current-work-body a { font-weight: 600; }
#current-work-body .cw-title { color: var(--text); }
#current-work-body .cw-elapsed { color: var(--muted); font-size: 0.85rem; }
#current-work-body.idle { color: var(--muted); font-style: italic; }

/* Band 2: Backlog counts */
#backlog-counts {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.75rem;
}
.count-tile {
  background: var(--surface); border-radius: 10px; padding: 0.9rem 1rem;
  border-left: 4px solid var(--muted); text-align: left;
}
.count-tile .count-value {
  font-size: 1.8rem; font-weight: 700; color: var(--text);
  font-family: 'Cascadia Code', 'Fira Code', monospace; line-height: 1.1;
}
.count-tile .count-label {
  font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.05em; margin-top: 0.25rem;
}
.count-tile.accent-blue { border-left-color: var(--accent); }
.count-tile.accent-yellow { border-left-color: var(--yellow); }
.count-tile.accent-green { border-left-color: var(--green); }
.count-tile.accent-red { border-left-color: var(--red); }

/* Band 3: Charts */
#charts {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
}
.chart-box {
  background: var(--surface); border-radius: 10px; padding: 1rem;
  min-height: 280px; position: relative;
}
.chart-box canvas { width: 100% !important; max-height: 260px; }

/* Band 4: Commits */
#commits {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
}
.commit-col {
  background: var(--surface); border-radius: 10px; padding: 1rem;
}
.commit-list { list-style: none; display: flex; flex-direction: column;
               gap: 0.35rem; font-size: 0.82rem; }
.commit-list li {
  padding: 0.4rem 0.5rem; border-radius: 6px; background: var(--surface-2);
  font-family: 'Cascadia Code', 'Fira Code', monospace; word-break: break-word;
}
.commit-list .sha { color: var(--accent); margin-right: 0.5rem; }
.commit-list .subject { color: var(--text); }
.commit-list .meta { display: block; color: var(--muted);
                     font-size: 0.72rem; margin-top: 0.15rem; }
.commit-empty { color: var(--muted); font-style: italic; font-size: 0.85rem; }

/* Band 5: Function hotspots */
#fn-hotspots {
  background: var(--surface); border-radius: 10px; padding: 1rem;
}
#fn-hotspots-table {
  width: 100%; border-collapse: collapse;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  font-size: 0.8rem;
}
#fn-hotspots-table thead th {
  text-align: left; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.7rem;
  padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border);
}
#fn-hotspots-table thead th.num { text-align: right; }
#fn-hotspots-table tbody td {
  padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border);
  color: var(--text); word-break: break-word;
}
#fn-hotspots-table tbody td.num {
  text-align: right; color: var(--accent);
}
#fn-hotspots-table tbody tr:last-child td { border-bottom: none; }
.fn-empty { color: var(--muted); font-style: italic; font-size: 0.85rem; }

#footer-status {
  margin-top: 1rem; color: var(--muted); font-size: 0.75rem;
}

@media (max-width: 900px) {
  #charts, #commits { grid-template-columns: 1fr; }
  #backlog-counts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .count-tile .count-value { font-size: 1.4rem; }
  #fn-hotspots-table { font-size: 0.72rem; }
}
@media (max-width: 520px) {
  body { padding: 12px; }
  #backlog-counts { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
  <div class="page-header">
    <div>
      <h1>AI Dev Team Dashboard</h1>
      <div class="subtitle">
        <a href="/">&larr; Hub</a> &middot;
        <a href="/aim">AIM timeline</a> &middot;
        Health &amp; throughput at a glance
      </div>
    </div>
    <div id="window-selector" role="tablist" aria-label="Time window">
      <button type="button" data-hours="1">1h</button>
      <button type="button" data-hours="6">6h</button>
      <button type="button" data-hours="24" class="active">24h</button>
      <button type="button" data-hours="168">7d</button>
    </div>
  </div>

  <section id="current-work" aria-label="Current work">
    <h2>Current Work</h2>
    <div id="current-work-body" class="idle">loading&hellip;</div>
  </section>

  <section id="backlog-counts" aria-label="Backlog counts">
    <div class="count-tile accent-blue">
      <div class="count-value" id="count-todo">&mdash;</div>
      <div class="count-label">To Do</div>
    </div>
    <div class="count-tile accent-yellow">
      <div class="count-value" id="count-in-progress">&mdash;</div>
      <div class="count-label">In Progress</div>
    </div>
    <div class="count-tile accent-green">
      <div class="count-value" id="count-done-today">&mdash;</div>
      <div class="count-label">Done Today</div>
    </div>
    <div class="count-tile accent-red">
      <div class="count-value" id="count-failed-today">&mdash;</div>
      <div class="count-label">Failed Today</div>
    </div>
    <div class="count-tile">
      <div class="count-value" id="count-veto">&mdash;</div>
      <div class="count-label">Veto Total</div>
    </div>
  </section>

  <section id="charts" aria-label="Throughput and reliability charts">
    <div class="chart-box">
      <h3>Completions per hour</h3>
      <canvas id="completions-chart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Success vs failure</h3>
      <canvas id="success-failure-chart"></canvas>
    </div>
  </section>

  <section id="commits" aria-label="Recent commits">
    <div class="commit-col">
      <h3>Private repo</h3>
      <ul class="commit-list" id="private-commits">
        <li class="commit-empty">loading&hellip;</li>
      </ul>
    </div>
    <div class="commit-col">
      <h3>Public repo</h3>
      <ul class="commit-list" id="public-commits">
        <li class="commit-empty">loading&hellip;</li>
      </ul>
    </div>
  </section>

  <section id="fn-hotspots" aria-label="Function hotspots">
    <h3>Function hotspots</h3>
    <table id="fn-hotspots-table">
      <thead>
        <tr>
          <th>Function</th>
          <th class="num">Calls</th>
          <th class="num">Total (s)</th>
          <th class="num">p50 (s)</th>
          <th class="num">p95 (s)</th>
          <th class="num">Stddev (s)</th>
        </tr>
      </thead>
      <tbody id="fn-hotspots-body">
        <tr><td colspan="6" class="fn-empty">loading&hellip;</td></tr>
      </tbody>
    </table>
  </section>

  <div id="footer-status">
    Polling: backlog 10s &middot; commits 30s &middot; metrics 60s &middot; hotspots 60s.
    Endpoints: /api/aim/backlog, /api/aim/commits, /api/aim/metrics, /api/perf/functions.
  </div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
(function() {
  "use strict";

  // Decision 5: polling intervals
  const BACKLOG_POLL_MS = 10000;
  const COMMITS_POLL_MS = 30000;
  const METRICS_POLL_MS = 60000;
  const FN_HOTSPOTS_POLL_MS = 60000;
  const FN_HOTSPOTS_LIMIT = 10;

  let currentHours = 24;
  let completionsChart = null;
  let successFailureChart = null;
  let elapsedTimer = null;
  let currentWorkStartedAt = null;

  function formatElapsed(secs) {
    if (secs < 60) return secs + 's';
    const mins = Math.floor(secs / 60);
    if (mins < 60) return mins + 'm ' + (secs % 60) + 's';
    const hrs = Math.floor(mins / 60);
    return hrs + 'h ' + (mins % 60) + 'm';
  }

  function formatTimestampLabel(iso) {
    try {
      const d = new Date(iso);
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      if (currentHours >= 48) {
        const mo = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return mo + '-' + day + ' ' + hh + ':' + mm;
      }
      return hh + ':' + mm;
    } catch (e) { return iso; }
  }

  // ------------------------------------------------------------------
  // Band 1 + 2: current work + backlog counts (polled every 10s)
  // ------------------------------------------------------------------

  function renderCurrentWork(inProgress, startedAt) {
    const panel = document.getElementById('current-work');
    const body = document.getElementById('current-work-body');
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }

    if (!inProgress || !inProgress.key) {
      panel.classList.remove('active');
      body.classList.add('idle');
      body.textContent = 'idle — no story in progress';
      currentWorkStartedAt = null;
      return;
    }

    panel.classList.add('active');
    body.classList.remove('idle');
    body.innerHTML = '';

    const link = document.createElement('a');
    link.href = '/live/' + encodeURIComponent(inProgress.key);
    link.textContent = inProgress.key;
    body.appendChild(link);

    if (inProgress.title) {
      const titleSpan = document.createElement('span');
      titleSpan.className = 'cw-title';
      titleSpan.textContent = '  —  ' + inProgress.title;
      body.appendChild(titleSpan);
    }

    const elapsedSpan = document.createElement('span');
    elapsedSpan.className = 'cw-elapsed';
    elapsedSpan.id = 'cw-elapsed';
    body.appendChild(document.createElement('br'));
    body.appendChild(elapsedSpan);

    currentWorkStartedAt = startedAt ? new Date(startedAt) : null;
    updateElapsed();
    elapsedTimer = setInterval(updateElapsed, 1000);
  }

  function updateElapsed() {
    const el = document.getElementById('cw-elapsed');
    if (!el) return;
    if (!currentWorkStartedAt) { el.textContent = ''; return; }
    const secs = Math.max(0, Math.floor(
      (Date.now() - currentWorkStartedAt.getTime()) / 1000
    ));
    el.textContent = 'elapsed ' + formatElapsed(secs);
  }

  function renderCounts(counts, today) {
    counts = counts || {};
    today = today || {};
    document.getElementById('count-todo').textContent = counts['To Do'] || 0;
    document.getElementById('count-in-progress').textContent =
      counts['In Progress'] || 0;
    document.getElementById('count-done-today').textContent = today.done || 0;
    document.getElementById('count-failed-today').textContent = today.failed || 0;
    document.getElementById('count-veto').textContent =
      counts['Veto'] || counts['Vetoed'] || 0;
  }

  async function fetchBacklog() {
    try {
      const backlogP = fetch('/api/aim/backlog').then(r => r.json());
      const statusP = fetch('/api/aim/status').then(r => r.json())
        .catch(() => null);
      const [backlog, status] = await Promise.all([backlogP, statusP]);
      const startedAt = status && status.worker && status.worker.started_at;
      renderCurrentWork(backlog.in_progress, startedAt);
      renderCounts(backlog.counts, backlog.today);
    } catch (e) {
      // Leave previous values on transient error.
    }
  }

  // ------------------------------------------------------------------
  // Band 4: commits (polled every 30s)
  // ------------------------------------------------------------------

  function renderCommitList(ulId, commits) {
    const ul = document.getElementById(ulId);
    ul.innerHTML = '';
    if (!commits || commits.length === 0) {
      const li = document.createElement('li');
      li.className = 'commit-empty';
      li.textContent = 'no commits';
      ul.appendChild(li);
      return;
    }
    commits.forEach(c => {
      const li = document.createElement('li');
      const sha = document.createElement('span');
      sha.className = 'sha';
      sha.textContent = c.sha;
      const subj = document.createElement('span');
      subj.className = 'subject';
      subj.textContent = c.subject;
      const meta = document.createElement('span');
      meta.className = 'meta';
      let when = c.timestamp;
      try {
        const d = new Date(c.timestamp);
        if (!isNaN(d.getTime())) when = d.toLocaleString();
      } catch (e) {}
      meta.textContent = (c.author || 'unknown') + ' · ' + when;
      li.appendChild(sha);
      li.appendChild(subj);
      li.appendChild(meta);
      ul.appendChild(li);
    });
  }

  async function fetchCommits() {
    try {
      const r = await fetch('/api/aim/commits?limit=10&repo=both');
      const data = await r.json();
      renderCommitList('private-commits', data.private || []);
      renderCommitList('public-commits', data.public || []);
    } catch (e) {
      // Leave previous values on transient error.
    }
  }

  // ------------------------------------------------------------------
  // Band 3: charts (polled every 60s, also on window change)
  // ------------------------------------------------------------------

  function buildCompletionsChart(labels, completions) {
    const ctx = document.getElementById('completions-chart').getContext('2d');
    if (completionsChart) {
      completionsChart.data.labels = labels;
      completionsChart.data.datasets[0].data = completions;
      completionsChart.update();
      return;
    }
    completionsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Completions',
          data: completions,
          borderColor: '#66b3ff',
          backgroundColor: 'rgba(102,179,255,0.15)',
          fill: true,
          tension: 0.25,
          pointRadius: 2,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e0e0e0' } } },
        scales: {
          x: { ticks: { color: '#888' }, grid: { color: '#333' } },
          y: { ticks: { color: '#888', precision: 0 },
               grid: { color: '#333' }, beginAtZero: true },
        },
      },
    });
  }

  function buildSuccessFailureChart(labels, completions, failures) {
    const ctx = document.getElementById('success-failure-chart').getContext('2d');
    if (successFailureChart) {
      successFailureChart.data.labels = labels;
      successFailureChart.data.datasets[0].data = completions;
      successFailureChart.data.datasets[1].data = failures;
      successFailureChart.update();
      return;
    }
    successFailureChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Success', data: completions, backgroundColor: '#4caf50' },
          { label: 'Failure', data: failures, backgroundColor: '#f44336' },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e0e0e0' } } },
        scales: {
          x: { stacked: true, ticks: { color: '#888' },
               grid: { color: '#333' } },
          y: { stacked: true, ticks: { color: '#888', precision: 0 },
               grid: { color: '#333' }, beginAtZero: true },
        },
      },
    });
  }

  async function fetchMetrics() {
    try {
      const r = await fetch('/api/aim/metrics?hours=' + currentHours);
      const data = await r.json();
      const labels = (data.timestamps || []).map(formatTimestampLabel);
      const completions = data.completions || [];
      const failures = data.failures || [];
      buildCompletionsChart(labels, completions);
      buildSuccessFailureChart(labels, completions, failures);
    } catch (e) {
      // Leave previous chart state on transient error.
    }
  }

  // ------------------------------------------------------------------
  // Band 5: function hotspots (polled every 60s)
  // ------------------------------------------------------------------

  function formatSeconds(v) {
    if (v == null || isNaN(v)) return '—';
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1);
    if (v >= 1) return v.toFixed(2);
    if (v >= 0.001) return v.toFixed(3);
    return v.toExponential(1);
  }

  function renderHotspots(items) {
    const tbody = document.getElementById('fn-hotspots-body');
    tbody.innerHTML = '';
    if (!items || items.length === 0) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 6;
      td.className = 'fn-empty';
      td.textContent = 'no data yet — waiting for profiled calls';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    items.forEach(item => {
      const tr = document.createElement('tr');
      const nameTd = document.createElement('td');
      nameTd.textContent = item.name;
      tr.appendChild(nameTd);
      const cells = [
        item.call_count,
        formatSeconds(item.total_seconds),
        formatSeconds(item.p50_seconds),
        formatSeconds(item.p95_seconds),
        formatSeconds(item.stddev_seconds),
      ];
      cells.forEach(v => {
        const td = document.createElement('td');
        td.className = 'num';
        td.textContent = v;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  async function fetchHotspots() {
    try {
      const r = await fetch('/api/perf/functions?limit=' + FN_HOTSPOTS_LIMIT);
      const data = await r.json();
      // Story scope: client-side sort by total_seconds only (already
      // ordered by the endpoint, but guard against future changes).
      const items = (data.by_total_time || []).slice().sort(
        (a, b) => (b.total_seconds || 0) - (a.total_seconds || 0)
      );
      renderHotspots(items);
    } catch (e) {
      // Leave previous values on transient error.
    }
  }

  // ------------------------------------------------------------------
  // Window selector
  // ------------------------------------------------------------------

  function wireWindowSelector() {
    const buttons = document.querySelectorAll('#window-selector button');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentHours = parseInt(btn.dataset.hours, 10) || 24;
        fetchMetrics();
      });
    });
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------

  function boot() {
    wireWindowSelector();
    fetchBacklog();
    fetchCommits();
    fetchMetrics();
    fetchHotspots();
    setInterval(fetchBacklog, BACKLOG_POLL_MS);
    setInterval(fetchCommits, COMMITS_POLL_MS);
    setInterval(fetchMetrics, METRICS_POLL_MS);
    setInterval(fetchHotspots, FN_HOTSPOTS_POLL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
</script>
</body>
</html>
"""


@app.route("/aim/dashboard")
def aim_team_dashboard() -> Response:
    """Serve the AI Dev Team Dashboard HTML page.

    Renders the four-band layout locked in by the parent epic's Decision 6:
    current work (top), backlog counts (second), two Chart.js charts (third),
    and two commit columns (fourth). The page polls /api/aim/backlog,
    /api/aim/commits, and /api/aim/metrics on the intervals from Decision 5
    and lets the user switch the metrics time window between 1h, 6h, 24h,
    and 7d.
    """
    return Response(_AIM_TEAM_DASHBOARD_HTML, mimetype="text/html")


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
# JIRA STORY CREATION
# ============================================================================


def _rank_issue(jira_key: str, rank_position: str) -> str | None:
    """Rank a Jira issue using the Agile REST API.

    Args:
        jira_key: The issue key to rank (e.g. "TK-123").
        rank_position: Either "top" (rank before the current top To Do item)
            or "after:<KEY>" (rank after a specific issue).

    Returns:
        None on success, or an error message string on failure.
    """
    if not is_jira_configured():
        return "Jira not configured"

    base_url = f"{settings.jira_url}/rest/agile/1.0"
    auth = (settings.jira_email, settings.jira_api_token)

    if rank_position == "top":
        # Find the current top To Do item to rank before it.
        try:
            resp = _jira_api(
                "post",
                "/search/jql",
                json={
                    "jql": (
                        f"project = {settings.jira_project_key} "
                        f'AND status = "To Do" ORDER BY rank ASC'
                    ),
                    "maxResults": 1,
                    "fields": ["summary"],
                },
            )
            if resp.status_code != 200:
                return f"Failed to find top To Do item ({resp.status_code})"
            issues = resp.json().get("issues", [])
            if not issues:
                return None  # No To Do items — nothing to rank against
            top_key = issues[0]["key"]
            if top_key == jira_key:
                return None  # Already at the top
        except Exception as exc:
            return f"Search for top item failed: {exc}"

        try:
            resp = _requests_lib.put(
                f"{base_url}/issue/rank",
                json={"issues": [jira_key], "rankBeforeIssue": top_key},
                auth=auth,
                timeout=15,
            )
            if resp.status_code not in (200, 204):
                return f"Rank API returned {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            return f"Rank API call failed: {exc}"

    elif rank_position.startswith("after:"):
        after_key = rank_position[6:].strip()
        if not after_key:
            return "rank_position 'after:' requires a Jira key"
        try:
            resp = _requests_lib.put(
                f"{base_url}/issue/rank",
                json={"issues": [jira_key], "rankAfterIssue": after_key},
                auth=auth,
                timeout=15,
            )
            if resp.status_code not in (200, 204):
                return f"Rank API returned {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            return f"Rank API call failed: {exc}"
    else:
        return f"Invalid rank_position: {rank_position!r} (use 'top' or 'after:<KEY>')"

    return None


@app.route("/api/jira/create", methods=["POST"])
def api_jira_create() -> tuple:
    """POST /api/jira/create — create a Jira story or epic via BoardProvider.

    Accepts JSON:
        title (required): Short descriptive title.
        description (required): Technical rationale / body text.
        category (optional, default "quality"): e.g. quality, feature, performance.
        source (optional, default "planning"): What prompted the idea.
        idea_type (optional, default "story"): "story" or "epic".
        parent_key (optional): Jira key of parent epic (e.g. "TK-10").
        rank_position (optional): "top" or "after:<KEY>" for backlog ordering.
        force (optional, default false): On duplicate match, suppress the
            ``[Duplicate Match]`` comment that would otherwise be appended
            to the existing issue.

    Returns 201 JSON: {key, title, state, url, rank_result?}

    On duplicate match returns 409 JSON: {error, key, title, state, url}.
    Unless ``force`` is true, the incoming ``title`` + ``description`` is
    posted as a ``[Duplicate Match]`` comment on the matched issue.
    """
    data = request.get_json(silent=True) or {}

    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    if not description:
        return jsonify({"error": "description is required"}), 400

    category = (data.get("category") or "quality").strip().lower()
    source = (data.get("source") or "planning").strip().lower()
    idea_type = (data.get("idea_type") or "story").strip().lower()
    parent_key = (data.get("parent_key") or "").strip() or None
    rank_position = (data.get("rank_position") or "").strip() or None
    force = bool(data.get("force"))

    if idea_type not in ("story", "epic"):
        return jsonify({"error": "idea_type must be 'story' or 'epic'"}), 400

    try:
        provider = _get_board_provider()
        # Snapshot existing ids so we can detect when provider.add returns an
        # existing item instead of creating a new one (TK-441 regression:
        # near-duplicate Ollama-performance ideas were silently accepted as
        # 201 Created, polluting the backlog).
        existing_ids = {i.id for i in provider.load_all()}
        item = provider.add(
            title=title,
            description=description,
            source=source,
            category=category,
            idea_type=idea_type,
            parent_id=parent_key,
        )
    except Exception as exc:
        logger.error("[JiraCreate] provider.add failed: %s", exc)
        return jsonify({"error": f"Failed to create issue: {exc}"}), 500

    # Build browse URL (works for both Jira keys and local IDs).
    browse_url = ""
    if settings.jira_url and item.id.startswith(settings.jira_project_key or ""):
        browse_url = f"{settings.jira_url}/browse/{item.id}"

    # Provider-level dedup returned an existing item — surface it as 409 so
    # clients can distinguish "we created this" from "this already existed".
    if item.id in existing_ids:
        # Attach the incoming idea's description as a comment on the matched
        # issue so the canonical item accumulates the context of every
        # near-duplicate submission instead of dropping it on the floor.
        # ``force=true`` bypasses this — callers that already decided to
        # re-post the same content don't need to re-log it as a match.
        if not force:
            comment_text = (
                f"[Duplicate Match] Incoming idea matched this issue.\n\n"
                f"Title: {title}\n\n{description}"
            )
            try:
                provider.add_comment(item.id, "jira_create", comment_text)
            except Exception as exc:
                logger.warning(
                    "[JiraCreate] add_comment on duplicate %s failed: %s",
                    item.id, exc,
                )
        return jsonify({
            "error": "Duplicate idea detected",
            "key": item.id,
            "title": item.title,
            "state": item.state,
            "url": browse_url,
        }), 409

    result: dict[str, Any] = {
        "key": item.id,
        "title": item.title,
        "state": item.state,
        "url": browse_url,
    }

    # Optional ranking (Jira-only, best-effort).
    if rank_position:
        rank_err = _rank_issue(item.id, rank_position)
        if rank_err:
            result["rank_result"] = f"warning: {rank_err}"
        else:
            result["rank_result"] = "ok"

    return jsonify(result), 201


@app.route("/api/jira/dlq", methods=["GET"])
def api_jira_dlq() -> tuple:
    """GET /api/jira/dlq — recent Jira-sync dead-letter entries.

    Read-only view of ``jira_sync_dlq`` rows with parsed payloads, newest
    first, capped at 100 entries. Returns ``{"entries": [...]}`` JSON.
    """
    try:
        entries = get_jira_dlq_entries(limit=100)
    except Exception as exc:
        logger.error("[JiraDLQ] get_jira_dlq_entries failed: %s", exc)
        return jsonify({"error": f"Failed to read DLQ: {exc}"}), 500
    return jsonify({"entries": entries}), 200


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
.stat-card.clickable {{ cursor: pointer; transition: transform 0.1s, background 0.2s; }}
.stat-card.clickable:hover {{ background: #2e2e2e; transform: translateY(-1px); }}
.stat-card.clickable .hint {{ font-size: 0.7rem; color: var(--accent); margin-top: 4px; }}
.modal-backdrop {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
                    z-index: 50; align-items: flex-start; justify-content: center; padding: 40px 16px;
                    overflow-y: auto; }}
.modal-backdrop.open {{ display: flex; }}
.modal {{ background: var(--surface); border-radius: 10px; max-width: 720px; width: 100%;
          padding: 1.5rem; border: 1px solid var(--border); }}
.modal-head {{ display: flex; justify-content: space-between; align-items: baseline;
               gap: 12px; margin-bottom: 0.5rem; }}
.modal-head h2 {{ margin: 0; color: var(--accent); }}
.modal-close {{ background: none; border: none; color: var(--muted); font-size: 1.4rem;
                cursor: pointer; padding: 0 4px; }}
.modal-close:hover {{ color: var(--text); }}
.modal-def {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem;
              padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
.modal .cmd-row {{ padding: 8px 0; border-bottom: 1px solid var(--border); }}
.modal .cmd-row:last-child {{ border-bottom: none; }}
.modal .cmd-name {{ font-family: 'Cascadia Code', 'Fira Code', monospace;
                     color: var(--accent); font-weight: 600; }}
.modal .cmd-meta {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
.modal .cmd-desc {{ font-size: 0.85rem; color: var(--text); margin-top: 2px; }}
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
        <div class="stat-card clickable" id="unused-card" onclick="openUnusedModal()" title="Click to see which commands haven't been used in {days} days">
            <div class="number">{len(underused)}</div>
            <div class="label">Unused Features</div>
            <div class="hint">Click for list &rarr;</div>
        </div>
    </div>

    <div class="modal-backdrop" id="unused-modal" onclick="if(event.target===this)closeUnusedModal()">
        <div class="modal" role="dialog" aria-labelledby="unused-modal-title">
            <div class="modal-head">
                <h2 id="unused-modal-title">Unused Features</h2>
                <button class="modal-close" onclick="closeUnusedModal()" aria-label="Close">&times;</button>
            </div>
            <div class="modal-def" id="unused-modal-def">Loading&hellip;</div>
            <div id="unused-modal-body">Loading&hellip;</div>
        </div>
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

    <script>
    function escapeHtml(s) {{
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }}
    async function openUnusedModal() {{
        const modal = document.getElementById('unused-modal');
        const body = document.getElementById('unused-modal-body');
        const defEl = document.getElementById('unused-modal-def');
        modal.classList.add('open');
        body.textContent = 'Loading...';
        defEl.textContent = '';
        try {{
            const resp = await fetch('/api/analytics/unused?days={days}');
            const data = await resp.json();
            defEl.textContent = data.definition || '';
            const cmds = data.commands || [];
            if (!cmds.length) {{
                body.innerHTML = '<p style="color:var(--muted)">Every command has been used recently.</p>';
                return;
            }}
            body.innerHTML = cmds.map(function(c) {{
                const lastSeen = c.last_seen
                    ? escapeHtml(String(c.last_seen).slice(0, 10))
                    : 'never';
                const cat = c.category ? ' &bull; ' + escapeHtml(c.category) : '';
                const desc = c.description
                    ? '<div class="cmd-desc">' + escapeHtml(c.description) + '</div>'
                    : '';
                return '<div class="cmd-row">'
                    + '<span class="cmd-name">' + escapeHtml(c.name) + '</span>'
                    + '<div class="cmd-meta">Last seen: ' + lastSeen + cat + '</div>'
                    + desc
                    + '</div>';
            }}).join('');
        }} catch (err) {{
            body.innerHTML = '<p style="color:var(--red)">Failed to load: '
                + escapeHtml(err.message || String(err)) + '</p>';
        }}
    }}
    function closeUnusedModal() {{
        document.getElementById('unused-modal').classList.remove('open');
    }}
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') closeUnusedModal();
    }});
    </script>
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


AIM_DASHBOARD_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --yellow: #ffb74d;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
h1 { margin-bottom: 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
a { color: var(--accent); }

#aim-status-widget {
    background: var(--surface); border-radius: 10px; padding: 1rem 1.2rem;
    border-left: 4px solid var(--muted); margin-bottom: 1.5rem;
    transition: border-color 0.3s;
}
#aim-status-widget.status-green { border-left-color: var(--green); }
#aim-status-widget.status-yellow { border-left-color: var(--yellow); }
#aim-status-widget.status-red { border-left-color: var(--red); }

.widget-row {
    display: flex; flex-wrap: wrap; gap: 1.5rem 2rem; align-items: baseline;
}
.widget-field { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.widget-label {
    font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.05em;
}
.widget-value {
    font-size: 0.95rem; font-family: 'Cascadia Code', 'Fira Code', monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 100%;
}
.widget-status-pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; text-transform: lowercase;
    background: var(--border); color: var(--text);
}
.widget-status-pill.status-green { background: var(--green); color: #0a1a0a; }
.widget-status-pill.status-yellow { background: var(--yellow); color: #2a1900; }
.widget-status-pill.status-red { background: var(--red); color: #1a0000; }

#widget-decision {
    margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);
    font-size: 0.85rem; color: var(--muted);
}
#widget-decision .widget-label { margin-bottom: 2px; }
#widget-decision-summary { color: var(--text); font-family: inherit;
    white-space: normal; word-break: break-word; }
#widget-updated {
    font-size: 0.7rem; color: var(--muted); margin-top: 0.5rem;
}
#widget-error {
    display: none; color: var(--red); font-size: 0.85rem; margin-top: 0.5rem;
}
#widget-error.visible { display: block; }

#aim-events {
    margin-top: 1.5rem;
}
#aim-events h2 {
    font-size: 1rem; color: var(--accent); margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: baseline;
}
#aim-events-status {
    font-size: 0.75rem; color: var(--muted); font-weight: normal;
}
#aim-events-status.connected { color: var(--green); }
#aim-events-status.disconnected { color: var(--red); }
#aim-events-timeline {
    display: flex; flex-direction: column; gap: 0.4rem;
    max-height: 70vh; overflow-y: auto;
    padding-right: 0.25rem;
}
.aim-event {
    background: var(--surface); border-radius: 6px;
    border-left: 4px solid var(--muted); padding: 0.5rem 0.75rem;
    font-size: 0.85rem; word-break: break-word;
}
.aim-event.color-blue { border-left-color: var(--accent); }
.aim-event.color-green { border-left-color: var(--green); }
.aim-event.color-red { border-left-color: var(--red); }
.aim-event.color-yellow { border-left-color: var(--yellow); }
.aim-event-header {
    display: flex; gap: 0.75rem; align-items: baseline;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    margin-bottom: 0.2rem;
}
.aim-event-time {
    color: var(--muted); font-size: 0.75rem;
}
.aim-event-type {
    color: var(--text); font-weight: 600; font-size: 0.8rem;
}
.aim-event-summary {
    color: var(--text); line-height: 1.4;
}
.aim-event-empty {
    color: var(--muted); font-style: italic; font-size: 0.85rem;
    padding: 0.5rem 0;
}

@media (max-width: 600px) {
    body { padding: 12px; }
    .widget-row { gap: 0.75rem 1.25rem; }
    .widget-value { font-size: 0.85rem; }
    #aim-events-timeline { max-height: 60vh; }
    .aim-event { font-size: 0.8rem; }
}
"""


def _render_aim_dashboard() -> str:
    """Render the /aim dashboard page with worker status widget."""
    jira_url = (settings.jira_url or "").rstrip("/")
    # Expose jira_url to the client-side JS as a JSON-encoded string so
    # it handles empty, quotes, etc. safely.
    jira_url_json = json.dumps(jira_url)
    now = datetime.now().strftime("%H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIM Dashboard</title>
    <style>{AIM_DASHBOARD_CSS}</style>
</head>
<body>
    <h1>AIM Dashboard</h1>
    <p class="subtitle"><a href="/">&larr; Hub</a> &middot;
       Live worker status &middot;
       <a href="/executor-runs">Executor runs</a> &middot;
       <a href="/api/aim/status">API: /api/aim/status</a></p>

    <header id="aim-status-widget" aria-live="polite">
        <div class="widget-row">
            <div class="widget-field">
                <span class="widget-label">Status</span>
                <span id="widget-status" class="widget-status-pill">&mdash;</span>
            </div>
            <div class="widget-field">
                <span class="widget-label">Worker PID</span>
                <span id="widget-pid" class="widget-value">&mdash;</span>
            </div>
            <div class="widget-field">
                <span class="widget-label">Assignment</span>
                <span id="widget-assignment" class="widget-value">&mdash;</span>
            </div>
            <div class="widget-field">
                <span class="widget-label">Cycles</span>
                <span id="widget-cycles" class="widget-value">&mdash;</span>
            </div>
        </div>
        <div id="widget-decision">
            <span class="widget-label">Last decision</span>
            <div id="widget-decision-summary">&mdash;</div>
        </div>
        <div id="widget-updated">Waiting for status&hellip;</div>
        <div id="widget-error"></div>
    </header>

    <script>
    const JIRA_URL = {jira_url_json};
    const JIRA_KEY_RE = /^[A-Z][A-Z0-9]+-\\d+$/;
    const POLL_MS = 5000;

    function statusColor(status) {{
        const s = (status || '').toLowerCase();
        if (s === 'idle' || s === 'watching') return 'green';
        if (s === 'executing' || s === 'assigned') return 'yellow';
        if (s === 'stuck' || s === 'dead' || s === 'failed' || s === 'crashed') return 'red';
        return '';
    }}

    function renderAssignment(currentIdeaId) {{
        if (!currentIdeaId) return '—';
        const el = document.createElement('span');
        if (JIRA_KEY_RE.test(currentIdeaId) && JIRA_URL) {{
            const a = document.createElement('a');
            a.href = JIRA_URL + '/browse/' + encodeURIComponent(currentIdeaId);
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.textContent = currentIdeaId;
            el.appendChild(a);
        }} else {{
            el.textContent = currentIdeaId;
        }}
        return el;
    }}

    function setWidgetClass(color) {{
        const widget = document.getElementById('aim-status-widget');
        widget.classList.remove('status-green', 'status-yellow', 'status-red');
        if (color) widget.classList.add('status-' + color);
    }}

    function setPillClass(color) {{
        const pill = document.getElementById('widget-status');
        pill.classList.remove('status-green', 'status-yellow', 'status-red');
        if (color) pill.classList.add('status-' + color);
    }}

    function formatDecision(decision) {{
        if (!decision) return '—';
        const data = decision.data || {{}};
        const ts = (decision.timestamp || '').slice(11, 19);
        const summary = data.summary || data.decision || data.reason ||
                        data.description || decision.type || '(no details)';
        return (ts ? '[' + ts + '] ' : '') + summary;
    }}

    async function poll() {{
        const errEl = document.getElementById('widget-error');
        try {{
            const resp = await fetch('/api/aim/status', {{ cache: 'no-store' }});
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const data = await resp.json();

            const worker = data.worker || {{}};
            const status = worker.status || 'unknown';
            const color = statusColor(status);

            document.getElementById('widget-status').textContent = status;
            setWidgetClass(color);
            setPillClass(color);

            document.getElementById('widget-pid').textContent =
                worker.pid != null ? String(worker.pid) : '—';

            const assignmentEl = document.getElementById('widget-assignment');
            assignmentEl.innerHTML = '';
            const rendered = renderAssignment(data.current_idea_id);
            if (typeof rendered === 'string') {{
                assignmentEl.textContent = rendered;
            }} else {{
                assignmentEl.appendChild(rendered);
            }}

            document.getElementById('widget-cycles').textContent =
                data.cycle_count != null ? String(data.cycle_count) : '—';

            const decisions = data.last_decisions || [];
            document.getElementById('widget-decision-summary').textContent =
                formatDecision(decisions[0]);

            document.getElementById('widget-updated').textContent =
                'Updated ' + new Date().toLocaleTimeString();
            errEl.classList.remove('visible');
            errEl.textContent = '';
        }} catch (e) {{
            errEl.textContent = 'Status fetch failed: ' + e.message;
            errEl.classList.add('visible');
        }}
    }}

    poll();
    setInterval(poll, POLL_MS);
    </script>

    <section id="aim-events" aria-live="polite">
        <h2>Timeline <span id="aim-events-status">connecting&hellip;</span></h2>
        <div id="aim-events-timeline">
            <div class="aim-event-empty">Waiting for events&hellip;</div>
        </div>
    </section>

    <script>
    const EVENT_COLORS = {{
        decision_made: 'blue',
        worker_started: 'green',
        worker_spawned: 'green',
        worker_assigned: 'green',
        execution_failed: 'red',
        worker_died: 'red',
        escalation: 'yellow'
    }};
    const MAX_EVENTS_RENDERED = 200;

    function colorForEvent(type) {{
        return EVENT_COLORS[type] || 'neutral';
    }}

    function eventSummary(event) {{
        const data = event.data || {{}};
        return data.summary || data.decision || data.reason ||
               data.description || data.message || data.idea_id ||
               event.type || '(no details)';
    }}

    function renderEvent(event) {{
        const wrapper = document.createElement('div');
        wrapper.className = 'aim-event color-' + colorForEvent(event.type);

        const header = document.createElement('div');
        header.className = 'aim-event-header';

        const timeEl = document.createElement('span');
        timeEl.className = 'aim-event-time';
        timeEl.textContent = (event.timestamp || '').slice(11, 19) || '--:--:--';

        const typeEl = document.createElement('span');
        typeEl.className = 'aim-event-type';
        typeEl.textContent = event.type || 'event';

        header.appendChild(timeEl);
        header.appendChild(typeEl);

        const summary = document.createElement('div');
        summary.className = 'aim-event-summary';
        summary.textContent = eventSummary(event);

        wrapper.appendChild(header);
        wrapper.appendChild(summary);
        return wrapper;
    }}

    function prependEvent(event) {{
        const timeline = document.getElementById('aim-events-timeline');
        const empty = timeline.querySelector('.aim-event-empty');
        if (empty) empty.remove();
        timeline.insertBefore(renderEvent(event), timeline.firstChild);
        while (timeline.children.length > MAX_EVENTS_RENDERED) {{
            timeline.removeChild(timeline.lastChild);
        }}
    }}

    function setStreamStatus(text, cls) {{
        const el = document.getElementById('aim-events-status');
        el.textContent = text;
        el.className = cls || '';
    }}

    function handleEventPayload(raw) {{
        if (!raw) return;
        try {{
            const event = JSON.parse(raw);
            prependEvent(event);
        }} catch (err) {{ /* skip malformed */ }}
    }}

    const evtSrc = new EventSource('/api/aim/events/stream');
    evtSrc.onopen = () => setStreamStatus('connected', 'connected');
    evtSrc.onmessage = (e) => handleEventPayload(e.data);
    evtSrc.addEventListener('event', (e) => handleEventPayload(e.data));
    evtSrc.onerror = () => setStreamStatus('disconnected', 'disconnected');
    </script>

    <p style="color:var(--muted);font-size:0.8rem;margin-top:2rem">
        Page loaded at {now} &middot; Polling every 5s &middot;
        Timeline streams from <a href="/api/aim/events/stream">/api/aim/events/stream</a>
    </p>
</body>
</html>"""


EXECUTOR_RUNS_CSS = """
:root {
    --bg: #1a1a1a; --surface: #252525; --text: #e0e0e0; --muted: #888;
    --accent: #66b3ff; --green: #4caf50; --red: #f44336; --yellow: #ffb74d;
    --border: #333;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
h1 { margin-bottom: 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }
a { color: var(--accent); }

.totals {
    display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.25rem;
}
.total-card {
    background: var(--surface); border-radius: 8px;
    border-left: 4px solid var(--accent);
    padding: 0.75rem 1rem; min-width: 150px;
}
.total-label {
    font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.05em;
}
.total-value {
    font-size: 1.25rem; font-weight: 600;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    margin-top: 2px;
}

.runs-wrapper { overflow-x: auto; background: var(--surface); border-radius: 8px; }
table.runs {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
table.runs th, table.runs td {
    padding: 0.55rem 0.75rem; text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
table.runs th {
    background: #2d2d2d; color: var(--accent); font-weight: 600;
    cursor: pointer; user-select: none; position: sticky; top: 0;
}
table.runs th:hover { background: #343434; }
table.runs th .sort-indicator { color: var(--muted); margin-left: 4px; }
table.runs tr:last-child td { border-bottom: none; }
table.runs td.err { color: var(--red); font-family: 'Cascadia Code', monospace;
    max-width: 280px; overflow: hidden; text-overflow: ellipsis; }
table.runs td.num {
    font-family: 'Cascadia Code', 'Fira Code', monospace; text-align: right;
}
table.runs td.jira a { color: var(--accent); text-decoration: none; }
.status-pill {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; text-transform: lowercase;
    background: var(--border); color: var(--text);
}
.status-pill.success { background: var(--green); color: #0a1a0a; }
.status-pill.failed, .status-pill.error, .status-pill.crashed {
    background: var(--red); color: #1a0000;
}
.status-pill.running { background: var(--yellow); color: #2a1900; }

.empty {
    padding: 1rem; color: var(--muted); font-style: italic; text-align: center;
}
#fetch-error {
    display: none; color: var(--red); margin-bottom: 1rem;
    background: var(--surface); padding: 0.6rem 0.9rem; border-radius: 6px;
    border-left: 4px solid var(--red);
}
#fetch-error.visible { display: block; }

@media (max-width: 600px) {
    body { padding: 12px; }
    .total-card { min-width: 120px; padding: 0.6rem 0.8rem; }
    .total-value { font-size: 1.05rem; }
    table.runs { font-size: 0.78rem; }
    table.runs th, table.runs td { padding: 0.45rem 0.5rem; }
    table.runs td.err { max-width: 160px; }
}
"""


def _render_executor_runs() -> str:
    """Render the /executor-runs dashboard page.

    Static shell that pulls rows from /api/executor/runs over fetch() so the
    table can refresh without reloading the page. Sorting and totals are
    computed client-side in the embedded vanilla JS.
    """
    jira_url = (settings.jira_url or "").rstrip("/")
    jira_url_json = json.dumps(jira_url)
    now = datetime.now().strftime("%H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executor Runs</title>
    <style>{EXECUTOR_RUNS_CSS}</style>
</head>
<body>
    <h1>Executor Runs</h1>
    <p class="subtitle">
        <a href="/">&larr; Hub</a> &middot;
        <a href="/aim">AIM dashboard</a> &middot;
        Last 100 runs &middot;
        <a href="/api/executor/runs">API: /api/executor/runs</a>
    </p>

    <div id="fetch-error"></div>

    <section class="totals" aria-label="Totals">
        <div class="total-card">
            <div class="total-label">Runs</div>
            <div class="total-value" id="total-runs">&mdash;</div>
        </div>
        <div class="total-card">
            <div class="total-label">Sum cost</div>
            <div class="total-value" id="total-cost">&mdash;</div>
        </div>
        <div class="total-card">
            <div class="total-label">Avg duration</div>
            <div class="total-value" id="avg-duration">&mdash;</div>
        </div>
        <div class="total-card">
            <div class="total-label">Success rate</div>
            <div class="total-value" id="success-rate">&mdash;</div>
        </div>
    </section>

    <div class="runs-wrapper">
        <table class="runs" id="runs-table">
            <thead>
                <tr>
                    <th data-col="id" data-type="num">ID<span class="sort-indicator"></span></th>
                    <th data-col="jira_key" data-type="str">Jira<span class="sort-indicator"></span></th>
                    <th data-col="started_at" data-type="str">Started<span class="sort-indicator">&darr;</span></th>
                    <th data-col="duration_ms" data-type="num">Duration<span class="sort-indicator"></span></th>
                    <th data-col="cost_usd" data-type="num">Cost&nbsp;(USD)<span class="sort-indicator"></span></th>
                    <th data-col="status" data-type="str">Status<span class="sort-indicator"></span></th>
                    <th data-col="error_message" data-type="str">Error<span class="sort-indicator"></span></th>
                </tr>
            </thead>
            <tbody id="runs-tbody">
                <tr><td colspan="7" class="empty">Loading&hellip;</td></tr>
            </tbody>
        </table>
    </div>

    <p style="color:var(--muted);font-size:0.8rem;margin-top:1.5rem">
        Page loaded at {now} &middot;
        Click any column header to sort &middot;
        Data from <a href="/api/executor/runs">/api/executor/runs</a>
    </p>

    <script>
    const JIRA_URL = {jira_url_json};
    const JIRA_KEY_RE = /^[A-Z][A-Z0-9]+-\\d+$/;
    let RUNS = [];
    let SORT_COL = 'started_at';
    let SORT_DIR = 'desc';

    function fmtCost(v) {{
        if (v == null || isNaN(v)) return '—';
        return '$' + Number(v).toFixed(4);
    }}

    function fmtDurationMs(v) {{
        if (v == null || isNaN(v)) return '—';
        const secs = Number(v) / 1000;
        if (secs < 60) return secs.toFixed(1) + 's';
        const mins = Math.floor(secs / 60);
        const rem = Math.round(secs - mins * 60);
        return mins + 'm ' + rem + 's';
    }}

    function fmtStarted(v) {{
        if (!v) return '—';
        // show YYYY-MM-DD HH:MM from ISO string, no TZ juggling
        return String(v).replace('T', ' ').slice(0, 16);
    }}

    function escapeHtml(s) {{
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    function jiraCell(key) {{
        if (!key) return '—';
        const safe = escapeHtml(key);
        if (JIRA_KEY_RE.test(key) && JIRA_URL) {{
            return '<a href="' + JIRA_URL + '/browse/' + encodeURIComponent(key) +
                   '" target="_blank" rel="noopener noreferrer">' + safe + '</a>';
        }}
        return safe;
    }}

    function statusPill(status) {{
        const s = (status || '').toLowerCase();
        return '<span class="status-pill ' + escapeHtml(s) + '">' +
               escapeHtml(status || '—') + '</span>';
    }}

    function renderTotals(runs) {{
        const totalRuns = runs.length;
        let totalCost = 0, sumDuration = 0, countDuration = 0;
        let success = 0, completed = 0;
        for (const r of runs) {{
            if (typeof r.cost_usd === 'number') totalCost += r.cost_usd;
            if (typeof r.duration_ms === 'number') {{
                sumDuration += r.duration_ms;
                countDuration += 1;
            }}
            const s = (r.status || '').toLowerCase();
            if (s === 'success') {{ success += 1; completed += 1; }}
            else if (s === 'failed' || s === 'error' || s === 'crashed') {{
                completed += 1;
            }}
        }}
        document.getElementById('total-runs').textContent = String(totalRuns);
        document.getElementById('total-cost').textContent = fmtCost(totalCost);
        document.getElementById('avg-duration').textContent =
            countDuration ? fmtDurationMs(sumDuration / countDuration) : '—';
        document.getElementById('success-rate').textContent =
            completed ? Math.round(success / completed * 100) + '%' : '—';
    }}

    function renderRows(runs) {{
        const tbody = document.getElementById('runs-tbody');
        if (!runs.length) {{
            tbody.innerHTML =
                '<tr><td colspan="7" class="empty">No runs yet</td></tr>';
            return;
        }}
        const rows = runs.map(r => {{
            const cost = typeof r.cost_usd === 'number' ? fmtCost(r.cost_usd) : '—';
            const dur = typeof r.duration_ms === 'number' ? fmtDurationMs(r.duration_ms) : '—';
            return '<tr>' +
                '<td class="num">' + escapeHtml(r.id) + '</td>' +
                '<td class="jira">' + jiraCell(r.jira_key) + '</td>' +
                '<td>' + escapeHtml(fmtStarted(r.started_at)) + '</td>' +
                '<td class="num">' + escapeHtml(dur) + '</td>' +
                '<td class="num">' + escapeHtml(cost) + '</td>' +
                '<td>' + statusPill(r.status) + '</td>' +
                '<td class="err" title="' + escapeHtml(r.error_message || '') + '">' +
                    escapeHtml(r.error_message || '') +
                '</td>' +
            '</tr>';
        }}).join('');
        tbody.innerHTML = rows;
    }}

    function sortBy(col, type) {{
        if (SORT_COL === col) {{
            SORT_DIR = SORT_DIR === 'asc' ? 'desc' : 'asc';
        }} else {{
            SORT_COL = col;
            SORT_DIR = type === 'num' ? 'desc' : 'asc';
        }}
        const mul = SORT_DIR === 'asc' ? 1 : -1;
        const sorted = [...RUNS].sort((a, b) => {{
            let av = a[col], bv = b[col];
            if (type === 'num') {{
                av = (av == null || isNaN(av)) ? -Infinity : Number(av);
                bv = (bv == null || isNaN(bv)) ? -Infinity : Number(bv);
            }} else {{
                av = (av == null) ? '' : String(av).toLowerCase();
                bv = (bv == null) ? '' : String(bv).toLowerCase();
            }}
            if (av < bv) return -1 * mul;
            if (av > bv) return 1 * mul;
            return 0;
        }});
        updateSortIndicators();
        renderRows(sorted);
    }}

    function updateSortIndicators() {{
        const arrow = SORT_DIR === 'asc' ? '\u2191' : '\u2193';
        document.querySelectorAll('#runs-table th').forEach(th => {{
            const ind = th.querySelector('.sort-indicator');
            if (!ind) return;
            ind.textContent = th.dataset.col === SORT_COL ? arrow : '';
        }});
    }}

    function attachSortHandlers() {{
        document.querySelectorAll('#runs-table th').forEach(th => {{
            th.addEventListener('click', () => {{
                sortBy(th.dataset.col, th.dataset.type || 'str');
            }});
        }});
    }}

    async function load() {{
        const errEl = document.getElementById('fetch-error');
        try {{
            const resp = await fetch('/api/executor/runs', {{ cache: 'no-store' }});
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            RUNS = await resp.json();
            if (!Array.isArray(RUNS)) RUNS = [];
            errEl.classList.remove('visible');
            errEl.textContent = '';
            renderTotals(RUNS);
            renderRows(RUNS);
            updateSortIndicators();
        }} catch (e) {{
            errEl.textContent = 'Failed to load runs: ' + e.message;
            errEl.classList.add('visible');
            document.getElementById('runs-tbody').innerHTML =
                '<tr><td colspan="7" class="empty">Failed to load</td></tr>';
        }}
    }}

    attachSortHandlers();
    load();
    </script>
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
            <div class="health-card" id="health-jira">
                <div class="svc-name"><span class="dot unknown"></span> Jira Sync</div>
                <div class="svc-detail">Checking...</div>
            </div>
            <div class="health-card" id="health-executor">
                <div class="svc-name"><span class="dot unknown"></span> Executor</div>
                <div class="svc-detail">Checking...</div>
            </div>
            <div class="health-card" id="health-disk">
                <div class="svc-name"><span class="dot unknown"></span> Disk</div>
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
            const checks = data.checks || {{}};
            for (const name of Object.keys(checks)) {{
                const card = document.getElementById('health-' + name);
                if (!card) continue;
                const dot = card.querySelector('.dot');
                const detail = card.querySelector('.svc-detail');
                const svc = checks[name];

                card.className = 'health-card ' + (svc.ok ? 'up' : 'down');
                dot.className = 'dot ' + (svc.ok ? 'up' : 'down');

                let info = svc.detail || '';
                if (typeof svc.latency_ms === 'number') {{
                    info += ' \u00b7 ' + svc.latency_ms + 'ms';
                }}
                detail.textContent = info;

                let errEl = card.querySelector('.svc-error');
                if (svc.last_error && !svc.ok) {{
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
