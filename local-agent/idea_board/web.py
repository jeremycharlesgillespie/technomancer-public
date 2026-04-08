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
import logging
import threading
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, request

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
# LLM CONVERSATION FOR IDEAS
# ============================================================================

IDEA_DISCUSSION_PROMPT = """You are an eager, thoughtful software engineer discussing an improvement idea
with your manager (Jeremy). You originally proposed this idea. Now Jeremy is
giving you feedback and asking questions about it.

YOUR IDEA:
Title: {title}
Description: {description}
Category: {category}

CONVERSATION SO FAR:
{conversation}

RULES:
- Be direct, specific, and technical — Jeremy is a senior software engineer
- If he asks you to explain, give concrete technical details
- If he pushes back, consider his point honestly — maybe the idea needs refinement
- If he's interested, suggest next steps or implementation approach
- If you realize the idea is bad based on his feedback, say so honestly
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
        role = "Jeremy (manager)" if c.author == "jeremy" else "You (engineer)"
        conv_lines.append(f"{role}: {c.text}")
    conversation = "\n".join(conv_lines)

    prompt = IDEA_DISCUSSION_PROMPT.format(
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
.comment.jeremy { background: #1a3a5c; margin-left: auto; }
.comment.llm { background: #2d2d2d; }
.comment-author { font-weight: 600; font-size: 0.75rem; margin-bottom: 2px; }
.comment-author.jeremy { color: var(--accent); }
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
"""


def _format_description(raw: str) -> str:
    """Format a structured idea description into HTML sections.

    Detects lines starting with WHAT:, WHY:, HOW:, BENEFITS:, COST:, UNLOCKS:
    and renders each as a labeled section. Falls back to plain text for
    descriptions without section headers.
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

    # Check if description has structured sections
    has_sections = any(raw.strip().startswith(k + ":") or ("\n" + k + ":") in raw for k in section_labels)

    if not has_sections:
        return html.escape(raw)

    # Parse sections
    parts = []
    # Split on section headers
    pattern = r"(?:^|\n)((?:WHAT|WHY|HOW|BENEFITS|COST|UNLOCKS):)"
    splits = re.split(pattern, raw.strip())

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


def _render_idea_card(idea: dict[str, Any]) -> str:
    """Render a single idea as an HTML card."""
    eid = html.escape(idea["id"])
    title = html.escape(idea["title"])
    desc = _format_description(idea["description"])
    state = idea["state"]
    category = html.escape(idea.get("category", ""))
    source = html.escape(idea.get("source", ""))
    created = idea.get("created", "")[:10]
    claude_vote = idea.get("votes", {}).get("claude") or "—"
    jeremy_vote = idea.get("votes", {}).get("jeremy") or "—"

    # Comments section — chat-style conversation
    comments_html = ""
    for c in idea.get("comments", []):
        author = html.escape(c["author"])
        text = html.escape(c["text"])
        ts = c.get("timestamp", "")[:16]
        label = "Owner" if author == "jeremy" else "LLM"
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
    done_btn = f'<button class="btn btn-done" onclick="doMarkDone(\'{eid}\')">Mark Done</button>'
    delete_btn = f'<button class="btn btn-delete" onclick="doDelete(\'{eid}\')">Delete</button>'
    if state in ("proposed", "refining"):
        actions = f'<div class="actions">{approve_btn} {veto_btn} {execute_btn} {delete_btn}</div>'
    elif state == "approved":
        actions = f'<div class="actions">{execute_btn} {done_btn} {delete_btn}</div>'
    elif state == "failed":
        actions = f'<div class="actions">{execute_btn} {done_btn} {delete_btn}</div>'
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

    return f"""
    <div class="card {state}" data-idea="{eid}">
        <div class="card-title">{eid}: {title}</div>
        <div class="card-meta">
            <span class="badge badge-state {state}">{state}</span>
            <span class="badge badge-cat">{category}</span>
            <span class="badge badge-src">{source}</span>
            &bull; {created} &bull; Claude: {claude_vote} &bull; Jeremy: {jeremy_vote}
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
    # Group by state
    groups: dict[str, list[dict]] = {}
    for idea in ideas:
        state = idea["state"]
        groups.setdefault(state, []).append(idea)

    # Render sections in priority order
    sections_html = ""
    for state in ["proposed", "refining", "approved", "executing", "done", "failed", "vetoed"]:
        items = groups.get(state, [])
        if items:
            sections_html += f'<h2 class="section-title">{state.upper()} ({len(items)})</h2>\n'
            for idea in items:
                sections_html += _render_idea_card(idea)

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
    </div>
    <p class="stats" id="stats">{total} ideas total &bull; {proposed} pending review &bull; Last refresh: {now}</p>
    {sections_html if sections_html else '<p style="color:var(--muted)">No ideas yet. They will start appearing hourly.</p>'}

    <script>
    async function doVote(id, v) {{
        const card = document.querySelector(`[data-idea="${{id}}"]`) || event.target.closest('.card');
        const btn = event.target;
        btn.disabled = true;
        btn.textContent = '...';
        await fetch(`/api/ideas/${{id}}/vote`, {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{voter: 'jeremy', vote: v}})
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
            body: JSON.stringify({{voter: 'jeremy', vote: 'approve'}})
        }});

        // Try to open claude.ai/code in a new tab
        // Note: popup blockers may prevent this — the toast tells user to paste manually
        const win = window.open('https://claude.ai/code', '_blank');
        if (!win) {{
            showToast('Popup blocked — open claude.ai/code manually and paste.');
        }}

        // Reset button after 5 seconds
        setTimeout(() => {{
            btn.textContent = 'Copy for Claude Code';
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
            `<div class="comment jeremy">` +
            `<div class="comment-author jeremy">Jeremy</div>` +
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
            body: JSON.stringify({{author: 'jeremy', text: text}})
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
    if voter not in ("jeremy", "claude") or not vote_value:
        return jsonify({"error": "Need voter (jeremy|claude) and vote"}), 400

    idea = vote(idea_id, voter, vote_value)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>/comment", methods=["POST"])
def api_comment(idea_id: str) -> tuple:
    """POST /api/ideas/<id>/comment — add a comment and get LLM reply.

    When Jeremy posts a comment, the LLM reads the full idea context
    and conversation history, then replies as an eager employee
    discussing the idea with their manager.
    """
    data = request.get_json(silent=True) or {}
    author = data.get("author", "jeremy")
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Need text"}), 400

    idea = add_comment(idea_id, author, text)
    if not idea:
        return jsonify({"error": "Idea not found"}), 404

    # If Jeremy posted, trigger an LLM reply in a background thread
    if author == "jeremy":
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
    idea = mark_done(idea_id, "Manually marked as done by Jeremy.")
    if not idea:
        return jsonify({"error": "Idea not found"}), 404
    return jsonify(idea.to_dict())


@app.route("/api/ideas/<idea_id>", methods=["DELETE"])
def api_delete(idea_id: str) -> tuple:
    """DELETE /api/ideas/<id> — permanently remove an idea from the board."""
    if delete_idea(idea_id):
        return jsonify({"status": "deleted", "idea_id": idea_id})
    return jsonify({"error": "Idea not found"}), 404


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
            label = "Jeremy (manager)" if c.author == "jeremy" else "LLM (engineer)"
            discussion += f"- {label}: {c.text}\n"

    prompt = (
        f"Implement this improvement for the Technomancer project.\n\n"
        f"## Idea: {idea.title}\n\n"
        f"**Description:** {idea.description}\n"
        f"**Category:** {idea.category}\n"
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
</body>
</html>"""


def start_idea_board() -> None:
    """Start the Flask hub (idea board + news config) in a daemon thread.

    Binds to 0.0.0.0 so it's accessible over Tailscale/LAN.
    """
    # Register news config blueprint
    from .news_config import news_bp
    app.register_blueprint(news_bp)

    def _run() -> None:
        app.run(host="0.0.0.0", port=BOARD_PORT, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="idea-board")
    thread.start()
    logger.info(f"Technomancer Hub running on http://0.0.0.0:{BOARD_PORT}")
