"""
Task Manager — Monitored asyncio task creation with error handling,
health checking, and graceful shutdown coordination.

Solves three problems:
1. Fire-and-forget asyncio.create_task() calls die silently on exception
2. No visibility into which background tasks are alive vs dead
3. No coordinated shutdown to flush pending writes (compaction, memory)

Usage:
    from .task_manager import create_monitored_task, start_health_checker

    # Long-running background loop (health-checked):
    create_monitored_task(news_digest_loop(), "news-digest", critical=True)

    # Short-lived one-off (exception-handled but not health-checked):
    create_monitored_task(extract_facts(), "extract-facts")
"""

import asyncio
import logging
import traceback
from typing import Any, Callable, Coroutine, Optional

from .alerts import send_alert

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Critical background tasks expected to run forever (name -> Task)
_background_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

# Health checker task reference
_health_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

# Sync callbacks to run during shutdown (e.g. stop compaction thread)
_shutdown_callbacks: list[Callable[[], None]] = []

# How often (seconds) the health checker scans for dead critical tasks
HEALTH_CHECK_INTERVAL: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Core: monitored task creation
# ---------------------------------------------------------------------------


def create_monitored_task(
    coro: Coroutine,  # type: ignore[type-arg]
    name: str,
    *,
    critical: bool = False,
) -> asyncio.Task:  # type: ignore[type-arg]
    """Create an asyncio task with automatic exception logging and alerting.

    Args:
        coro: The coroutine to schedule.
        name: Human-readable name for logging and health-check registry.
        critical: If True, the task is expected to run forever. The health
                  checker will flag it if it dies.

    Returns:
        The created asyncio.Task.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_done(t, name, critical))
    if critical:
        _background_tasks[name] = task
    return task


def _handle_task_done(task: asyncio.Task, name: str, critical: bool) -> None:  # type: ignore[type-arg]
    """Done callback: log exceptions, send alerts, update registry."""
    if critical:
        _background_tasks.pop(name, None)

    if task.cancelled():
        log.info("[TaskManager] Task '%s' was cancelled", name)
        return

    exc = task.exception()
    if exc is None:
        if critical:
            log.warning(
                "[TaskManager] Critical task '%s' ended without error (unexpected for a loop)",
                name,
            )
        return

    # Format the traceback for logging
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb_lines)
    log.error("[TaskManager] Task '%s' crashed:\n%s", name, tb_str)

    # Send alert to Discord alerts channel
    try:
        send_alert(
            f"Background task **{name}** crashed:\n```\n{type(exc).__name__}: {exc}\n```",
            title="Background Task Failure",
            level="error",
        )
    except Exception:
        pass  # Alert delivery must never disrupt anything


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------


async def _health_check_loop() -> None:
    """Periodically verify all critical background tasks are still alive."""
    await asyncio.sleep(60)  # Grace period for startup
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        dead: list[str] = []
        alive: list[str] = []
        for name, task in list(_background_tasks.items()):
            if task.done():
                dead.append(name)
            else:
                alive.append(name)

        if dead:
            log.error("[TaskManager] Dead critical tasks: %s", dead)
            try:
                send_alert(
                    f"**Dead tasks:** {', '.join(dead)}\n**Alive:** {len(alive)} tasks",
                    title="Background Task Health Check",
                    level="error",
                )
            except Exception:
                pass
        else:
            log.debug("[TaskManager] All %d critical tasks healthy", len(alive))


def start_health_checker() -> None:
    """Start the periodic health-check background task."""
    global _health_task
    _health_task = create_monitored_task(
        _health_check_loop(),
        "task-health-checker",
        critical=True,
    )
    log.info("[TaskManager] Health checker started (interval=%ds)", HEALTH_CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def register_shutdown_callback(callback: Callable[[], None]) -> None:
    """Register a synchronous callback to run during shutdown.

    Use this for things like stopping the compaction thread or flushing
    pending file writes.
    """
    _shutdown_callbacks.append(callback)


def shutdown_sync() -> None:
    """Run all registered shutdown callbacks (called from sync context).

    This runs after the event loop has stopped, so async tasks are already
    dead. This is for sync cleanup: stopping daemon threads, flushing files.
    """
    log.info("[TaskManager] Running %d shutdown callbacks...", len(_shutdown_callbacks))
    for cb in _shutdown_callbacks:
        try:
            cb_name = getattr(cb, "__qualname__", getattr(cb, "__name__", str(cb)))
            log.info("[TaskManager] Running: %s", cb_name)
            cb()
        except Exception as e:
            log.error("[TaskManager] Shutdown callback '%s' failed: %s", cb_name, e)

    # Log final state of background tasks
    for name, task in _background_tasks.items():
        if task.done():
            exc = None
            if not task.cancelled():
                try:
                    exc = task.exception()
                except Exception:
                    pass
            status = "cancelled" if task.cancelled() else f"crashed: {exc}" if exc else "ended"
        else:
            status = "was still alive"
        log.info("[TaskManager] Task '%s' at shutdown: %s", name, status)

    log.info("[TaskManager] Shutdown complete")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def get_task_status() -> dict[str, dict[str, Any]]:
    """Return status of all registered critical background tasks.

    Returns:
        Dict of task name -> status info with keys:
            alive (bool), cancelled (bool, if dead), error (str|None, if dead)
    """
    status: dict[str, dict[str, Any]] = {}
    for name, task in _background_tasks.items():
        if task.done():
            cancelled = task.cancelled()
            error = None
            if not cancelled:
                try:
                    exc = task.exception()
                    error = f"{type(exc).__name__}: {exc}" if exc else None
                except Exception:
                    error = "<unable to retrieve>"
            status[name] = {"alive": False, "cancelled": cancelled, "error": error}
        else:
            status[name] = {"alive": True, "cancelled": False, "error": None}
    return status


def get_registered_count() -> tuple[int, int]:
    """Return (total_critical_tasks, alive_critical_tasks)."""
    total = len(_background_tasks)
    alive = sum(1 for t in _background_tasks.values() if not t.done())
    return total, alive


# ---------------------------------------------------------------------------
# Reset (for testing)
# ---------------------------------------------------------------------------


def _reset() -> None:
    """Reset all state — for tests only."""
    global _health_task
    _background_tasks.clear()
    _shutdown_callbacks.clear()
    _health_task = None
