"""Tests for agent/task_manager.py — monitored task creation, health checking, shutdown."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import agent.task_manager as tm_module
from agent.task_manager import (
    _background_tasks,
    _handle_task_done,
    _reset,
    create_monitored_task,
    get_registered_count,
    get_task_status,
    register_shutdown_callback,
    shutdown_sync,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset task manager state before and after each test."""
    _reset()
    yield
    _reset()


# ================================================================
# create_monitored_task
# ================================================================


class TestCreateMonitoredTask:
    @pytest.mark.asyncio
    async def test_runs_coroutine_to_completion(self):
        """Non-critical task runs and completes normally."""
        result = []

        async def work():
            result.append("done")

        task = create_monitored_task(work(), "test-work")
        await task
        assert result == ["done"]

    @pytest.mark.asyncio
    async def test_critical_task_registered(self):
        """Critical tasks appear in _background_tasks."""
        async def loop():
            await asyncio.sleep(999)

        task = create_monitored_task(loop(), "my-loop", critical=True)
        assert "my-loop" in _background_tasks
        assert _background_tasks["my-loop"] is task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_non_critical_not_registered(self):
        """Non-critical tasks do NOT appear in _background_tasks."""
        async def quick():
            pass

        task = create_monitored_task(quick(), "ephemeral")
        await task
        assert "ephemeral" not in _background_tasks

    @pytest.mark.asyncio
    async def test_task_name_set(self):
        """Task gets the correct asyncio name."""
        async def noop():
            await asyncio.sleep(999)

        task = create_monitored_task(noop(), "named-task", critical=True)
        assert task.get_name() == "named-task"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ================================================================
# _handle_task_done — exception handling
# ================================================================


class TestHandleTaskDone:
    @pytest.mark.asyncio
    async def test_exception_logged(self):
        """Task that raises gets its exception logged."""
        async def bad():
            raise ValueError("boom")

        with patch.object(tm_module, "log") as mock_log:
            task = create_monitored_task(bad(), "bad-task")
            # Let the task run and fail
            with pytest.raises(ValueError):
                await task
            # Give the done callback a chance to fire
            await asyncio.sleep(0)
            # Check that error was logged
            error_calls = [c for c in mock_log.error.call_args_list
                           if "bad-task" in str(c)]
            assert len(error_calls) >= 1

    @pytest.mark.asyncio
    async def test_exception_sends_alert(self):
        """Task crash triggers send_alert."""
        async def bad():
            raise RuntimeError("kaboom")

        with patch.object(tm_module, "send_alert") as mock_alert:
            task = create_monitored_task(bad(), "alert-task")
            with pytest.raises(RuntimeError):
                await task
            await asyncio.sleep(0)
            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert "alert-task" in call_args[0][0]
            assert call_args[1]["level"] == "error"

    @pytest.mark.asyncio
    async def test_cancelled_task_no_alert(self):
        """Cancelled tasks are logged at info level, no alert."""
        async def slow():
            await asyncio.sleep(999)

        with patch.object(tm_module, "send_alert") as mock_alert, \
             patch.object(tm_module, "log") as mock_log:
            task = create_monitored_task(slow(), "cancel-me")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0)
            mock_alert.assert_not_called()
            info_calls = [c for c in mock_log.info.call_args_list
                          if "cancel-me" in str(c)]
            assert len(info_calls) >= 1

    @pytest.mark.asyncio
    async def test_critical_task_removed_on_crash(self):
        """Critical task is removed from registry when it crashes."""
        async def bad():
            raise ValueError("oops")

        with patch.object(tm_module, "send_alert"):
            task = create_monitored_task(bad(), "will-crash", critical=True)
            assert "will-crash" in _background_tasks
            with pytest.raises(ValueError):
                await task
            await asyncio.sleep(0)
            assert "will-crash" not in _background_tasks

    @pytest.mark.asyncio
    async def test_critical_task_warns_on_clean_exit(self):
        """Critical task that ends without error logs a warning."""
        async def short_lived():
            return "done"

        with patch.object(tm_module, "log") as mock_log:
            task = create_monitored_task(short_lived(), "short", critical=True)
            await task
            await asyncio.sleep(0)
            warning_calls = [c for c in mock_log.warning.call_args_list
                             if "short" in str(c)]
            assert len(warning_calls) >= 1

    @pytest.mark.asyncio
    async def test_alert_failure_does_not_propagate(self):
        """If send_alert itself fails, the done callback doesn't crash."""
        async def bad():
            raise ValueError("boom")

        with patch.object(tm_module, "send_alert", side_effect=Exception("alert broken")):
            task = create_monitored_task(bad(), "resilient")
            with pytest.raises(ValueError):
                await task
            await asyncio.sleep(0)
            # No unhandled exception — test passes if we get here


# ================================================================
# Health checker
# ================================================================


class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_detects_dead_critical_task(self):
        """Health check loop detects a dead critical task and sends alert."""
        # Manually register a dead task
        async def dead():
            pass

        dead_task = asyncio.create_task(dead())
        await dead_task  # Let it finish
        _background_tasks["dead-one"] = dead_task

        # Run one iteration of health check manually
        with patch.object(tm_module, "send_alert") as mock_alert, \
             patch.object(tm_module, "HEALTH_CHECK_INTERVAL", 0):
            from agent.task_manager import _health_check_loop

            # Create the health check loop and let it run once
            async def run_one_cycle():
                # Skip the initial 60s sleep
                with patch("asyncio.sleep", return_value=None):
                    # We need to break after one cycle
                    call_count = 0

                    async def counting_sleep(seconds):
                        nonlocal call_count
                        call_count += 1
                        if call_count > 2:
                            raise asyncio.CancelledError()

                    with patch("agent.task_manager.asyncio.sleep", side_effect=counting_sleep):
                        try:
                            await _health_check_loop()
                        except asyncio.CancelledError:
                            pass

            await run_one_cycle()
            assert mock_alert.called
            call_msg = mock_alert.call_args[0][0]
            assert "dead-one" in call_msg

    @pytest.mark.asyncio
    async def test_all_alive_no_alert(self):
        """When all critical tasks are alive, no alert is sent."""
        async def alive():
            await asyncio.sleep(999)

        alive_task = create_monitored_task(alive(), "alive-one", critical=True)

        with patch.object(tm_module, "send_alert") as mock_alert:
            call_count = 0

            async def counting_sleep(seconds):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise asyncio.CancelledError()

            with patch("agent.task_manager.asyncio.sleep", side_effect=counting_sleep):
                try:
                    await tm_module._health_check_loop()
                except asyncio.CancelledError:
                    pass

            mock_alert.assert_not_called()

        alive_task.cancel()
        try:
            await alive_task
        except asyncio.CancelledError:
            pass


# ================================================================
# Shutdown
# ================================================================


class TestShutdown:
    def test_callbacks_executed(self):
        """shutdown_sync runs all registered callbacks."""
        called = []
        register_shutdown_callback(lambda: called.append("a"))
        register_shutdown_callback(lambda: called.append("b"))
        shutdown_sync()
        assert called == ["a", "b"]

    def test_callback_failure_doesnt_stop_others(self):
        """A failing callback doesn't prevent subsequent ones from running."""
        called = []

        def bad():
            raise RuntimeError("fail")

        register_shutdown_callback(bad)
        register_shutdown_callback(lambda: called.append("ok"))
        shutdown_sync()
        assert called == ["ok"]

    def test_no_callbacks_is_fine(self):
        """shutdown_sync with nothing registered doesn't crash."""
        shutdown_sync()  # Should not raise


# ================================================================
# Introspection
# ================================================================


class TestIntrospection:
    @pytest.mark.asyncio
    async def test_get_task_status_alive(self):
        """get_task_status reports alive tasks correctly."""
        async def running():
            await asyncio.sleep(999)

        task = create_monitored_task(running(), "runner", critical=True)
        status = get_task_status()
        assert "runner" in status
        assert status["runner"]["alive"] is True
        assert status["runner"]["error"] is None

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_get_task_status_dead_with_error(self):
        """get_task_status reports crashed tasks with error info."""
        async def bad():
            raise TypeError("wrong type")

        with patch.object(tm_module, "send_alert"):
            task = create_monitored_task(bad(), "crasher", critical=True)
            with pytest.raises(TypeError):
                await task
            await asyncio.sleep(0)

        # Task is removed from registry by done callback, so re-add for status test
        _background_tasks["crasher"] = task
        status = get_task_status()
        assert "crasher" in status
        assert status["crasher"]["alive"] is False
        assert "TypeError" in status["crasher"]["error"]

    @pytest.mark.asyncio
    async def test_get_registered_count(self):
        """get_registered_count returns correct totals."""
        async def alive():
            await asyncio.sleep(999)

        task1 = create_monitored_task(alive(), "t1", critical=True)
        task2 = create_monitored_task(alive(), "t2", critical=True)
        total, alive_count = get_registered_count()
        assert total == 2
        assert alive_count == 2

        task1.cancel()
        task2.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass
        try:
            await task2
        except asyncio.CancelledError:
            pass


class TestReset:
    def test_reset_clears_state(self):
        """_reset clears all registries."""
        _background_tasks["test"] = MagicMock()
        register_shutdown_callback(lambda: None)
        _reset()
        assert len(_background_tasks) == 0
        # Verify shutdown callbacks are also cleared
        called = []
        shutdown_sync()
        assert called == []  # No callbacks ran
