"""Tests for fallback_responses module — hardcoded responses during LLM outages."""

from datetime import datetime
from unittest.mock import patch

from agent.fallback_responses import get_fallback_response


class TestTimeQueries:
    """Test time-related fallback patterns."""

    def test_what_time_is_it(self):
        result = get_fallback_response("what time is it?")
        assert result is not None
        assert ":" in result  # contains a time

    def test_whats_the_time(self):
        result = get_fallback_response("what's the time")
        assert result is not None

    def test_current_time(self):
        result = get_fallback_response("current time")
        assert result is not None

    def test_time_contains_actual_time(self):
        with patch("agent.fallback_responses.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 7, 14, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_fallback_response("what time is it")
            assert "02:30 PM" in result


class TestDateQueries:
    """Test date-related fallback patterns."""

    def test_what_date(self):
        result = get_fallback_response("what date is it?")
        assert result is not None
        assert "Today is" in result

    def test_whats_the_date(self):
        result = get_fallback_response("what's the date")
        assert result is not None

    def test_what_day_is_it(self):
        result = get_fallback_response("what day is it")
        assert result is not None

    def test_todays_date(self):
        result = get_fallback_response("today's date")
        assert result is not None

    def test_current_date(self):
        result = get_fallback_response("current date")
        assert result is not None


class TestGreetings:
    """Test greeting fallback patterns."""

    def test_hi(self):
        result = get_fallback_response("hi")
        assert result is not None
        assert "LLM backends" in result

    def test_hello(self):
        result = get_fallback_response("hello!")
        assert result is not None

    def test_hey(self):
        result = get_fallback_response("hey")
        assert result is not None

    def test_good_morning(self):
        result = get_fallback_response("good morning!")
        assert result is not None

    def test_whats_up(self):
        result = get_fallback_response("what's up")
        assert result is not None

    def test_greeting_with_more_text_no_match(self):
        """Greetings with additional content should NOT match."""
        result = get_fallback_response("hello can you help me write code")
        assert result is None


class TestHelpAndStatus:
    """Test help and status query patterns."""

    def test_help(self):
        result = get_fallback_response("help")
        assert result is not None
        assert "commands" in result.lower() or "showCommands" in result

    def test_are_you_there(self):
        result = get_fallback_response("are you there?")
        assert result is not None
        assert "online" in result.lower() or "unavailable" in result.lower()

    def test_are_you_alive(self):
        result = get_fallback_response("are you alive")
        assert result is not None

    def test_you_up(self):
        result = get_fallback_response("you up?")
        assert result is not None

    def test_are_you_working(self):
        result = get_fallback_response("are you working?")
        assert result is not None


class TestNoMatch:
    """Test that non-matching queries return None."""

    def test_complex_question(self):
        result = get_fallback_response("explain quantum computing to me")
        assert result is None

    def test_code_request(self):
        result = get_fallback_response("write me a python function that sorts a list")
        assert result is None

    def test_empty_string(self):
        result = get_fallback_response("")
        assert result is None

    def test_random_text(self):
        result = get_fallback_response("asdfghjkl")
        assert result is None
