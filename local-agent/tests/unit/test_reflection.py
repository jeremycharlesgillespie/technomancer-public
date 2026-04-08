"""
Tests for agent/reflection.py - reflection loop, question classification, and casual detection.
"""

from unittest.mock import MagicMock, patch

from agent.reflection import (
    FACTUAL_PASSES,
    LIGHT_PASSES,
    REFLECTION_PASSES,
    auto_search_for_factual,
    classify_question,
    is_factual_question,
    reflect,
)


class TestClassifyQuestion:
    """Tests for classify_question() - smart question routing."""

    # --- SKIP cases ---

    def test_greeting_is_skip(self):
        assert classify_question("hello", "Hi there!") == "skip"

    def test_thanks_is_skip(self):
        assert classify_question("thanks!", "You're welcome!") == "skip"

    def test_short_message_is_skip(self):
        assert classify_question("hey", "Hello!") == "skip"

    def test_what_time_is_skip(self):
        assert classify_question("What time is it?", "It's 2:33 AM.") == "skip"

    def test_what_day_is_skip(self):
        assert classify_question("What day is today?", "It's Saturday.") == "skip"

    def test_short_response_is_skip(self):
        assert classify_question("Tell me something interesting", "Cats sleep 16 hours.") == "skip"

    def test_ok_is_skip(self):
        assert classify_question("ok sounds good", "Great!") == "skip"

    def test_lol_is_skip(self):
        assert classify_question("lol that's funny", "Ha!") == "skip"

    def test_status_command_is_skip(self):
        assert classify_question("start the bot", "Starting...") == "skip"

    # --- FACTUAL cases ---

    def test_where_is_factual(self):
        long_response = "Blanchard is in Oklahoma..." + "x" * 200
        assert classify_question("Where is Blanchard, OK?", long_response) == "factual"

    def test_who_is_factual(self):
        long_response = "The current president is..." + "x" * 200
        assert classify_question("Who is the president?", long_response) == "factual"

    def test_when_did_is_factual(self):
        long_response = "World War 2 ended in..." + "x" * 200
        assert classify_question("When did World War 2 end?", long_response) == "factual"

    def test_what_is_factual(self):
        long_response = "The capital of France is..." + "x" * 200
        assert classify_question("What is the capital of France?", long_response) == "factual"

    def test_tell_me_about_is_factual(self):
        long_response = "France is a country..." + "x" * 200
        assert classify_question("Tell me about France", long_response) == "factual"

    def test_who_invented_is_factual(self):
        long_response = "The telephone was invented by..." + "x" * 200
        assert classify_question("Who invented the telephone?", long_response) == "factual"

    def test_population_is_factual(self):
        long_response = "The population of Tokyo is..." + "x" * 200
        assert classify_question("What is the population of Tokyo?", long_response) == "factual"

    # --- Short questions that should NOT be skipped ---

    def test_short_deep_question_not_skipped(self):
        long_response = "He died in a battle..." + "x" * 200
        result = classify_question("how did he die?", long_response)
        assert result != "skip"

    def test_short_question_with_question_mark_not_casual(self):
        long_response = "Yes, you should..." + "x" * 200
        result = classify_question("ok but why?", long_response)
        assert result != "skip"

    # --- FULL cases ---

    def test_opinion_question_is_full(self):
        long_response = "In my opinion..." + "x" * 200
        assert classify_question("What do you think about microservices?", long_response) == "full"

    def test_should_i_question_is_full(self):
        long_response = "Well, it depends..." + "x" * 200
        assert classify_question("Should I use Django or Flask for this project?", long_response) == "full"

    def test_best_practice_is_full(self):
        long_response = "The best approach..." + "x" * 200
        assert classify_question("What's the best way to handle authentication?", long_response) == "full"

    def test_future_impact_is_full(self):
        long_response = "AI will likely..." + "x" * 200
        assert classify_question("How will AI affect the finance industry?", long_response) == "full"

    def test_why_question_is_full(self):
        long_response = "The reason is..." + "x" * 200
        assert classify_question("Why is Python slower than Go for web servers?", long_response) == "full"

    def test_architecture_question_is_full(self):
        long_response = "When designing..." + "x" * 200
        assert classify_question("How should I architect my API gateway?", long_response) == "full"

    def test_pros_cons_is_full(self):
        long_response = "There are tradeoffs..." + "x" * 200
        assert classify_question("What are the pros and cons of serverless?", long_response) == "full"

    def test_long_response_upgrades_to_full(self):
        long_response = "x" * 600
        assert classify_question("How do I read a file in Python?", long_response) == "full"

    # --- LIGHT cases ---

    def test_simple_technical_is_light(self):
        medium_response = "You can use the open() function..." + "x" * 100
        assert classify_question("How do I read a file in Python?", medium_response) == "light"


# TestIsCasual removed — is_casual() was deleted as deprecated legacy code


class TestReflect:
    """Tests for reflect() function."""

    def test_reflect_runs_all_passes_full_mode(self):
        """reflect() in full mode calls agent.chat() once per reflection pass."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        mock_agent.chat.side_effect = [
            "Pass 0: user wants production-ready advice",
            "Pass 1: missing edge cases",
            "Pass 2: should use connection pool",
            "Pass 3: devil's advocate objection",
            "Final refined answer with all improvements incorporated.",
        ]

        final, thoughts = reflect(mock_agent, "How do I connect to a database?", mode="full")

        assert mock_agent.chat.call_count == len(REFLECTION_PASSES)
        assert final == "Final refined answer with all improvements incorporated."

    def test_reflect_light_mode_runs_two_passes(self):
        """reflect() in light mode only runs completeness + final."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        mock_agent.chat.side_effect = [
            "Missing some edge cases",
            "Final answer with fixes.",
        ]

        final, thoughts = reflect(mock_agent, "How do I read a file?", mode="light")

        assert mock_agent.chat.call_count == len(LIGHT_PASSES)
        assert final == "Final answer with fixes."

    def test_reflect_factual_mode_runs_two_passes(self):
        """reflect() in factual mode runs fact_check + final."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        mock_agent.chat.side_effect = [
            "Checked: location is correct per search results",
            "Blanchard is in McClain County, south of OKC.",
        ]

        final, thoughts = reflect(mock_agent, "Where is Blanchard OK?", mode="factual")

        assert mock_agent.chat.call_count == len(FACTUAL_PASSES)
        assert final == "Blanchard is in McClain County, south of OKC."

    def test_reflect_returns_final_pass_only(self):
        """reflect() returns only the final answer in the first tuple element."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        mock_agent.chat.side_effect = [
            "intent analysis",
            "critique 1",
            "critique 2",
            "critique 3",
            "THE FINAL ANSWER",
        ]

        final, thoughts = reflect(mock_agent, "some question")
        assert final == "THE FINAL ANSWER"
        assert "critique" not in final

    def test_reflect_handles_chat_failure_gracefully(self):
        """reflect() returns empty string and empty thoughts if first pass fails."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = Exception("Ollama error")

        final, thoughts = reflect(mock_agent, "some question")
        assert final == ""
        assert thoughts == []

    def test_reflect_collects_thoughts(self):
        """reflect() returns thinking content from each pass."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "response"
        mock_agent.last_thinking = "I need to think about this more carefully..."

        final, thoughts = reflect(mock_agent, "original question")

        assert len(thoughts) == len(REFLECTION_PASSES)
        for pass_name, thinking in thoughts:
            assert thinking == "I need to think about this more carefully..."

    def test_reflect_skips_empty_thoughts(self):
        """reflect() omits passes with no thinking content."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "response"
        mock_agent.last_thinking = ""

        final, thoughts = reflect(mock_agent, "original question")
        assert thoughts == []

    def test_reflect_uses_correct_prompts(self):
        """reflect() passes the structured reflection prompts to agent.chat()."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        mock_agent.chat.return_value = "response"

        reflect(mock_agent, "original question")

        called_prompts = [call.args[0] for call in mock_agent.chat.call_args_list]
        pass_prompts = [prompt for _, prompt in REFLECTION_PASSES]
        assert called_prompts == pass_prompts


class TestAutoSearchForFactual:
    """Tests for auto_search_for_factual()."""

    @patch("agent.utility_tools.wikipedia_summary")
    @patch("agent.web_search.web_search")
    def test_returns_search_context(self, mock_search, mock_wiki):
        mock_wiki.return_value = "**Blanchard** (Wikipedia)\n\nBlanchard is in McClain County, OK"
        mock_search.return_value = "1. Blanchard is a city in Oklahoma"
        mock_agent = MagicMock()

        result = auto_search_for_factual(mock_agent, "Where is Blanchard OK?")

        assert "WIKIPEDIA" in result
        assert "McClain County" in result

    @patch("agent.utility_tools.wikipedia_summary")
    @patch("agent.web_search.web_search")
    def test_returns_empty_on_no_results(self, mock_search, mock_wiki):
        mock_search.return_value = "No results found for: blah"
        mock_wiki.return_value = "No Wikipedia article found for: blah"
        mock_agent = MagicMock()

        result = auto_search_for_factual(mock_agent, "blah")
        assert result == ""

    @patch("agent.utility_tools.wikipedia_summary")
    @patch("agent.web_search.web_search")
    def test_returns_empty_on_error(self, mock_search, mock_wiki):
        mock_search.side_effect = Exception("Network error")
        mock_wiki.side_effect = Exception("Network error")
        mock_agent = MagicMock()

        result = auto_search_for_factual(mock_agent, "test")
        assert result == ""


class TestReflectionPasses:
    """Sanity checks on the REFLECTION_PASSES config."""

    def test_has_five_passes(self):
        assert len(REFLECTION_PASSES) == 5

    def test_final_pass_is_last(self):
        assert REFLECTION_PASSES[-1][0] == "final"

    def test_all_passes_have_non_empty_prompts(self):
        for name, prompt in REFLECTION_PASSES:
            assert name
            assert len(prompt.strip()) > 20

    def test_light_passes_are_subset(self):
        all_names = [n for n, _ in REFLECTION_PASSES]
        for name in LIGHT_PASSES:
            assert name in all_names


class TestIsFactualQuestion:
    """Tests for is_factual_question() pre-classifier."""

    def test_where_is_not_factual(self):
        """Basic 'where is X' doesn't need web search — LLM knows geography."""
        assert is_factual_question("Where is Blanchard, OK?") is False

    def test_who_invented_is_factual(self):
        assert is_factual_question("Who invented the telephone?") is True

    def test_tell_me_about_not_factual(self):
        """Basic 'tell me about X' doesn't need web search — LLM knows this."""
        assert is_factual_question("Tell me about France") is False

    def test_capital_of_is_factual(self):
        assert is_factual_question("What is the capital of France?") is True

    def test_current_price_is_factual(self):
        assert is_factual_question("What is the current price of bitcoin?") is True

    def test_latest_version_is_factual(self):
        assert is_factual_question("What is the latest version of Python?") is True

    def test_basic_knowledge_not_factual(self):
        """Basic knowledge questions don't need web search."""
        assert is_factual_question("What is spaghetti?") is False

    def test_greeting_not_factual(self):
        assert is_factual_question("hello") is False

    def test_opinion_not_factual(self):
        assert is_factual_question("Should I use Django?") is False


class TestReflectHistoryIsolation:
    """Tests that reflection does not pollute agent conversation history."""

    def test_reflect_restores_message_history(self):
        """After reflection, agent.messages should not contain reflection prompts."""
        mock_agent = MagicMock()
        mock_agent.last_thinking = ""
        # Simulate a real messages list with system + user + assistant
        mock_agent.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "How do I connect to a database?"},
            {"role": "assistant", "content": "Use psycopg2 to connect..."},
        ]
        mock_agent.chat.side_effect = [
            "intent check",
            "completeness check",
            "best practices check",
            "adversarial check",
            "REFINED FINAL ANSWER",
        ]

        final, _ = reflect(mock_agent, "How do I connect to a database?", mode="full")

        # History should be restored to original 3 messages
        assert len(mock_agent.messages) == 3
        # Last assistant message should be updated with refined answer
        assert mock_agent.messages[-1]["content"] == "REFINED FINAL ANSWER"
        assert final == "REFINED FINAL ANSWER"

    def test_factual_passes_end_with_final(self):
        assert FACTUAL_PASSES[-1][0] == "final"

    def test_factual_passes_has_fact_check(self):
        names = [n for n, _ in FACTUAL_PASSES]
        assert "fact_check" in names
