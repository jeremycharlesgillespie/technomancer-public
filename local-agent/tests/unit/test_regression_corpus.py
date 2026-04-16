"""Tests for agent/regression_corpus.py - Regression test corpus validation.

These tests don't run the corpus against a live model; they only validate that
the corpus itself is well-formed, sufficiently diverse, and stable enough for
quality_test.py and future regression harnesses to consume.
"""

import pytest

from agent.regression_corpus import (
    CATEGORIES,
    CORPUS,
    RegressionCase,
    categories_present,
    get_cases,
)


# =============================================================================
# SHAPE / SIZE
# =============================================================================


class TestCorpusShape:
    def test_corpus_is_tuple(self):
        assert isinstance(CORPUS, tuple)

    def test_corpus_has_20_to_30_cases(self):
        assert 20 <= len(CORPUS) <= 30, (
            f"Corpus should have 20-30 cases, has {len(CORPUS)}"
        )

    def test_every_entry_is_regression_case(self):
        for case in CORPUS:
            assert isinstance(case, RegressionCase)


# =============================================================================
# FIELD VALIDITY
# =============================================================================


class TestCaseFields:
    def test_questions_non_empty(self):
        for case in CORPUS:
            assert case.question.strip(), f"Empty question: {case}"

    def test_criteria_non_empty(self):
        for case in CORPUS:
            assert case.criteria.strip(), f"Empty criteria for: {case.question}"

    def test_criteria_describes_failure(self):
        """Every criteria should tell the grader when to fail, otherwise the
        grader has no way to give negative feedback."""
        for case in CORPUS:
            lowered = case.criteria.lower()
            assert "fail" in lowered or "pass if" in lowered, (
                f"Criteria for '{case.question}' doesn't describe pass/fail conditions"
            )

    def test_category_non_empty(self):
        for case in CORPUS:
            assert case.category.strip()

    def test_category_in_known_set(self):
        for case in CORPUS:
            assert case.category in CATEGORIES, (
                f"Unknown category '{case.category}' on question: {case.question}"
            )

    def test_tags_is_tuple(self):
        """Tags must be a tuple (not list) so RegressionCase stays hashable."""
        for case in CORPUS:
            assert isinstance(case.tags, tuple)

    def test_tags_are_strings(self):
        for case in CORPUS:
            for tag in case.tags:
                assert isinstance(tag, str) and tag.strip()


# =============================================================================
# DIVERSITY / COVERAGE
# =============================================================================


class TestCorpusDiversity:
    def test_questions_unique(self):
        questions = [c.question for c in CORPUS]
        assert len(questions) == len(set(questions)), "Duplicate questions found"

    def test_at_least_six_categories_represented(self):
        present = categories_present()
        assert len(present) >= 6, (
            f"Need broad coverage; only {len(present)} categories present: {present}"
        )

    @pytest.mark.parametrize("required_category", [
        "math",
        "factual",
        "honesty",
        "conciseness",
        "time_aware",
        "tool_use",
        "project",
    ])
    def test_critical_category_present(self, required_category):
        """Each critical regression category must have at least one case."""
        assert any(c.category == required_category for c in CORPUS), (
            f"Corpus missing a case in critical category '{required_category}'"
        )

    def test_no_category_dominates(self):
        """No single category should be more than ~40% of the corpus; a lopsided
        corpus hides regressions in underrepresented areas."""
        from collections import Counter

        counts = Counter(c.category for c in CORPUS)
        max_share = max(counts.values()) / len(CORPUS)
        assert max_share <= 0.4, (
            f"Category '{counts.most_common(1)[0][0]}' dominates ({max_share:.0%}); "
            "rebalance the corpus"
        )


# =============================================================================
# IMMUTABILITY
# =============================================================================


class TestRegressionCaseImmutable:
    def test_frozen_cannot_mutate(self):
        case = CORPUS[0]
        with pytest.raises(Exception):
            case.question = "mutated"  # type: ignore[misc]

    def test_case_is_hashable(self):
        """Frozen dataclasses with tuple fields should be hashable."""
        assert hash(CORPUS[0])

    def test_case_set_dedup_works(self):
        """Cases in a set should dedup by value."""
        duplicate = RegressionCase(
            question=CORPUS[0].question,
            criteria=CORPUS[0].criteria,
            category=CORPUS[0].category,
            tags=CORPUS[0].tags,
        )
        assert len({CORPUS[0], duplicate}) == 1


# =============================================================================
# FILTER HELPERS
# =============================================================================


class TestGetCases:
    def test_no_args_returns_full_corpus(self):
        assert get_cases() == CORPUS

    def test_filter_by_category(self):
        math_cases = get_cases(category="math")
        assert len(math_cases) > 0
        assert all(c.category == "math" for c in math_cases)

    def test_filter_by_unknown_category_returns_empty(self):
        assert get_cases(category="does-not-exist") == ()

    def test_filter_by_tag(self):
        tagged = get_cases(tag="memory")
        for c in tagged:
            assert "memory" in c.tags

    def test_filter_by_category_and_tag(self):
        result = get_cases(category="tool_use", tag="memory")
        for c in result:
            assert c.category == "tool_use"
            assert "memory" in c.tags

    def test_categories_present_matches_corpus(self):
        present = categories_present()
        from_corpus = {c.category for c in CORPUS}
        assert present == from_corpus
