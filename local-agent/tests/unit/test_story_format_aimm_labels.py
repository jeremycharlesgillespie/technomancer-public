"""Tests for the AIMM label constants in agent.story_format."""

from agent import story_format
from agent.story_format import (
    AIMM_APPROVED_LABEL,
    AIMM_ARCHIVED_LABEL,
    AIMM_DRAFTED_LABEL,
)


def test_aimm_approved_label_value():
    assert AIMM_APPROVED_LABEL == "aimm-approved"


def test_aimm_drafted_label_value():
    assert AIMM_DRAFTED_LABEL == "aimm-drafted"


def test_aimm_archived_label_value():
    assert AIMM_ARCHIVED_LABEL == "aimm-archived"


def test_aimm_labels_are_strings():
    assert isinstance(AIMM_APPROVED_LABEL, str)
    assert isinstance(AIMM_DRAFTED_LABEL, str)
    assert isinstance(AIMM_ARCHIVED_LABEL, str)


def test_aimm_labels_importable_from_module_namespace():
    assert story_format.AIMM_APPROVED_LABEL == "aimm-approved"
    assert story_format.AIMM_DRAFTED_LABEL == "aimm-drafted"
    assert story_format.AIMM_ARCHIVED_LABEL == "aimm-archived"


def test_aimm_labels_are_distinct():
    labels = {AIMM_APPROVED_LABEL, AIMM_DRAFTED_LABEL, AIMM_ARCHIVED_LABEL}
    assert len(labels) == 3
