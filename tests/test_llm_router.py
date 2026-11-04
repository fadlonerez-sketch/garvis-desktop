"""Tests for the request classifier in app/llm/router.py (`_classify()` and
helpers, used by classify_tier()).

The former two-provider `LLMRouter`, which sent each request to one of two
different models, was removed under the single-model policy; these are its
classification cases, now asserted on the tier decision itself. The tier
only picks Claude Haiku 5.5's effort level -- never a different model.
"""
from __future__ import annotations

import pytest

from app.llm.base import Message, Role, ToolCall
from app.llm.registry import ModelTier
from app.llm.router import _looks_tool_oriented, classify_tier


def _sys_and_user(text: str) -> list[Message]:
    return [Message(role=Role.SYSTEM, content="You are Garvis."), Message(role=Role.USER, content=text)]


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of France?",
        "How many centimeters are in 2 meters?",
        "Rewrite this sentence more politely.",
        "What does CPU stand for?",
        "Say this in English.",
        "Hello!",
    ],
)
def test_simple_requests_are_small(text):
    assert classify_tier(_sys_and_user(text)) == (ModelTier.SMALL, "simple_request")


@pytest.mark.parametrize(
    "text",
    [
        "Explain why the sky is blue in detail.",
        "Compare Python and JavaScript for backend development.",
        "Which is better, a Tesla or a BMW, and why?",
        "Can you help me debug this Python function?",
        "Write a function that reverses a linked list.",
        "What is the best way to configure this system and why?",
        "Compare these three options and tell me which one you would choose.",
    ],
)
def test_complex_requests_are_large(text):
    assert classify_tier(_sys_and_user(text)) == (ModelTier.LARGE, "complex_request")


@pytest.mark.parametrize(
    "text",
    [
        "Search the web for the latest news about AI.",
        "Check my email for messages from John.",
        "Open the browser and go to example.com.",
        "Find restaurants near me in Jerusalem.",
        "Book a table for two tonight.",
    ],
)
def test_tool_oriented_requests_are_medium(text):
    assert classify_tier(_sys_and_user(text)) == (ModelTier.MEDIUM, "tool_intent")


def test_word_search_definition_question_is_not_treated_as_tool_intent():
    """Regression: merely containing the word 'search' must not itself
    trigger tool routing -- only clear tool-intent phrases (e.g. "search
    for", "search the web") should."""
    assert not _looks_tool_oriented("What does the word 'search' mean?")
    assert _looks_tool_oriented("Search for the best pizza place nearby.")


def test_ambiguous_request_is_medium():
    assert classify_tier(_sys_and_user("Can you help me with my trip?")) == (ModelTier.MEDIUM, "uncertain")


def test_long_conversation_context_is_not_small_even_for_a_simple_looking_question():
    history = [Message(role=Role.SYSTEM, content="sys")]
    for i in range(3):
        history.append(Message(role=Role.USER, content=f"question {i}"))
        history.append(Message(role=Role.ASSISTANT, content=f"answer {i}"))
    history.append(Message(role=Role.USER, content="What is the capital of France?"))
    assert classify_tier(history) == (ModelTier.MEDIUM, "substantial_context")


def test_ongoing_tool_loop_is_not_small():
    """Once a tool call is in flight for this turn, the follow-up call
    (carrying the tool result) stays out of SMALL, even though the original
    question looked trivial."""
    messages = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="What is the capital of France?"),
        Message(role=Role.ASSISTANT, content="",
                tool_calls=[ToolCall(id="call-1", name="web_search", arguments={"query": "capital of france"})]),
        Message(role=Role.TOOL, content='{"result": "Paris"}', tool_call_id="call-1", name="web_search"),
    ]
    assert classify_tier(messages) == (ModelTier.MEDIUM, "tool_activity")


def test_the_two_model_router_is_gone():
    import app.llm.router as router_module

    assert not hasattr(router_module, "LLMRouter")
