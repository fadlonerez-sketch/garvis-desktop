"""Tests for token-aware trimming in ConversationContext."""

from __future__ import annotations

from app.agent.context import ConversationContext
from app.agent.policies import build_system_prompt
from app.llm.base import Message, Role


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def test_normal_recent_messages_are_preserved() -> None:
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt("You are Garvis.")
    for i in range(5):
        context.add(_msg(Role.USER, f"short message {i}"))
        context.add(_msg(Role.ASSISTANT, f"short reply {i}"))

    messages = context.messages
    assert messages[0].role == Role.SYSTEM
    # None of these small messages should have been evicted by the token budget.
    assert len(messages) == 1 + 10
    assert messages[-1].content == "short reply 4"


def test_large_old_tool_result_is_evicted_before_message_count_limit() -> None:
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt("You are Garvis.")

    huge_tool_result = "x" * 8000  # ~2000 tokens, alone over budget
    context.add(Message(role=Role.TOOL, content=huge_tool_result, tool_call_id="1", name="big_tool"))

    # Well under the 30-message count limit, so only token-based trimming
    # can be responsible for removing it.
    for i in range(5):
        context.add(_msg(Role.USER, f"message {i}"))
        context.add(_msg(Role.ASSISTANT, f"reply {i}"))

    messages = context.messages
    assert all(m.name != "big_tool" for m in messages)
    assert len(messages) < 1 + 1 + 10


def test_most_recent_message_always_survives_even_if_oversized() -> None:
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt("You are Garvis.")

    huge_message = "y" * 20000  # far larger than the entire budget
    context.add(_msg(Role.USER, huge_message))

    messages = context.messages
    assert messages[-1].content == huge_message


def test_small_history_is_unaffected_by_token_trimming() -> None:
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt("You are Garvis.")
    context.add(_msg(Role.USER, "hello"))
    context.add(_msg(Role.ASSISTANT, "hi there"))

    messages = context.messages
    assert len(messages) == 3
    assert messages[1].content == "hello"
    assert messages[2].content == "hi there"


def test_message_count_trim_still_applies_independently_of_tokens() -> None:
    context = ConversationContext(max_messages=5, max_tokens=1_000_000)
    context.set_system_prompt("You are Garvis.")
    for i in range(10):
        context.add(_msg(Role.USER, f"m{i}"))

    messages = context.messages
    # system + 4 most recent user messages == max_messages
    assert len(messages) == 5
    assert messages[0].role == Role.SYSTEM
    assert [m.content for m in messages[1:]] == ["m6", "m7", "m8", "m9"]


def test_realistic_system_prompt_does_not_consume_entire_history_budget() -> None:
    """Regression test: the real system+personality prompt (built via
    build_system_prompt) is itself ~1200 approx-tokens -- roughly the whole
    default max_tokens budget. If that prompt were counted against the
    history budget (as it was before this fix), every one of these small
    recent messages would be evicted on the very next `add()`."""
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt(build_system_prompt(None))

    for i in range(5):
        context.add(_msg(Role.USER, f"short message {i}"))
        context.add(_msg(Role.ASSISTANT, f"short reply {i}"))

    messages = context.messages
    assert messages[0].role == Role.SYSTEM
    assert len(messages) == 1 + 10
    assert messages[-1].content == "short reply 4"


def test_earlier_exchange_survives_a_later_turn_with_real_system_prompt() -> None:
    """Regression test for the reported bug: with the real system prompt in
    place, an earlier user/assistant exchange (e.g. the user stating their
    name) must still be present in context after one more turn is added."""
    context = ConversationContext(max_messages=30, max_tokens=1200)
    context.set_system_prompt(build_system_prompt(None))

    context.add(_msg(Role.USER, "my name is Eris"))
    context.add(
        _msg(
            Role.ASSISTANT,
            "Understood, Eris. This is Garvis, not Jarvis. How may I assist you, Sir?",
        )
    )
    context.add(_msg(Role.USER, "what is my name?"))

    messages = context.messages
    assert any(m.content == "my name is Eris" for m in messages)
    assert any("Eris" in m.content for m in messages if m.role == Role.ASSISTANT)
