"""Turn-based trimming and tool-result elision in ConversationContext."""

from __future__ import annotations

import json

from app.agent.agent import Agent
from app.agent.context import STALE_TOOL_RESULT_CHARS, ConversationContext
from app.agent.memory import SQLiteMemoryStore
from app.core.events import EventBus
from app.llm.base import Message, Role, ToolCall
from app.tools.registry import ToolRegistry
from tests.conftest import final_response


def _tool_turn(context: ConversationContext, n: int, result_chars: int, success: bool = True) -> None:
    context.add(Message(role=Role.USER, content=f"request {n}"))
    context.add(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id=f"call-{n}", name="read_screen_text", arguments={})],
        )
    )
    payload = (
        {"success": success, "data": {"text": "z" * result_chars}}
        if success
        else {"success": False, "error": "Denied: the user declined the confirmation."}
    )
    context.add(Message(role=Role.TOOL, content=json.dumps(payload), tool_call_id=f"call-{n}", name="read_screen_text"))
    context.add(Message(role=Role.ASSISTANT, content=f"answer {n}"))


def _assert_valid_pairs(messages: list[Message]) -> None:
    non_system = [m for m in messages if m.role != Role.SYSTEM]
    assert non_system[0].role == Role.USER
    open_calls: set[str] = set()
    for m in non_system:
        if m.role == Role.ASSISTANT:
            open_calls = {c.id for c in m.tool_calls}
        elif m.role == Role.TOOL:
            assert m.tool_call_id in open_calls, "tool result without its assistant tool call"


def test_trimming_drops_whole_turns_and_never_orphans_a_tool_result():
    context = ConversationContext(max_messages=9, max_tokens=100_000)
    context.set_system_prompt("sys")
    for n in range(5):
        _tool_turn(context, n, 10)
    messages = context.messages
    _assert_valid_pairs(messages)
    assert [m.content for m in messages if m.role == Role.USER] == ["request 3", "request 4"]


def test_large_results_of_finished_turns_are_elided_with_an_explicit_marker():
    context = ConversationContext(max_messages=100, max_tokens=100_000)
    _tool_turn(context, 1, 20_000)
    big_before = context.messages[2].content
    assert len(big_before) > 20_000  # the current turn keeps its full evidence
    context.add(Message(role=Role.USER, content="next question"))
    stub = json.loads(context.messages[2].content)
    assert stub["success"] is True and "call the tool again" in stub["elided"]
    assert "20,0" in stub["elided"] and len(context.messages[2].content) < 700
    _assert_valid_pairs(context.messages)


def test_small_results_such_as_a_declined_confirmation_are_never_elided():
    context = ConversationContext(max_messages=100, max_tokens=100_000)
    _tool_turn(context, 1, 0, success=False)
    before = context.messages[2].content
    assert len(before) < STALE_TOOL_RESULT_CHARS
    context.add(Message(role=Role.USER, content="next"))
    assert context.messages[2].content == before


def test_the_current_turn_is_never_dropped_only_its_older_large_results_shrink():
    context = ConversationContext(max_messages=100, max_tokens=3_000)
    context.add(Message(role=Role.USER, content="multi-step task"))
    for n in range(4):
        context.add(
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id=f"c{n}", name="inspect_ui", arguments={"n": n})],
            )
        )
        context.add(
            Message(
                role=Role.TOOL,
                content=json.dumps({"success": True, "data": "q" * 8_000}),
                tool_call_id=f"c{n}",
                name="inspect_ui",
            )
        )
    messages = context.messages
    assert messages[0].content == "multi-step task" and len(messages) == 9
    tool_results = [m.content for m in messages if m.role == Role.TOOL]
    assert all("call the tool again" in r for r in tool_results[:2])
    assert all(len(r) > 8_000 for r in tool_results[2:])  # the newest two stay complete
    _assert_valid_pairs(messages)


def test_leading_orphan_tool_results_are_dropped():
    context = ConversationContext(max_messages=100, max_tokens=100_000)
    context.add(Message(role=Role.TOOL, content="{}", tool_call_id="ghost", name="x"))
    context.add(Message(role=Role.USER, content="hi"))
    assert [m.role for m in context.messages] == [Role.USER]


def test_dropped_turns_leave_a_digest_and_clear_resets_it():
    context = ConversationContext(max_messages=3, max_tokens=100_000)
    context.add(Message(role=Role.USER, content="my name is Eris"))
    context.add(Message(role=Role.ASSISTANT, content="Noted, Eris."))
    context.add(Message(role=Role.USER, content="open notepad"))
    context.add(Message(role=Role.ASSISTANT, content="Opening Notepad."))
    digest = context.earlier_turns_digest()
    assert digest == "- User: my name is Eris -> You: Noted, Eris."
    context.clear()
    assert context.earlier_turns_digest() is None


async def test_agent_puts_the_digest_in_the_system_prompt(tmp_path, make_scripted_llm):
    llm = make_scripted_llm([final_response("Noted, Eris."), final_response("Opening."), final_response("Eris.")])
    agent = Agent(
        llm=llm,
        registry=ToolRegistry(),
        memory=SQLiteMemoryStore(tmp_path / "m.db"),
        events=EventBus(),
        max_context_messages=2,
    )
    await agent.handle_user_message("my name is Eris")
    await agent.handle_user_message("open notepad")
    await agent.handle_user_message("what's my name?")
    system = llm.calls[2][0]
    assert system.role == Role.SYSTEM and "my name is Eris -> You: Noted, Eris." in system.content
    assert [m.content for m in llm.calls[2][1:]] == ["what's my name?"]
