"""Agent additions for multiple interfaces: serialized turns, the optional
`source` tag, CONFIRMATION_RESOLVED, and a reset that never lands mid-turn."""

from __future__ import annotations

import asyncio

from app.core.events import AgentEvent, EventBus, EventType
from app.llm.base import Role
from tests.conftest import final_response, tool_call_response
from tests.mobile_fakes import GatedLLM, HighRiskTool, make_agent, wait_until


async def test_concurrent_turns_are_serialized_and_context_stays_ordered(tmp_path):
    llm = GatedLLM()
    agent = make_agent(tmp_path, llm)

    first = asyncio.create_task(agent.handle_user_message("first"))
    await wait_until(lambda: llm.in_flight == 1)
    second = asyncio.create_task(agent.handle_user_message("second"))
    await asyncio.sleep(0.05)
    # The second turn hasn't touched the context while the first is in flight.
    assert agent.is_busy
    assert [m.content for m in agent._context.messages if m.role == Role.USER] == ["first"]

    llm.release()
    assert await first == "reply to: first"
    assert await second == "reply to: second"
    assert llm.max_in_flight == 1
    roles = [(m.role, m.content) for m in agent._context.messages if m.role != Role.SYSTEM]
    assert roles == [
        (Role.USER, "first"),
        (Role.ASSISTANT, "reply to: first"),
        (Role.USER, "second"),
        (Role.ASSISTANT, "reply to: second"),
    ]
    assert not agent.is_busy


async def test_source_tag_is_echoed_only_when_given(tmp_path):
    events = EventBus()
    seen: list[AgentEvent] = []
    events.subscribe(seen.append)
    agent = make_agent(tmp_path, GatedLLM(gated=False), events=events)

    await agent.handle_user_message("from the desk")
    await agent.handle_user_message("from the phone", source="mobile")

    messages = [e.payload for e in seen if e.type in (EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE)]
    assert messages[0] == {"text": "from the desk"}
    assert messages[1] == {"text": "reply to: from the desk"}
    assert messages[2] == {"text": "from the phone", "source": "mobile"}
    assert messages[3] == {"text": "reply to: from the phone", "source": "mobile"}


async def _run_confirmation(tmp_path, decide):
    events = EventBus()
    seen: list[AgentEvent] = []
    events.subscribe(seen.append)
    tool = HighRiskTool()
    llm = GatedLLM([tool_call_response("high_risk_echo", {}), final_response("ok")], gated=False)
    agent = make_agent(tmp_path, llm, events=events, tools=[tool], confirmation_timeout=0.3)
    turn = asyncio.create_task(agent.handle_user_message("do it"))
    await wait_until(lambda: any(e.type == EventType.CONFIRMATION_REQUESTED for e in seen))
    request_id = next(e.payload["id"] for e in seen if e.type == EventType.CONFIRMATION_REQUESTED)
    decide(agent, request_id)
    await turn
    resolved = [e.payload for e in seen if e.type == EventType.CONFIRMATION_RESOLVED]
    return tool, request_id, resolved


async def test_confirmation_resolved_is_published_on_approval(tmp_path):
    tool, request_id, resolved = await _run_confirmation(
        tmp_path, lambda agent, rid: agent.confirmations.resolve(rid, True)
    )
    assert resolved == [{"id": request_id, "outcome": "approved"}]
    assert tool.executions == 1


async def test_confirmation_resolved_is_published_on_denial(tmp_path):
    tool, request_id, resolved = await _run_confirmation(
        tmp_path, lambda agent, rid: agent.confirmations.resolve(rid, False)
    )
    assert resolved == [{"id": request_id, "outcome": "denied"}]
    assert tool.executions == 0


async def test_confirmation_resolved_is_published_on_timeout(tmp_path):
    tool, request_id, resolved = await _run_confirmation(tmp_path, lambda agent, rid: None)
    assert resolved == [{"id": request_id, "outcome": "timeout"}]
    assert tool.executions == 0


async def test_reset_between_turns_waits_for_the_running_turn(tmp_path):
    llm = GatedLLM()
    agent = make_agent(tmp_path, llm)
    turn = asyncio.create_task(agent.handle_user_message("keep me whole"))
    await wait_until(lambda: llm.in_flight == 1)

    reset = asyncio.create_task(agent.reset_conversation_between_turns())
    await asyncio.sleep(0.05)
    assert not reset.done()  # waiting for the turn, not cutting into it

    llm.release()
    await turn
    await reset
    assert all(m.role == Role.SYSTEM for m in agent._context.messages)
