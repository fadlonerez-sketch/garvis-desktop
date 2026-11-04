"""Regression tests for the per-session context boundary.

AgentBridge keeps ONE Agent (and therefore one ConversationContext) for the
whole app lifetime, and every voice session reuses it. Before the fix,
nothing cleared that history between sessions, so a new session re-sent the
previous session's turns to the LLM on every request. Now
AgentBridge._start_session_async calls Agent.reset_conversation() at each
new session. That drops the user/assistant/tool history and keeps the
system prompt, which includes long-term memory.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.agent.policies import build_system_prompt
from app.core.config import Settings
from app.core.events import EventBus
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolSpec
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from app.ui.agent_bridge import AgentBridge
from tests.conftest import final_response, tool_call_response


class _RecordingLLM(LLMProvider):
    """Answers every call with a fixed reply and records exactly what was sent."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        self.calls.append((messages, tools))
        return self._responses.pop(0) if self._responses else final_response("ok")

    async def health_check(self) -> bool:
        return True

    @property
    def last_messages(self) -> list[Message]:
        return self.calls[-1][0]


def _agent(tmp_path, llm: LLMProvider) -> Agent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return Agent(llm=llm, registry=registry, memory=SQLiteMemoryStore(tmp_path / "m.db"), events=EventBus())


def _texts(messages: list[Message]) -> list[str]:
    return [m.content for m in messages if m.role != Role.SYSTEM]


@pytest.mark.asyncio
async def test_messages_accumulate_inside_one_session(tmp_path):
    llm = _RecordingLLM()
    agent = _agent(tmp_path, llm)

    await agent.handle_user_message("my name is Alex")
    await agent.handle_user_message("what is my name?")
    await agent.handle_user_message("and again?")

    assert _texts(llm.last_messages) == ["my name is Alex", "ok", "what is my name?", "ok", "and again?"]


@pytest.mark.asyncio
async def test_new_session_does_not_contain_previous_session_messages(tmp_path):
    llm = _RecordingLLM([
        tool_call_response("calculator", {"expression": "2 + 2"}),
        final_response("4"),
        final_response("hello again"),
    ])
    agent = _agent(tmp_path, llm)
    await agent.handle_user_message("secret from session one: 2 + 2")

    agent.reset_conversation()
    await agent.handle_user_message("hi")

    sent = llm.last_messages
    assert _texts(sent) == ["hi"]
    assert not any(m.role == Role.TOOL or m.tool_calls for m in sent)
    assert not any("session one" in (m.content or "") for m in sent)


@pytest.mark.asyncio
async def test_system_prompt_and_long_term_memory_survive_the_reset(tmp_path):
    llm = _RecordingLLM()
    agent = _agent(tmp_path, llm)
    await agent._memory.remember("The user's name is Alex")
    await agent.handle_user_message("first session")

    agent.reset_conversation()
    await agent.handle_user_message("second session")

    sent = llm.last_messages
    assert sent[0].role == Role.SYSTEM
    assert sent[0].content == build_system_prompt(await agent._memory.recall(limit=20))
    assert "The user's name is Alex" in sent[0].content
    assert [m.role for m in sent].count(Role.SYSTEM) == 1


@pytest.mark.asyncio
async def test_reset_does_not_touch_provider_or_tool_specs(tmp_path):
    llm = _RecordingLLM()
    agent = _agent(tmp_path, llm)
    await agent.handle_user_message("before")
    tools_before = llm.calls[-1][1]

    agent.reset_conversation()
    assert len(llm.calls) == 1  # the reset itself never calls the LLM
    assert agent._llm is llm

    await agent.handle_user_message("after")
    assert llm.calls[-1][1] == tools_before


@pytest.mark.asyncio
async def test_repeated_sessions_do_not_grow_the_context(tmp_path):
    llm = _RecordingLLM()
    agent = _agent(tmp_path, llm)
    first_request_sizes = []
    for session in range(5):
        agent.reset_conversation()
        await agent.handle_user_message(f"session {session} turn 1")
        first_request_sizes.append(len(llm.last_messages))
        await agent.handle_user_message(f"session {session} turn 2")
        assert _texts(llm.last_messages) == [f"session {session} turn 1", "ok", f"session {session} turn 2"]

    assert first_request_sizes == [2] * 5  # system + this session's first user message


@pytest.mark.asyncio
async def test_reset_on_an_empty_context_is_safe(tmp_path):
    llm = _RecordingLLM()
    agent = _agent(tmp_path, llm)
    agent.reset_conversation()
    agent.reset_conversation()
    await agent.handle_user_message("hi")
    assert [m.role for m in llm.last_messages] == [Role.SYSTEM, Role.USER]


class _FakeSession:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


def _bridge(tmp_path) -> AgentBridge:
    return AgentBridge(settings=Settings(
        _env_file=None, memory_db_path=tmp_path / "b.db", llm_provider="tiered", omniroute_api_key="fake-test-key",
    ))


def _run_on_bridge_loop(bridge: AgentBridge, coro):
    return asyncio.run_coroutine_threadsafe(coro, bridge._loop).result(timeout=5)


def test_bridge_starts_each_session_with_a_fresh_context(tmp_path):
    """The real AgentBridge seam: a new session clears the previous one's
    history before the session loop starts. A fake session stands in for
    ConversationSession so no microphone is opened."""
    bridge = _bridge(tmp_path)
    try:
        context = bridge._agent._context
        context.set_system_prompt("system")
        context.add(Message(role=Role.USER, content="from the previous session"))
        context.add(Message(role=Role.ASSISTANT, content="previous reply"))

        session = _FakeSession()
        bridge._session = session
        _run_on_bridge_loop(bridge, bridge._start_session_async(session, None))

        assert session.started
        assert [(m.role, m.content) for m in context.messages] == [(Role.SYSTEM, "system")]
    finally:
        bridge._session = None
        bridge.shutdown()


def test_bridge_does_not_reset_for_a_superseded_session(tmp_path):
    """If end_session() already replaced the session, the stale start must
    neither start it nor wipe the conversation."""
    bridge = _bridge(tmp_path)
    try:
        context = bridge._agent._context
        context.add(Message(role=Role.USER, content="keep me"))

        stale = _FakeSession()
        bridge._session = None  # superseded
        _run_on_bridge_loop(bridge, bridge._start_session_async(stale, None))

        assert not stale.started
        assert [m.content for m in context.messages] == ["keep me"]
    finally:
        bridge.shutdown()
