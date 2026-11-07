import asyncio

import pytest
from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import LLMAuthenticationError, LLMRateLimitError, LLMUnavailableError
from app.security.confirmations import ConfirmationManager
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from tests.conftest import final_response, tool_call_response


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_agent_answers_directly_without_tools(make_scripted_llm, memory_store):
    llm = make_scripted_llm([final_response("Hello there!")])
    registry = ToolRegistry()
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("hi")
    assert reply == "Hello there!"


@pytest.mark.asyncio
async def test_one_user_message_without_tools_produces_exactly_one_llm_request(make_scripted_llm, memory_store):
    """Regression test (Groq 429 / uncontrolled-request investigation): a
    single handle_user_message() call for a plain conversational turn (no
    tool use) must make exactly one underlying LLM request -- never more,
    regardless of caller (push-to-talk, typed text, or ConversationSession
    all call this exact method unchanged)."""
    llm = make_scripted_llm([final_response("Hello there!")])
    registry = ToolRegistry()
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    await agent.handle_user_message("hi")

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agent_executes_low_risk_tool_then_answers(make_scripted_llm, memory_store):
    llm = make_scripted_llm(
        [
            tool_call_response("calculator", {"expression": "2 + 2"}),
            final_response("2 + 2 is 4."),
        ]
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("what is 2 + 2?")
    assert reply == "2 + 2 is 4."
    # second LLM call should include the tool result message
    second_call_messages = llm.calls[1]
    assert any(m.role == "tool" for m in second_call_messages)


@pytest.mark.asyncio
async def test_agent_preserves_earlier_turn_across_sequential_messages(make_scripted_llm, memory_store):
    """Regression test: an earlier user/assistant exchange (e.g. the user
    stating their name) must still reach the LLM on a later turn. Agent
    builds its system prompt via the real build_system_prompt() (not a
    stub), so this exercises the actual system-prompt size against
    ConversationContext's token-aware trimming."""
    llm = make_scripted_llm(
        [
            final_response("Understood, Eris. This is Garvis, not Jarvis. How may I assist you, Sir?"),
            final_response("Your name is Eris, Sir."),
        ]
    )
    registry = ToolRegistry()
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    await agent.handle_user_message("my name is Eris")
    reply2 = await agent.handle_user_message("what is my name?")

    assert reply2 == "Your name is Eris, Sir."
    second_call_messages = llm.calls[1]
    assert any("Eris" in m.content for m in second_call_messages)


@pytest.mark.asyncio
async def test_agent_reports_unknown_tool_gracefully(make_scripted_llm, memory_store):
    llm = make_scripted_llm(
        [
            tool_call_response("nonexistent_tool", {}),
            final_response("Sorry, I couldn't do that."),
        ]
    )
    registry = ToolRegistry()
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("do the impossible")
    assert reply == "Sorry, I couldn't do that."


class _Args(BaseModel):
    pass


class _HighRiskEchoTool(Tool):
    name = "high_risk_echo"
    description = "high risk test tool"
    risk_level = RiskLevel.HIGH
    input_model = _Args

    async def execute(self, args: _Args) -> ToolResult:
        return ToolResult.ok("did the risky thing")


@pytest.mark.asyncio
async def test_agent_waits_for_confirmation_before_high_risk_tool(make_scripted_llm, memory_store):
    llm = make_scripted_llm(
        [
            tool_call_response("high_risk_echo", {}),
            final_response("Done with the risky thing."),
        ]
    )
    registry = ToolRegistry()
    registry.register(_HighRiskEchoTool())
    events = EventBus()
    confirmation_ids: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.CONFIRMATION_REQUESTED:
            confirmation_ids.append(event.payload["id"])

    events.subscribe(capture)
    agent = Agent(
        llm=llm,
        registry=registry,
        memory=memory_store,
        events=events,
        confirmations=ConfirmationManager(default_timeout_seconds=2),
    )

    async def approve_when_ready():
        while not confirmation_ids:
            await asyncio.sleep(0.01)
        agent.confirmations.resolve(confirmation_ids[0], True)

    approver = asyncio.create_task(approve_when_ready())
    reply = await agent.handle_user_message("do the risky thing")
    await approver
    assert reply == "Done with the risky thing."


@pytest.mark.asyncio
async def test_agent_denied_confirmation_reports_failure_not_success(make_scripted_llm, memory_store):
    llm = make_scripted_llm(
        [
            tool_call_response("high_risk_echo", {}),
            final_response("I wasn't able to do that since you declined."),
        ]
    )
    registry = ToolRegistry()
    registry.register(_HighRiskEchoTool())
    events = EventBus()
    confirmation_ids: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.CONFIRMATION_REQUESTED:
            confirmation_ids.append(event.payload["id"])

    events.subscribe(capture)
    agent = Agent(
        llm=llm,
        registry=registry,
        memory=memory_store,
        events=events,
        confirmations=ConfirmationManager(default_timeout_seconds=2),
    )

    async def deny_when_ready():
        while not confirmation_ids:
            await asyncio.sleep(0.01)
        agent.confirmations.resolve(confirmation_ids[0], False)

    denier = asyncio.create_task(deny_when_ready())
    reply = await agent.handle_user_message("do the risky thing")
    await denier
    assert "declined" in reply.lower()
    # ensure the tool result fed back to the LLM reflects failure, not success
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert any("denied" in m.content.lower() or "false" in m.content.lower() for m in tool_messages)


class _ToolThenErrorLLM:
    """Scripted LLM: succeeds with a tool call on the first chat() call (so
    the tool actually runs and takes effect), then raises `error` on the
    follow-up chat() call that would otherwise compose the final reply --
    reproducing the reported bug where a tool (e.g. open_website opening
    WhatsApp Web) succeeds but the wrap-up LLM call hits a rate limit."""

    def __init__(self, tool_response, error: Exception) -> None:
        self._tool_response = tool_response
        self._error = error
        self.calls: list[list] = []

    async def chat(self, messages, tools=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return self._tool_response
        raise self._error

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_returns_specific_message_when_rate_limited_after_successful_tool_call(
    memory_store,
):
    """Regression test for the WhatsApp bug: open_website (via a stand-in
    calculator tool call here) succeeds, but the follow-up LLM call that
    composes the final confirmation hits Groq's rate limit. LLMRateLimitError
    must surface its own specific message, not the generic "Something went
    wrong" fallback (see app/agent/agent.py handle_user_message)."""
    rate_limit_message = "Groq rate limit exceeded. Please try again shortly."
    llm = _ToolThenErrorLLM(
        tool_call_response("calculator", {"expression": "2 + 2"}),
        LLMRateLimitError(rate_limit_message),
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    events = EventBus()
    tool_successes: list[bool] = []
    errors: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.TOOL_FINISHED:
            tool_successes.append(event.payload["success"])
        elif event.type == EventType.ERROR:
            errors.append(event.payload["message"])

    events.subscribe(capture)
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=events)

    reply = await agent.handle_user_message("what is 2 + 2?")

    assert tool_successes == [True]  # the tool call genuinely succeeded first
    assert reply == rate_limit_message
    assert errors == [rate_limit_message]
    assert "something went wrong" not in reply.lower()


@pytest.mark.asyncio
async def test_agent_returns_specific_message_when_authentication_fails_after_successful_tool_call(
    memory_store,
):
    """Same shape as the rate-limit regression test, for LLMAuthenticationError."""
    auth_message = "Groq rejected the request credentials. Check GARVIS_GROQ_API_KEY."
    llm = _ToolThenErrorLLM(
        tool_call_response("calculator", {"expression": "2 + 2"}),
        LLMAuthenticationError(auth_message),
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    events = EventBus()
    tool_successes: list[bool] = []
    errors: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.TOOL_FINISHED:
            tool_successes.append(event.payload["success"])
        elif event.type == EventType.ERROR:
            errors.append(event.payload["message"])

    events.subscribe(capture)
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=events)

    reply = await agent.handle_user_message("what is 2 + 2?")

    assert tool_successes == [True]
    assert reply == auth_message
    assert errors == [auth_message]
    assert "something went wrong" not in reply.lower()


@pytest.mark.asyncio
async def test_agent_llm_unavailable_behavior_unchanged_after_successful_tool_call(memory_store):
    """LLMUnavailableError must keep behaving exactly as before this fix:
    its specific message is returned, not the generic fallback."""
    unavailable_message = "Garvis can't reach OmniRoute. Is it running?"
    llm = _ToolThenErrorLLM(
        tool_call_response("calculator", {"expression": "2 + 2"}),
        LLMUnavailableError(unavailable_message),
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    events = EventBus()
    tool_successes: list[bool] = []
    errors: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.TOOL_FINISHED:
            tool_successes.append(event.payload["success"])
        elif event.type == EventType.ERROR:
            errors.append(event.payload["message"])

    events.subscribe(capture)
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=events)

    reply = await agent.handle_user_message("what is 2 + 2?")

    assert tool_successes == [True]
    assert reply == unavailable_message
    assert errors == [unavailable_message]
    assert "something went wrong" not in reply.lower()


class _ExplodingLLM:
    """Raises an unexpected (non-LLMError) exception from chat(), simulating
    a bug somewhere in the loop rather than a known LLM-backend problem."""

    async def chat(self, messages, tools=None):
        raise RuntimeError("boom: some internal bug, not a connectivity issue")

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_survives_unexpected_error_without_crashing(memory_store):
    """Regression test: previously only LLMUnavailableError was caught in
    handle_user_message, so any other unexpected exception propagated out of
    the coroutine uncaught. Since AgentBridge fires this coroutine without
    awaiting/retrieving its result, that silently left the UI stuck in
    "Thinking..." forever (Send/Stop/mic buttons never re-enabled) -- see
    app/agent/agent.py handle_user_message."""
    registry = ToolRegistry()
    events = EventBus()
    statuses: list[str] = []
    errors: list[str] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.STATUS:
            statuses.append(event.payload["status"])
        elif event.type == EventType.ERROR:
            errors.append(event.payload["message"])

    events.subscribe(capture)
    agent = Agent(llm=_ExplodingLLM(), registry=registry, memory=memory_store, events=events)

    reply = await agent.handle_user_message("hello")

    # Never raises, never leaks the raw exception text/internals to the user.
    assert "boom" not in reply
    assert reply  # a safe, non-empty message was returned
    assert errors and "boom" not in errors[0]
    # Status must return to Idle so the UI never gets stuck disabled.
    assert statuses[-1] == "Idle."

    # Session state remains usable: a normal turn afterwards still works.
    from tests.conftest import final_response

    class _OkLLM:
        async def chat(self, messages, tools=None):
            return final_response("all good now")

        async def health_check(self) -> bool:
            return True

    agent._llm = _OkLLM()
    reply2 = await agent.handle_user_message("hi again")
    assert reply2 == "all good now"
