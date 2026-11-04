"""Phase 8 integration tests: the Agent driving the *real* ToolRegistry,
PermissionChecker, and ConfirmationManager together (not each in isolation),
including a browser tool. Existing unit tests already cover each of these
components on its own (test_agent_loop.py, test_browser_tools.py,
test_permissions.py, test_confirmations.py); this file verifies they still
behave correctly when wired together the way `app/agent/build.py` wires
them for real, for Phase 8's "Agent <-> Tool integration" and "Browser
integration" goals.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import BrowserError
from app.security.confirmations import ConfirmationManager
from app.tools.browser import SubmitFormTool, NavigateTool, ReadPageTool
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry

from tests.conftest import final_response, tool_call_response


class _FakeBrowserManager:
    """Same shape as the real BrowserManager (see test_browser_tools.py)."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.current_url = None
        self.navigate_calls: list[str] = []
        self.clicked: list[str] = []

    async def navigate(self, url: str) -> str:
        if self.fail:
            raise BrowserError(f"Could not navigate to '{url}': DNS lookup failed")
        self.navigate_calls.append(url)
        self.current_url = url
        return url

    async def extract_text(self) -> str:
        return "Example Domain. This domain is for illustrative use."

    async def click(self, selector: str) -> None:
        self.clicked.append(selector)


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


def _build_real_registry(browser_manager) -> ToolRegistry:
    """Mirrors app/agent/build.py's build_registry wiring for the subset of
    tools relevant here, using the real Tool/ToolRegistry classes end to end."""
    registry = ToolRegistry()
    for tool in (
        CalculatorTool(), NavigateTool(browser_manager), ReadPageTool(browser_manager), SubmitFormTool(browser_manager)
    ):
        registry.register(tool)
    return registry


@pytest.mark.asyncio
async def test_agent_discovers_and_invokes_real_registered_tool(memory_store, make_scripted_llm):
    """The LLM is only ever given tool specs pulled live from the registry --
    prove a tool registered there is both offered to the LLM and invokable."""
    llm = make_scripted_llm(
        [
            tool_call_response("calculator", {"expression": "6 * 7"}),
            final_response("6 * 7 is 42."),
        ]
    )
    registry = _build_real_registry(_FakeBrowserManager())
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("what is 6 * 7?")

    assert reply == "6 * 7 is 42."
    offered_tool_names = {spec.name for spec in registry.tool_specs()}
    assert "calculator" in offered_tool_names
    assert "browser_navigate" in offered_tool_names  # discovery covers all registered tools


@pytest.mark.asyncio
async def test_browser_tool_flows_through_real_permission_and_confirmation_system(
    memory_store, make_scripted_llm
):
    """browser_submit_form is HIGH risk: the agent must pause for a real
    confirmation (not bypass it), and only then call into the (fake)
    BrowserManager, before returning a final answer built on the tool
    result -- exercising the full discover -> select -> permission-check ->
    confirm -> invoke -> result path with real production components.
    (browser_navigate used to be the example here; routine navigation no
    longer asks -- see test_routine_navigation_runs_without_confirmation.)"""
    llm = make_scripted_llm(
        [
            tool_call_response("browser_submit_form", {"selector": "button[type=submit]"}),
            final_response("Submitted."),
        ]
    )
    browser_manager = _FakeBrowserManager()
    registry = _build_real_registry(browser_manager)
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

    async def submit_after_approval_only():
        while not confirmation_ids:
            await asyncio.sleep(0.01)
        assert browser_manager.clicked == []  # nothing ran before the answer
        agent.confirmations.resolve(confirmation_ids[0], True)

    approver = asyncio.create_task(submit_after_approval_only())
    reply = await agent.handle_user_message("submit the form")
    await approver

    assert reply == "Submitted."
    assert browser_manager.clicked == ["button[type=submit]"]  # tool actually ran, once approved


@pytest.mark.asyncio
async def test_routine_navigation_runs_without_confirmation(memory_store, make_scripted_llm):
    llm = make_scripted_llm(
        [
            tool_call_response("browser_navigate", {"url": "https://example.com"}),
            final_response("Opened example.com."),
        ]
    )
    browser_manager = _FakeBrowserManager()
    events = EventBus()
    requested: list[AgentEvent] = []
    events.subscribe(lambda e: requested.append(e) if e.type == EventType.CONFIRMATION_REQUESTED else None)
    agent = Agent(llm=llm, registry=_build_real_registry(browser_manager), memory=memory_store, events=events)

    assert await agent.handle_user_message("go to example.com") == "Opened example.com."
    assert browser_manager.navigate_calls == ["https://example.com"]
    assert requested == []


@pytest.mark.asyncio
async def test_browser_failure_becomes_safe_tool_result_not_a_crash(memory_store, make_scripted_llm):
    """A real BrowserError (navigation/DNS/timeout failure) must reach the LLM
    as a structured ToolResult.fail(...), never as an unhandled exception."""
    llm = make_scripted_llm(
        [
            tool_call_response("browser_navigate", {"url": "https://nonexistent.example"}),
            final_response("I couldn't reach that site."),
        ]
    )
    browser_manager = _FakeBrowserManager(fail=True)
    registry = _build_real_registry(browser_manager)
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

    reply = await agent.handle_user_message("go to a broken site")

    assert reply == "I couldn't reach that site."
    assert confirmation_ids == []  # routine navigation: no confirmation
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert any("could not navigate" in m.content.lower() for m in tool_messages)


@pytest.mark.asyncio
async def test_unknown_tool_call_does_not_break_the_session_for_the_next_turn(
    memory_store, make_scripted_llm
):
    """Tool-not-found is handled per-call; the conversation/session must
    remain usable for a subsequent, unrelated turn afterwards."""
    llm = make_scripted_llm(
        [
            tool_call_response("does_not_exist", {}),
            final_response("Sorry, I can't do that."),
            final_response("2 + 2 is 4."),
        ]
    )
    registry = _build_real_registry(_FakeBrowserManager())
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    first_reply = await agent.handle_user_message("do the impossible")
    assert "can't" in first_reply.lower()

    second_reply = await agent.handle_user_message("what is 2 + 2?")
    assert second_reply == "2 + 2 is 4."
