"""Phase 9 integration tests: the Agent driving real ToolRegistry +
PermissionChecker + ConfirmationManager together with the Gmail tools,
mirroring tests/test_agent_tool_integration.py's browser-tool coverage.
Verifies gmail_send only ever executes after explicit confirmation, is
never sent on denial, and that Gmail API/auth failures reach the LLM as
safe ToolResult data rather than crashing the agent loop.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import GmailAuthError, GmailError
from app.security.confirmations import ConfirmationManager
from app.tools.gmail import ListRecentEmailsTool, SearchEmailsTool, SendEmailTool
from app.tools.registry import ToolRegistry
from tests.conftest import final_response, tool_call_response


class _FakeGmailManager:
    def __init__(self, fail: bool = False, auth_fail: bool = False):
        self.fail = fail
        self.auth_fail = auth_fail
        self.sent: list[tuple[str, str, str]] = []

    def _maybe_raise(self):
        if self.auth_fail:
            raise GmailAuthError("Garvis's Gmail authorization has expired or was revoked.")
        if self.fail:
            raise GmailError("Gmail send failed: quota exceeded")

    async def search(self, query: str, max_results: int) -> list[dict]:
        self._maybe_raise()
        return [{"id": "m1", "subject": "Hi", "from": "a@b.com", "date": "", "snippet": "hello"}]

    async def send(self, to: str, subject: str, body: str) -> str:
        self._maybe_raise()
        self.sent.append((to, subject, body))
        return "sent-123"


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


def _build_registry(gmail_manager) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        SearchEmailsTool(gmail_manager),
        ListRecentEmailsTool(gmail_manager),
        SendEmailTool(gmail_manager),
    ):
        registry.register(tool)
    return registry


@pytest.mark.asyncio
async def test_gmail_search_is_discoverable_and_invokable_without_confirmation(
    memory_store, make_scripted_llm
):
    llm = make_scripted_llm(
        [
            tool_call_response("gmail_search", {"query": "from:alice"}),
            final_response("You have one email from Alice."),
        ]
    )
    registry = _build_registry(_FakeGmailManager())
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("do I have any email from alice?")

    assert reply == "You have one email from Alice."
    offered = {spec.name for spec in registry.tool_specs()}
    assert {"gmail_search", "gmail_list_recent", "gmail_send"}.issubset(offered)


@pytest.mark.asyncio
async def test_send_email_waits_for_confirmation_then_actually_sends(memory_store, make_scripted_llm):
    llm = make_scripted_llm(
        [
            tool_call_response(
                "gmail_send", {"to": "bob@example.com", "subject": "Hi", "body": "hello"}
            ),
            final_response("I've sent the email to Bob."),
        ]
    )
    gmail_manager = _FakeGmailManager()
    registry = _build_registry(gmail_manager)
    events = EventBus()
    confirmation_ids: list[str] = []
    confirmation_payloads: list[dict] = []

    async def capture(event: AgentEvent) -> None:
        if event.type == EventType.CONFIRMATION_REQUESTED:
            confirmation_ids.append(event.payload["id"])
            confirmation_payloads.append(event.payload)

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
    reply = await agent.handle_user_message("send bob an email saying hello")
    await approver

    assert reply == "I've sent the email to Bob."
    assert gmail_manager.sent == [("bob@example.com", "Hi", "hello")]  # only ran after approval
    assert confirmation_payloads[0]["tool_name"] == "gmail_send"


@pytest.mark.asyncio
async def test_send_email_never_sent_when_confirmation_denied(memory_store, make_scripted_llm):
    """Critical safety property: denying the confirmation must mean the
    email is genuinely never sent, not merely reported as failed."""
    llm = make_scripted_llm(
        [
            tool_call_response(
                "gmail_send", {"to": "bob@example.com", "subject": "Hi", "body": "hello"}
            ),
            final_response("I did not send the email since you declined."),
        ]
    )
    gmail_manager = _FakeGmailManager()
    registry = _build_registry(gmail_manager)
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
    reply = await agent.handle_user_message("send bob an email saying hello")
    await denier

    assert "declined" in reply.lower()
    assert gmail_manager.sent == []  # the tool's send() was never called
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert any("denied" in m.content.lower() or "false" in m.content.lower() for m in tool_messages)


@pytest.mark.asyncio
async def test_gmail_api_failure_reaches_llm_as_safe_result_not_a_crash(memory_store, make_scripted_llm):
    llm = make_scripted_llm(
        [
            tool_call_response("gmail_search", {"query": "from:alice"}),
            final_response("I couldn't search your email right now."),
        ]
    )
    registry = _build_registry(_FakeGmailManager(fail=True))
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("search my email from alice")

    assert reply == "I couldn't search your email right now."
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert any("gmail send failed" in m.content.lower() or "quota" in m.content.lower() for m in tool_messages)


@pytest.mark.asyncio
async def test_gmail_auth_failure_reaches_llm_as_safe_result_not_a_crash(memory_store, make_scripted_llm):
    """Expired/revoked credentials must produce a clean tool failure the
    agent can relay to the user, never an unhandled exception."""
    llm = make_scripted_llm(
        [
            tool_call_response("gmail_search", {"query": "from:alice"}),
            final_response("I need you to re-authenticate Gmail before I can search it."),
        ]
    )
    registry = _build_registry(_FakeGmailManager(auth_fail=True))
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    reply = await agent.handle_user_message("search my email from alice")

    assert "re-authenticate" in reply.lower()
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert any("expired or was revoked" in m.content.lower() for m in tool_messages)
