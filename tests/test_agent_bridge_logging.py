"""Tests for AgentBridge's assistant-message dispatch logging
(app/ui/agent_bridge.py:_on_event), added to diagnose whether a single
ASSISTANT_MESSAGE event from the Agent's EventBus is ever dispatched more
than once by the bridge (vs. the LLM itself producing repeated text).

These call `AgentBridge._on_event` directly against a lightweight
duck-typed stand-in rather than constructing a real AgentBridge, since a
real one spins up an LLM-provider-backed Agent, a background asyncio loop,
and the voice pipeline -- none of which this logging logic touches.
"""

from __future__ import annotations

import hashlib
import logging
from unittest.mock import Mock

from app.core.events import AgentEvent, EventType
from app.ui.agent_bridge import AgentBridge


class _FakeBridge:
    """Exposes only what AgentBridge._on_event's USER_MESSAGE/
    ASSISTANT_MESSAGE branches touch."""

    def __init__(self) -> None:
        self._assistant_message_seq = 0
        self._user_message_seq = 0
        self._voice_replies_enabled = False
        self._session_active = False
        self.assistant_message = Mock()
        self.user_message = Mock()


def test_assistant_message_dispatch_is_logged_with_fingerprint(caplog):
    fake = _FakeBridge()
    text = "Acknowledged, Sir. Your name is Alex."

    with caplog.at_level(logging.INFO, logger="garvis.ui.agent_bridge"):
        AgentBridge._on_event(fake, AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": text}))

    records = [r for r in caplog.records if r.getMessage() == "agent_bridge.assistant_message_dispatch"]
    assert len(records) == 1
    fields = records[0].extra_fields
    assert fields["seq"] == 1
    assert fields["content_length"] == len(text)
    assert fields["content_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    fake.assistant_message.emit.assert_called_once_with(text)


def test_repeated_dispatch_of_identical_content_is_distinguishable_in_logs(caplog):
    """This is exactly the log signal that lets a reviewer tell 'the bridge
    dispatched the same assistant message twice' (two records, same
    content_hash, consecutive seq) apart from 'the LLM produced repeated
    text in one response' (a single dispatch whose content_hash covers
    already-duplicated text)."""
    fake = _FakeBridge()
    text = "Your name is Alex, Sir."
    event = AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": text})

    with caplog.at_level(logging.INFO, logger="garvis.ui.agent_bridge"):
        AgentBridge._on_event(fake, event)
        AgentBridge._on_event(fake, event)

    records = [r for r in caplog.records if r.getMessage() == "agent_bridge.assistant_message_dispatch"]
    assert [r.extra_fields["seq"] for r in records] == [1, 2]
    assert records[0].extra_fields["content_hash"] == records[1].extra_fields["content_hash"]
    assert fake.assistant_message.emit.call_count == 2


# ------------------------------------------------------- USER_MESSAGE dispatch
def test_user_message_dispatch_is_logged_with_fingerprint(caplog):
    """Companion to the ASSISTANT_MESSAGE logging above: Agent.handle_user_message()
    publishes exactly one USER_MESSAGE per invocation regardless of the
    caller (push-to-talk, typed text, or ConversationSession), so this is a
    direct, source-agnostic count of Agent invocations -- the key
    diagnostic for confirming or ruling out the reported self-trigger
    voice loop (one real utterance must produce exactly one seq bump)."""
    fake = _FakeBridge()
    text = "what is two plus two"

    with caplog.at_level(logging.INFO, logger="garvis.ui.agent_bridge"):
        AgentBridge._on_event(fake, AgentEvent(type=EventType.USER_MESSAGE, payload={"text": text}))

    records = [r for r in caplog.records if r.getMessage() == "agent_bridge.user_message_dispatch"]
    assert len(records) == 1
    fields = records[0].extra_fields
    assert fields["seq"] == 1
    assert fields["content_length"] == len(text)
    assert fields["content_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    fake.user_message.emit.assert_called_once_with(text)


def test_repeated_user_message_dispatch_is_distinguishable_in_logs(caplog):
    fake = _FakeBridge()
    event = AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "open whatsapp web"})

    with caplog.at_level(logging.INFO, logger="garvis.ui.agent_bridge"):
        AgentBridge._on_event(fake, event)
        AgentBridge._on_event(fake, event)

    records = [r for r in caplog.records if r.getMessage() == "agent_bridge.user_message_dispatch"]
    assert [r.extra_fields["seq"] for r in records] == [1, 2]
    assert records[0].extra_fields["content_hash"] == records[1].extra_fields["content_hash"]
