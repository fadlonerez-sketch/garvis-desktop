import hashlib
import logging

import httpx
import pytest

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.llm.base import Message, Role
from app.llm.groq import GroqProvider


def test_constructor_rejects_missing_api_key():
    with pytest.raises(ConfigurationError):
        GroqProvider(api_key="", model="test-model")


@pytest.mark.asyncio
async def test_chat_parses_plain_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="hello")])
    assert response.message.content == "Hi!"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_parses_tool_calls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "1+1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="what is 1+1?")])
    assert response.finish_reason == "tool_calls"
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].id == "call_1"
    assert response.message.tool_calls[0].name == "calculator"
    assert response.message.tool_calls[0].arguments == {"expression": "1+1"}


@pytest.mark.asyncio
async def test_chat_parses_multiple_tool_calls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "1+1"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "datetime",
                                        "arguments": "{}",
                                    },
                                },
                            ],
                        }
                    }
                ]
            },
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="do two things")])
    assert response.finish_reason == "tool_calls"
    assert [tc.id for tc in response.message.tool_calls] == ["call_1", "call_2"]
    assert [tc.name for tc in response.message.tool_calls] == ["calculator", "datetime"]


@pytest.mark.asyncio
async def test_chat_sends_tool_call_id_and_json_arguments(monkeypatch):
    """Assistant tool_calls (history) must round-trip id + JSON-string args,
    and tool-result messages must carry tool_call_id -- Groq's OpenAI-style
    API links results back to calls by id, unlike simpler formats that match by position."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    from app.llm.base import ToolCall

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    history = [
        Message(role=Role.USER, content="what is 1+1?"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})],
        ),
        Message(role=Role.TOOL, content='{"result": 2}', tool_call_id="call_1", name="calculator"),
    ]
    await provider.chat(history)

    sent_messages = captured["body"]["messages"]
    assistant_msg = sent_messages[1]
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    tool_msg = sent_messages[2]
    assert tool_msg["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_chat_raises_authentication_error_on_401(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    provider = GroqProvider(api_key="bad-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMAuthenticationError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_rate_limit_error_on_429(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMRateLimitError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_does_not_retry_after_a_429_rate_limit_error(monkeypatch):
    """Regression test (real-runtime Groq 429 investigation): a rate-limit
    response must produce exactly one outbound HTTP request and one
    controlled error -- never a retry loop that would itself worsen the
    rate limit. If a recursive-retry bug were ever introduced here, this
    call count would catch it immediately."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMRateLimitError):
        await provider.chat([Message(role=Role.USER, content="hello")])

    assert call_count == 1


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_connect_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_server_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_malformed_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_malformed_tool_call_arguments(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "calculator", "arguments": "not-json"},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_response_log_carries_content_fingerprint_not_full_text(monkeypatch, caplog):
    """Regression/observability test: to diagnose duplicate-output bugs from
    logs alone (see app/core/logging.py:content_fingerprint), the
    "groq.chat.response" log line must carry content_length/content_hash/
    content_excerpt -- but never the full response text verbatim under a
    field that could grow unbounded or leak more than intended."""
    reply = "Acknowledged, Sir. Your name is Alex."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": reply}}]}
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with caplog.at_level(logging.INFO, logger="garvis.llm.groq"):
        await provider.chat([Message(role=Role.USER, content="hi")])

    records = [r for r in caplog.records if r.getMessage() == "groq.chat.response"]
    assert len(records) == 1
    fields = records[0].extra_fields
    assert fields["content_length"] == len(reply)
    assert fields["content_hash"] == hashlib.sha256(reply.encode("utf-8")).hexdigest()[:12]
    # Excerpt logging is opt-in (GARVIS_LOG_CONTENT_EXCERPTS); off by default.
    assert "content_excerpt" not in fields
    assert reply not in str(fields)
    assert "content" not in fields  # no verbatim full-text field name


@pytest.mark.asyncio
async def test_chat_response_log_excerpt_is_truncated_for_long_content(monkeypatch, caplog):
    from app.core import logging as core_logging

    monkeypatch.setattr(core_logging, "_log_content_excerpts", True)  # restored after the test
    long_reply = "x" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": long_reply}}]}
        )

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with caplog.at_level(logging.INFO, logger="garvis.llm.groq"):
        await provider.chat([Message(role=Role.USER, content="hi")])

    records = [r for r in caplog.records if r.getMessage() == "groq.chat.response"]
    fields = records[0].extra_fields
    assert fields["content_length"] == 500
    assert len(fields["content_excerpt"]) == 120


@pytest.mark.asyncio
async def test_health_check_true_on_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_false_on_connect_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = GroqProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    assert await provider.health_check() is False


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Patch httpx.AsyncClient (via app.llm.groq's imported reference) so
    requests hit `handler` instead of the network, restored after each test."""
    import app.llm.groq as groq_module

    original_client = groq_module.httpx.AsyncClient

    class _PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(groq_module.httpx, "AsyncClient", _PatchedClient)
