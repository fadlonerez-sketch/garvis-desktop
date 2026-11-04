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
from app.llm.openrouter import OpenRouterProvider


def test_constructor_rejects_missing_api_key():
    with pytest.raises(ConfigurationError):
        OpenRouterProvider(api_key="", model="test-model")


def test_constructor_rejects_missing_model():
    """Unlike Groq (which always has a concrete default model), OpenRouter
    serves hundreds of models with no safe default -- an unset model must
    fail fast and clearly rather than sending an empty model string to the
    API and surfacing a confusing 400 later."""
    with pytest.raises(ConfigurationError, match="model"):
        OpenRouterProvider(api_key="test-key", model="")


@pytest.mark.asyncio
async def test_chat_parses_plain_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        )

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="hello")])
    assert response.message.content == "Hi!"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_uses_the_configured_model(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-haiku-4.5")
    _patch_transport(monkeypatch, handler)

    await provider.chat([Message(role=Role.USER, content="hello")])
    assert captured["body"]["model"] == "anthropic/claude-haiku-4.5"


@pytest.mark.asyncio
async def test_chat_posts_to_the_configured_base_url(monkeypatch):
    """Confirms the request actually lands on OpenRouter's chat-completions
    endpoint, not merely that some URL was hit."""
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    await provider.chat([Message(role=Role.USER, content="hello")])
    assert seen_urls == ["https://openrouter.ai/api/v1/chat/completions"]


@pytest.mark.asyncio
async def test_chat_respects_custom_base_url(monkeypatch):
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    provider = OpenRouterProvider(
        api_key="test-key", model="test-model", base_url="https://example.test/api/v1"
    )
    _patch_transport(monkeypatch, handler)

    await provider.chat([Message(role=Role.USER, content="hello")])
    assert seen_urls == ["https://example.test/api/v1/chat/completions"]


@pytest.mark.asyncio
async def test_chat_sends_tool_definitions(monkeypatch):
    from app.llm.base import ToolSpec

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    tools = [
        ToolSpec(
            name="calculator",
            description="Evaluate a math expression",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        )
    ]
    await provider.chat([Message(role=Role.USER, content="what is 1+1?")], tools=tools)

    sent_tools = captured["body"]["tools"]
    assert sent_tools == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_chat_omits_tools_field_when_no_tools_given(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    await provider.chat([Message(role=Role.USER, content="hello")])
    assert "tools" not in captured["body"]


@pytest.mark.asyncio
async def test_chat_parses_single_tool_call(monkeypatch):
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

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="what is 1+1?")])
    assert response.finish_reason == "tool_calls"
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].id == "call_1"
    assert response.message.tool_calls[0].name == "calculator"
    assert response.message.tool_calls[0].arguments == {"expression": "1+1"}


@pytest.mark.asyncio
async def test_chat_parses_multiple_tool_calls_distinguishably(monkeypatch):
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

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    response = await provider.chat([Message(role=Role.USER, content="do two things")])
    assert response.finish_reason == "tool_calls"
    assert [tc.id for tc in response.message.tool_calls] == ["call_1", "call_2"]
    assert [tc.name for tc in response.message.tool_calls] == ["calculator", "datetime"]
    assert [tc.arguments for tc in response.message.tool_calls] == [
        {"expression": "1+1"},
        {},
    ]


@pytest.mark.asyncio
async def test_chat_sends_tool_call_id_and_json_arguments(monkeypatch):
    """Assistant tool_calls (history) must round-trip id + JSON-string args,
    and tool-result messages must carry tool_call_id -- same OpenAI-style
    id-threading Groq already requires, confirming the existing Agent tool
    loop can feed this provider without any OpenRouter-specific logic."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    from app.llm.base import ToolCall

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
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
    assert tool_msg["name"] == "calculator"


@pytest.mark.asyncio
async def test_chat_raises_authentication_error_on_401(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    provider = OpenRouterProvider(api_key="bad-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMAuthenticationError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_authentication_error_on_403(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMAuthenticationError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_rate_limit_error_on_429(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMRateLimitError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_does_not_retry_after_a_429_rate_limit_error(monkeypatch):
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMRateLimitError):
        await provider.chat([Message(role=Role.USER, content="hello")])

    assert call_count == 1


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_insufficient_credits_402(monkeypatch):
    """OpenRouter-specific error code Groq never returns -- must not crash
    with an unhandled exception, and must map into the existing LLMError
    hierarchy rather than a new one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient credits"}})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_no_provider_available_503(monkeypatch):
    """Another OpenRouter-specific code ("no provider currently meets the
    routing requirements") -- same generic-unavailable handling as any
    other non-auth/rate-limit server error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "no provider available"}})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_connect_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_timeout(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_unavailable_on_server_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMUnavailableError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_malformed_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_missing_choices_field(monkeypatch):
    """Missing expected response field -- distinct from a wrong-shape
    'unexpected' key, closer to what a real malformed/truncated upstream
    response could look like."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
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

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with pytest.raises(LLMError):
        await provider.chat([Message(role=Role.USER, content="hello")])


@pytest.mark.asyncio
async def test_chat_response_log_carries_content_fingerprint_not_full_text(monkeypatch, caplog):
    reply = "Acknowledged, Sir."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": reply}}]}
        )

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    with caplog.at_level(logging.INFO, logger="garvis.llm.openrouter"):
        await provider.chat([Message(role=Role.USER, content="hi")])

    records = [r for r in caplog.records if r.getMessage() == "openrouter.chat.response"]
    assert len(records) == 1
    fields = records[0].extra_fields
    assert fields["content_length"] == len(reply)
    assert fields["content_hash"] == hashlib.sha256(reply.encode("utf-8")).hexdigest()[:12]
    assert "content" not in fields  # no verbatim full-text field name


@pytest.mark.asyncio
async def test_health_check_true_on_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_false_on_connect_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OpenRouterProvider(api_key="test-key", model="test-model")
    _patch_transport(monkeypatch, handler)

    assert await provider.health_check() is False


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Patch httpx.AsyncClient (via app.llm.openrouter's imported reference)
    so requests hit `handler` instead of the network, restored after each
    test. Mirrors tests/test_groq_provider.py's helper exactly."""
    import app.llm.openrouter as openrouter_module

    original_client = openrouter_module.httpx.AsyncClient

    class _PatchedClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openrouter_module.httpx, "AsyncClient", _PatchedClient)
