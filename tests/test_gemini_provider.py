"""Tests for GeminiProvider (app/llm/gemini.py) -- httpx MockTransport, no network."""
from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.llm.base import Message, Role, ToolCall, ToolSpec
from app.llm.gemini import SKIP_SIGNATURE, GeminiProvider, sanitize_schema


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_transport(monkeypatch, handler) -> None:
    """Safe to call more than once per test: always wraps the real client."""
    import app.llm.gemini as gemini_module

    class _Patched(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(gemini_module.httpx, "AsyncClient", _Patched)


def _ok(message: dict, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})


_SIGNED_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
    "extra_content": {"google": {"thought_signature": "SIG-abc"}},
}


def test_constructor_requires_key_and_model():
    with pytest.raises(ConfigurationError, match="GARVIS_GEMINI_API_KEY"):
        GeminiProvider(api_key="", model="gemini-3.8-flash")
    with pytest.raises(ConfigurationError):
        GeminiProvider(api_key="k", model="")


async def test_request_shape_auth_model_reasoning_and_tools(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "Hi."})

    _patch_transport(monkeypatch, handler)
    provider = GeminiProvider(api_key="test-key", model="gemini-3.1-flash-lite", reasoning_effort="minimal")
    spec = ToolSpec(name="calculator", description="calc", parameters={
        "title": "CalcInput", "type": "object", "required": ["expression"],
        "properties": {"expression": {"title": "Expression", "type": "string", "description": "e"}},
    })
    response = await provider.chat([Message(role=Role.USER, content="hi")], tools=[spec])

    assert response.message.content == "Hi." and response.finish_reason == "stop"
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "gemini-3.1-flash-lite" and body["reasoning_effort"] == "minimal"
    params = body["tools"][0]["function"]["parameters"]
    assert "title" not in params and "title" not in params["properties"]["expression"]


async def test_no_reasoning_effort_when_unset(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "ok"})

    _patch_transport(monkeypatch, handler)
    await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])
    assert "reasoning_effort" not in seen["body"]


async def test_tool_calls_parsed_and_thought_signature_kept(monkeypatch):
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "content": None, "tool_calls": [_SIGNED_CALL]}, "tool_calls"))
    response = await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="2+2?")])
    call = response.message.tool_calls[0]
    assert response.finish_reason == "tool_calls"
    assert (call.id, call.name, call.arguments) == ("call_1", "calculator", {"expression": "2+2"})
    assert call.provider_metadata == {"extra_content": {"google": {"thought_signature": "SIG-abc"}}}
    assert response.message.content == ""


async def test_thought_signature_is_echoed_back_and_foreign_calls_get_placeholder(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "4"})

    _patch_transport(monkeypatch, handler)
    own = ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"},
                   provider_metadata={"extra_content": {"google": {"thought_signature": "SIG-abc"}}})
    foreign = ToolCall(id="call_2", name="datetime", arguments={})  # made by another model (failover)
    history = [
        Message(role=Role.USER, content="2+2 and time?"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[own, foreign]),
        Message(role=Role.TOOL, content='{"success": true, "data": 4}', tool_call_id="call_1", name="calculator"),
        Message(role=Role.TOOL, content='{"success": true}', tool_call_id="call_2", name="datetime"),
    ]
    await GeminiProvider(api_key="k", model="m").chat(history)

    assistant = seen["body"]["messages"][1]
    assert assistant["content"] is None  # empty text alongside tool calls
    assert assistant["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG-abc"}}
    assert assistant["tool_calls"][1]["extra_content"] == {"google": {"thought_signature": SKIP_SIGNATURE}}
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}
    tool_msg = seen["body"]["messages"][2]
    assert tool_msg["tool_call_id"] == "call_1" and tool_msg["name"] == "calculator"


async def test_missing_tool_call_id_gets_generated(monkeypatch):
    call = {"type": "function", "function": {"name": "x", "arguments": ""}}
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "tool_calls": [call]}))
    response = await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])
    assert response.message.tool_calls[0].id and response.message.tool_calls[0].arguments == {}


async def test_length_finish_reason(monkeypatch):
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "content": "partial"}, "length"))
    response = await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])
    assert response.finish_reason == "length"


@pytest.mark.parametrize(
    "status, payload, exc_type, fragment",
    [
        (400, [{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}],
         LLMAuthenticationError, "GARVIS_GEMINI_API_KEY"),
        (403, {"error": {"message": "permission denied"}}, LLMAuthenticationError, "API key"),
        (429, {"error": {"message": "Quota exceeded for metric generate_content_free_tier_requests"}},
         LLMRateLimitError, "Quota exceeded"),
        (404, {"error": {"message": "models/nope is not found"}}, LLMUnavailableError, "not found"),
        (500, {"error": {"message": "internal"}}, LLMUnavailableError, "500"),
        (400, {"error": {"message": "Invalid JSON payload"}}, LLMUnavailableError, "Invalid JSON"),
    ],
)
async def test_error_mapping(monkeypatch, status, payload, exc_type, fragment):
    _patch_transport(monkeypatch, lambda r: httpx.Response(status, json=payload))
    with pytest.raises(exc_type, match=fragment):
        await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])


async def test_error_message_never_contains_the_key(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(400, json={"error": {"message": "API key not valid"}}))
    with pytest.raises(LLMAuthenticationError) as info:
        await GeminiProvider(api_key="super-secret-123", model="m").chat([Message(role=Role.USER, content="x")])
    assert "super-secret-123" not in str(info.value)


async def test_network_errors(monkeypatch):
    def connect_fail(request):
        raise httpx.ConnectError("no route", request=request)

    _patch_transport(monkeypatch, connect_fail)
    with pytest.raises(LLMUnavailableError, match="can't reach Gemini"):
        await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    _patch_transport(monkeypatch, timeout)
    with pytest.raises(LLMUnavailableError, match="in time"):
        await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])


async def test_malformed_responses(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="unexpected response"):
        await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])
    bad_call = {"id": "c", "function": {"name": "x", "arguments": "{not json"}}
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "tool_calls": [bad_call]}))
    with pytest.raises(LLMError, match="malformed tool call"):
        await GeminiProvider(api_key="k", model="m").chat([Message(role=Role.USER, content="x")])


async def test_health_check(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"data": []}))
    assert await GeminiProvider(api_key="k", model="m").health_check()
    _patch_transport(monkeypatch, lambda r: httpx.Response(400, json={}))
    assert not await GeminiProvider(api_key="k", model="m").health_check()


# ------------------------------------------------------------ sanitize_schema
def test_sanitize_schema_nullable_and_drops_unsupported_keys():
    schema = {
        "title": "In", "type": "object", "required": ["a"],
        "properties": {
            "a": {"title": "A", "type": "string", "description": "d"},
            "b": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None, "title": "B",
                  "description": "opt"},
            "c": {"type": "string", "enum": ["up", "down"], "title": "C"},
            "title": {"type": "string"},  # a *property* called "title" must survive
        },
    }
    out = sanitize_schema(schema)
    assert out == {
        "type": "object", "required": ["a"],
        "properties": {
            "a": {"type": "string", "description": "d"},
            "b": {"type": "integer", "minimum": 1, "description": "opt", "nullable": True},
            "c": {"type": "string", "enum": ["up", "down"]},
            "title": {"type": "string"},
        },
    }


def test_sanitize_schema_handles_every_registered_tool():
    """Every real Garvis tool schema sanitizes to the supported key subset."""
    from app.agent.build import build_browser_stack, build_extra_tools, build_registry
    from app.core.config import Settings

    settings = Settings(_env_file=None)
    desktop, workspace, browser = build_browser_stack(settings)
    registry = build_registry(object(), browser, object(), extra_tools=build_extra_tools(settings, desktop, workspace, browser))
    allowed = {"type", "description", "properties", "required", "items", "enum", "minimum", "maximum",
               "minItems", "maxItems", "format", "nullable"}

    def check(node):
        assert set(node) <= allowed, set(node) - allowed
        for prop in node.get("properties", {}).values():
            check(prop)
        if isinstance(node.get("items"), dict):
            check(node["items"])

    for spec in registry.tool_specs():
        check(sanitize_schema(spec.parameters))
