"""Tests for OmniRouteProvider (app/llm/omniroute.py) -- httpx MockTransport, no network."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.llm.base import Message, Role, ToolCall, ToolSpec
from app.llm.omniroute import DEFAULT_BASE_URL, OmniRouteProvider

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_SECRET = "omr-test-secret-value"


def _patch_transport(monkeypatch, handler) -> None:
    import app.llm.omniroute as omniroute_module

    class _Patched(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(omniroute_module.httpx, "AsyncClient", _Patched)


def _ok(message: dict, finish_reason: str = "stop", model: str = "gemini-3.1-flash-lite") -> httpx.Response:
    return httpx.Response(200, json={"model": model, "choices": [{"message": message, "finish_reason": finish_reason}]})


def _provider(model="gemini/gemini-3.1-flash-lite", **kw) -> OmniRouteProvider:
    return OmniRouteProvider(api_key=_SECRET, model=model, **kw)


_CALC = ToolSpec(
    name="calculator",
    description="calc",
    parameters={
        "title": "CalcInput",
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"title": "Expression", "type": "string", "default": "1"}},
    },
)


# ------------------------------------------------------------ construction
def test_defaults_to_local_omniroute():
    provider = _provider()
    assert provider.base_url == DEFAULT_BASE_URL == "http://127.0.0.1:20128/v1"
    assert provider.model == "gemini/gemini-3.1-flash-lite"


def test_requires_key_and_model():
    with pytest.raises(ConfigurationError, match="check_omniroute.py --set-key"):
        OmniRouteProvider(api_key="", model="gemini/gemini-3.1-flash-lite")
    with pytest.raises(ConfigurationError):
        OmniRouteProvider(api_key="k", model="")


@pytest.mark.parametrize("url", ["http://localhost:20128/v1/", "http://[::1]:20128/v1", "http://127.0.0.1:9999/v1"])
def test_accepts_loopback_base_urls(url):
    assert _provider(base_url=url).base_url == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.10:20128/v1",
        "https://omniroute.example.com/v1",
        "http://0.0.0.0:20128/v1",
        "ftp://127.0.0.1/v1",
    ],
)
def test_refuses_non_local_base_urls(url):
    with pytest.raises(ConfigurationError, match="localhost-only"):
        _provider(base_url=url)


@pytest.mark.parametrize(
    "model", ["auto", "auto/best-free", "auto/coding:free", "AUTO/x", "combo/mine", "my-combo", "gemini/"]
)
def test_refuses_automatic_or_implicit_routes(model):
    with pytest.raises(ConfigurationError):
        _provider(model=model)


def test_repr_never_contains_the_key():
    assert _SECRET not in repr(_provider()) and _SECRET not in str(_provider())


# ------------------------------------------------------------ request shape
async def test_request_url_auth_and_explicit_model(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "Hi."})

    _patch_transport(monkeypatch, handler)
    response = await _provider(reasoning_effort="minimal").chat([Message(role=Role.USER, content="hi")])

    assert response.message.content == "Hi." and response.finish_reason == "stop"
    assert seen["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    assert seen["auth"] == f"Bearer {_SECRET}"
    body = seen["body"]
    assert body["model"] == "gemini/gemini-3.1-flash-lite"  # sent verbatim: no auto routing
    assert body["stream"] is False and body["reasoning_effort"] == "minimal"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_no_reasoning_effort_or_tools_when_unset(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "ok"})

    _patch_transport(monkeypatch, handler)
    await _provider(model="groq/openai/gpt-oss-120b").chat([Message(role=Role.USER, content="x")])
    assert "reasoning_effort" not in seen["body"] and "tools" not in seen["body"]


async def test_gemini_tool_schemas_are_sanitized_but_others_are_not(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _ok({"role": "assistant", "content": "ok"})

    _patch_transport(monkeypatch, handler)
    await _provider(model="gemini/gemini-3.1-flash-lite").chat([Message(role=Role.USER, content="x")], tools=[_CALC])
    await _provider(model="groq/openai/gpt-oss-120b").chat([Message(role=Role.USER, content="x")], tools=[_CALC])
    gemini_params = bodies[0]["tools"][0]["function"]["parameters"]
    groq_params = bodies[1]["tools"][0]["function"]["parameters"]
    assert "title" not in gemini_params and "default" not in gemini_params["properties"]["expression"]
    assert groq_params == _CALC.parameters
    assert bodies[1]["tools"][0] == {
        "type": "function",
        "function": {"name": "calculator", "description": "calc", "parameters": _CALC.parameters},
    }


# ------------------------------------------------------------ tool calling
_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
    "extra_content": {"google": {"thought_signature": "SIG-abc"}},
}


async def test_tool_calls_are_parsed_and_round_trip_with_opaque_fields(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return _ok({"role": "assistant", "content": None, "tool_calls": [_CALL]}, finish_reason="tool_calls")
        return _ok({"role": "assistant", "content": "4"})

    _patch_transport(monkeypatch, handler)
    provider = _provider()
    first = await provider.chat([Message(role=Role.USER, content="2+2?")], tools=[_CALC])
    [call] = first.message.tool_calls
    assert first.finish_reason == "tool_calls"
    assert (call.id, call.name, call.arguments) == ("call_1", "calculator", {"expression": "2+2"})
    assert call.provider_metadata == {"extra_content": {"google": {"thought_signature": "SIG-abc"}}}

    history = [
        Message(role=Role.USER, content="2+2?"),
        first.message,
        Message(role=Role.TOOL, content='{"data": 4}', tool_call_id="call_1", name="calculator"),
    ]
    second = await provider.chat(history, tools=[_CALC])
    assert second.message.content == "4"
    assistant, tool = bodies[1]["messages"][1], bodies[1]["messages"][2]
    assert assistant["content"] is None
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
            "extra_content": {"google": {"thought_signature": "SIG-abc"}},
        }
    ]
    assert tool == {"role": "tool", "content": '{"data": 4}', "tool_call_id": "call_1", "name": "calculator"}


async def test_tool_call_without_opaque_fields_gets_no_placeholder(monkeypatch):
    """OmniRoute manages Gemini thought signatures itself -- Garvis never injects one."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok({"role": "assistant", "content": "ok"})

    _patch_transport(monkeypatch, handler)
    history = [
        Message(role=Role.USER, content="x"),
        Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id="c9", name="calculator", arguments={})]),
        Message(role=Role.TOOL, content="{}", tool_call_id="c9", name="calculator"),
    ]
    await _provider().chat(history)
    assert "extra_content" not in seen["body"]["messages"][1]["tool_calls"][0]


async def test_missing_tool_call_id_gets_generated(monkeypatch):
    call = {"type": "function", "function": {"name": "calculator", "arguments": "{}"}}
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "tool_calls": [call]}))
    [tc] = (await _provider().chat([Message(role=Role.USER, content="x")])).message.tool_calls
    assert tc.id and tc.arguments == {} and tc.provider_metadata is None


async def test_malformed_tool_call_is_an_error(monkeypatch):
    bad = {"id": "c", "function": {"name": "calculator", "arguments": "{not json"}}
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "tool_calls": [bad]}))
    with pytest.raises(LLMError, match="malformed tool call"):
        await _provider().chat([Message(role=Role.USER, content="x")])


async def test_length_finish_and_list_content(monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda r: _ok(
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            finish_reason="length",
        ),
    )
    response = await _provider().chat([Message(role=Role.USER, content="x")])
    assert response.message.content == "ab" and response.finish_reason == "length"


async def test_reasoning_field_is_accepted_and_not_mixed_into_content(monkeypatch):
    _patch_transport(
        monkeypatch, lambda r: _ok({"role": "assistant", "content": "answer", "reasoning_content": "thinking..."})
    )
    response = await _provider(model="groq/openai/gpt-oss-120b").chat([Message(role=Role.USER, content="x")])
    assert response.message.content == "answer"


# ------------------------------------------------------------ errors
@pytest.mark.parametrize(
    "status, body, exc_type, match",
    [
        (401, {"error": {"message": "Invalid API key"}}, LLMAuthenticationError, "GARVIS_OMNIROUTE_API_KEY"),
        # Real OmniRoute body when the upstream provider isn't connected: not a Garvis key problem.
        (
            401,
            {"error": {"message": "No active credentials for provider: groq."}},
            LLMUnavailableError,
            "no connection in OmniRoute.*No active credentials for provider: groq",
        ),
        (
            403,
            {"error": {"message": 'Model "ddgw/gpt-5.4-mini" is not allowed for this API key'}},
            LLMAuthenticationError,
            "not allowed for this API key",
        ),
        (403, {"error": {"message": "provider blocked"}}, LLMAuthenticationError, "provider blocked"),
        (429, {"error": {"message": "quota exhausted"}}, LLMRateLimitError, "refused, not billed.*quota exhausted"),
        (404, {"error": {"message": "no connection"}}, LLMUnavailableError, "connected in the OmniRoute dashboard"),
        (400, {"error": "bad request"}, LLMBadRequestError, r"rejected the request \(400\): bad request"),
        (503, "upstream down", LLMUnavailableError, "503 upstream down"),
    ],
)
async def test_http_errors_are_mapped_and_name_the_model(monkeypatch, status, body, exc_type, match):
    def handler(request):
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(exc_type, match=match) as info:
        await _provider(model="mistral/mistral-small-latest").chat([Message(role=Role.USER, content="x")])
    if status != 401:
        assert "mistral/mistral-small-latest" in str(info.value)
    assert _SECRET not in str(info.value)


async def test_connection_refused_is_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="can't reach OmniRoute"):
        await _provider().chat([Message(role=Role.USER, content="x")])


async def test_timeout_is_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LLMUnavailableError, match="did not respond in time"):
        await _provider().chat([Message(role=Role.USER, content="x")])


async def test_unexpected_shape_is_an_error(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="unexpected response shape"):
        await _provider().chat([Message(role=Role.USER, content="x")])


# ------------------------------------------------------------ secrecy / health
async def test_key_never_appears_in_logs(monkeypatch, caplog):
    _patch_transport(monkeypatch, lambda r: _ok({"role": "assistant", "content": "ok"}))
    logging.getLogger("garvis").propagate = True
    try:
        with caplog.at_level(logging.DEBUG, logger="garvis.llm.omniroute"):
            await _provider().chat([Message(role=Role.USER, content="x")])
    finally:
        logging.getLogger("garvis").propagate = False
    assert caplog.records
    for record in caplog.records:
        assert _SECRET not in record.getMessage()
        assert _SECRET not in json.dumps(getattr(record, "extra_fields", {}), default=str)


async def test_health_check(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": []})

    _patch_transport(monkeypatch, handler)
    assert await _provider().health_check()
    assert seen["url"] == "http://127.0.0.1:20128/v1/models"
    _patch_transport(monkeypatch, lambda r: httpx.Response(401))
    assert not await _provider().health_check()
