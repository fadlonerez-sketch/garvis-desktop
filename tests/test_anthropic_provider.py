"""Direct Anthropic provider (app/llm/anthropic_provider.py) and the
single-model policy end to end: Claude Haiku 5.5 only, never another model.

Requests go through the real `anthropic` SDK into an httpx2 MockTransport,
so the wire format and response parsing are the SDK's own -- no network,
no key, nothing billed."""
from __future__ import annotations

import json
import logging

import anthropic
import httpx
import httpx2
import pytest

from app.agent.agent import Agent
from app.agent.build import build_llm_provider
from app.agent.memory import SQLiteMemoryStore
from app.core.config import Settings
from app.core.events import EventBus, EventType
from app.core.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBillingError,
    LLMModelMismatchError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.anthropic_provider import AnthropicProvider, to_anthropic_messages, tool_use_id
from app.llm.base import Message, Role, ToolCall, ToolSpec
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.omniroute import OmniRouteProvider
from app.llm.openrouter import OpenRouterProvider
from app.llm.registry import ModelTier
from app.llm.usage import PriceCard, TokenUsage, task_meter
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry

_KEY = "sk-ant-test-key-never-real"
_PRICES = PriceCard(input=0.10, output=0.50, cache_read=0.01, cache_write_5m=0.125, cache_write_1h=0.20,
                    long_context_threshold=100_000, long_input=0.50, long_output=2.50)
_CALC = ToolSpec(name="calculator", description="Evaluate arithmetic.",
                 parameters={"type": "object", "properties": {"expression": {"type": "string"}},
                             "required": ["expression"]})


# ------------------------------------------------------------------ helpers
def _message(content, stop_reason="end_turn", model="claude-haiku-5-5", usage=None, msg_id="msg_01", **extra):
    return {
        "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 1200, "output_tokens": 30, "cache_read_input_tokens": 9000,
                           "cache_creation_input_tokens": 400,
                           "cache_creation": {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 0}},
        **extra,
    }


class FakeAnthropicAPI:
    """Scripted Messages API behind an httpx2 MockTransport."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        headers = {"request-id": f"req_{len(self.requests)}"}
        if status == 429:
            headers["retry-after"] = "7"
        return httpx2.Response(status, json=body, headers=headers)

    def body(self, index=-1) -> dict:
        return json.loads(self.requests[index].content)

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key=_KEY, base_url="https://api.anthropic.com", max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler)),
        )


def _provider(api: FakeAnthropicAPI, **kw) -> AnthropicProvider:
    kw.setdefault("reasoning_effort", "low")
    return AnthropicProvider(api_key=_KEY, model="claude-haiku-5-5", client=api.client(), pricing=_PRICES,
                             max_output_tokens=8192, **kw)


def _error(status, message, error_type="invalid_request_error"):
    return (status, {"type": "error", "error": {"type": error_type, "message": message}})


# ------------------------------------------------------- request formatting
async def test_request_uses_the_messages_api_shape_for_haiku_with_cache_breakpoints():
    api = FakeAnthropicAPI([_message([{"type": "text", "text": "408."}])])
    system = Message(role=Role.SYSTEM, content="STABLE RULES\n\nCurrent desktop: notepad", cacheable_prefix_chars=12)
    response = await _provider(api).chat([system, Message(role=Role.USER, content="17*24?")], tools=[_CALC])

    request = api.requests[0]
    assert request.url.path == "/v1/messages" and request.url.host == "api.anthropic.com"
    assert _KEY in (request.headers.get("authorization", "") + request.headers.get("x-api-key", ""))
    assert "anthropic-version" in request.headers
    body = api.body()
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 8192
    assert body["thinking"] == {"type": "disabled"} and body["output_config"] == {"effort": "low"}
    for absent in ("temperature", "top_p", "top_k"):
        assert absent not in body
    assert body["system"] == [
        {"type": "text", "text": "STABLE RULES", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Current desktop: notepad"},
    ]
    assert body["cache_control"] == {"type": "ephemeral"}  # automatic breakpoint for the tool-loop tail
    assert body["tools"] == [{"name": "calculator", "description": "Evaluate arithmetic.",
                              "input_schema": _CALC.parameters}]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "17*24?"}]}]
    assert response.message.content == "408." and response.finish_reason == "stop"


async def test_one_hour_cache_and_no_cache_options():
    api = FakeAnthropicAPI([_message([{"type": "text", "text": "a"}]), _message([{"type": "text", "text": "b"}])])
    system = Message(role=Role.SYSTEM, content="RULES\n\nlive", cacheable_prefix_chars=5)
    await _provider(api, prompt_cache="1h").chat([system, Message(role=Role.USER, content="x")])
    assert api.body()["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert api.body()["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    await _provider(api, prompt_cache="off").chat([system, Message(role=Role.USER, content="x")])
    body = api.body()
    assert "cache_control" not in body and body["system"] == [{"type": "text", "text": "RULES\n\nlive"}]


async def test_without_a_stable_system_prefix_the_tool_definitions_get_the_breakpoint():
    api = FakeAnthropicAPI([_message([{"type": "text", "text": "ok"}])])
    await _provider(api).chat([Message(role=Role.SYSTEM, content="all volatile"), Message(role=Role.USER, content="x")],
                              tools=[_CALC])
    body = api.body()
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["system"][0]


def test_history_translation_pairs_every_tool_call_with_its_result():
    history = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="do two things"),
        Message(role=Role.ASSISTANT, content="On it.", tool_calls=[
            ToolCall(id="toolu_a", name="calculator", arguments={"expression": "1+1"}),
            ToolCall(id="call:b.7", name="inspect_ui", arguments={}),  # id from a fallback provider
        ]),
        Message(role=Role.TOOL, content='{"success":true,"data":2}', tool_call_id="toolu_a", name="calculator"),
        Message(role=Role.TOOL, content='{"success":false,"error":"no window"}', tool_call_id="call:b.7",
                name="inspect_ui"),
        Message(role=Role.ASSISTANT, content="Done."),
        Message(role=Role.USER, content="thanks"),
    ]
    messages = to_anthropic_messages(history)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    call_ids = [b["id"] for b in messages[1]["content"] if b["type"] == "tool_use"]
    assert call_ids[0] == "toolu_a" and call_ids[1] == tool_use_id("call:b.7") != "call:b.7"
    results = messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == call_ids  # both results in ONE user turn, same ids
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    assert tool_use_id("call:b.7") == tool_use_id("call:b.7")  # deterministic


def test_interrupted_or_orphaned_steps_are_repaired_instead_of_breaking_the_request():
    history = [
        Message(role=Role.TOOL, content="{}", tool_call_id="ghost", name="x"),  # orphan at the start
        Message(role=Role.USER, content="go"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="toolu_1", name="calculator", arguments={})]),
        Message(role=Role.USER, content="never mind"),  # the tool result never arrived
    ]
    messages = to_anthropic_messages(history)
    # The orphaned result has no call to answer: it is kept as plain text, not as a tool_result.
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "[Earlier tool output] {}"},
                                                       {"type": "text", "text": "go"}]}
    last = messages[-1]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["tool_use_id"] == "toolu_1" and last[0]["is_error"]
    assert last[1] == {"type": "text", "text": "never mind"}


# --------------------------------------------------------- response parsing
async def test_tool_use_response_and_usage_are_parsed():
    api = FakeAnthropicAPI([_message(
        [{"type": "text", "text": "Checking."},
         {"type": "tool_use", "id": "toolu_01X", "name": "calculator", "input": {"expression": "17*24"}}],
        stop_reason="tool_use", msg_id="msg_tool")])
    response = await _provider(api).chat([Message(role=Role.USER, content="17*24")], tools=[_CALC])
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls == [ToolCall(id="toolu_01X", name="calculator",
                                                    arguments={"expression": "17*24"})]
    assert response.served_model == "claude-haiku-5-5" and response.request_id == "msg_tool"
    assert response.usage == TokenUsage(input_tokens=1200, output_tokens=30, cache_read_tokens=9000,
                                        cache_write_5m_tokens=400, cache_write_1h_tokens=0,
                                        cache_write_unknown_ttl_tokens=0)
    assert _PRICES.cost(response.usage) == pytest.approx((1200 * 0.10 + 30 * 0.50 + 9000 * 0.01 + 400 * 0.125) / 1e6)


async def test_usage_without_the_ttl_split_is_kept_as_unsplit_and_missing_cache_fields_are_zero():
    api = FakeAnthropicAPI([_message([{"type": "text", "text": "x"}],
                                     usage={"input_tokens": 50, "output_tokens": 5, "cache_creation_input_tokens": 700})])
    usage = (await _provider(api).chat([Message(role=Role.USER, content="x")])).usage
    assert usage.cache_write_unknown_ttl_tokens == 700 and usage.cache_write_5m_tokens == 0
    assert usage.cache_read_tokens == 0


async def test_a_tool_call_cut_off_by_max_tokens_is_never_returned_for_execution():
    api = FakeAnthropicAPI([_message(
        [{"type": "tool_use", "id": "toolu_1", "name": "write_text_file", "input": {"path": "a.txt"}}],
        stop_reason="max_tokens")])
    response = await _provider(api).chat([Message(role=Role.USER, content="write a long file")])
    assert response.finish_reason == "length" and response.message.tool_calls == []


async def test_refusal_and_model_mismatch_are_raised_with_their_billed_usage():
    api = FakeAnthropicAPI([
        _message([], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber",
                                                          "explanation": None}),
        _message([{"type": "text", "text": "hi"}], model="claude-sonnet-5-5"),
    ])
    with pytest.raises(LLMRefusalError, match="cyber") as refusal:
        await _provider(api).chat([Message(role=Role.USER, content="x")])
    assert refusal.value.usage.input_tokens == 1200
    with pytest.raises(LLMModelMismatchError, match="claude-sonnet-5-5") as mismatch:
        await _provider(api).chat([Message(role=Role.USER, content="x")])
    assert mismatch.value.usage is not None


# ------------------------------------------------------------------- errors
@pytest.mark.parametrize("reply, error_type, match", [
    (_error(401, "invalid x-api-key", "authentication_error"), LLMAuthenticationError, "ANTHROPIC_API_KEY"),
    (_error(403, "not allowed", "permission_error"), LLMAuthenticationError, "403"),
    (_error(429, "slow down", "rate_limit_error"), LLMRateLimitError, "retry after 7s"),
    (_error(529, "Overloaded", "overloaded_error"), LLMUnavailableError, "529"),
    (_error(500, "boom", "api_error"), LLMUnavailableError, "500"),
    (_error(404, "model: claude-haiku-5-5", "not_found_error"), LLMUnavailableError, "404"),
    (_error(400, "tools.0.input_schema: invalid"), LLMBadRequestError, "input_schema"),
    (_error(400, "Your credit balance is too low to access the Anthropic API."), LLMBillingError, "billing"),
    (httpx2.ReadTimeout("slow"), LLMTimeoutError, "didn't answer within 45s"),
    (httpx2.ConnectError("offline"), LLMUnavailableError, "can't reach the Anthropic API"),
])
async def test_api_errors_map_to_garvis_errors_without_sdk_retries(reply, error_type, match):
    api = FakeAnthropicAPI([reply])
    with pytest.raises(error_type, match=match) as info:
        await _provider(api, timeout_seconds=45.0).chat([Message(role=Role.USER, content="x")])
    assert len(api.requests) == 1  # the SDK's own retries are off: the router decides
    assert _KEY not in str(info.value)


async def test_without_a_key_requests_fail_at_once_and_nothing_is_sent():
    provider = AnthropicProvider(api_key="", model="claude-haiku-5-5")
    with pytest.raises(LLMAuthenticationError, match="ANTHROPIC_API_KEY is not set.*No other model is used"):
        await provider.chat([Message(role=Role.USER, content="hi")])
    assert provider._client is None  # no SDK client, so no request, was ever created
    assert not await provider.health_check()


def test_the_key_never_shows_and_the_sdk_retries_are_off():
    provider = AnthropicProvider(api_key=_KEY, model="claude-haiku-5-5")
    assert _KEY not in repr(provider)
    client = provider._get_client()
    assert client.max_retries == 0 and str(client.base_url).startswith("https://api.anthropic.com")


def test_settings_read_the_key_from_ANTHROPIC_API_KEY(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GARVIS_LOG_LEVEL=INFO\nANTHROPIC_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert Settings(_env_file=str(env)).anthropic_api_key == "sk-from-dotenv"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert Settings(_env_file=str(env)).anthropic_api_key == "sk-from-env"


def test_a_missing_key_is_logged_with_instructions(caplog):
    logging.getLogger("garvis").propagate = True
    caplog.set_level(logging.WARNING, logger="garvis.agent.build")
    try:
        router = build_llm_provider(Settings(_env_file=None, llm_provider="tiered", omniroute_api_key="k"))
    finally:
        logging.getLogger("garvis").propagate = False
    assert all(isinstance(p, AnthropicProvider) for t in ModelTier for _, p in router.pools[t])
    [record] = [r for r in caplog.records if r.getMessage() == "llm.anthropic.no_key"]
    assert "check_anthropic.py --set-key" in record.extra_fields["note"]


# ------------------------------------------- Claude Haiku 5.5 only, no fallback
class _OtherProviderCalled(AssertionError):
    pass


@pytest.fixture
def no_other_provider(monkeypatch):
    """Every other provider class Garvis still ships is booby-trapped, and so is
    the plain-httpx transport they use: if anything but the Anthropic SDK (an
    httpx2 MockTransport here) is asked for a completion, the test fails."""
    calls: list[str] = []

    def trap(name):
        async def chat(self, *args, **kwargs):
            calls.append(name)
            raise _OtherProviderCalled(f"{name}.chat was called")
        return chat

    for cls in (OmniRouteProvider, GeminiProvider, GroqProvider, OpenRouterProvider):
        monkeypatch.setattr(cls, "chat", trap(cls.__name__))

    async def no_http(self, request, *args, **kwargs):
        calls.append(f"httpx:{request.url}")
        raise _OtherProviderCalled(f"plain httpx request to {request.url}")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_http)
    return calls


def _real_router(api: FakeAnthropicAPI, tmp_path, **settings):
    """The shipped registry, built the way Garvis builds it -- with keys for
    every other provider present -- and the Anthropic SDK pointed at `api`."""
    router = build_llm_provider(Settings(
        _env_file=None, llm_provider="tiered", anthropic_api_key=_KEY, omniroute_api_key="k", gemini_api_key="g",
        groq_api_key="q", openrouter_api_key="o", openrouter_model="x/y", llm_usage_db_path=tmp_path / "u.db",
        **settings,
    ))
    client = api.client()
    for tier in ModelTier:
        ((_, provider),) = router.pools[tier]
        provider._client = client
    return router


def _ledger_rows(router):
    return router._ledger.query("SELECT provider, model_requested, model_served, status, error_type FROM llm_calls "
                                "ORDER BY id")


async def test_a_successful_request_goes_to_haiku_and_is_recorded_as_haiku(tmp_path, no_other_provider):
    api = FakeAnthropicAPI([_message([{"type": "text", "text": "Hello, Sir."}])])
    router = _real_router(api, tmp_path)
    with task_meter() as meter:
        response = await router.chat([Message(role=Role.USER, content="hello")])
    assert response.message.content == "Hello, Sir." and len(api.requests) == 1
    assert api.body()["model"] == "claude-haiku-5-5"
    assert meter.models == ["anthropic:claude-haiku-5-5@low"]
    assert _ledger_rows(router) == [("AnthropicProvider", "claude-haiku-5-5", "claude-haiku-5-5", "ok", None)]
    assert no_other_provider == []


@pytest.mark.parametrize("replies, error_type, requests_sent", [
    ([_error(429, "slow down", "rate_limit_error")] * 2, LLMRateLimitError, 2),     # 1 retry, same model
    ([_error(529, "Overloaded", "overloaded_error")] * 2, LLMUnavailableError, 2),
    ([_error(500, "boom", "api_error")] * 2, LLMUnavailableError, 2),
    ([httpx2.ReadTimeout("slow")], LLMTimeoutError, 1),                                 # never resent
    ([_error(401, "invalid x-api-key", "authentication_error")], LLMAuthenticationError, 1),
    ([_error(400, "Your credit balance is too low to access the Anthropic API.")], LLMBillingError, 1),
    ([_error(400, "messages.1: invalid")], LLMBadRequestError, 1),
    ([_message([], stop_reason="refusal")], LLMRefusalError, 1),
    ([_message([{"type": "text", "text": "x"}], model="claude-sonnet-5-5")], LLMModelMismatchError, 1),
])
async def test_when_haiku_fails_the_error_is_reported_and_no_other_model_is_called(
        tmp_path, no_other_provider, replies, error_type, requests_sent):
    api = FakeAnthropicAPI(list(replies))
    router = _real_router(api, tmp_path, llm_max_retries=1, llm_retry_backoff_seconds=0)
    events = EventBus()
    errors: list[str] = []
    events.subscribe(lambda e: errors.append(e.payload["message"]) if e.type == EventType.ERROR else None)
    agent = Agent(llm=router, registry=ToolRegistry(), memory=SQLiteMemoryStore(tmp_path / "m.db"), events=events)
    reply = await agent.handle_user_message("open notepad")
    assert errors == [reply] and reply  # reported to the user, turn stopped
    assert len(api.requests) == requests_sent
    assert all(json.loads(r.content)["model"] == "claude-haiku-5-5" for r in api.requests)
    assert no_other_provider == []
    rows = _ledger_rows(router)
    assert {row[0] for row in rows} == {"AnthropicProvider"} and len(rows) == requests_sent
    assert rows[-1][4] == error_type.__name__


class CountingTool(Tool):
    name = "click_button"
    description = "Click a button."
    risk_level = RiskLevel.LOW

    class input_model(CalculatorTool.input_model):  # any one-field schema will do
        pass

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args) -> ToolResult:
        self.runs += 1
        return ToolResult.ok({"clicked": True})


def _tool_agent(router, tmp_path):
    tool = CountingTool()
    registry = ToolRegistry()
    registry.register(tool)
    return Agent(llm=router, registry=registry, memory=SQLiteMemoryStore(tmp_path / "m.db"), events=EventBus()), tool


_CLICK = _message([{"type": "tool_use", "id": "toolu_click", "name": "click_button", "input": {"expression": "ok"}}],
                  stop_reason="tool_use")


async def test_a_failure_after_a_tool_ran_stops_the_turn_and_the_tool_never_reruns(tmp_path, no_other_provider):
    api = FakeAnthropicAPI([_CLICK, _error(529, "Overloaded", "overloaded_error")])
    router = _real_router(api, tmp_path, llm_max_retries=0)
    agent, tool = _tool_agent(router, tmp_path)
    reply = await agent.handle_user_message("click it")
    assert "529" in reply and tool.runs == 1 and len(api.requests) == 2
    assert no_other_provider == []


async def test_a_same_model_retry_after_a_tool_ran_resends_the_result_not_the_action(tmp_path, no_other_provider):
    api = FakeAnthropicAPI([_CLICK, _error(529, "Overloaded", "overloaded_error"),
                            _message([{"type": "text", "text": "Clicked it, Sir."}])])
    router = _real_router(api, tmp_path, llm_max_retries=1, llm_retry_backoff_seconds=0)
    agent, tool = _tool_agent(router, tmp_path)
    assert await agent.handle_user_message("click it") == "Clicked it, Sir."
    assert tool.runs == 1 and len(api.requests) == 3
    retried = api.body(2)["messages"]
    assert api.body(1)["messages"] == retried  # the retry is the identical request, same model
    assert retried[-1]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_click",
                                         "content": '{"success":true,"data":{"clicked":true}}'}
    assert no_other_provider == []


async def test_multi_step_tool_loop_runs_entirely_on_haiku(tmp_path, no_other_provider):
    api = FakeAnthropicAPI([
        _message([{"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "17*24"}}],
                 stop_reason="tool_use", msg_id="msg_1"),
        _message([{"type": "tool_use", "id": "toolu_2", "name": "calculator", "input": {"expression": "408+1"}}],
                 stop_reason="tool_use", msg_id="msg_2"),
        _message([{"type": "text", "text": "409."}], msg_id="msg_3"),
    ])
    router = _real_router(api, tmp_path)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = Agent(llm=router, registry=registry, memory=SQLiteMemoryStore(tmp_path / "m.db"), events=EventBus())
    assert await agent.handle_user_message("17*24, plus one") == "409."
    assert len(api.requests) == 3 and no_other_provider == []
    final = api.body(2)
    assert [m["role"] for m in final["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert final["messages"][4]["content"][0]["tool_use_id"] == "toolu_2"
    assert final["system"][0]["cache_control"] == {"type": "ephemeral"}  # real Garvis system prompt is split
    assert "Current desktop" not in final["system"][0]["text"]
