"""Cost controls: usage accounting, budgets, bounded retries, model
verification, and the Haiku 5.5 request shape through OmniRoute."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from app.agent.agent import Agent
from app.agent.build import build_llm_provider
from app.agent.memory import SQLiteMemoryStore
from app.core.config import Settings
from app.core.events import EventBus, EventType
from app.core.exceptions import (
    ConfigurationError,
    LLMBadRequestError,
    LLMBudgetExceededError,
    LLMModelMismatchError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec, served_model_matches
from app.llm.omniroute import OmniRouteProvider, parse_usage
from app.llm.registry import ModelTier, load_model_registry
from app.llm.router import TieredLLMRouter, estimate_prompt_tokens
from app.llm.usage import BudgetGuard, BudgetLimits, PriceCard, TaskMeter, TokenUsage, UsageLedger, task_meter
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry, compact_schema
from tests.conftest import final_response, tool_call_response

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_HAIKU_PRICES = PriceCard(
    input=0.10, output=0.50, cache_read=0.01, cache_write_5m=0.125, cache_write_1h=0.20,
    long_context_threshold=100_000, long_input=0.50, long_output=2.50, long_cache_read=0.05,
    long_cache_write_5m=0.625, long_cache_write_1h=1.00,
)


# ----------------------------------------------------------------- pricing
def test_cost_uses_reported_tokens_and_the_haiku_rate_card():
    usage = TokenUsage(input_tokens=1_000, output_tokens=200, cache_read_tokens=10_000, cache_write_5m_tokens=2_000)
    expected = (1_000 * 0.10 + 200 * 0.50 + 10_000 * 0.01 + 2_000 * 0.125) / 1e6
    assert _HAIKU_PRICES.cost(usage) == pytest.approx(expected)


def test_prompts_over_100k_tokens_use_the_long_context_prices():
    usage = TokenUsage(input_tokens=90_000, output_tokens=100, cache_read_tokens=20_000)
    expected = (90_000 * 0.50 + 100 * 2.50 + 20_000 * 0.05) / 1e6  # cache reads count toward prompt length
    assert _HAIKU_PRICES.cost(usage) == pytest.approx(expected)


def test_unsplit_cache_writes_are_priced_at_the_1h_rate_and_missing_counts_are_not_invented():
    usage = TokenUsage(input_tokens=0, output_tokens=0, cache_write_unknown_ttl_tokens=50_000)
    assert _HAIKU_PRICES.cost(usage) == pytest.approx(50_000 * 0.20 / 1e6)
    assert _HAIKU_PRICES.cost(TokenUsage(input_tokens=None, output_tokens=5)) is None


# ------------------------------------------------------------------ budget
def _ledger(tmp_path) -> UsageLedger:
    return UsageLedger(tmp_path / "usage.db")


def test_ledger_stores_counts_but_never_text(tmp_path):
    ledger = _ledger(tmp_path)
    task = TaskMeter(category="medium:tool_intent")
    ledger.record(status="ok", model_requested="anthropic/claude-haiku-5-5", task=task,
                  model_served="claude-haiku-5-5", request_id="chatcmpl-msg_1",
                  usage=TokenUsage(input_tokens=10, output_tokens=2), cost_usd=0.000002,
                  tools_requested=["inspect_ui"])
    with sqlite3.connect(ledger.path) as conn:
        row = conn.execute("SELECT * FROM llm_calls").fetchone()
        columns = [d[0] for d in conn.execute("SELECT * FROM llm_calls").description]
    assert "claude-haiku-5-5" in row and "inspect_ui" in row and task.task_id in row
    assert not any(name in columns for name in ("prompt", "content", "messages", "reply"))


def test_budget_stops_on_each_limit(tmp_path):
    ledger = _ledger(tmp_path)
    now = datetime(2026, 10, 22, 12, 0)
    guard = BudgetGuard(BudgetLimits(max_calls_per_task=2, max_cost_per_task_usd=0.01, daily_budget_usd=0.05,
                                     monthly_budget_usd=0.08, max_prompt_tokens=5_000), ledger, clock=lambda: now)
    guard.check(TaskMeter(), 100, 0.001)  # fine

    with pytest.raises(LLMBudgetExceededError) as size:
        guard.check(TaskMeter(), 6_000, 0.001)
    assert size.value.limit == "max_prompt_tokens"
    with pytest.raises(LLMBudgetExceededError) as calls:
        guard.check(TaskMeter(calls=2), 100, 0.001)
    assert calls.value.limit == "max_calls_per_task"
    with pytest.raises(LLMBudgetExceededError) as per_task:
        guard.check(TaskMeter(cost_usd=0.0095), 100, 0.001)
    assert per_task.value.limit == "max_cost_per_task_usd"

    ledger.record(status="ok", model_requested="m", cost_usd=0.0495, now=now)
    with pytest.raises(LLMBudgetExceededError) as daily:
        guard.check(TaskMeter(), 100, 0.001)
    assert daily.value.limit == "daily_budget_usd" and "resets at midnight" in str(daily.value)

    ledger.record(status="ok", model_requested="m", cost_usd=0.04, now=datetime(2026, 10, 3, 9, 0))
    tomorrow = datetime(2026, 10, 23, 8, 0)
    guard_tomorrow = BudgetGuard(guard.limits, ledger, clock=lambda: tomorrow)
    with pytest.raises(LLMBudgetExceededError) as monthly:
        guard_tomorrow.check(TaskMeter(), 100, 0.001)
    assert monthly.value.limit == "monthly_budget_usd" and "2026-11-01" in str(monthly.value)


def test_unpriced_models_still_get_call_and_size_limits(tmp_path):
    guard = BudgetGuard(BudgetLimits(max_calls_per_task=1, daily_budget_usd=0), _ledger(tmp_path))
    guard.check(TaskMeter(), 100, None)  # no price -> cost limits can't apply
    with pytest.raises(LLMBudgetExceededError):
        guard.check(TaskMeter(calls=1), 100, None)


# ------------------------------------------------------------------ router
class PricedModel(LLMProvider):
    """Returns scripted responses/errors and reports usage like OmniRoute."""

    model = "anthropic/claude-haiku-5-5"

    def __init__(self, outcomes, pricing=_HAIKU_PRICES) -> None:
        self.outcomes = list(outcomes)
        self.pricing = pricing
        self.calls = 0

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def health_check(self) -> bool:
        return True


def _ok(text="done", input_tokens=1_000, output_tokens=50, tools=()):
    message = Message(role=Role.ASSISTANT, content=text,
                      tool_calls=[ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(tools)])
    return LLMResponse(message=message, finish_reason="tool_calls" if tools else "stop",
                       usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                       served_model="claude-haiku-5-5", request_id="chatcmpl-x")


def _router_with(model, tmp_path, **kw):
    pools = {tier: [(f"omniroute:{model.model}", model)] for tier in ModelTier}
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    router = TieredLLMRouter(pools, ledger=_ledger(tmp_path), sleep=fake_sleep, **kw)
    return router, sleeps


def _rows(router):
    return router._ledger.query("SELECT status, error_type, input_tokens, cost_usd, task_id FROM llm_calls ORDER BY id")


async def test_transient_errors_are_retried_on_the_same_model_with_backoff(tmp_path):
    model = PricedModel([LLMRateLimitError("429"), LLMUnavailableError("503"), _ok()])
    router, sleeps = _router_with(model, tmp_path, max_retries=2, retry_backoff_seconds=2.0)
    response = await router.chat([Message(role=Role.USER, content="hello")])
    assert response.message.content == "done" and model.calls == 3
    assert sleeps == [2.0, 4.0]
    assert [r[0] for r in _rows(router)] == ["error", "error", "ok"]


async def test_retries_are_bounded_and_the_real_error_surfaces(tmp_path):
    model = PricedModel([LLMRateLimitError("quota exhausted")] * 5)
    router, sleeps = _router_with(model, tmp_path, max_retries=1)
    with pytest.raises(LLMRateLimitError, match="quota exhausted"):
        await router.chat([Message(role=Role.USER, content="hello")])
    assert model.calls == 2 and len(sleeps) == 1


@pytest.mark.parametrize("error", [LLMBadRequestError("400"), LLMModelMismatchError("other model")])
async def test_permanent_errors_are_never_retried(tmp_path, error):
    model = PricedModel([error, _ok()])
    router, sleeps = _router_with(model, tmp_path, max_retries=3)
    with pytest.raises(type(error)):
        await router.chat([Message(role=Role.USER, content="hello")])
    assert model.calls == 1 and sleeps == []


async def test_model_mismatch_is_recorded_with_its_cost_and_ends_the_request(tmp_path):
    mismatch = LLMModelMismatchError("served by another model", usage=TokenUsage(input_tokens=500, output_tokens=10),
                                     served_model="claude-sonnet-5-5")
    haiku = PricedModel([mismatch, _ok("never sent")])
    router, sleeps = _router_with(haiku, tmp_path, max_retries=3)
    with pytest.raises(LLMModelMismatchError):
        await router.chat([Message(role=Role.USER, content="hello")])
    assert haiku.calls == 1 and sleeps == []
    (status, _, input_tokens, cost, _), = _rows(router)
    assert status == "model_mismatch" and input_tokens == 500 and cost > 0


async def test_budget_is_checked_before_every_request_including_retries(tmp_path):
    model = PricedModel([LLMRateLimitError("429"), _ok()])
    budget = BudgetGuard(BudgetLimits(max_calls_per_task=1), _ledger(tmp_path))
    router, _ = _router_with(model, tmp_path, max_retries=3, budget=budget)
    with task_meter():
        with pytest.raises(LLMBudgetExceededError, match="limit of 1 model calls"):
            await router.chat([Message(role=Role.USER, content="hello")])
    assert model.calls == 1


async def test_task_meter_accumulates_usage_and_category(tmp_path):
    model = PricedModel([_ok(input_tokens=2_000, output_tokens=100), _ok(input_tokens=3_000, output_tokens=20)])
    router, _ = _router_with(model, tmp_path)
    with task_meter() as meter:
        await router.chat([Message(role=Role.USER, content="open notepad")])
        await router.chat([Message(role=Role.USER, content="open notepad")])
    assert meter.calls == 2 and meter.input_tokens == 5_000 and meter.output_tokens == 120
    assert meter.cost_usd == pytest.approx((5_000 * 0.10 + 120 * 0.50) / 1e6)
    assert meter.category == "medium:tool_intent"
    assert {r[4] for r in _rows(router)} == {meter.task_id}


def test_prompt_estimate_is_pessimistic_and_counts_tools():
    messages = [Message(role=Role.USER, content="x" * 3_000)]
    tools = [ToolSpec(name="t", description="d" * 300, parameters={"type": "object"})]
    assert estimate_prompt_tokens(messages, None) >= 3_000 / 3.5
    assert estimate_prompt_tokens(messages, tools) > estimate_prompt_tokens(messages, None) + 100


# ------------------------------------------------------- OmniRoute + Haiku
def _patch_transport(monkeypatch, handler):
    import app.llm.omniroute as omniroute_module

    class _Patched(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(omniroute_module.httpx, "AsyncClient", _Patched)


def _haiku(**kw) -> OmniRouteProvider:
    return OmniRouteProvider(api_key="k", model="anthropic/claude-haiku-5-5", thinking="disabled",
                             max_output_tokens=4096, verify_served_model=True, pricing=_HAIKU_PRICES, **kw)


def _claude_reply(model="claude-haiku-5-5", usage=None, headers=None):
    body = {
        "id": "chatcmpl-msg_01", "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "408."}}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 12_000, "completion_tokens": 7, "total_tokens": 12_007,
            "prompt_tokens_details": {"cached_tokens": 10_000, "cache_creation_tokens": 1_500},
        },
    }
    return httpx.Response(200, json=body, headers=headers or {"X-OmniRoute-Provider": "anthropic"})


async def test_haiku_request_disables_thinking_sends_no_effort_or_sampling_and_opts_out_of_gateway_extras(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return _claude_reply()

    _patch_transport(monkeypatch, handler)
    response = await _haiku().chat([Message(role=Role.USER, content="17*24?")])
    body = seen["body"]
    assert body["model"] == "anthropic/claude-haiku-5-5"
    assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 4096
    for forbidden in ("reasoning_effort", "temperature", "top_p", "top_k", "output_config"):
        assert forbidden not in body
    assert seen["headers"]["x-omniroute-no-cache"] == "true"
    assert seen["headers"]["x-omniroute-no-memory"] == "true"
    assert response.served_model == "claude-haiku-5-5" and response.request_id == "chatcmpl-msg_01"
    assert response.usage == TokenUsage(input_tokens=2_000, output_tokens=7, cache_read_tokens=10_000,
                                        cache_write_unknown_ttl_tokens=1_500)


@pytest.mark.parametrize("served, provider_header", [
    ("claude-sonnet-5-5", "anthropic"),
    ("gpt-oss-120b", "nvidia"),
    ("claude-haiku-5-5", "claude"),
])
async def test_a_reply_from_another_model_or_provider_is_rejected(monkeypatch, served, provider_header):
    _patch_transport(monkeypatch, lambda request: _claude_reply(served, headers={"X-OmniRoute-Provider": provider_header}))
    with pytest.raises(LLMModelMismatchError, match="discarded") as info:
        await _haiku().chat([Message(role=Role.USER, content="hi")])
    assert info.value.usage is not None and info.value.served_model == served


def test_served_model_matching():
    assert served_model_matches("claude-haiku-5-5", "claude-haiku-5-5")
    assert served_model_matches("claude-haiku-5-5", "anthropic/claude-haiku-5-5")
    assert served_model_matches("claude-haiku-5-5", "claude-haiku-5-5-20261007")
    assert not served_model_matches("claude-haiku-5-5", "claude-haiku-5-5-fast")
    assert not served_model_matches("claude-haiku-5-5", "claude-haiku-4-5")
    assert not served_model_matches("claude-haiku-5-5", None)


def test_missing_usage_stays_missing():
    assert parse_usage({}) is None
    usage = parse_usage({"usage": {"completion_tokens": 3}})
    assert usage.input_tokens is None and usage.output_tokens == 3
    assert _HAIKU_PRICES.cost(usage) is None


def test_haiku_through_omniroute_is_refused_too(tmp_path):
    """Single-model policy: Claude Haiku 5.5 only via the direct Anthropic API --
    not even the same model through the OmniRoute gateway."""
    entry = {"provider": "omniroute", "model": "anthropic/claude-haiku-5-5", "thinking": "disabled"}
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"tiers": {"small": [entry], "medium": [entry], "large": [entry]}}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="may only use anthropic/claude-haiku-5-5"):
        build_llm_provider(Settings(_env_file=None, llm_provider="tiered", omniroute_api_key="k",
                                    llm_model_registry_path=path))


def test_registry_refuses_reasoning_effort_for_haiku_through_omniroute(tmp_path):
    entry = {"provider": "omniroute", "model": "anthropic/claude-haiku-5-5", "reasoning_effort": "low"}
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"tiers": {"small": [entry], "medium": [entry], "large": [entry]}}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="reasoning_effort"):
        load_model_registry(path)


def test_tool_schemas_drop_generated_titles_but_keep_title_properties():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    params = registry.tool_specs()[0].parameters
    assert "title" not in params and "title" not in params["properties"]["expression"]
    schema = {"type": "object", "title": "In", "properties": {"title": {"title": "Title", "type": "string"}}}
    assert compact_schema(schema) == {"type": "object", "properties": {"title": {"type": "string"}}}


# ------------------------------------------------------------------- agent
@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "m.db")


async def test_budget_stop_reports_what_was_already_done(memory_store, tmp_path):
    """The per-task call cap stops the loop before the 3rd request, and the
    reply says which tool actions had already run."""
    first, second = _ok(text="", tools=["calculator"]), _ok(text="", tools=["calculator"])
    first.message.tool_calls[0].arguments = {"expression": "2*2"}
    second.message.tool_calls[0].arguments = {"expression": "3*3"}
    model = PricedModel([first, second, _ok("never sent")])
    budget = BudgetGuard(BudgetLimits(max_calls_per_task=2), _ledger(tmp_path))
    router, _ = _router_with(model, tmp_path, budget=budget)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    errors: list[str] = []
    events = EventBus()
    events.subscribe(lambda e: errors.append(e.payload["message"]) if e.type == EventType.ERROR else None)
    agent = Agent(llm=router, registry=registry, memory=memory_store, events=events)
    reply = await agent.handle_user_message("compute things")
    assert "limit of 2 model calls" in reply and "Completed before stopping: calculator, calculator" in reply
    assert model.calls == 2 and errors == [reply]


async def test_an_identical_failing_call_is_not_rerun(memory_store, make_scripted_llm):
    bad = {"expression": "import os"}
    llm = make_scripted_llm([
        tool_call_response("calculator", bad),
        tool_call_response("calculator", bad),
        final_response("That expression isn't supported."),
    ])
    registry = ToolRegistry()
    calculator = CalculatorTool()
    runs = []
    original = calculator.execute

    async def counting(args):
        runs.append(args)
        return await original(args)

    calculator.execute = counting
    registry.register(calculator)
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())
    reply = await agent.handle_user_message("calc")
    assert reply == "That expression isn't supported." and len(runs) == 1
    second_result = json.loads(llm.calls[2][-1].content)
    assert second_result["success"] is False and "Not run again" in second_result["error"]


async def test_repeating_a_failed_step_stops_the_turn(memory_store, make_scripted_llm):
    llm = make_scripted_llm([tool_call_response("missing_tool", {})] * 6)
    agent = Agent(llm=llm, registry=ToolRegistry(), memory=memory_store, events=EventBus(), max_tool_iterations=15)
    reply = await agent.handle_user_message("loop")
    assert reply.startswith("I stopped") and len(llm.calls) == 4


async def test_tool_results_are_compact_and_keep_non_ascii_text(memory_store, make_scripted_llm):
    llm = make_scripted_llm([tool_call_response("remember_fact", {"content": "שלום עולם"}), final_response("ok")])
    from app.tools.memory import RememberFactTool

    registry = ToolRegistry()
    registry.register(RememberFactTool(memory_store))
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())
    await agent.handle_user_message("remember this")
    content = llm.calls[1][-1].content
    assert "שלום" in content and "\\u05e9" not in content
    assert '"error"' not in content and ": " not in content


async def test_truncated_empty_reply_is_reported_not_returned_blank(memory_store, make_scripted_llm):
    llm = make_scripted_llm([LLMResponse(message=Message(role=Role.ASSISTANT, content=""), finish_reason="length")])
    agent = Agent(llm=llm, registry=ToolRegistry(), memory=memory_store, events=EventBus())
    assert "output limit" in await agent.handle_user_message("write a novel")


async def test_task_summary_is_logged_once_per_turn(memory_store, tmp_path, monkeypatch):
    import app.agent.agent as agent_module

    summaries: list[dict] = []
    real_log_extra = agent_module.log_extra

    def capture(logger, level, message, **fields):
        if message == "llm.task_summary":
            summaries.append(fields)
        real_log_extra(logger, level, message, **fields)

    monkeypatch.setattr(agent_module, "log_extra", capture)
    model = PricedModel([_ok("hi")])
    router, _ = _router_with(model, tmp_path)
    agent = Agent(llm=router, registry=ToolRegistry(), memory=memory_store, events=EventBus())
    await agent.handle_user_message("hello")
    assert len(summaries) == 1
    assert summaries[0]["calls"] == 1 and summaries[0]["input_tokens"] == 1_000 and summaries[0]["cost_usd"] > 0


# ------------------------------------------------------------------ report
def test_usage_report_summarizes_spend_without_text(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "usage_report", Path(__file__).resolve().parents[1] / "scripts" / "usage_report.py")
    usage_report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(usage_report)
    ledger = _ledger(tmp_path)
    now = datetime(2026, 10, 22, 15, 0)
    task = TaskMeter(category="medium:tool_activity")
    ledger.record(status="ok", model_requested="anthropic/claude-haiku-5-5", model_served="claude-haiku-5-5",
                  task=task, usage=TokenUsage(input_tokens=2_000, output_tokens=50, cache_read_tokens=10_000),
                  cost_usd=0.000325, tools_requested=["read_screen_text"], now=now)
    text = usage_report.report(ledger, days=3, top=5, now=now)
    assert "$   0.0003" in text and "read_screen_text" in text and "medium:tool_activity" in text
    assert "anthropic/claude-haiku-5-5 -> claude-haiku-5-5: 1" in text
