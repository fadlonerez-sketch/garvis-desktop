"""Tests for tiered routing: classify_tier(), TieredLLMRouter's single-model
policy (one model per tier, same-model retries only, failures reported --
never another model), the 'tiered' build, and the Agent on top of it."""

from __future__ import annotations

import json
import logging

import pytest

from app.agent.agent import Agent
from app.agent.build import build_llm_provider
from app.agent.memory import SQLiteMemoryStore
from app.core.config import Settings
from app.core.events import EventBus, EventType
from app.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBillingError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall
from app.llm.registry import ModelTier
from app.llm.router import TieredLLMRouter, classify_tier
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from tests.conftest import final_response, tool_call_response


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


def _user(text: str) -> list[Message]:
    return [Message(role=Role.SYSTEM, content="sys"), Message(role=Role.USER, content=text)]


# ------------------------------------------------------------ classify_tier
@pytest.mark.parametrize("text", ["What is the capital of France?", "hello", "17 * 24", "define entropy"])
def test_simple_requests_are_small(text):
    assert classify_tier(_user(text)) == (ModelTier.SMALL, "simple_request")


@pytest.mark.parametrize(
    "text, reason",
    [
        ("Open WhatsApp", "tool_intent"),
        ("open spotify", "uncertain"),
        ("search for the best pizza near me", "tool_intent"),
        ("research the current price of bitcoin", "tool_intent"),
        ("take a screenshot", "tool_intent"),
        ("remember that my name is Alex", "uncertain"),
    ],
)
def test_ordinary_and_tool_requests_are_medium(text, reason):
    assert classify_tier(_user(text)) == (ModelTier.MEDIUM, reason)


@pytest.mark.parametrize(
    "text",
    ["Why is the sky blue?", "compare postgres and mysql", "explain this traceback", "debug my code step by step"],
)
def test_complex_requests_are_large(text):
    assert classify_tier(_user(text)) == (ModelTier.LARGE, "complex_request")


def test_long_requests_are_large():
    assert classify_tier(_user("word " * 60)) == (ModelTier.LARGE, "long_request")


def test_tier_is_stable_across_a_tool_loop():
    """After a tool result, the turn must stay on the tier of its user message."""
    turn = _user("compare the weather in Paris and Rome")
    assert classify_tier(turn)[0] == ModelTier.LARGE
    turn += [
        Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id="1", name="web_search", arguments={})]),
        Message(role=Role.TOOL, content="{}", tool_call_id="1", name="web_search"),
    ]
    assert classify_tier(turn)[0] == ModelTier.LARGE
    simple_then_tool = _user("hello") + turn[2:]
    assert classify_tier(simple_then_tool) == (ModelTier.MEDIUM, "tool_activity")


# ------------------------------------------------------------------ router
class FakeModel(LLMProvider):
    def __init__(
        self, name: str, errors: list[Exception] | None = None, healthy: bool = True, model: str | None = None
    ) -> None:
        self.name = name
        self.model = model or name
        self.errors = list(errors or [])
        self.calls = 0
        self.healthy = healthy

    async def chat(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return final_response(f"from {self.name}")

    async def health_check(self) -> bool:
        return self.healthy


def _router(small=None, medium=None, large=None, **kw):
    pools = {
        ModelTier.SMALL: [((small or FakeModel("small-1")).name, small or FakeModel("small-1"))],
        ModelTier.MEDIUM: [((medium or FakeModel("medium-1")).name, medium or FakeModel("medium-1"))],
        ModelTier.LARGE: [((large or FakeModel("large-1")).name, large or FakeModel("large-1"))],
    }
    return TieredLLMRouter(pools, **kw)


@pytest.fixture
def router_logs(caplog):
    logging.getLogger("garvis").propagate = True
    caplog.set_level(logging.INFO, logger="garvis.llm.router")
    yield caplog
    logging.getLogger("garvis").propagate = False


def _events(caplog, name):
    return [r for r in caplog.records if r.getMessage() == name]


@pytest.mark.parametrize(
    "text, expected", [("hello", "small-1"), ("open whatsapp", "medium-1"), ("why is the sky blue", "large-1")]
)
async def test_each_tier_routes_to_its_one_model(text, expected):
    router = _router()
    response = await router.chat(_user(text))
    assert response.message.content == f"from {expected}"
    assert router.last_selection["model"] == expected


async def test_selection_is_logged(router_logs):
    await _router().chat(_user("hello"))
    [record] = _events(router_logs, "llm.router.tier_selection")
    assert record.extra_fields == {"tier": "small", "reason": "simple_request", "model": "small-1", "attempt": 1}


def test_router_requires_every_tier():
    with pytest.raises(ValueError, match="empty"):
        TieredLLMRouter({ModelTier.SMALL: [("a", FakeModel("a"))], ModelTier.MEDIUM: [], ModelTier.LARGE: []})


def test_a_tier_with_a_second_model_is_refused():
    """No fallback list can be configured: two models in one tier is an error."""
    pools = {tier: [("haiku", FakeModel("haiku"))] for tier in ModelTier}
    pools[ModelTier.MEDIUM] = [("haiku", FakeModel("haiku")), ("gemini", FakeModel("gemini"))]
    with pytest.raises(LLMError, match="2 models configured"):
        TieredLLMRouter(pools)


def test_required_model_is_checked_at_construction():
    pools = {tier: [("x", FakeModel("x", model="claude-haiku-5-5"))] for tier in ModelTier}
    pools[ModelTier.LARGE] = [("g", FakeModel("g", model="gemini-3.8-flash"))]
    with pytest.raises(LLMError, match="only claude-haiku-5-5 may answer.*gemini-3.8-flash"):
        TieredLLMRouter(pools, required_model="claude-haiku-5-5")


async def test_a_pool_swapped_at_runtime_is_blocked_before_any_request():
    haiku = FakeModel("haiku", model="claude-haiku-5-5")
    router = _router(
        small=haiku,
        medium=FakeModel("m", model="claude-haiku-5-5"),
        large=FakeModel("l", model="claude-haiku-5-5"),
        required_model="claude-haiku-5-5",
    )
    intruder = FakeModel("gemini", model="gemini-3.1-flash-lite")
    router.pools[ModelTier.SMALL][:] = [("gemini", intruder)]
    with pytest.raises(LLMError, match="Blocked"):
        await router.chat(_user("hello"))
    router.pools[ModelTier.SMALL][:] = [("haiku", haiku), ("gemini", intruder)]
    with pytest.raises(LLMError, match="Blocked"):
        await router.chat(_user("hello"))
    assert intruder.calls == 0 and haiku.calls == 0


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("Anthropic rate limit reached"),
        LLMUnavailableError("Anthropic API error: 529 Overloaded"),
        LLMTimeoutError("Claude didn't answer within 45s."),
        LLMAuthenticationError("Anthropic rejected the API key (401)."),
        LLMBadRequestError("Claude rejected the request (400)."),
        LLMBillingError("Anthropic API billing problem: credit balance too low"),
    ],
)
async def test_any_failure_is_raised_and_no_other_tier_model_is_called(router_logs, error):
    small, medium, large = FakeModel("small-1", errors=[error] * 5), FakeModel("medium-1"), FakeModel("large-1")
    router = _router(small=small, medium=medium, large=large)
    with pytest.raises(type(error), match=str(error)[:20]):
        await router.chat(_user("hello"))
    assert small.calls == 1 and medium.calls == 0 and large.calls == 0
    [failed] = _events(router_logs, "llm.router.failed")
    assert failed.extra_fields["note"] == "no other model is used"


async def test_retries_target_only_the_same_model():
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    small = FakeModel("small-1", errors=[LLMRateLimitError("429"), LLMUnavailableError("529")])
    medium, large = FakeModel("medium-1"), FakeModel("large-1")
    router = _router(small=small, medium=medium, large=large, max_retries=2, sleep=fake_sleep)
    assert (await router.chat(_user("hello"))).message.content == "from small-1"
    assert small.calls == 3 and sleeps == [2.0, 4.0]
    assert medium.calls == 0 and large.calls == 0


async def test_a_timeout_is_never_resent():
    small = FakeModel("small-1", errors=[LLMTimeoutError("slow"), LLMTimeoutError("slow")])
    router = _router(small=small, max_retries=3)
    with pytest.raises(LLMTimeoutError):
        await router.chat(_user("hello"))
    assert small.calls == 1


async def test_health_check_uses_the_medium_model():
    assert await _router(medium=FakeModel("m", healthy=True)).health_check()
    assert not await _router(medium=FakeModel("m", healthy=False)).health_check()


async def test_agent_runs_transparently_on_tiered_router(memory_store, make_scripted_llm):
    """Tool loop end-to-end: user -> MEDIUM model -> tool -> same model -> answer."""
    medium = make_scripted_llm([tool_call_response("calculator", {"expression": "17*24"}), final_response("408.")])
    small = make_scripted_llm([])
    router = TieredLLMRouter(
        {
            ModelTier.SMALL: [("small", small)],
            ModelTier.MEDIUM: [("medium", medium)],
            ModelTier.LARGE: [("large", make_scripted_llm([]))],
        }
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = Agent(llm=router, registry=registry, memory=memory_store, events=EventBus())
    reply = await agent.handle_user_message("use the calculator tool for 17 times 24")
    assert reply == "408."
    assert len(medium.calls) == 2 and small.calls == []
    tool_msg = medium.calls[1][-1]
    assert tool_msg.role == Role.TOOL and json.loads(tool_msg.content)["data"] == 408


async def test_agent_reports_the_failure_and_stops(memory_store):
    events = EventBus()
    errors: list[str] = []
    events.subscribe(lambda e: errors.append(e.payload["message"]) if e.type == EventType.ERROR else None)
    medium = FakeModel("medium-1", errors=[LLMUnavailableError("Anthropic API error for claude-haiku-5-5: 529")])
    others = FakeModel("small-1"), FakeModel("large-1")
    agent = Agent(
        llm=_router(small=others[0], medium=medium, large=others[1]),
        registry=ToolRegistry(),
        memory=memory_store,
        events=events,
    )
    reply = await agent.handle_user_message("open whatsapp")
    assert reply == "Anthropic API error for claude-haiku-5-5: 529" and errors == [reply]
    assert medium.calls == 1 and all(m.calls == 0 for m in others)


# ------------------------------------------------------------------- build
def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


_HAIKU = {
    ModelTier.SMALL: "anthropic:claude-haiku-5-5@low",
    ModelTier.MEDIUM: "anthropic:claude-haiku-5-5@medium",
    ModelTier.LARGE: "anthropic:claude-haiku-5-5@high",
}


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"anthropic_api_key": "sk-test"},
        {
            "anthropic_api_key": "sk-test",
            "omniroute_api_key": "k",
            "gemini_api_key": "g",
            "groq_api_key": "q",
            "openrouter_api_key": "o",
        },
    ],
)
def test_build_serves_every_tier_with_haiku_only_whatever_keys_exist(keys):
    router = build_llm_provider(_settings(llm_provider="tiered", **keys))
    assert isinstance(router, TieredLLMRouter) and router.required_model == "claude-haiku-5-5"
    for tier in ModelTier:
        ((label, provider),) = router.pools[tier]
        assert label == _HAIKU[tier] and isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-haiku-5-5"


async def test_without_an_anthropic_key_requests_fail_clearly_and_nothing_else_is_called(memory_store):
    router = build_llm_provider(_settings(llm_provider="tiered", omniroute_api_key="k", gemini_api_key="g"))
    agent = Agent(llm=router, registry=ToolRegistry(), memory=memory_store, events=EventBus())
    reply = await agent.handle_user_message("hello")
    assert "ANTHROPIC_API_KEY is not set" in reply and "No other model is used" in reply


@pytest.mark.parametrize(
    "entries",
    [
        [
            {"provider": "anthropic", "model": "claude-haiku-5-5"},
            {"provider": "omniroute", "model": "gemini/gemini-3.1-flash-lite", "enabled": False},
        ],
        [{"provider": "omniroute", "model": "gemini/gemini-3.1-flash-lite"}],
        [{"provider": "anthropic", "model": "claude-sonnet-5-5"}],
        [
            {"provider": "anthropic", "model": "claude-haiku-5-5"},
            {"provider": "anthropic", "model": "claude-haiku-5-5"},
        ],
        [{"provider": "anthropic", "model": "claude-haiku-5-5", "enabled": False}],
    ],
)
def test_build_refuses_any_registry_that_could_reach_another_model(tmp_path, entries):
    path = tmp_path / "models.json"
    ok = [{"provider": "anthropic", "model": "claude-haiku-5-5"}]
    path.write_text(json.dumps({"tiers": {"small": ok, "medium": entries, "large": ok}}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="may only use anthropic/claude-haiku-5-5"):
        build_llm_provider(
            _settings(
                llm_provider="tiered", anthropic_api_key="sk-test", omniroute_api_key="k", llm_model_registry_path=path
            )
        )


@pytest.mark.parametrize("mode", ["gemini", "groq", "openrouter", "omniroute"])
def test_other_single_provider_modes_are_refused(mode):
    with pytest.raises(ConfigurationError, match="only Claude Haiku 5.5"):
        build_llm_provider(
            _settings(
                llm_provider=mode,
                gemini_api_key="g",
                groq_api_key="q",
                openrouter_api_key="o",
                openrouter_model="x/y",
                omniroute_api_key="k",
            )
        )


def test_context_budget_defaults():
    assert _settings(llm_provider="tiered").effective_max_context_tokens() == 12000
    assert _settings(llm_provider="tiered", max_context_tokens=5000).effective_max_context_tokens() == 5000


def test_agent_uses_the_configured_context_budget(memory_store):
    agent = Agent(
        llm=FakeModel("x"), registry=ToolRegistry(), memory=memory_store, events=EventBus(), max_context_tokens=12000
    )
    assert agent._context._max_tokens == 12000
    default = Agent(llm=FakeModel("x"), registry=ToolRegistry(), memory=memory_store, events=EventBus())
    assert default._context._max_tokens == 1200
