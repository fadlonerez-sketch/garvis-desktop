from __future__ import annotations

from collections.abc import Callable

import pytest

from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSpec


@pytest.fixture(autouse=True)
def _no_real_phone_server(monkeypatch):
    """Tests never open the real iPhone port because of the developer's .env
    (GARVIS_MOBILE_ENABLED=true). An environment variable outranks .env;
    tests that want the server pass mobile_enabled=True explicitly."""
    monkeypatch.setenv("GARVIS_MOBILE_ENABLED", "false")


@pytest.fixture(autouse=True)
def _no_real_usage_ledger(monkeypatch, tmp_path):
    """Tests never write rows into the developer's real data/usage.db (or
    count its spend against their budget checks)."""
    monkeypatch.setenv("GARVIS_LLM_USAGE_DB_PATH", str(tmp_path / "usage.db"))


@pytest.fixture(autouse=True)
def _no_real_data_dir(monkeypatch, tmp_path):
    """Tests never read or write the developer's real data/ folder (saved
    protocols, the task history, UI state) through Settings.data_dir."""
    monkeypatch.setenv("GARVIS_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def _no_real_anthropic_key(monkeypatch):
    """A real Anthropic key -- in the environment or in the developer's .env,
    which some bridge tests read -- must never reach a test (no accidental
    live, billed requests). An empty environment variable outranks .env;
    tests that need the primary provider pass a fake key explicitly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    for name in ("GARVIS_ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


class ScriptedLLMProvider(LLMProvider):
    """Returns a pre-scripted sequence of responses, one per `chat()` call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLMProvider ran out of scripted responses")
        return self._responses.pop(0)

    async def health_check(self) -> bool:
        return True


def tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)],
        ),
        finish_reason="tool_calls",
    )


def final_response(text: str) -> LLMResponse:
    return LLMResponse(message=Message(role=Role.ASSISTANT, content=text), finish_reason="stop")


@pytest.fixture
def make_scripted_llm() -> Callable[[list[LLMResponse]], ScriptedLLMProvider]:
    return ScriptedLLMProvider
