"""Backend pieces behind the dashboard: source metadata from real tool
results (app/tools/sources.py + the agent's TOOL_FINISHED payload), the
display-only playback listener (VoicePipeline.add_playback_listener), the
bridge relaying both, and the read-only usage summary (app/ui/usage_view.py).
All with fakes -- no audio device, model or network."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.llm.usage import TaskMeter, TokenUsage, UsageLedger
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.sources import extract_sources, publisher_of
from app.ui.agent_bridge import AgentBridge
from app.ui.usage_view import read_usage_summary
from app.voice.pipeline import VoicePipeline
from tests.conftest import final_response, tool_call_response
from tests.test_voice_pipeline import _FailingTTS, _FakePlayer, _FakeRecorder, _FakeSTT, _FakeTTS


# ------------------------------------------------------------ extraction
def test_web_search_results_keep_their_exact_titles_and_urls():
    data = [
        {"title": "  Artemis  II crew named ", "url": "https://www.nasa.gov/artemis", "snippet": "The crew..."},
        {"title": "Second", "url": "https://example.org/b", "snippet": ""},
    ]
    sources = extract_sources("web_search", data)
    assert [s["url"] for s in sources] == ["https://www.nasa.gov/artemis", "https://example.org/b"]
    assert sources[0] == {"tool": "web_search", "kind": "search_result", "title": "Artemis II crew named",
                          "url": "https://www.nasa.gov/artemis", "publisher": "nasa.gov", "snippet": "The crew..."}


def test_web_research_lists_read_pages_first_and_dedupes_their_search_listing():
    data = {
        "sources": [{"title": "Read page", "url": "https://a.example/x", "excerpt": "read text"}],
        "results": [
            {"title": "Read page (listing)", "url": "https://a.example/x", "snippet": "listing"},
            {"title": "Only listed", "url": "https://b.example/y", "snippet": "snip"},
        ],
    }
    sources = extract_sources("web_research", data)
    assert [(s["kind"], s["title"]) for s in sources] == [("read", "Read page"), ("search_result", "Only listed")]


def test_page_readers_report_the_page_they_read():
    sources = extract_sources("fetch_webpage_text", {"url": "https://news.example/a", "text": "Body " * 200})
    assert len(sources) == 1 and sources[0]["kind"] == "read"
    assert len(sources[0]["snippet"]) <= 320


@pytest.mark.parametrize("tool, data", [
    ("calculator", 42),
    ("web_search", [{"title": "x", "url": "javascript:alert(1)"}, {"title": "y", "url": "file:///C:/x"}]),
    ("web_research", {"sources": "not a list"}),
    ("web_search", None),
    ("open_website", {"url": "https://example.com"}),  # navigation, not an information source
])
def test_no_sources_are_invented(tool, data):
    assert extract_sources(tool, data) == []


def test_publisher_strips_www():
    assert publisher_of("https://www.bbc.co.uk/news") == "bbc.co.uk"


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "memory.db")


class _SearchInput(BaseModel):
    query: str


class _FakeWebSearch(Tool):
    name = "web_search"
    description = "fake search"
    risk_level = RiskLevel.LOW
    input_model = _SearchInput

    async def execute(self, args: _SearchInput) -> ToolResult:
        return ToolResult.ok([{"title": "Result", "url": "https://example.com/r", "snippet": "s"}])


@pytest.mark.asyncio
async def test_tool_finished_event_carries_the_sources_but_the_model_sees_the_same_result(
        memory_store, make_scripted_llm):
    registry = ToolRegistry()
    registry.register(_FakeWebSearch())
    llm = make_scripted_llm([tool_call_response("web_search", {"query": "q"}), final_response("done")])
    events = EventBus()
    seen: list[AgentEvent] = []
    events.subscribe(seen.append)
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=events)

    await agent.handle_user_message("search")

    finished = [e for e in seen if e.type == EventType.TOOL_FINISHED]
    assert finished[0].payload["sources"] == [{"tool": "web_search", "kind": "search_result", "title": "Result",
                                               "url": "https://example.com/r", "publisher": "example.com",
                                               "snippet": "s"}]
    tool_message = next(m for m in agent._context.messages if m.role.value == "tool")
    assert "publisher" not in tool_message.content  # display metadata never reaches the model


def test_bridge_relays_sources_only_when_present():
    fake = Mock()
    payload = {"name": "web_search", "success": True, "error": None,
               "sources": [{"url": "https://example.com", "title": "t"}]}
    AgentBridge._on_event(fake, AgentEvent(type=EventType.TOOL_FINISHED, payload=payload))
    fake.tool_finished.emit.assert_called_once_with("web_search", True, "")
    fake.tool_sources.emit.assert_called_once_with("web_search", payload["sources"])

    fake = Mock()
    AgentBridge._on_event(fake, AgentEvent(type=EventType.TOOL_FINISHED,
                                           payload={"name": "calculator", "success": True}))
    fake.tool_sources.emit.assert_not_called()


# -------------------------------------------------------- playback listener
def _pipeline(tts=None, player=None) -> tuple[VoicePipeline, _FakePlayer, list[bool]]:
    player = player or _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=tts or _FakeTTS(), recorder=_FakeRecorder(""), player=player)
    events: list[bool] = []
    pipeline.add_playback_listener(events.append)
    return pipeline, player, events


@pytest.mark.asyncio
async def test_playback_listener_follows_the_real_clip_lifecycle():
    pipeline, player, events = _pipeline()
    await pipeline.speak("hello")
    assert events == [True]  # handed to the player; still playing
    player.on_finished_callbacks[0]()  # the player reports the end
    assert events == [True, False]


@pytest.mark.asyncio
async def test_a_replaced_clip_does_not_end_the_speaking_state_of_its_successor():
    pipeline, player, events = _pipeline()
    await pipeline.speak("one")
    await pipeline.speak("two")
    player.on_finished_callbacks[0]()  # the first clip's late end
    assert events == [True, True]
    player.on_finished_callbacks[1]()
    assert events == [True, True, False]


@pytest.mark.asyncio
async def test_failed_synthesis_never_reports_speaking():
    pipeline, _player, events = _pipeline(tts=_FailingTTS())
    with pytest.raises(RuntimeError):
        await pipeline.speak("hello")
    assert events == []


@pytest.mark.asyncio
async def test_a_broken_listener_cannot_break_playback():
    pipeline, player, _events = _pipeline()

    def broken(_playing: bool) -> None:
        raise RuntimeError("listener bug")

    pipeline.add_playback_listener(broken)
    await pipeline.speak("hello")
    assert len(player.played) == 1


# ------------------------------------------------------------ usage summary
def test_usage_summary_reads_the_real_ledger(tmp_path):
    db = tmp_path / "usage.db"
    ledger = UsageLedger(db)
    now = datetime(2026, 10, 23, 15, 0, 0)
    task = TaskMeter()
    ledger.record(status="ok", model_requested="claude-haiku-5-5", provider="anthropic", tier="small", task=task,
                  usage=TokenUsage(input_tokens=1000, output_tokens=50, cache_read_tokens=400),
                  cost_usd=0.002, latency_ms=900, now=now)
    ledger.record(status="error", model_requested="claude-haiku-5-5", provider="anthropic", tier="small",
                  error_type="LLMRateLimitError", now=now)
    ledger.record(status="ok", model_requested="claude-haiku-5-5", cost_usd=0.5, now=datetime(2026, 10, 2, 9))

    summary = read_usage_summary(db, now=now)

    assert summary["available"] is True
    assert summary["today"]["calls"] == 2 and summary["today"]["ok"] == 1 and summary["today"]["errors"] == 1
    assert summary["today"]["input_tokens"] == 1000 and summary["today"]["cache_read_tokens"] == 400
    assert summary["today"]["cost_usd"] == pytest.approx(0.002)
    assert summary["month"]["cost_usd"] == pytest.approx(0.502)
    assert summary["recent"][0]["status"] == "error"


def test_missing_ledger_is_reported_and_not_created(tmp_path):
    db = tmp_path / "usage.db"
    summary = read_usage_summary(db)
    assert summary["available"] is False and "No model requests" in summary["reason"]
    assert not db.exists()


# ------------------------------------------------------------- system info
def test_system_info_reads_real_settings_and_never_the_key_itself():
    from types import SimpleNamespace

    from app.core.config import Settings

    secret = "sk-ant-test-not-a-real-key"
    fake = SimpleNamespace(_settings=Settings(anthropic_api_key=secret))
    info = AgentBridge.system_info(fake)
    assert info["model"] == "claude-haiku-5-5" and info["provider"] == "Anthropic API"
    assert info["api_key_configured"] is True
    assert secret not in repr(info)
    assert AgentBridge.system_info(SimpleNamespace(_settings=Settings(anthropic_api_key="")))["api_key_configured"] is False


@pytest.mark.asyncio
async def test_voice_level_reads_the_playing_clip_and_is_zero_otherwise(tmp_path):
    import struct
    import wave
    from types import SimpleNamespace

    class _ToneTTS(_FakeTTS):
        async def synthesize_to_file(self, text: str, output_path: str) -> str:
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(struct.pack("<h", 3000) * 16000)
            return output_path

    pipeline, player, _events = _pipeline(tts=_ToneTTS())
    fake = SimpleNamespace(_voice=pipeline)
    assert AgentBridge.voice_level(fake) == 0.0
    await pipeline.speak("tone")
    assert AgentBridge.voice_level(fake) == pytest.approx(3000, rel=0.01)
    player.on_finished_callbacks[0]()
    assert AgentBridge.voice_level(fake) == 0.0
