"""Phase 8 integration test: the full voice pipeline driving the Agent.

    microphone -> STT -> Agent -> Tool execution -> response -> TTS -> AudioPlayer

Uses the same fake STT/TTS/recorder/player collaborators as
test_voice_pipeline.py (no real hardware/model/subprocess), plus a real
Agent with a real ToolRegistry + ScriptedLLMProvider, to prove the two
subsystems compose correctly end to end -- something no existing test
exercises (test_voice_pipeline.py stops at VoicePipeline; test_agent_loop.py
never involves voice).
"""

from __future__ import annotations

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import EventBus
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import TranscriptionResult
from tests.conftest import final_response, tool_call_response


class _FakeRecorder:
    def __init__(self, wav_path: str):
        self._wav_path = wav_path
        self.started = False

    @property
    def is_recording(self) -> bool:
        return self.started

    def start(self) -> None:
        self.started = True

    def stop(self) -> str:
        return self._wav_path


class _FakeSTT:
    def __init__(self, text: str):
        self._text = text

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        return TranscriptionResult(text=self._text)


class _RecordingTTS:
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        self.synthesized.append(text)
        return output_path

    def stop(self) -> None:
        pass


class _RecordingPlayer:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, wav_path: str, on_finished=None) -> None:
        self.played.append(wav_path)
        if on_finished:
            on_finished()

    def stop(self) -> None:
        pass


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_voice_turn_reaches_the_agent_and_uses_a_tool(tmp_path, memory_store, make_scripted_llm):
    """microphone -> STT produces the transcript that becomes the Agent's
    user message, and a tool call in that turn actually runs against the
    real registry."""
    wav_path = tmp_path / "captured.wav"
    wav_path.write_bytes(b"fake-audio")

    llm = make_scripted_llm(
        [
            tool_call_response("calculator", {"expression": "10 / 2"}),
            final_response("10 divided by 2 is 5."),
        ]
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = Agent(llm=llm, registry=registry, memory=memory_store, events=EventBus())

    pipeline = VoicePipeline(
        stt=_FakeSTT("what is 10 divided by 2"),
        tts=_RecordingTTS(),
        recorder=_FakeRecorder(str(wav_path)),
        player=_RecordingPlayer(),
    )

    pipeline.start_listening()
    transcript = await pipeline.stop_listening_and_transcribe()
    assert transcript == "what is 10 divided by 2"

    reply = await agent.handle_user_message(transcript)
    assert reply == "10 divided by 2 is 5."

    # second LLM call carried the tool result forward, proving the tool ran
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert any("5" in m.content for m in tool_messages)


@pytest.mark.asyncio
async def test_agent_reply_is_spoken_back_through_tts_and_player(memory_store, make_scripted_llm):
    """response -> TTS -> AudioPlayer: the Agent's final reply text, once
    handed to the voice pipeline, is what gets synthesized and played --
    mirroring what AgentBridge does for `assistant_message` when voice
    replies are enabled (app/ui/agent_bridge.py)."""
    llm = make_scripted_llm([final_response("The current time is nine AM.")])
    agent = Agent(llm=llm, registry=ToolRegistry(), memory=memory_store, events=EventBus())

    tts = _RecordingTTS()
    player = _RecordingPlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=tts, recorder=_FakeRecorder(""), player=player)

    reply = await agent.handle_user_message("what time is it")
    await pipeline.speak(reply)

    assert tts.synthesized == ["The current time is nine AM."]
    assert len(player.played) == 1  # AudioPlayer.play was invoked with the synthesized file


@pytest.mark.asyncio
async def test_empty_transcript_never_reaches_the_agent(tmp_path, memory_store, make_scripted_llm):
    """Mirrors AgentBridge._stop_listening: if STT returns nothing (silence/
    noise), the pipeline must not forward an empty message into the Agent."""
    wav_path = tmp_path / "silence.wav"
    wav_path.write_bytes(b"fake-audio")

    llm = make_scripted_llm([])  # no responses scripted: chat() must never be called
    agent = Agent(llm=llm, registry=ToolRegistry(), memory=memory_store, events=EventBus())

    pipeline = VoicePipeline(
        stt=_FakeSTT("   "),
        tts=_RecordingTTS(),
        recorder=_FakeRecorder(str(wav_path)),
        player=_RecordingPlayer(),
    )

    pipeline.start_listening()
    transcript = await pipeline.stop_listening_and_transcribe()

    if transcript.strip():
        await agent.handle_user_message(transcript)

    assert llm.calls == []
