"""Shared fakes for the mobile-interface tests: no microphone, speaker,
network or real STT/TTS/LLM is ever touched."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path

from pydantic import BaseModel

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import EventBus
from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.llm.base import LLMProvider, LLMResponse, Message, ToolSpec
from app.security.confirmations import ConfirmationManager
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import SpeechToTextProvider, TranscriptionResult
from app.voice.tts.base import TextToSpeechProvider
from tests.conftest import final_response


def make_wav(seconds: float = 1.0, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01" * int(seconds * sample_rate) * channels * sample_width)
    return buffer.getvalue()


class FakeSTT(SpeechToTextProvider):
    def __init__(self, text: str = "what time is it", error: str | None = None, delay: float = 0.0) -> None:
        self.text = text
        self.error = error
        self.delay = delay
        self.paths: list[str] = []
        self.existed_during_call: list[bool] = []

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.paths.append(audio_path)
        self.existed_during_call.append(Path(audio_path).exists())
        if self.delay:
            await asyncio.to_thread(time.sleep, self.delay)
        if self.error:
            raise SpeechToTextError(self.error)
        return TranscriptionResult(text=self.text)


class FakeTTS(TextToSpeechProvider):
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.texts: list[str] = []
        self.paths: list[str] = []

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        self.texts.append(text)
        self.paths.append(output_path)
        if self.error:
            raise TextToSpeechError(self.error)
        Path(output_path).write_bytes(make_wav(0.5, 22050))
        return output_path

    def stop(self) -> None:
        pass


class NoHardwareRecorder:
    sample_rate = 16000
    is_recording = False

    def start(self) -> None:
        raise AssertionError("the mobile path must never open the PC microphone")

    def stop(self) -> str:
        raise AssertionError("the mobile path must never use the PC recorder")

    def read_available(self) -> bytes:
        return b""

    def close(self) -> None:
        pass


class NoHardwarePlayer:
    is_playing = False

    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, wav_path: str, on_finished=None) -> None:
        self.played.append(wav_path)
        if on_finished:
            on_finished()

    def stop(self) -> None:
        pass


def make_voice(stt: FakeSTT | None = None, tts: FakeTTS | None = None) -> VoicePipeline:
    return VoicePipeline(
        stt=stt or FakeSTT(), tts=tts or FakeTTS(), recorder=NoHardwareRecorder(), player=NoHardwarePlayer()
    )


class GatedLLM(LLMProvider):
    """Answers each chat() with the next scripted response, but only after
    `release()` -- lets a test hold a turn "in flight"."""

    def __init__(self, responses: list[LLMResponse] | None = None, gated: bool = True) -> None:
        self._responses = list(responses or [])
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()
        self.calls: list[list[Message]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def release(self) -> None:
        self.gate.set()

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        self.calls.append(list(messages))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await self.gate.wait()
        finally:
            self.in_flight -= 1
        if self._responses:
            return self._responses.pop(0)
        return final_response(f"reply to: {messages[-1].content}")

    async def health_check(self) -> bool:
        return True


class _NoArgs(BaseModel):
    pass


class HighRiskTool(Tool):
    name = "high_risk_echo"
    description = "Does something consequential (test double)."
    risk_level = RiskLevel.HIGH
    input_model = _NoArgs

    def __init__(self) -> None:
        self.executions = 0

    async def execute(self, args: BaseModel) -> ToolResult:
        self.executions += 1
        return ToolResult.ok("did the risky thing")


def make_agent(
    tmp_path,
    llm: LLMProvider,
    events: EventBus | None = None,
    tools: list[Tool] | None = None,
    confirmation_timeout: float = 5.0,
) -> Agent:
    registry = ToolRegistry()
    for tool in tools or []:
        registry.register(tool)
    return Agent(
        llm=llm,
        registry=registry,
        memory=SQLiteMemoryStore(tmp_path / "mobile-test.db"),
        events=events or EventBus(),
        confirmations=ConfirmationManager(default_timeout_seconds=confirmation_timeout),
    )


async def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)
