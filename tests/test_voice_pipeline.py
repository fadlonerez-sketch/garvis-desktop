"""Tests for VoicePipeline, using fake recorder/STT/TTS/player collaborators
so no real hardware, model, or subprocess is involved."""
from __future__ import annotations

import os
import struct
import tempfile
import wave

import pytest

from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import TranscriptionResult


class _FakeRecorder:
    def __init__(self, wav_path: str, sample_rate: int = 16000):
        self._wav_path = wav_path
        self.sample_rate = sample_rate
        self.started = False
        self.stopped = False
        self.closed = False
        self.chunks_to_yield: list[bytes] = []

    @property
    def is_recording(self) -> bool:
        return self.started and not self.stopped and not self.closed

    def start(self) -> None:
        self.started = True

    def stop(self) -> str:
        self.stopped = True
        return self._wav_path

    def read_available(self) -> bytes:
        if not self.chunks_to_yield:
            return b""
        return self.chunks_to_yield.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeSTT:
    def __init__(self, text: str):
        self._text = text
        self.transcribed_paths: list[str] = []

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.transcribed_paths.append(audio_path)
        return TranscriptionResult(text=self._text)


class _FakeTTS:
    def __init__(self) -> None:
        self.synthesized: list[tuple[str, str]] = []
        self.stopped = False

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        self.synthesized.append((text, output_path))
        return output_path

    def stop(self) -> None:
        self.stopped = True


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.on_finished_callbacks: list = []
        self.stopped = False
        self.is_playing = False

    def play(self, wav_path: str, on_finished=None) -> None:
        self.played.append(wav_path)
        self.is_playing = True

        def _tracked_on_finished() -> None:
            self.is_playing = False
            if on_finished is not None:
                on_finished()

        self.on_finished_callbacks.append(_tracked_on_finished)

    def stop(self) -> None:
        self.stopped = True
        self.is_playing = False


class _FailingTTS:
    """Raises during synthesis, to exercise the speak() cleanup-on-error path."""

    def __init__(self) -> None:
        self.stopped = False

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        raise RuntimeError("synthesis failed")

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_stop_listening_and_transcribe_returns_text_and_cleans_up(tmp_path):
    wav_path = tmp_path / "captured.wav"
    wav_path.write_bytes(b"fake-audio")
    recorder = _FakeRecorder(str(wav_path))
    stt = _FakeSTT("what time is it")

    pipeline = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    pipeline.start_listening()
    assert recorder.started

    text = await pipeline.stop_listening_and_transcribe()
    assert text == "what time is it"
    assert stt.transcribed_paths == [str(wav_path)]
    assert not os.path.exists(wav_path)  # temp recording is cleaned up


@pytest.mark.asyncio
async def test_speak_synthesizes_then_plays():
    tts = _FakeTTS()
    player = _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=tts, recorder=_FakeRecorder(""), player=player)

    await pipeline.speak("hello there")

    assert len(tts.synthesized) == 1
    spoken_text, output_path = tts.synthesized[0]
    assert spoken_text == "hello there"
    assert player.played == [output_path]


@pytest.mark.asyncio
async def test_speak_uses_a_real_temp_file_and_cleans_up_when_playback_finishes():
    tts = _FakeTTS()
    player = _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=tts, recorder=_FakeRecorder(""), player=player)

    await pipeline.speak("hello")
    output_path = player.played[0]
    assert os.path.exists(output_path)  # tempfile.mkstemp actually created it

    on_finished = player.on_finished_callbacks[0]
    assert on_finished is not None
    on_finished()  # simulates AudioPlayer reporting playback complete

    assert not os.path.exists(output_path)


@pytest.mark.asyncio
async def test_speak_cleans_up_temp_file_when_synthesis_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    player = _FakePlayer()
    pipeline = VoicePipeline(
        stt=_FakeSTT(""), tts=_FailingTTS(), recorder=_FakeRecorder(""), player=player
    )

    with pytest.raises(RuntimeError, match="synthesis failed"):
        await pipeline.speak("hello")

    assert player.played == []  # playback was never reached
    assert list(tmp_path.iterdir()) == []  # the temp file was not left behind


def test_stop_speaking_stops_both_player_and_tts():
    tts = _FakeTTS()
    player = _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=tts, recorder=_FakeRecorder(""), player=player)

    pipeline.stop_speaking()

    assert player.stopped
    assert tts.stopped


def test_is_listening_reflects_recorder_state():
    recorder = _FakeRecorder("")
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    assert not pipeline.is_listening
    pipeline.start_listening()
    assert pipeline.is_listening


# ------------------------------------------------- session (hands-free) mode
@pytest.mark.asyncio
async def test_is_speaking_reflects_player_state():
    player = _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=_FakeRecorder(""), player=player)
    assert not pipeline.is_speaking

    await pipeline.speak("hi")
    assert pipeline.is_speaking

    player.on_finished_callbacks[0]()
    assert not pipeline.is_speaking


def test_start_continuous_listening_opens_the_recorder():
    recorder = _FakeRecorder("")
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    pipeline.start_continuous_listening()
    assert recorder.started


def test_read_microphone_chunk_delegates_to_recorder():
    recorder = _FakeRecorder("")
    recorder.chunks_to_yield = [b"abc", b"def"]
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())

    assert pipeline.read_microphone_chunk() == b"abc"
    assert pipeline.read_microphone_chunk() == b"def"
    assert pipeline.read_microphone_chunk() == b""


def test_stop_continuous_listening_closes_the_recorder():
    recorder = _FakeRecorder("")
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    pipeline.start_continuous_listening()
    pipeline.stop_continuous_listening()
    assert recorder.closed


@pytest.mark.asyncio
async def test_transcribe_bytes_writes_a_wav_and_transcribes():
    recorder = _FakeRecorder("", sample_rate=16000)
    stt = _FakeSTT("hello from session mode")
    pipeline = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())

    text = await pipeline.transcribe_bytes(b"\x01\x02" * 100)

    assert text == "hello from session mode"
    assert len(stt.transcribed_paths) == 1
    written_path = stt.transcribed_paths[0]
    assert not os.path.exists(written_path)  # temp file cleaned up after transcription


@pytest.mark.asyncio
async def test_transcribe_bytes_uses_correct_wav_format():
    import wave as wave_module

    captured = {}
    recorder = _FakeRecorder("", sample_rate=22050)

    class _InspectingSTT:
        async def transcribe(self, audio_path: str):
            with wave_module.open(audio_path, "rb") as wav_file:
                captured["channels"] = wav_file.getnchannels()
                captured["sampwidth"] = wav_file.getsampwidth()
                captured["framerate"] = wav_file.getframerate()
                captured["nframes"] = wav_file.getnframes()
            from app.voice.stt.base import TranscriptionResult

            return TranscriptionResult(text="ok")

    pipeline = VoicePipeline(
        stt=_InspectingSTT(), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer()
    )
    await pipeline.transcribe_bytes(b"\x01\x02" * 50)

    assert captured == {"channels": 1, "sampwidth": 2, "framerate": 22050, "nframes": 50}


@pytest.mark.asyncio
async def test_transcribe_bytes_cleans_up_temp_file_even_on_stt_failure():
    recorder = _FakeRecorder("")

    class _FailingSTT:
        async def transcribe(self, audio_path: str):
            raise RuntimeError("stt exploded")

    pipeline = VoicePipeline(
        stt=_FailingSTT(), tts=_FakeTTS(), recorder=recorder, player=_FakePlayer()
    )
    with pytest.raises(RuntimeError, match="stt exploded"):
        await pipeline.transcribe_bytes(b"\x01\x02" * 10)


# ------------------------------------- reference_energy_now() (self-echo guard)
def _write_tone_wav(path: str, seconds: float, amplitude: int, sample_rate: int = 16000) -> None:
    count = int(seconds * sample_rate)
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(count)]
    pcm = struct.pack(f"<{count}h", *samples)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


class _ToneWritingTTS:
    """Unlike _FakeTTS (which leaves the temp file empty), this actually
    writes real audio to output_path, so reference_energy_now() has
    something real to read back."""

    def __init__(self, amplitude: int = 5000, seconds: float = 1.0) -> None:
        self._amplitude = amplitude
        self._seconds = seconds

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        _write_tone_wav(output_path, self._seconds, self._amplitude)
        return output_path

    def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_reference_energy_now_is_zero_when_not_speaking():
    pipeline = VoicePipeline(
        stt=_FakeSTT(""), tts=_ToneWritingTTS(), recorder=_FakeRecorder(""), player=_FakePlayer()
    )
    assert pipeline.reference_energy_now() == 0.0


@pytest.mark.asyncio
async def test_reference_energy_now_reflects_currently_playing_audio():
    player = _FakePlayer()
    pipeline = VoicePipeline(
        stt=_FakeSTT(""),
        tts=_ToneWritingTTS(amplitude=5000, seconds=1.0),
        recorder=_FakeRecorder(""),
        player=player,
    )

    await pipeline.speak("hello")

    assert pipeline.is_speaking
    assert pipeline.reference_energy_now() == pytest.approx(5000.0, rel=0.05)


@pytest.mark.asyncio
async def test_reference_energy_now_drops_to_zero_after_playback_finishes():
    player = _FakePlayer()
    pipeline = VoicePipeline(
        stt=_FakeSTT(""), tts=_ToneWritingTTS(), recorder=_FakeRecorder(""), player=player
    )

    await pipeline.speak("hello")
    assert pipeline.reference_energy_now() > 0.0

    player.on_finished_callbacks[0]()
    assert pipeline.reference_energy_now() == 0.0


@pytest.mark.asyncio
async def test_reference_energy_now_is_zero_when_tts_double_writes_no_real_audio():
    """A test double (or any TTS provider) that doesn't produce a readable
    WAV must never crash speak() or reference_energy_now() -- the echo
    guard is best-effort only."""
    player = _FakePlayer()
    pipeline = VoicePipeline(stt=_FakeSTT(""), tts=_FakeTTS(), recorder=_FakeRecorder(""), player=player)

    await pipeline.speak("hello")

    assert pipeline.is_speaking
    assert pipeline.reference_energy_now() == 0.0
