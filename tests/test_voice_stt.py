"""Tests for the microphone recorder and the faster-whisper STT provider.

Both external libraries (`sounddevice`, `faster_whisper`) are faked via
sys.modules injection rather than exercising real hardware/models, so
these tests are hermetic and fast regardless of what's installed.
"""
from __future__ import annotations

import sys
import wave

import pytest

from app.core.exceptions import SpeechToTextError
from app.voice.stt.faster_whisper_provider import FasterWhisperProvider
from app.voice.stt.recorder import MicrophoneRecorder


# --------------------------------------------------------------- recorder
class _FakeRawInputStream:
    """Stands in for sounddevice.RawInputStream, invoking the callback once
    with a fixed chunk of fake int16 audio as soon as start() is called."""

    fail_on_start = False

    def __init__(self, samplerate, channels, dtype, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._callback = callback
        self.started = False
        self.closed = False

    def start(self):
        if _FakeRawInputStream.fail_on_start:
            raise RuntimeError("no default input device")
        self.started = True
        self._callback(b"\x01\x02" * 50, 50, None, None)

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class _FakeSoundDevice:
    RawInputStream = _FakeRawInputStream


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    _FakeRawInputStream.fail_on_start = False
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSoundDevice())
    yield


def test_recorder_start_stop_produces_valid_wav():
    recorder = MicrophoneRecorder(sample_rate=16000)
    assert not recorder.is_recording

    recorder.start()
    assert recorder.is_recording

    path = recorder.stop()
    assert not recorder.is_recording

    with wave.open(path, "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() > 0

    import os

    os.unlink(path)


def test_recorder_stop_without_start_raises():
    recorder = MicrophoneRecorder()
    with pytest.raises(SpeechToTextError):
        recorder.stop()


def test_recorder_start_twice_is_idempotent():
    recorder = MicrophoneRecorder()
    recorder.start()
    recorder.start()  # should not raise or open a second stream
    assert recorder.is_recording
    recorder.stop()


def test_recorder_reports_friendly_error_when_mic_unavailable():
    _FakeRawInputStream.fail_on_start = True
    recorder = MicrophoneRecorder()
    with pytest.raises(SpeechToTextError, match="microphone"):
        recorder.start()
    assert not recorder.is_recording


def test_recorder_missing_sounddevice_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    recorder = MicrophoneRecorder()
    with pytest.raises(SpeechToTextError, match="sounddevice"):
        recorder.start()


# ------------------------------------------- continuous/session mode methods
def test_sample_rate_property_reflects_constructor_argument():
    assert MicrophoneRecorder(sample_rate=22050).sample_rate == 22050
    assert MicrophoneRecorder().sample_rate == 16000


def test_read_available_drains_queued_frames_without_stopping_stream():
    recorder = MicrophoneRecorder(sample_rate=16000)
    recorder.start()  # the fake stream pushes one chunk synchronously on start()

    chunk = recorder.read_available()
    assert chunk == b"\x01\x02" * 50
    assert recorder.is_recording  # stream still open, unlike stop()

    # nothing new queued yet -> empty, not blocking, not raising
    assert recorder.read_available() == b""
    recorder.close()


def test_read_available_without_start_raises():
    recorder = MicrophoneRecorder()
    with pytest.raises(SpeechToTextError):
        recorder.read_available()


def test_close_releases_stream_and_allows_a_fresh_start():
    recorder = MicrophoneRecorder()
    recorder.start()
    recorder.close()
    assert not recorder.is_recording

    # a session can be started again afterward with no special-casing
    recorder.start()
    assert recorder.is_recording
    recorder.close()


def test_close_without_start_is_a_noop():
    recorder = MicrophoneRecorder()
    recorder.close()  # must not raise
    assert not recorder.is_recording


def test_close_is_idempotent():
    recorder = MicrophoneRecorder()
    recorder.start()
    recorder.close()
    recorder.close()  # calling twice must not raise
    assert not recorder.is_recording


# --------------------------------------------------------- faster-whisper
class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


class _FakeInfo:
    language = "en"


class _FakeWhisperModel:
    last_init_kwargs: dict | None = None
    last_transcribe_kwargs: dict | None = None

    def __init__(self, model_size, device, compute_type):
        _FakeWhisperModel.last_init_kwargs = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }

    def transcribe(self, audio_path, **kwargs):
        _FakeWhisperModel.last_transcribe_kwargs = kwargs
        return [_FakeSegment(" hello "), _FakeSegment("world ")], _FakeInfo()


class _FakeFasterWhisperModule:
    WhisperModel = _FakeWhisperModel


@pytest.fixture(autouse=True)
def fake_faster_whisper(monkeypatch):
    _FakeWhisperModel.last_init_kwargs = None
    _FakeWhisperModel.last_transcribe_kwargs = None
    monkeypatch.setitem(sys.modules, "faster_whisper", _FakeFasterWhisperModule())
    yield


@pytest.mark.asyncio
async def test_faster_whisper_transcribe_joins_segments():
    provider = FasterWhisperProvider(model_size="base", device="cpu", compute_type="int8")
    result = await provider.transcribe("some_audio.wav")
    assert result.text == "hello world"
    assert result.language == "en"
    assert _FakeWhisperModel.last_init_kwargs == {
        "model_size": "base",
        "device": "cpu",
        "compute_type": "int8",
    }


@pytest.mark.asyncio
async def test_faster_whisper_pins_language_by_default():
    """Regression test: language auto-detection on short/noisy clips was
    observed producing garbled non-English hallucinated output. The
    language must be pinned (default 'en') rather than left unset."""
    provider = FasterWhisperProvider(model_size="base")
    await provider.transcribe("some_audio.wav")
    assert _FakeWhisperModel.last_transcribe_kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_faster_whisper_language_override_is_passed_through():
    provider = FasterWhisperProvider(model_size="base", language="fr")
    await provider.transcribe("some_audio.wav")
    assert _FakeWhisperModel.last_transcribe_kwargs["language"] == "fr"


@pytest.mark.asyncio
async def test_faster_whisper_falsy_language_falls_back_to_auto_detect():
    provider = FasterWhisperProvider(model_size="base", language=None)
    await provider.transcribe("some_audio.wav")
    assert _FakeWhisperModel.last_transcribe_kwargs["language"] is None


@pytest.mark.asyncio
async def test_faster_whisper_sets_anti_hallucination_transcribe_kwargs():
    provider = FasterWhisperProvider(model_size="base")
    await provider.transcribe("some_audio.wav")
    kwargs = _FakeWhisperModel.last_transcribe_kwargs
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["compression_ratio_threshold"] == 2.4


def _write_wav(path: str, seconds: float, sample_rate: int = 16000) -> None:
    frame_count = int(seconds * sample_rate)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01\x02" * frame_count)


@pytest.mark.asyncio
async def test_faster_whisper_skips_transcription_for_audio_shorter_than_min_duration(tmp_path):
    """Regression test: very short clips are a classic faster-whisper
    hallucination trigger. Below the configured floor, the model must not
    even be invoked."""
    audio_path = tmp_path / "short.wav"
    _write_wav(str(audio_path), seconds=0.1)
    provider = FasterWhisperProvider(model_size="base", min_duration_seconds=0.3)

    result = await provider.transcribe(str(audio_path))

    assert result.text == ""
    assert _FakeWhisperModel.last_transcribe_kwargs is None  # model never called


@pytest.mark.asyncio
async def test_faster_whisper_transcribes_audio_at_or_above_min_duration(tmp_path):
    audio_path = tmp_path / "long.wav"
    _write_wav(str(audio_path), seconds=0.5)
    provider = FasterWhisperProvider(model_size="base", min_duration_seconds=0.3)

    result = await provider.transcribe(str(audio_path))

    assert result.text == "hello world"
    assert _FakeWhisperModel.last_transcribe_kwargs is not None


@pytest.mark.asyncio
async def test_faster_whisper_min_duration_disabled_when_zero(tmp_path):
    audio_path = tmp_path / "tiny.wav"
    _write_wav(str(audio_path), seconds=0.01)
    provider = FasterWhisperProvider(model_size="base", min_duration_seconds=0.0)

    result = await provider.transcribe(str(audio_path))

    assert result.text == "hello world"


@pytest.mark.asyncio
async def test_faster_whisper_reuses_loaded_model():
    provider = FasterWhisperProvider(model_size="tiny")
    await provider.transcribe("a.wav")
    first_model = provider._model
    await provider.transcribe("b.wav")
    assert provider._model is first_model


@pytest.mark.asyncio
async def test_faster_whisper_missing_dependency_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    provider = FasterWhisperProvider(model_size="base")
    with pytest.raises(SpeechToTextError, match="faster-whisper"):
        await provider.transcribe("audio.wav")


@pytest.mark.asyncio
async def test_faster_whisper_wraps_unexpected_errors(monkeypatch):
    class _BrokenModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio_path, **kwargs):
            raise RuntimeError("model exploded")

    class _BrokenModule:
        WhisperModel = _BrokenModel

    monkeypatch.setitem(sys.modules, "faster_whisper", _BrokenModule())
    provider = FasterWhisperProvider(model_size="base")
    with pytest.raises(SpeechToTextError, match="Transcription failed"):
        await provider.transcribe("audio.wav")
