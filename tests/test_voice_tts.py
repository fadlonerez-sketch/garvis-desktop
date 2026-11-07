"""Tests for the Piper TTS provider and the audio player.

The `piper` package (in-process default path), `subprocess` (standalone-
executable path), and `winsound` are all faked so these tests don't need a
real Piper install, a real voice model, or real audio hardware.
"""

from __future__ import annotations

import struct
import sys
import time
import wave
from pathlib import Path

import pytest

from app.core.exceptions import TextToSpeechError
from app.voice.tts.piper_provider import (
    _LEADING_SILENCE_SECONDS,
    PiperProvider,
    _sanitize_for_tts,
)


class _FakeVoiceModel:
    """Stands in for piper.PiperVoice: records synthesized text and writes a
    minimal-but-valid WAV so the real `wave.open(...).close()` doesn't
    complain about an unset format."""

    def __init__(self, fail_on: str | None = None, audio_frames: bytes | None = None) -> None:
        self.synthesized_texts: list[str] = []
        self._fail_on = fail_on
        # Overridable so tests can assert on non-silent, distinguishable
        # content (e.g. for the leading-silence regression tests below);
        # defaults to the original all-zero stand-in used elsewhere.
        self._audio_frames = audio_frames if audio_frames is not None else b"\x00\x00" * 10

    def synthesize_wav(self, text: str, wav_file, **kwargs) -> None:
        if self._fail_on is not None and self._fail_on in text:
            raise RuntimeError("simulated synthesis failure")
        self.synthesized_texts.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(self._audio_frames)


class _FakePiperModule:
    """Stands in for the real `piper` package (heavy: onnxruntime + a real
    ONNX model on disk) so tests never need it installed."""

    def __init__(self, voice_model: _FakeVoiceModel) -> None:
        self.load_calls: list[str] = []
        outer = self

        class PiperVoice:
            @staticmethod
            def load(model_path):
                outer.load_calls.append(model_path)
                return voice_model

        self.PiperVoice = PiperVoice


@pytest.fixture
def fake_piper(monkeypatch):
    """Patches the lazily-imported `piper` module. Returns (fake_module,
    voice_model) so tests can assert on synthesized text and load count."""
    voice_model = _FakeVoiceModel()
    fake_module = _FakePiperModule(voice_model)
    monkeypatch.setitem(sys.modules, "piper", fake_module)
    return fake_module, voice_model


def _write_short_wav(path, seconds: float = 0.05, framerate: int = 16000) -> None:
    n_frames = int(seconds * framerate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


# ------------------------------------------------------------ PiperProvider
def test_model_path_missing_raises_clear_error(tmp_path):
    provider = PiperProvider(voice="en_US-lessac-medium", voice_dir=tmp_path)
    with pytest.raises(TextToSpeechError, match="not found"):
        provider._model_path()


def test_model_path_found(tmp_path):
    model_file = tmp_path / "en_US-lessac-medium.onnx"
    model_file.write_bytes(b"fake onnx data")
    provider = PiperProvider(voice="en_US-lessac-medium", voice_dir=tmp_path)
    assert provider._model_path() == model_file


def test_command_uses_python_module_by_default(tmp_path):
    provider = PiperProvider(voice="v", voice_dir=tmp_path)
    command = provider._command(tmp_path / "v.onnx", "out.wav")
    assert command[:3] == [sys.executable, "-m", "piper"]
    assert "--model" in command
    assert "--output_file" in command


def test_command_uses_standalone_executable_when_configured(tmp_path):
    provider = PiperProvider(voice="v", voice_dir=tmp_path, piper_executable="C:/tools/piper.exe")
    command = provider._command(tmp_path / "v.onnx", "out.wav")
    assert command[0] == "C:/tools/piper.exe"
    assert "-m" not in command


class _FakeProcess:
    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self._stderr = stderr

    def communicate(self, input: bytes):
        return b"", self._stderr


@pytest.mark.asyncio
async def test_synthesize_to_file_success_default_in_process_path(tmp_path, fake_piper):
    """Default (no piper_executable configured) path uses the in-process
    piper.PiperVoice API, not a subprocess."""
    _fake_module, voice_model = fake_piper
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    output_path = str(tmp_path / "out.wav")
    result = await provider.synthesize_to_file("hello there", output_path)

    assert result == output_path
    assert voice_model.synthesized_texts == ["hello there"]
    with wave.open(output_path, "rb") as wav_file:  # a real, valid WAV was written
        assert wav_file.getnframes() > 0


@pytest.mark.asyncio
async def test_synthesize_to_file_loads_model_once_and_reuses_it(tmp_path, fake_piper):
    """Regression guard for the startup-latency fix: the voice model must be
    loaded once per provider instance and reused, not reloaded on every
    call -- reloading it every time (the old subprocess-per-call design)
    was the measured root cause of Garvis's slow-to-start-speaking delay."""
    fake_module, voice_model = fake_piper
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    await provider.synthesize_to_file("first reply", str(tmp_path / "a.wav"))
    await provider.synthesize_to_file("second reply", str(tmp_path / "b.wav"))
    await provider.synthesize_to_file("third reply", str(tmp_path / "c.wav"))

    assert len(fake_module.load_calls) == 1  # model loaded exactly once
    assert voice_model.synthesized_texts == ["first reply", "second reply", "third reply"]


@pytest.mark.asyncio
async def test_synthesize_to_file_in_process_error_raises_tts_error(tmp_path, fake_piper):
    _fake_module, voice_model = fake_piper
    voice_model._fail_on = "trigger"
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    with pytest.raises(TextToSpeechError):
        await provider.synthesize_to_file("please trigger failure", str(tmp_path / "out.wav"))


@pytest.mark.asyncio
async def test_synthesize_to_file_subprocess_path_used_when_executable_configured(tmp_path, monkeypatch):
    """Explicitly configuring a standalone executable must still use the
    subprocess path -- that path can't get the warm-model benefit since
    synthesis happens in a separate binary's own process."""
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path, piper_executable="C:/tools/piper.exe")

    captured_command = {}

    def fake_popen(command, **kwargs):
        captured_command["command"] = command
        return _FakeProcess(returncode=0)

    monkeypatch.setattr("app.voice.tts.piper_provider.subprocess.Popen", fake_popen)

    output_path = str(tmp_path / "out.wav")
    result = await provider.synthesize_to_file("hello there", output_path)
    assert result == output_path
    assert str(model_file) in captured_command["command"]
    assert captured_command["command"][0] == "C:/tools/piper.exe"


@pytest.mark.asyncio
async def test_synthesize_to_file_nonzero_exit_raises(tmp_path, monkeypatch):
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path, piper_executable="C:/tools/piper.exe")

    monkeypatch.setattr(
        "app.voice.tts.piper_provider.subprocess.Popen",
        lambda command, **kwargs: _FakeProcess(returncode=1, stderr=b"boom"),
    )

    with pytest.raises(TextToSpeechError, match="boom"):
        await provider.synthesize_to_file("hello", str(tmp_path / "out.wav"))


# ------------------------------------------- leading-silence onset-glitch fix
@pytest.mark.asyncio
async def test_synthesize_to_file_prepends_approximately_200ms_of_silence(tmp_path, fake_piper):
    """Regression test for the TTS onset-glitch fix: winsound.PlaySound's
    SND_ASYNC playback can glitch/drop the very start of audio while the
    Windows audio device wakes from idle, so every in-process-synthesized
    WAV must start with ~200ms of true digital silence -- see
    app.voice.tts.piper_provider._prepend_silence / _LEADING_SILENCE_SECONDS."""
    _fake_module, voice_model = fake_piper
    framerate = 16000
    original_samples = list(range(1, 51))  # 50 non-zero, distinguishable samples
    voice_model._audio_frames = struct.pack(f"<{len(original_samples)}h", *original_samples)

    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    output_path = str(tmp_path / "out.wav")
    result = await provider.synthesize_to_file("hello there", output_path)

    with wave.open(result, "rb") as wav_file:
        all_samples = struct.unpack(f"<{wav_file.getnframes()}h", wav_file.readframes(wav_file.getnframes()))

    expected_silence_frames = int(framerate * _LEADING_SILENCE_SECONDS)
    assert expected_silence_frames == 3200  # 0.2s * 16000Hz

    leading = all_samples[:expected_silence_frames]
    assert len(leading) == expected_silence_frames
    assert all(sample == 0 for sample in leading)  # true digital silence, not near-silence


@pytest.mark.asyncio
async def test_synthesize_to_file_original_audio_follows_silence_unchanged(tmp_path, fake_piper):
    """The synthesized audio itself must be byte-for-byte unchanged -- only
    prepended to, never altered -- and the WAV format (rate/width/channels)
    must be preserved exactly."""
    _fake_module, voice_model = fake_piper
    framerate = 16000
    original_samples = list(range(1, 51))
    voice_model._audio_frames = struct.pack(f"<{len(original_samples)}h", *original_samples)

    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    output_path = str(tmp_path / "out.wav")
    result = await provider.synthesize_to_file("hello there", output_path)

    with wave.open(result, "rb") as wav_file:
        params = wav_file.getparams()
        all_samples = struct.unpack(f"<{wav_file.getnframes()}h", wav_file.readframes(wav_file.getnframes()))

    # format preserved
    assert params.framerate == framerate
    assert params.sampwidth == 2
    assert params.nchannels == 1

    expected_silence_frames = int(framerate * _LEADING_SILENCE_SECONDS)
    trailing = all_samples[expected_silence_frames:]
    assert list(trailing) == original_samples  # original audio unchanged, in order


@pytest.mark.asyncio
async def test_synthesized_file_can_still_be_cleaned_up_after_silence_prepend(tmp_path, fake_piper):
    """_prepend_silence reads then rewrites the same file (read + write
    `wave.open` context managers) -- it must fully release its handles so
    the temp file can still be deleted immediately afterward, exactly as
    VoicePipeline.speak()'s cleanup callback does."""
    _fake_module, voice_model = fake_piper
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    output_path = tmp_path / "out.wav"
    result = await provider.synthesize_to_file("hello there", str(output_path))

    assert output_path.exists()
    Path(result).unlink()  # must not raise (e.g. a held file handle on Windows)
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_audio_player_duration_estimate_accounts_for_prepended_silence(tmp_path, fake_piper, fake_winsound):
    """AudioPlayer estimates on_finished timing purely from the WAV file's
    own frame count/rate (_estimate_duration_seconds) -- it must naturally
    include the prepended silence's extra frames with no special-casing, so
    on_finished still fires only once the full (silence + original) playback
    would have completed."""
    from app.voice.tts.player import AudioPlayer

    _fake_module, voice_model = fake_piper
    framerate = 16000
    voice_model._audio_frames = b"\x01\x00" * 50

    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)
    output_path = str(tmp_path / "out.wav")
    await provider.synthesize_to_file("hi", output_path)

    with wave.open(output_path, "rb") as wav_file:
        total_frames = wav_file.getnframes()
    total_duration = total_frames / framerate
    assert total_duration > _LEADING_SILENCE_SECONDS  # padding actually extended the file

    player = AudioPlayer()
    finished_called = []
    player.play(output_path, on_finished=lambda: finished_called.append(True))

    assert player.is_playing
    assert finished_called == []  # not finished immediately
    for _ in range(100):  # poll up to ~2s for the background timer to fire
        if not player.is_playing:
            break
        time.sleep(0.02)

    assert not player.is_playing
    assert finished_called == [True]


# ---------------------------------------------- Unicode/surrogate sanitizing
def test_sanitize_for_tts_removes_lone_surrogate():
    text = "Hello \udc9d World"
    sanitized = _sanitize_for_tts(text)
    sanitized.encode("utf-8")  # must not raise
    assert "\udc9d" not in sanitized
    assert "Hello" in sanitized and "World" in sanitized


def test_sanitize_for_tts_preserves_hebrew_and_emoji():
    text = "שלום \U0001f600 hello"
    sanitized = _sanitize_for_tts(text)
    assert sanitized == text  # nothing well-formed is touched


def test_sanitize_for_tts_leaves_plain_text_untouched():
    assert _sanitize_for_tts("hello there") == "hello there"


@pytest.mark.asyncio
async def test_synthesize_to_file_survives_lone_surrogate_in_text(tmp_path, fake_piper):
    """Regression test: an assistant response containing a lone UTF-16
    surrogate (e.g. from malformed upstream Unicode) must not crash
    synthesis with UnicodeEncodeError -- it should be sanitized and
    synthesis should proceed normally, through the default in-process path."""
    _fake_module, voice_model = fake_piper
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    output_path = str(tmp_path / "out.wav")
    result = await provider.synthesize_to_file("Hello \udc9d World", output_path)

    assert result == output_path
    assert voice_model.synthesized_texts == ["Hello  World"]


@pytest.mark.asyncio
async def test_synthesize_to_file_in_process_wraps_bypassed_sanitization_as_tts_error(
    tmp_path, fake_piper, monkeypatch
):
    """Defense in depth for the default in-process path: even if
    sanitization were bypassed, any synthesis failure (e.g. from a lone
    surrogate the underlying library can't handle) must surface as
    TextToSpeechError, never a raw exception -- so a TTS failure can never
    crash the interaction."""
    _fake_module, voice_model = fake_piper
    voice_model._fail_on = "\udc9d"
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path)

    monkeypatch.setattr("app.voice.tts.piper_provider._sanitize_for_tts", lambda text: text)

    with pytest.raises(TextToSpeechError):
        await provider.synthesize_to_file("Hello \udc9d World", str(tmp_path / "out.wav"))


@pytest.mark.asyncio
async def test_synthesize_to_file_subprocess_path_wraps_encode_failure_as_tts_error(tmp_path, monkeypatch):
    """Defense in depth for the standalone-executable (subprocess) path:
    even if sanitization is bypassed, a text that still can't be UTF-8
    encoded must surface as TextToSpeechError, never a raw
    UnicodeEncodeError."""
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path, piper_executable="C:/tools/piper.exe")

    monkeypatch.setattr("app.voice.tts.piper_provider._sanitize_for_tts", lambda text: text)
    monkeypatch.setattr(
        "app.voice.tts.piper_provider.subprocess.Popen",
        lambda command, **kwargs: pytest.fail("Popen should not be called"),
    )

    with pytest.raises(TextToSpeechError):
        await provider.synthesize_to_file("Hello \udc9d World", str(tmp_path / "out.wav"))


@pytest.mark.asyncio
async def test_synthesize_to_file_missing_executable_raises_clear_error(tmp_path, monkeypatch):
    model_file = tmp_path / "v.onnx"
    model_file.write_bytes(b"fake")
    provider = PiperProvider(voice="v", voice_dir=tmp_path, piper_executable="nope")

    def fake_popen(command, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr("app.voice.tts.piper_provider.subprocess.Popen", fake_popen)

    with pytest.raises(TextToSpeechError, match="pip install piper-tts"):
        await provider.synthesize_to_file("hello", str(tmp_path / "out.wav"))


# --------------------------------------------------------------- AudioPlayer
class _FakeWinsound:
    SND_FILENAME = 1
    SND_ASYNC = 2
    SND_PURGE = 4

    def __init__(self):
        self.calls = []

    def PlaySound(self, sound, flags):
        self.calls.append((sound, flags))


@pytest.fixture
def fake_winsound(monkeypatch):
    fake = _FakeWinsound()
    monkeypatch.setitem(sys.modules, "winsound", fake)
    return fake


def test_audio_player_play_marks_playing(fake_winsound):
    from app.voice.tts.player import AudioPlayer

    player = AudioPlayer()
    assert not player.is_playing
    player.play("some.wav")
    assert player.is_playing
    assert fake_winsound.calls == [("some.wav", _FakeWinsound.SND_FILENAME | _FakeWinsound.SND_ASYNC)]


def test_audio_player_stop_purges_when_playing(fake_winsound):
    from app.voice.tts.player import AudioPlayer

    player = AudioPlayer()
    player.play("some.wav")
    player.stop()
    assert not player.is_playing
    assert fake_winsound.calls[-1] == (None, _FakeWinsound.SND_PURGE)


def test_audio_player_stop_when_not_playing_is_a_noop(fake_winsound):
    from app.voice.tts.player import AudioPlayer

    player = AudioPlayer()
    player.stop()
    assert fake_winsound.calls == []


def test_audio_player_raises_when_winsound_unavailable(monkeypatch):
    """Simulates a non-Windows environment by making `import winsound` fail,
    regardless of the actual OS this test runs on -- this is what makes the
    test hermetic/portable rather than depending on sys.platform."""
    from app.voice.tts.player import AudioPlayer

    monkeypatch.setitem(sys.modules, "winsound", None)
    player = AudioPlayer()
    with pytest.raises(TextToSpeechError, match="Windows"):
        player.play("some.wav")


def test_audio_player_stop_is_noop_when_winsound_unavailable(monkeypatch):
    from app.voice.tts.player import AudioPlayer

    monkeypatch.setitem(sys.modules, "winsound", None)
    player = AudioPlayer()
    player.stop()  # must not raise


def test_audio_player_on_finished_called_when_playback_estimated_complete(tmp_path, fake_winsound):
    from app.voice.tts.player import AudioPlayer

    wav_path = tmp_path / "short.wav"
    _write_short_wav(wav_path, seconds=0.05)

    player = AudioPlayer()
    finished_called = []
    player.play(str(wav_path), on_finished=lambda: finished_called.append(True))

    assert player.is_playing
    for _ in range(50):  # poll up to ~1s for the background timer to fire
        if not player.is_playing:
            break
        import time

        time.sleep(0.02)

    assert not player.is_playing
    assert finished_called == [True]


def test_audio_player_stop_invokes_pending_callback_immediately(fake_winsound):
    from app.voice.tts.player import AudioPlayer

    player = AudioPlayer()
    called = []
    player.play("some.wav", on_finished=lambda: called.append(True))
    player.stop()
    assert called == [True]
    assert not player.is_playing


def test_audio_player_stop_does_not_double_invoke_callback(fake_winsound):
    from app.voice.tts.player import AudioPlayer

    player = AudioPlayer()
    called = []
    player.play("some.wav", on_finished=lambda: called.append(True))
    player.stop()
    player.stop()  # calling stop() again must not re-invoke the callback
    assert called == [True]


# --------------------------------------------------- real Jarvis-High voice
# Deliberate exception to this file's "no real model" design (see module
# docstring): data/voices/*.onnx are large binaries excluded from version
# control (see .gitignore: "data/"), so these tests skip gracefully when the
# asset isn't present locally instead of failing a fresh clone/CI checkout.
_REAL_VOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "voices"
_JARVIS_HIGH_ONNX = _REAL_VOICES_DIR / "Jarvis-High.onnx"
_JARVIS_HIGH_CONFIG = _REAL_VOICES_DIR / "Jarvis-High.onnx.json"

_jarvis_high_available = pytest.mark.skipif(
    not (_JARVIS_HIGH_ONNX.exists() and _JARVIS_HIGH_CONFIG.exists()),
    reason="Jarvis-High.onnx / .onnx.json not present in data/voices/",
)


@_jarvis_high_available
@pytest.mark.asyncio
async def test_jarvis_high_voice_loads_and_synthesizes_real_audio():
    """The real, downloaded Jarvis-High voice must load via the same
    in-process PiperVoice.load() path as any other voice and produce valid,
    non-empty audio -- proving the .onnx + .onnx.json pair is a genuine,
    compatible Piper voice, not just present on disk."""
    provider = PiperProvider(voice="Jarvis-High", voice_dir=_REAL_VOICES_DIR)
    output_path = str(_REAL_VOICES_DIR / "_jarvis_high_test_output.wav")
    try:
        result = await provider.synthesize_to_file("Your name is Alex, Sir.", output_path)
        assert result == output_path
        with wave.open(result, "rb") as wav_file:
            assert wav_file.getnframes() > 0
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnchannels() == 1
    finally:
        Path(output_path).unlink(missing_ok=True)


@_jarvis_high_available
@pytest.mark.asyncio
async def test_jarvis_high_voice_output_still_has_leading_silence():
    """Regression guard: the 200ms leading-silence fix (see
    app.voice.tts.piper_provider._prepend_silence) must apply to every
    in-process voice, including a newly added one -- not just the voice it
    was originally verified against."""
    provider = PiperProvider(voice="Jarvis-High", voice_dir=_REAL_VOICES_DIR)
    output_path = str(_REAL_VOICES_DIR / "_jarvis_high_test_silence.wav")
    try:
        await provider.synthesize_to_file("Your name is Alex, Sir.", output_path)
        with wave.open(output_path, "rb") as wav_file:
            rate = wav_file.getframerate()
            expected_silence_frames = int(rate * _LEADING_SILENCE_SECONDS)
            samples = struct.unpack(f"<{wav_file.getnframes()}h", wav_file.readframes(wav_file.getnframes()))
        assert len(samples) > expected_silence_frames
        assert all(sample == 0 for sample in samples[:expected_silence_frames])
    finally:
        Path(output_path).unlink(missing_ok=True)
