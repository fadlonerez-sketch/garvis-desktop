"""Playback timing and reference lifecycle for echo discrimination: the
streaming player reports the DAC start and real completion; the recorder
timestamps blocks; VoicePipeline registers/ends references on every path
(finish, stop, synthesis failure, playback failure). No real audio."""
from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.exceptions import TextToSpeechError
from app.voice.echo import EchoGuard
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import TranscriptionResult
from app.voice.stt.recorder import MicrophoneRecorder
from app.voice.tts.player import StreamingAudioPlayer


def _wav(path, seconds=0.5, rate=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.sin(np.arange(int(seconds * rate)) / 5) * 3000).astype(np.int16).tobytes())
    return str(path)


class FakeOutputStream:
    instances: list = []

    def __init__(self, samplerate, channels, dtype, callback, finished_callback):
        self.callback, self.finished_callback = callback, finished_callback
        self.started = self.aborted = self.closed = False
        FakeOutputStream.instances.append(self)

    def start(self):
        self.started = True

    def pump(self, frames=512, dac=50.0):
        """One PortAudio callback; returns False once the stream stopped itself."""
        out = bytearray(frames * 2)
        try:
            self.callback(out, frames, SimpleNamespace(outputBufferDacTime=dac), None)
        except FakeSoundDevice.CallbackStop:
            self.finished_callback()
            return False
        return True

    def abort(self):
        self.aborted = True
        self.finished_callback()

    def close(self):
        self.closed = True


class FakeSoundDevice:
    class CallbackStop(Exception):
        pass

    RawOutputStream = FakeOutputStream


@pytest.fixture(autouse=True)
def _reset_streams():
    FakeOutputStream.instances.clear()


# ------------------------------------------------------- streaming player
def test_player_reports_the_dac_time_of_the_first_sample_and_real_completion(tmp_path):
    started, finished = [], []
    player = StreamingAudioPlayer(sd_module=FakeSoundDevice)
    player.play(_wav(tmp_path / "a.wav"), on_finished=lambda: finished.append(1), on_started=started.append)
    stream = FakeOutputStream.instances[-1]
    assert player.is_playing and started == []
    assert stream.pump(dac=1234.5)
    assert started == [1234.5]
    while stream.pump(dac=1235.0):
        pass
    assert finished == [1] and not player.is_playing and started == [1234.5]


def test_stop_ends_playback_once_and_drops_buffered_audio(tmp_path):
    finished = []
    player = StreamingAudioPlayer(sd_module=FakeSoundDevice)
    player.play(_wav(tmp_path / "a.wav"), on_finished=lambda: finished.append(1))
    stream = FakeOutputStream.instances[-1]
    player.stop()
    player.stop()
    assert stream.aborted and stream.closed and finished == [1] and not player.is_playing


def test_a_new_clip_supersedes_the_old_one_without_stale_callbacks(tmp_path):
    finished = []
    player = StreamingAudioPlayer(sd_module=FakeSoundDevice)
    player.play(_wav(tmp_path / "a.wav"), on_finished=lambda: finished.append("a"))
    first = FakeOutputStream.instances[-1]
    player.play(_wav(tmp_path / "b.wav"), on_finished=lambda: finished.append("b"))
    first.finished_callback()  # a late callback from the old stream
    assert finished == ["a"] and player.is_playing


def test_player_falls_back_to_winsound_when_no_output_stream_opens(tmp_path):
    class Broken(FakeSoundDevice):
        class RawOutputStream:
            def __init__(self, *a, **k):
                raise OSError("no device")

    class Fallback:
        def __init__(self):
            self.played = []

        def play(self, path, on_finished=None):
            self.played.append(path)
            self.on_finished = on_finished

        def stop(self):
            pass

    started, fallback = [], Fallback()
    player = StreamingAudioPlayer(sd_module=Broken, fallback=fallback, clock=lambda: 77.0)
    path = _wav(tmp_path / "a.wav")
    player.play(path, on_started=started.append)
    assert fallback.played == [path] and started == [77.0] and player.is_playing


# --------------------------------------------------------------- recorder
def test_recorder_timestamps_each_block(monkeypatch):
    class FakeInput:
        latency = 0.03

        def __init__(self, samplerate, channels, dtype, callback):
            self.callback = callback

        def start(self):
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace(RawInputStream=FakeInput))
    monkeypatch.setattr("app.voice.stt.recorder.time.perf_counter", lambda: 10.0)
    recorder = MicrophoneRecorder(sample_rate=16000)
    recorder.start()
    recorder._stream.callback(b"\x01\x00" * 160, 160, None, None)
    recorder._stream.callback(b"\x02\x00" * 160, 160, None, None)
    blocks = recorder.read_available_timed()
    assert [round(t, 3) for t, _ in blocks] == [round(10.0 - 0.01 - 0.03, 3)] * 2
    recorder._stream.callback(b"\x03\x00" * 160, 160, None, None)
    assert recorder.read_available() == b"\x03\x00" * 160  # the bytes API is unchanged


# ---------------------------------------------------------- pipeline
class _STT:
    async def transcribe(self, path):
        return TranscriptionResult(text="")


class _TTS:
    def __init__(self, fail=False):
        self.fail = fail

    async def synthesize_to_file(self, text, output_path):
        if self.fail:
            raise TextToSpeechError("voice model missing")
        _wav(output_path)
        return output_path

    def stop(self):
        pass


class _TimedPlayer:
    def __init__(self, fail=False):
        self.fail = fail
        self.is_playing = False

    def play(self, path, on_finished=None, on_started=None):
        if self.fail:
            raise TextToSpeechError("device busy")
        self.is_playing = True
        self.on_finished = on_finished
        on_started(42.0)

    def stop(self):
        if self.is_playing:
            self.is_playing = False
            self.on_finished()


class _Recorder:
    sample_rate = 16000
    is_recording = True

    def read_available_timed(self):
        return [(42.2, (np.ones(416) * 2000).astype(np.int16).tobytes())]


async def test_pipeline_registers_the_played_clip_with_its_dac_time_and_ends_it_on_stop():
    guard = EchoGuard()
    now = [42.5]
    voice = VoicePipeline(stt=_STT(), tts=_TTS(), recorder=_Recorder(), player=_TimedPlayer(), echo_guard=guard,
                          clock=lambda: now[0])
    await voice.speak("hello")
    clip = guard._clips[0]
    assert clip.start_time == 42.0 and clip.end_time is None
    chunk, residual = voice.read_microphone_chunk_with_echo()
    assert chunk and residual is not None  # judged against the reference
    voice.stop_speaking()
    assert clip.end_time == 42.5


@pytest.mark.parametrize("tts_fails, player_fails", [(True, False), (False, True)])
async def test_failed_speech_leaves_no_open_reference(tts_fails, player_fails):
    guard = EchoGuard()
    voice = VoicePipeline(stt=_STT(), tts=_TTS(fail=tts_fails), recorder=_Recorder(),
                          player=_TimedPlayer(fail=player_fails), echo_guard=guard, clock=lambda: 50.0)
    with pytest.raises(TextToSpeechError):
        await voice.speak("hello")
    assert all(clip.end_time is not None for clip in guard._clips)


def test_without_an_echo_guard_mic_reads_are_unchanged():
    class Plain:
        sample_rate = 16000

        def read_available(self):
            return b"\x05\x00" * 10

    voice = VoicePipeline(stt=_STT(), tts=_TTS(), recorder=Plain(), player=_TimedPlayer())
    assert voice.read_microphone_chunk_with_echo() == (b"\x05\x00" * 10, None)
