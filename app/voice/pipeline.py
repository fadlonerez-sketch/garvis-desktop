"""Voice pipeline: the only module that knows about both voice input and
voice output. The agent core still only ever sees plain text in
(`Agent.handle_user_message`) and plain text out (the returned reply) --
this module is entirely a UI-side concern, same as `app/ui/agent_bridge.py`
which wires it up.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.voice.echo import EchoGuard
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.recorder import MicrophoneRecorder
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.player import AudioPlayer
from app.voice.vad import PlaybackEnvelope


class VoicePipeline:
    def __init__(
        self,
        stt: SpeechToTextProvider,
        tts: TextToSpeechProvider,
        recorder: MicrophoneRecorder | None = None,
        player: AudioPlayer | None = None,
        echo_guard: EchoGuard | None = None,
        clock=time.perf_counter,
    ) -> None:
        """`echo_guard`, when given, receives every clip that is played (timed
        by the player) and every timestamped mic block, so
        read_microphone_chunk_with_echo() can report the part of the
        microphone signal Garvis's own playback can't explain (see
        app/voice/echo.py). Without it, behavior is exactly as before."""
        self._stt = stt
        self._tts = tts
        self._recorder = recorder or MicrophoneRecorder()
        self._player = player or AudioPlayer()
        self._echo = echo_guard
        self._clock = clock
        """Shared timeline of playback and microphone timestamps (PortAudio's
        clock is time.perf_counter on Windows)."""
        self._playback_envelope: PlaybackEnvelope | None = None
        self._playback_started_at = 0.0
        self._playback_listeners: list[Callable[[bool], None]] = []
        self._playback_seq = 0

    def add_playback_listener(self, listener: Callable[[bool], None]) -> None:
        """`listener(True)` when a clip is handed to the player, `listener(False)`
        when that clip finishes or is stopped (a clip replaced by a newer one
        reports nothing; the newer one's end does). Display only -- the desktop
        orb follows it; nothing in the voice path depends on it. Called on
        whichever thread the player reports from: it must be quick and not raise."""
        self._playback_listeners.append(listener)

    def _notify_playback(self, playing: bool) -> None:
        for listener in list(self._playback_listeners):
            try:
                listener(playing)
            except Exception:  # noqa: BLE001 -- a display hook must never break playback
                pass

    @property
    def is_listening(self) -> bool:
        return self._recorder.is_recording

    @property
    def is_speaking(self) -> bool:
        return self._player.is_playing

    @property
    def sample_rate(self) -> int:
        return self._recorder.sample_rate

    def start_listening(self) -> None:
        """Begin recording from the microphone. Raises SpeechToTextError on failure."""
        self._recorder.start()

    async def stop_listening_and_transcribe(self) -> str:
        """Stop recording and transcribe what was captured. Raises SpeechToTextError."""
        audio_path = self._recorder.stop()
        try:
            result = await self._stt.transcribe(audio_path)
        finally:
            Path(audio_path).unlink(missing_ok=True)
        return result.text

    async def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an existing audio file with the configured STT provider
        (no recorder involved -- e.g. a recording uploaded by the mobile
        interface). The caller owns the file. Raises SpeechToTextError."""
        result = await self._stt.transcribe(audio_path)
        return result.text

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize `text` with the configured TTS provider and return the
        audio file's bytes without playing it locally (e.g. for playback on
        the phone). The temp file is always deleted. Raises TextToSpeechError."""
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="garvis_tts_")
        os.close(fd)
        try:
            await self._tts.synthesize_to_file(text, output_path)
            return Path(output_path).read_bytes()
        finally:
            Path(output_path).unlink(missing_ok=True)

    async def speak(self, text: str) -> None:
        """Synthesize `text` and play it back. Raises TextToSpeechError on failure.

        The synthesized WAV is written to a securely-created temp file
        (`tempfile.mkstemp`, not the race-prone `tempfile.mktemp`) and is
        always deleted afterwards: once playback finishes (via the
        player's `on_finished` callback) or is interrupted by
        `stop_speaking()`, and immediately if synthesis or playback itself
        raises.
        """
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="garvis_tts_")
        os.close(fd)

        def _cleanup() -> None:
            Path(output_path).unlink(missing_ok=True)

        clip = None
        playback = {"seq": None}

        def _finished() -> None:
            _cleanup()
            if clip is not None:
                # Playback ended (naturally or by stop_speaking): nothing after
                # this moment was played, so no later echo of it can arrive.
                self._echo.end_reference(clip, self._clock())
            if playback["seq"] is not None and playback["seq"] == self._playback_seq:
                self._notify_playback(False)

        try:
            await self._tts.synthesize_to_file(text, output_path)
            self._playback_envelope = self._build_playback_envelope(output_path)
            clip = self._register_echo_reference(output_path)
            self._playback_started_at = time.monotonic()
            self._playback_seq += 1
            playback["seq"] = self._playback_seq
            self._notify_playback(True)
            if clip is not None and self._player_reports_start():
                self._player.play(
                    output_path,
                    on_finished=_finished,
                    on_started=lambda dac_time: self._echo.set_reference_start(clip, dac_time),
                )
            else:
                self._player.play(output_path, on_finished=_finished)
                if clip is not None:  # no device timing: approximate; the delay tracker corrects it
                    self._echo.set_reference_start(clip, self._clock())
        except Exception:
            _finished()
            raise

    def _player_reports_start(self) -> bool:
        import inspect

        try:
            return "on_started" in inspect.signature(self._player.play).parameters
        except TypeError, ValueError:
            return False

    def _register_echo_reference(self, wav_path: str):
        """Hand the exact samples about to be played to the echo guard."""
        if self._echo is None:
            return None
        try:
            with wave.open(wav_path, "rb") as wav_file:
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        except OSError, wave.Error, EOFError, ValueError:
            return None
        if channels > 1:
            pcm = pcm[::channels]
        if pcm.size == 0 or rate <= 0:
            return None
        return self._echo.start_reference(pcm, rate, None)

    @staticmethod
    def _build_playback_envelope(wav_path: str) -> PlaybackEnvelope | None:
        """Best-effort only: backs reference_energy_now(), itself only a
        barge-in refinement (see VoiceActivityDetector.process()'s
        `reference_energy` param). Must never block real playback, so any
        failure to read the file back (e.g. a test double that never
        writes real audio to it) just means no envelope this turn rather
        than raising out of speak()."""
        try:
            with wave.open(wav_path, "rb") as wav_file:
                pcm = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
            if not pcm or sample_rate <= 0:
                return None
            return PlaybackEnvelope(pcm, sample_rate)
        except OSError, wave.Error, EOFError:
            return None

    def reference_energy_now(self) -> float:
        """Best-effort estimate of how loud Garvis's own TTS output is
        *right now*, from the audio already synthesized for the
        current/most recent speak() call plus wall-clock time elapsed
        since that playback started -- see PlaybackEnvelope. 0.0 whenever
        nothing is currently playing or no envelope could be built, so
        callers that never check this (push-to-talk) are unaffected."""
        if not self.is_speaking or self._playback_envelope is None:
            return 0.0
        elapsed = time.monotonic() - self._playback_started_at
        return self._playback_envelope.energy_at(elapsed, held_if_still_playing=True)

    def stop_speaking(self) -> None:
        self._player.stop()
        self._tts.stop()

    # ------------------------------------------------- session (hands-free) mode
    # Additive: push-to-talk (start_listening/stop_listening_and_transcribe
    # above) is untouched by these. ConversationSession (app/voice/session.py)
    # uses these instead, because session mode needs the mic to stay open
    # across multiple utterances -- and even while speak() is playing, for
    # barge-in detection -- rather than one start-then-stop cycle per turn.

    def start_continuous_listening(self) -> None:
        """Open the mic stream for the whole session. Raises
        SpeechToTextError on failure (e.g. no microphone available)."""
        self._recorder.start()

    def read_microphone_chunk(self) -> bytes:
        """Non-blocking drain of audio captured since the last call.
        Session mode only; requires start_continuous_listening() first."""
        return self._recorder.read_available()

    def read_microphone_chunk_with_echo(self) -> tuple[bytes, float | None]:
        """read_microphone_chunk() plus the RMS of the part of it that
        Garvis's own playback can't explain (app/voice/echo.py), or None when
        no playback reference is active / no echo guard is configured -- in
        which case the raw level applies."""
        if self._echo is None:
            return self._recorder.read_available(), None
        timed = getattr(self._recorder, "read_available_timed", None)
        if timed is not None:
            blocks = timed()
        else:  # recorders without timestamps: assume the audio just arrived
            pcm = self._recorder.read_available()
            blocks = [(self._clock() - len(pcm) / 2 / self.sample_rate, pcm)] if pcm else []
        if not blocks:
            return b"", None
        power = 0.0
        samples = 0
        any_reference = False
        for captured_at, pcm in blocks:
            count = len(pcm) // 2
            if count == 0:
                continue
            residual = self._echo.process(pcm, captured_at)
            if residual is None:
                values = np.frombuffer(pcm[: count * 2], dtype=np.int16).astype(np.float64)
                power += float(np.sum(values**2))
            else:
                any_reference = True
                power += residual**2 * count
            samples += count
        chunk = b"".join(pcm for _, pcm in blocks)
        if not any_reference or samples == 0:
            return chunk, None
        return chunk, float((power / samples) ** 0.5)

    def drain_microphone(self) -> bytes:
        """Everything captured since the last read, for audio that predates
        the playback just started (it was recorded while Garvis was silent --
        thinking or synthesizing), so its raw level is what counts. The echo
        guard still sees it, keeping its timeline and room-level tracking
        continuous. Session mode only."""
        timed = getattr(self._recorder, "read_available_timed", None)
        if self._echo is None or timed is None:
            return self._recorder.read_available()
        blocks = timed()
        for captured_at, pcm in blocks:
            if len(pcm) >= 2:
                self._echo.process(pcm, captured_at)
        return b"".join(pcm for _, pcm in blocks)

    def stop_continuous_listening(self) -> None:
        """Release the mic stream opened by start_continuous_listening()."""
        self._recorder.close()

    async def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw 16-bit mono PCM captured via
        read_microphone_chunk() -- the session-mode equivalent of
        stop_listening_and_transcribe(), without a recorder.stop() call
        (the stream stays open for the rest of the session)."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="garvis_utt_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._recorder.sample_rate)
                wav_file.writeframes(audio_bytes)
            result = await self._stt.transcribe(path)
        finally:
            Path(path).unlink(missing_ok=True)
        return result.text
