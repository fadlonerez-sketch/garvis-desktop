"""Piper implementation of TextToSpeechProvider.

By default, this uses the `piper-tts` PyPI package's in-process Python API
(`piper.PiperVoice`) directly, loading the ONNX voice model once per
provider instance and reusing it for every `synthesize_to_file()` call.

That caching is the fix for a measured startup-latency bug: the previous
implementation shelled out to `python -m piper` as a brand-new subprocess
for *every* reply. Profiling that path locally showed ~1.5s of overhead per
call before any audio was produced -- dominated by loading the ONNX voice
model from disk (~1.2s), not by the Python interpreter starting (~0.04s) or
the actual synthesis itself (~0.1s once the model is loaded). Because the
model was reloaded from scratch every time, *every* reply -- not just the
first -- paid that cost, which is what made Garvis feel slow to start
speaking. Loading the model once and reusing it means only the first
utterance in a session pays that cost; every one after is near-instant.

If `piper_executable` is set (Settings.tts_executable), a standalone Piper
binary is invoked instead via subprocess, for users who prefer that install
path. That path doesn't get the warm-model benefit (there's no Python API
to hold a loaded model in when the actual synthesis happens in a separate
binary's own process) -- its overhead is intrinsic to choosing a separate
executable rather than the bundled pip package, which is why it remains an
explicit opt-in rather than the default.

The voice model file itself is never hardcoded: it's resolved from
`voice_dir / f"{voice}.onnx"`, both of which come from Settings.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import TextToSpeechError
from app.voice.tts.base import TextToSpeechProvider

if TYPE_CHECKING:
    from piper import PiperVoice

_LEADING_SILENCE_SECONDS = 0.2
"""Prepended to every in-process-synthesized WAV. `winsound.PlaySound(...,
SND_ASYNC)` (see app/voice/tts/player.py) can glitch or drop the very start
of playback while the Windows audio device wakes from an idle state -- a
device/playback-timing artifact, not a defect in the synthesized audio
itself (verified by inspecting Piper's raw PCM output, which is clean at
the same point). Prepending silence means whatever gets clipped during that
device warm-up window is inaudible silence, not the first word."""


def _prepend_silence(path: str, seconds: float = _LEADING_SILENCE_SECONDS) -> None:
    """Rewrite the WAV at `path` with `seconds` of digital silence inserted
    before its existing audio, preserving format (sample rate/width/channel
    count) exactly."""
    with wave.open(path, "rb") as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(wav_file.getnframes())

    silence_frame_count = int(params.framerate * seconds)
    silence = b"\x00" * (silence_frame_count * params.sampwidth * params.nchannels)

    with wave.open(path, "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(silence + frames)


def _sanitize_for_tts(text: str) -> str:
    """Drop characters that cannot survive a UTF-8 round trip -- notably lone
    UTF-16 surrogate halves (e.g. U+DC9D) that can end up in an LLM response
    from malformed upstream JSON/text handling. Piper (both the in-process
    API and the standalone-executable subprocess path) is ultimately handed
    UTF-8 text, and `str.encode("utf-8")` raises UnicodeEncodeError on a lone
    surrogate ("surrogates not allowed"), which would otherwise crash the
    TTS step. Well-formed text -- Hebrew, English, real emoji -- encodes
    fine and passes through untouched; this only ever removes already-
    invalid code points."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


class PiperProvider(TextToSpeechProvider):
    def __init__(self, voice: str, voice_dir: Path, piper_executable: str | None = None) -> None:
        self._voice = voice
        self._voice_dir = voice_dir
        self._piper_executable = piper_executable
        self._process: subprocess.Popen | None = None
        self._voice_model: "PiperVoice | None" = None
        # Guards both lazy-loading the model and the synthesis call itself:
        # an ONNX Runtime inference session isn't guaranteed safe for
        # concurrent Run() calls, and Garvis never needs two TTS syntheses
        # in flight at once, so full serialization is the simplest correct
        # choice here (see stop(), below, for the one behavior trade-off).
        self._synthesis_lock = threading.Lock()

    @property
    def voice(self) -> str:
        """The configured voice name (Settings.tts_voice), exposed so
        callers/tests can verify which voice is actually wired up rather
        than assuming the constructor argument was honored."""
        return self._voice

    def _model_path(self) -> Path:
        model_path = self._voice_dir / f"{self._voice}.onnx"
        if not model_path.exists():
            raise TextToSpeechError(
                f"Voice model '{self._voice}' not found at {model_path}. Download it with: "
                f"python -m piper.download_voices --download-dir \"{self._voice_dir}\" {self._voice}"
            )
        return model_path

    def _command(self, model_path: Path, output_path: str) -> list[str]:
        if self._piper_executable:
            base = [self._piper_executable]
        else:
            base = [sys.executable, "-m", "piper"]
        return [*base, "--model", str(model_path), "--output_file", output_path]

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        text = _sanitize_for_tts(text)
        if self._piper_executable:
            return await asyncio.to_thread(self._synthesize_via_subprocess, text, output_path)
        return await asyncio.to_thread(self._synthesize_in_process, text, output_path)

    # --------------------------------------------------------- in-process
    def _get_voice_model_locked(self) -> "PiperVoice":
        """Load and cache the ONNX voice model once per provider instance.
        Must be called while holding `self._synthesis_lock` -- see its
        docstring for why, and the module docstring for why loading (not
        synthesis) is the cost worth caching."""
        if self._voice_model is None:
            from piper import PiperVoice  # imported lazily: heavy (onnxruntime), only needed here

            self._voice_model = PiperVoice.load(str(self._model_path()))
        return self._voice_model

    def _synthesize_in_process(self, text: str, output_path: str) -> str:
        with self._synthesis_lock:
            voice = self._get_voice_model_locked()
            try:
                with wave.open(output_path, "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file)
                _prepend_silence(output_path)
            except Exception as exc:
                raise TextToSpeechError(f"Piper failed to synthesize speech: {exc}") from exc
        return output_path

    # ------------------------------------------------- standalone executable
    def _synthesize_via_subprocess(self, text: str, output_path: str) -> str:
        model_path = self._model_path()
        command = self._command(model_path, output_path)

        try:
            try:
                stdin_bytes = text.encode("utf-8")
            except UnicodeEncodeError as exc:
                # Defense in depth: _sanitize_for_tts should already have
                # removed anything UTF-8 can't encode, but never let a gap
                # there crash the interaction -- fail this TTS turn cleanly
                # instead (VoicePipeline.speak/AgentBridge._safe_speak both
                # already turn TextToSpeechError into a non-fatal error
                # event rather than propagating it).
                raise TextToSpeechError(
                    "Couldn't synthesize speech: the response contained characters "
                    "Piper can't handle."
                ) from exc
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _, stderr = self._process.communicate(input=stdin_bytes)
            if self._process.returncode != 0:
                raise TextToSpeechError(
                    f"piper exited with code {self._process.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace')}"
                )
        except FileNotFoundError as exc:
            raise TextToSpeechError(
                "Piper executable not found. Install it with: pip install piper-tts "
                "(or set GARVIS_TTS_EXECUTABLE to a standalone piper binary)."
            ) from exc
        return output_path

    def stop(self) -> None:
        # Only meaningful for the standalone-executable path: in-process
        # synthesis has no subprocess to terminate. It's also a much
        # smaller window now -- warm synthesis takes ~0.1s, versus the
        # ~1.5s a subprocess call used to take -- so there's little for an
        # in-flight synthesis to interrupt in the first place.
        if self._process and self._process.poll() is None:
            self._process.terminate()
