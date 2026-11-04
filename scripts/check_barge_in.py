"""Interactive check of barge-in on YOUR speakers and microphone.

Uses the production voice path -- build_voice_pipeline() (Piper TTS, the
sounddevice player, the timestamped recorder, EchoGuard) and the session's
own barge-in loop (ConversationSession._speak_interruptibly). No LLM, no
API credits. Speakers must be on (not muted) at your normal volume.

  Round 1: Garvis reads 3 sentences. Stay silent.
           Expect: no interruptions, no phantom utterances afterwards.
  Round 2: Garvis reads 3 sentences. About 2 seconds into each, say
           "stop" (or anything) at your normal voice.
           Expect: each one is interrupted shortly after you start talking.

Run with: python scripts/check_barge_in.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.events import EventBus  # noqa: E402
from app.voice.build import build_voice_pipeline  # noqa: E402
from app.voice.session import ConversationSession  # noqa: E402
from app.voice.vad import VoiceActivityDetector  # noqa: E402

SENTENCES = [
    "Certainly, Sir. I have opened the project folder, and the failing test is in the parser module.",
    "Here is your schedule for tomorrow. At nine, a call with the design team, and at eleven thirty, lunch.",
    "The download finished. Two files were saved to your desktop, and one failed because the server refused it.",
]


class _NoAgent:
    async def handle_user_message(self, text: str) -> str:
        return ""


async def main() -> None:
    settings = get_settings()
    voice = build_voice_pipeline(settings)
    session = ConversationSession(
        agent=_NoAgent(), voice=voice, events=EventBus(),
        poll_interval_seconds=settings.session_poll_interval_seconds,
        energy_threshold=settings.session_energy_threshold,
        min_speech_seconds=settings.session_min_speech_seconds,
        min_silence_seconds=settings.session_min_silence_seconds,
        barge_in_energy_threshold=settings.session_barge_in_energy_threshold,
        max_utterance_seconds=settings.session_max_utterance_seconds,
    )
    voice.start_continuous_listening()
    print("Learning the room's noise level (2 s, stay quiet)...")
    calibration = VoiceActivityDetector(16000, settings.session_energy_threshold, 0.15, 0.7)
    end = time.perf_counter() + 2.0
    while time.perf_counter() < end:
        chunk, residual = voice.read_microphone_chunk_with_echo()
        if chunk:
            calibration.process(chunk, residual_energy=residual)
        await asyncio.sleep(0.03)
    session._noise_floor = calibration.noise_floor

    async def phantoms_after(seconds: float = 1.5) -> int:
        vad = VoiceActivityDetector(16000, settings.session_energy_threshold, settings.session_min_speech_seconds,
                                    settings.session_min_silence_seconds, initial_noise_floor=session._noise_floor)
        count = 0
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            chunk, residual = voice.read_microphone_chunk_with_echo()
            if chunk and vad.process(chunk, residual_energy=residual) == "speech_start":
                count += 1
            await asyncio.sleep(settings.session_poll_interval_seconds)
        return count

    print("\nROUND 1 -- stay silent while Garvis speaks.")
    false_interruptions = phantoms = 0
    for text in SENTENCES:
        started = time.perf_counter()
        interrupted, _ = await session._speak_interruptibly(text)
        if interrupted is not None:
            false_interruptions += 1
            voice.stop_speaking()
            print(f"  FALSE interruption after {time.perf_counter() - started:.1f} s")
        phantoms += await phantoms_after()
    print(f"  false interruptions: {false_interruptions}/{len(SENTENCES)}, phantom utterances after replies: {phantoms}")

    print("\nROUND 2 -- about 2 seconds into each sentence, say something at your normal voice.")
    await asyncio.sleep(1.5)
    caught = 0
    for text in SENTENCES:
        started = time.perf_counter()
        interrupted, _ = await session._speak_interruptibly(text)
        elapsed = time.perf_counter() - started
        if interrupted is not None:
            caught += 1
            voice.stop_speaking()
            print(f"  interrupted {elapsed:.1f} s into the sentence")
        else:
            print(f"  NOT interrupted (sentence finished after {elapsed:.1f} s)")
        await asyncio.sleep(2.0)
    print(f"  interruptions detected: {caught}/{len(SENTENCES)}")
    guard = voice._echo
    if guard is not None:
        print(f"\nLearned echo delay {guard.delay * 1000:.0f} ms.")
    voice.stop_continuous_listening()


if __name__ == "__main__":
    asyncio.run(main())
