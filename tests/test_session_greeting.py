"""Tests for ConversationSession's two additive wake-word hooks
(app/voice/session.py): an optional spoken `greeting` and an optional
`initial_noise_floor` seed. Both default to None, which must leave the
existing session behavior exactly as it was."""
from __future__ import annotations

import pytest

from app.voice.session import SessionState
from tests.test_conversation_session import (
    _build,
    _collect_states,
    _FakeAgent,
    _FakePlayer,
    _run_briefly,
    _ScriptedRecorder,
    _speech_burst,
    _split,
    _silence_chunk,
    _utterance_chunks,
)

_GREETING = "Yes, Sir. I'm listening."


@pytest.mark.asyncio
async def test_no_greeting_by_default_session_starts_by_listening():
    agent = _FakeAgent(["ok"])
    session, voice, events, player = _build(agent, _ScriptedRecorder())
    states = _collect_states(events)
    session.start()
    await _run_briefly(session, 50)
    await session.stop()
    assert states[0] == SessionState.LISTENING.value
    assert voice._tts.synthesize_calls == []
    assert player.play_calls == []


@pytest.mark.asyncio
async def test_greeting_is_spoken_once_via_tts_before_the_first_listen():
    agent = _FakeAgent(["ok"])
    session, voice, events, player = _build(agent, _ScriptedRecorder(), greeting=_GREETING)
    states = _collect_states(events)
    session.start()
    await _run_briefly(session, 100)
    await session.stop()
    assert voice._tts.synthesize_calls == [_GREETING]
    assert len(player.play_calls) == 1
    assert states[:2] == [SessionState.SPEAKING.value, SessionState.LISTENING.value]


@pytest.mark.asyncio
async def test_greeting_never_reaches_the_agent():
    agent = _FakeAgent(["Two plus two is four."])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder, greeting=_GREETING)
    session.start()
    await _run_briefly(session, 50)
    assert agent.calls == []  # greeting spoken, nothing sent to the LLM
    recorder.queue(*_utterance_chunks())
    await _run_briefly(session, 300)
    await session.stop()
    assert agent.calls == ["what is two plus two"]  # only the user's command
    assert voice._tts.synthesize_calls[0] == _GREETING
    assert voice._tts.synthesize_calls.count(_GREETING) == 1


@pytest.mark.asyncio
async def test_user_can_barge_in_during_the_greeting():
    agent = _FakeAgent(["ok"])
    player = _FakePlayer(finish_after_polls=None)  # greeting keeps playing until stopped
    recorder = _ScriptedRecorder(
        _split(0.35, 0.05, _silence_chunk) + _speech_burst(0.4, amplitude=4000) + _split(0.5, 0.05, _silence_chunk)
    )
    session, voice, events, _ = _build(
        agent, recorder, player=player, greeting=_GREETING, initial_noise_floor=50.0
    )
    states = _collect_states(events)
    session.start()
    await _run_briefly(session, 400)
    await session.stop()
    assert player.stop_calls >= 1
    assert SessionState.INTERRUPTED.value in states
    assert agent.calls == ["what is two plus two"]


@pytest.mark.asyncio
async def test_blank_greeting_is_treated_as_no_greeting():
    agent = _FakeAgent([])
    session, voice, events, _ = _build(agent, _ScriptedRecorder(), greeting="   ")
    states = _collect_states(events)
    session.start()
    await _run_briefly(session, 50)
    await session.stop()
    assert voice._tts.synthesize_calls == []
    assert states[0] == SessionState.LISTENING.value


# --------------------------------------------------------- noise floor
@pytest.mark.asyncio
async def test_initial_noise_floor_defaults_to_none():
    session, _, _, _ = _build(_FakeAgent([]), _ScriptedRecorder())
    assert session._noise_floor is None


@pytest.mark.asyncio
async def test_seeded_noise_floor_skips_calibration_so_immediate_speech_is_heard():
    """Speech from the very first sample: an unseeded VAD would absorb it
    into its calibration window; a floor seeded from standby is already
    calibrated, so the command is captured."""
    recorder = _ScriptedRecorder(_speech_burst(0.3) + _split(0.4, 0.05, _silence_chunk))
    agent = _FakeAgent(["ok"])
    session, _, _, _ = _build(agent, recorder, initial_noise_floor=40.0)
    session.start()
    await _run_briefly(session, 300)
    await session.stop()
    assert agent.calls == ["what is two plus two"]


@pytest.mark.asyncio
async def test_unseeded_session_still_calibrates_as_before():
    recorder = _ScriptedRecorder(_speech_burst(0.3) + _split(0.4, 0.05, _silence_chunk))
    agent = _FakeAgent(["ok"])
    session, _, _, _ = _build(agent, recorder)
    session.start()
    await _run_briefly(session, 300)
    await session.stop()
    assert agent.calls == []  # existing behavior: absorbed during calibration


@pytest.mark.asyncio
async def test_seeded_floor_is_carried_into_the_greeting_barge_in_vad():
    """The greeting's speaking-phase VAD must not calibrate (that window
    would overlap the greeting's own playback) when a floor is seeded."""
    from app.voice import session as session_module

    created = []
    real_vad = session_module.VoiceActivityDetector

    def _spy(*args, **kwargs):
        vad = real_vad(*args, **kwargs)
        created.append(vad)
        return vad

    session_module.VoiceActivityDetector = _spy
    try:
        session, _, _, _ = _build(
            _FakeAgent([]), _ScriptedRecorder(), greeting=_GREETING, initial_noise_floor=75.0
        )
        session.start()
        await _run_briefly(session, 30)
        await session.stop()
    finally:
        session_module.VoiceActivityDetector = real_vad
    assert created, "greeting should create a barge-in VAD"
    assert not created[0].is_calibrating
    assert created[0].noise_floor == pytest.approx(75.0, rel=0.2)
