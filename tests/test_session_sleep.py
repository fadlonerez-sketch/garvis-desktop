"""ConversationSession's sleep hook (app/voice/session.py): a standby command
or closing statement is never sent to the Agent; a fixed acknowledgement is
spoken and the owner is asked to end the session. Without a sleep matcher
(wake mode off) every utterance reaches the Agent exactly as before."""

from __future__ import annotations

import pytest

from app.voice.wake import SleepCommandMatcher
from tests.test_conversation_session import (
    _build,
    _FakeAgent,
    _FakePlayer,
    _run_briefly,
    _ScriptedRecorder,
    _silence_chunk,
    _speech_burst,
    _split,
    _utterance_chunks,
)


def _sleepy_session(agent, recorder, player=None, **overrides):
    requests = []
    session, voice, events, player = _build(
        agent,
        recorder,
        player=player,
        sleep_matcher=SleepCommandMatcher(),
        on_sleep=requests.append,
        **overrides,
    )
    return session, voice, player, requests


@pytest.mark.asyncio
async def test_standby_command_never_reaches_the_agent_and_requests_sleep():
    agent = _FakeAgent(["should never be used"])
    session, voice, _, requests = _sleepy_session(agent, _ScriptedRecorder(_utterance_chunks()))
    voice._stt.text = "Go to standby."
    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == []
    assert voice._tts.synthesize_calls == ["Entering standby, Sir."]
    assert requests == [session]
    assert not session.is_active  # loop finished: no further listening
    await session.stop()


@pytest.mark.asyncio
async def test_nothing_is_captured_after_the_sleep_request():
    agent = _FakeAgent(["reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, _, requests = _sleepy_session(agent, recorder)
    voice._stt.text = "Go to sleep."
    session.start()
    await _run_briefly(session, 300)
    voice._stt.text = "what time is it"
    recorder.queue(*_utterance_chunks())
    await _run_briefly(session, 300)
    await session.stop()
    assert agent.calls == []
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_goodnight_is_acknowledged_with_goodnight():
    session, voice, _, requests = _sleepy_session(_FakeAgent([]), _ScriptedRecorder(_utterance_chunks()))
    voice._stt.text = "Goodnight, Garvis."
    session.start()
    await _run_briefly(session, 300)
    await session.stop()
    assert voice._tts.synthesize_calls == ["Goodnight, Sir."]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_closing_statement_after_a_normal_turn_returns_to_standby():
    agent = _FakeAgent(["It is four."])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, _, requests = _sleepy_session(agent, recorder)
    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == ["what is two plus two"]
    assert requests == []

    voice._stt.text = "That's all for now, thank you."
    recorder.queue(*_utterance_chunks())
    await _run_briefly(session, 300)
    await session.stop()
    assert agent.calls == ["what is two plus two"]  # closing line not sent
    assert voice._tts.synthesize_calls == ["It is four.", "Entering standby, Sir."]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_thanks_mid_conversation_does_not_end_the_session():
    agent = _FakeAgent(["You're welcome.", "Sunny."])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, _, requests = _sleepy_session(agent, recorder)
    voice._stt.text = "Thanks."
    session.start()
    await _run_briefly(session, 300)
    voice._stt.text = "Thanks, and what's the weather tomorrow?"
    recorder.queue(*_utterance_chunks())
    await _run_briefly(session, 300)
    assert session.is_active
    await session.stop()
    assert agent.calls == ["Thanks.", "Thanks, and what's the weather tomorrow?"]
    assert requests == []


@pytest.mark.asyncio
async def test_short_silence_does_not_end_the_session():
    agent = _FakeAgent(["ok"])
    recorder = _ScriptedRecorder(_split(3.0, 0.05, _silence_chunk))
    session, _, _, requests = _sleepy_session(agent, recorder)
    session.start()
    await _run_briefly(session, 300)
    assert session.is_active
    await session.stop()
    assert requests == []
    assert agent.calls == []


@pytest.mark.asyncio
async def test_without_a_sleep_matcher_the_command_goes_to_the_agent_as_before():
    agent = _FakeAgent(["Okay."])
    session, voice, _, _ = _build(agent, _ScriptedRecorder(_utterance_chunks()))
    voice._stt.text = "Go to standby."
    session.start()
    await _run_briefly(session, 300)
    assert session.is_active
    await session.stop()
    assert agent.calls == ["Go to standby."]


@pytest.mark.asyncio
async def test_interrupting_the_acknowledgement_still_goes_to_sleep():
    player = _FakePlayer(finish_after_polls=None)  # ack keeps playing until barged in
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, player, requests = _sleepy_session(
        _FakeAgent([]), recorder, player=player, initial_noise_floor=50.0
    )
    voice._stt.text = "Go to standby."
    session.start()
    await _run_briefly(session, 300)
    assert player.play_calls  # acknowledgement started
    recorder.queue(*_speech_burst(0.4, amplitude=4000), *_split(0.5, 0.05, _silence_chunk))
    await _run_briefly(session, 300)
    await session.stop()
    assert player.stop_calls >= 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_ending_the_session_during_the_acknowledgement_does_not_request_sleep():
    player = _FakePlayer(finish_after_polls=None)
    session, voice, _, requests = _sleepy_session(_FakeAgent([]), _ScriptedRecorder(_utterance_chunks()), player=player)
    voice._stt.text = "Go to standby."
    session.start()
    await _run_briefly(session, 300)
    await session.stop()  # user clicked End Session mid-acknowledgement
    assert requests == []


# ------------------------------------------------- natural-language standby
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spoken", "ack"),
    [
        ("Jarvis, go to sleep.", "Entering standby, Sir."),
        ("Good night, Jarvis.", "Goodnight, Sir."),
        ("Switch to standby mode.", "Entering standby, Sir."),
        ("Go ahead and stand by.", "Entering standby, Sir."),
        ("I'm done for now, go to sleep.", "Entering standby, Sir."),
    ],
)
async def test_natural_standby_requests_never_reach_the_agent(spoken, ack):
    agent = _FakeAgent(["should never be used"])
    session, voice, _, requests = _sleepy_session(agent, _ScriptedRecorder(_utterance_chunks()))
    voice._stt.text = spoken
    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == []
    assert voice._tts.synthesize_calls == [ack]
    assert requests == [session]
    await session.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoken",
    ["Don't go to sleep.", "What does standby mode mean?", "Shut down, Garvis.", "Quit."],
)
async def test_negations_questions_and_shutdown_still_reach_the_agent(spoken):
    agent = _FakeAgent(["Okay."])
    session, voice, _, requests = _sleepy_session(agent, _ScriptedRecorder(_utterance_chunks()))
    voice._stt.text = spoken
    session.start()
    await _run_briefly(session, 300)
    assert session.is_active
    await session.stop()
    assert agent.calls == [spoken]
    assert requests == []


@pytest.mark.asyncio
async def test_standby_spoken_over_a_reply_stops_it_and_goes_to_sleep():
    player = _FakePlayer(finish_after_polls=None)  # the reply keeps playing until barged in
    recorder = _ScriptedRecorder(_utterance_chunks())
    agent = _FakeAgent(["A long answer that is still playing."])
    session, voice, player, requests = _sleepy_session(agent, recorder, player=player)
    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == ["what is two plus two"]
    assert player.is_playing  # Garvis is speaking

    voice._stt.text = "Jarvis, go to sleep."
    recorder.queue(*_speech_burst(0.4, amplitude=4000), *_split(0.5, 0.05, _silence_chunk))
    await _run_briefly(session, 500)
    assert player.stop_calls >= 1  # the reply was cut off by the barge-in
    assert agent.calls == ["what is two plus two"]  # the command itself never reached the Agent
    assert voice._tts.synthesize_calls[-1] == "Entering standby, Sir."
    player.stop()  # let the acknowledgement "finish"
    await _run_briefly(session, 200)
    await session.stop()
    assert requests == [session]


@pytest.mark.asyncio
async def test_a_second_standby_request_after_the_first_is_not_processed():
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, _, requests = _sleepy_session(_FakeAgent([]), recorder)
    voice._stt.text = "Go to sleep."
    session.start()
    await _run_briefly(session, 300)
    recorder.queue(*_utterance_chunks())  # said again right away
    await _run_briefly(session, 300)
    await session.stop()
    assert requests == [session]  # one request; the loop ended after the first
    assert voice._tts.synthesize_calls == ["Entering standby, Sir."]
