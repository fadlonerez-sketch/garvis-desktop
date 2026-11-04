"""Tests for WakePhraseMatcher and GreetingPicker (app/voice/wake.py) --
pure text logic, no audio."""
from __future__ import annotations

import random

import pytest

from app.voice.wake import GREETINGS, WAKE_PHRASES, GreetingPicker, WakePhraseMatcher

_EXPECTED_GREETINGS = (
    "Yes, Sir. Welcome back.",
    "Welcome back, Sir. I'm ready.",
    "Yes, Sir. I'm listening.",
    "At your service, Sir.",
    "Good to have you back, Sir.",
    "I'm here, Sir. How may I assist you?",
    "Certainly, Sir. I'm ready.",
    "Welcome, Sir. Systems are ready.",
    "Yes, Sir. Garvis is standing by.",
)


def test_wake_phrases_are_exactly_the_approved_three():
    assert WAKE_PHRASES == ("Hey Garvis", "Hey Daddy", "Wake up, Daddy is home")


@pytest.mark.parametrize(
    "transcript, phrase",
    [
        ("Hey Garvis.", "Hey Garvis"),
        ("hey garvis", "Hey Garvis"),
        ("HEY, GARVIS!", "Hey Garvis"),
        ("Hey Daddy.", "Hey Daddy"),
        ("Hey, daddy!", "Hey Daddy"),
        ("Wake up, Daddy is home.", "Wake up, Daddy is home"),
        ("wake up daddy is home", "Wake up, Daddy is home"),
        ("Wake up! Daddy's home.", "Wake up, Daddy is home"),
        ("Wake up, Daddy’s home.", "Wake up, Daddy is home"),
    ],
)
def test_matches_each_phrase_with_punctuation_and_case_variants(transcript, phrase):
    match = WakePhraseMatcher().match(transcript)
    assert match is not None
    assert match.phrase == phrase


@pytest.mark.parametrize(
    "transcript",
    ["Hey Jarvis.", "Hey Garbis", "Hey Carvis.", "Hey Garvas", "Hey Gervis", "Hay Garvis", "Hey Gar vis"],
)
def test_tolerates_whisper_misspellings_of_garvis(transcript):
    match = WakePhraseMatcher().match(transcript)
    assert match is not None and match.phrase == "Hey Garvis"


@pytest.mark.parametrize("transcript", ["Hey Daddie", "Hey Dady", "Hey dad dy"])
def test_tolerates_whisper_misspellings_of_daddy(transcript):
    match = WakePhraseMatcher().match(transcript)
    assert match is not None and match.phrase == "Hey Daddy"


def test_phrase_inside_a_longer_transcript_is_found_and_remainder_is_kept():
    match = WakePhraseMatcher().match("Okay. Hey Garvis, open Chrome please.")
    assert match is not None
    assert match.phrase == "Hey Garvis"
    assert match.remainder == "open chrome please"


def test_remainder_is_empty_for_a_bare_wake_phrase():
    assert WakePhraseMatcher().match("Hey Garvis.").remainder == ""


@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "   ",
        "Hey",
        "Garvis",
        "Daddy",
        "Daddy is home",
        "wake up",
        "hey there",
        "Hey buddy",
        "Hey baby",
        "harvest moon",
        "Hey Harvest",
        "garbage day",
        "hey garbage",
        "Hey Travis",
        "Hey Gary",
        "Hey Marvin",
        "What is the weather today?",
        "Wake up Danny is home",
    ],
)
def test_rejects_non_wake_speech(transcript):
    assert WakePhraseMatcher().match(transcript) is None


def test_threshold_is_configurable():
    assert WakePhraseMatcher(threshold=1.0).match("Hey Garbis") is not None  # known variant
    assert WakePhraseMatcher(threshold=1.0).match("Hey Garvys") is None
    assert WakePhraseMatcher(threshold=0.75).match("Hey Garvys") is not None


# ------------------------------------------------------------ greetings
def test_greeting_pool_is_exactly_the_approved_nine():
    assert GREETINGS == _EXPECTED_GREETINGS


def test_greetings_only_come_from_the_pool():
    picker = GreetingPicker(rng=random.Random(1))
    for _ in range(200):
        assert picker.next() in _EXPECTED_GREETINGS


def test_never_the_same_greeting_twice_in_a_row():
    picker = GreetingPicker(rng=random.Random(7))
    previous = None
    for _ in range(1000):
        greeting = picker.next()
        assert greeting != previous
        previous = greeting


def test_every_greeting_is_eventually_used():
    picker = GreetingPicker(rng=random.Random(3))
    assert {picker.next() for _ in range(500)} == set(_EXPECTED_GREETINGS)


def test_single_greeting_pool_still_works():
    picker = GreetingPicker(greetings=["Only one."])
    assert [picker.next() for _ in range(3)] == ["Only one."] * 3


def test_empty_greeting_pool_is_rejected():
    with pytest.raises(ValueError):
        GreetingPicker(greetings=[])
