"""Tests for SleepCommandMatcher (app/voice/wake.py): explicit standby
commands and unambiguous closing statements -- pure text logic."""

from __future__ import annotations

import pytest

from app.voice.wake import GREETINGS, SleepCommandMatcher, WakePhraseMatcher


@pytest.mark.parametrize(
    "transcript",
    [
        "Go to standby.",
        "Go into standby.",
        "Go to sleep.",
        "Goodnight, Garvis.",
        "Enter standby mode.",
    ],
)
def test_the_five_explicit_standby_commands(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "go to standby",
        "GO TO SLEEP!",
        "Go to standby mode.",
        "Go to stand by.",  # Whisper splitting "standby"
        "Good night, Garvis.",  # Whisper splitting "goodnight"
        "Goodnight, Jarvis!",  # Whisper misspelling "Garvis"
        "Okay Garvis, go to standby please.",
        "Garvis, go to sleep now.",
        "Enter standby.",
    ],
)
def test_explicit_command_variations(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "That's all.",
        "That's all for now.",
        "I'm done.",
        "We're done.",
        "We're finished.",
        "That's it.",
        "Thank you, Garvis.",
        "Thanks, Garvis.",
        "Goodnight.",
        "You can go back to standby.",
        "Okay, that's all for now. Thank you.",
        "No, that will be all.",
        "Thanks, that is all.",
        "Perfect, thank you Garvis.",
    ],
)
def test_natural_closing_statements(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "Shut yourself down.",
        "Shut down.",
        "Thanks.",
        "Thank you.",
        "Thank you so much.",
        "Okay.",
        "Good.",
        "Thanks for watching!",  # classic Whisper hallucination
        "Thanks, what's the weather tomorrow?",
        "That's it, now open Chrome.",
        "Is that it?",
        "I'm done with the report, can you email it?",
        "We are done with step one, next step please.",
        "How do I make my computer go to sleep?",
        "What does standby mode do?",
        "Go to the store.",
        "Go to settings.",
        "Standby.",
        "Sleep.",
        "Garvis.",
        "",
    ],
)
def test_ordinary_speech_does_not_end_the_session(transcript):
    assert SleepCommandMatcher().match(transcript) is None


def test_acknowledgement_is_goodnight_only_for_goodnight():
    matcher = SleepCommandMatcher()
    assert matcher.match("Goodnight, Garvis.").acknowledgement == "Goodnight, Sir."
    assert matcher.match("Good night.").acknowledgement == "Goodnight, Sir."
    assert matcher.match("Go to standby.").acknowledgement == "Entering standby, Sir."
    assert matcher.match("That's all for now.").acknowledgement == "Entering standby, Sir."


def test_acknowledgements_and_greetings_can_never_wake_garvis():
    """Self-trigger protection at the text level: nothing Garvis itself says
    around a wake/sleep transition contains a wake phrase."""
    wake = WakePhraseMatcher()
    for spoken in ("Goodnight, Sir.", "Entering standby, Sir.", *GREETINGS):
        assert wake.match(spoken) is None, spoken


def test_threshold_is_configurable():
    assert SleepCommandMatcher(threshold=0.75).match("Go to standbi.") is not None
    assert SleepCommandMatcher(threshold=1.0).match("Go to standbi.") is None


# ------------------------------------------------- natural-language standby
@pytest.mark.parametrize(
    "transcript",
    [
        "Go to sleep.",
        "Good night, Jarvis.",
        "Jarvis, go to sleep.",
        "Jarvis, enter standby mode.",
        "Enter standby.",
        "Go into standby.",
        "Switch to standby mode.",
        "Jarvis, stand by.",
        "You can go to sleep now.",
        "That's all for now, go into standby.",
        "Rest for now, Jarvis.",
        "Time to sleep.",
        "Good night.",
        "You can sleep now.",
        "Go ahead and stand by.",
        "I'm done for now, go to sleep.",
    ],
)
def test_natural_standby_requests(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "Hey Jarvis, go to sleep.",
        "Garvis, standby.",
        "Okay Garvis, you can go to standby now, thanks.",
        "Go back to sleep.",
        "Go back to standby mode.",
        "Return to standby.",
        "Switch into sleep mode.",
        "Put yourself in standby mode.",
        "Alright, time for bed.",
        "Get some rest, Jarvis.",
        "Sleep now.",
        "Garvis, sleep mode.",
        "Could you go to standby, please?",
        "Can you go to sleep now?",
        "Thanks, Garvis. Good night.",
        "Okay, that's it for today. Goodnight, Sir.",
        "No, that's all. Go to sleep.",
    ],
)
def test_additional_standby_variations(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "jarvis go to sleep",
        "JARVIS, ENTER STANDBY MODE!!!",
        "  Go   to   sleep  ",
        "go to sleep...",
        "Good-night, Jarvis.",
        "Goodnight Jarvis",
        "You can go to sleep now!",
        "That’s all for now, Jarvis.",  # curly apostrophe
    ],
)
def test_case_punctuation_and_spacing_do_not_matter(transcript):
    assert SleepCommandMatcher().match(transcript) is not None


@pytest.mark.parametrize(
    "transcript",
    [
        "Don't go to sleep.",
        "Do not go to sleep.",
        "Never enter standby mode.",
        "Please don't go into standby.",
        "Jarvis, don't stand by.",
        "No sleep.",
        "No standby, Jarvis.",
        "Stop going to sleep.",
        "Not now, go to sleep later.",
    ],
)
def test_negated_commands_do_not_trigger_standby(transcript):
    assert SleepCommandMatcher().match(transcript) is None


@pytest.mark.parametrize(
    "transcript",
    [
        "What does standby mode mean?",
        "How do I put a computer into standby?",
        "Can you explain why you went to sleep?",
        "Why did you go to sleep?",
        "Go to sleep?",
        "That's all?",
        "Should you go to sleep?",
        "Can you sleep?",
        "Jarvis, can you sleep?",
        "Do you ever sleep?",
        "Is it time to sleep?",
        "I couldn't sleep last night.",
        "I need to go to sleep early tonight.",
        "My laptop keeps going to sleep.",
        "Put my computer to sleep.",
        "Remind me to go to sleep at eleven.",
        "Open Chrome and go to sleep.",
        "Stand by me.",
        "Stand by.",
        "Standby mode.",
        "Sleep mode.",
        "Rest.",
        "Test.",  # "test" must never fuzzy-match "rest"
        "Test now.",
        "Jarvis.",
        "Good.",
        "Garvis, thanks.",
        "Standby mode is a low-power state that saves energy.",
    ],
)
def test_questions_and_mentions_do_not_trigger_standby(transcript):
    assert SleepCommandMatcher().match(transcript) is None


@pytest.mark.parametrize(
    "transcript",
    [
        "Shut down.",
        "Shut down, Garvis.",
        "Garvis, shut yourself down.",
        "Quit.",
        "Quit the application.",
        "Exit.",
        "Exit the application.",
        "Turn off.",
        "Power off.",
        "Close Garvis.",
        "Restart the computer.",
        "Good night, shut down.",
        "Go to sleep and shut down.",
    ],
)
def test_shutdown_requests_are_never_standby(transcript):
    assert SleepCommandMatcher().match(transcript) is None


def test_new_phrases_use_the_existing_acknowledgements():
    matcher = SleepCommandMatcher()
    assert matcher.match("Good night, Jarvis.").acknowledgement == "Goodnight, Sir."
    assert matcher.match("Jarvis, stand by.").acknowledgement == "Entering standby, Sir."
    assert matcher.match("Switch to standby mode.").acknowledgement == "Entering standby, Sir."
    assert matcher.match("You can sleep now.").acknowledgement == "Entering standby, Sir."


def test_long_messages_are_never_standby_commands():
    long_text = "Go to sleep. " + "Thanks. " * 40
    assert SleepCommandMatcher().match(long_text) is None


def test_matching_is_repeatable():
    matcher = SleepCommandMatcher()
    first = matcher.match("Jarvis, go to sleep.")
    assert first is not None
    assert matcher.match("Jarvis, go to sleep.") == first
