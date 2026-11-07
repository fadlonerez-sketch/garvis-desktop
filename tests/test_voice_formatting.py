"""Tests for the deterministic spoken-text formatter (app/voice/formatting.py).

This is the fix for Garvis's "restaurant answers sound like a raw database
record" complaint: the same text the Agent produces for the GUI, adapted
for speech only at the voice boundary. Purely deterministic string
transforms -- no LLM involved -- so these tests assert exact behavior, not
model creativity.
"""

from __future__ import annotations

import re

import pytest

from app.voice.formatting import to_spoken_text


def test_plain_prose_is_left_essentially_unchanged():
    text = "The sky is blue because of Rayleigh scattering, Sir."
    assert to_spoken_text(text) == text


def test_multi_paragraph_prose_is_flattened_without_becoming_a_list():
    text = "First paragraph of the explanation.\n\nSecond paragraph continues here."
    result = to_spoken_text(text)
    assert "\n" not in result
    assert "First paragraph of the explanation." in result
    assert "Second paragraph continues here." in result


def test_bulleted_restaurant_list_becomes_natural_sentences():
    text = (
        "Sir, here are three options:\n"
        "- **Pizzeria Roma** (Italian) - https://example.com/pizzeria-roma?ref=search\n"
        "- **Sakura Sushi** (Japanese)\n"
        "- **Olive Grove** (Mediterranean) - www.olivegrove.example.com"
    )
    result = to_spoken_text(text)

    # names preserved intact -- never truncated or invented
    assert "Pizzeria Roma" in result
    assert "Sakura Sushi" in result
    assert "Olive Grove" in result
    # no raw URLs read aloud
    assert "http" not in result
    assert "www." not in result
    # no literal markdown syntax
    assert "*" not in result
    assert "-\n" not in result and "\n" not in result


def test_numbered_list_becomes_natural_sentences():
    text = "1. First step\n2. Second step\n3. Third step"
    assert to_spoken_text(text) == "First step. Second step. Third step"


def test_urls_are_removed_but_surrounding_content_kept():
    text = "Check out https://example.com/some/long/path?query=1&x=2 for more, Sir."
    result = to_spoken_text(text)
    assert "http" not in result
    assert "Check out" in result
    assert "for more" in result


def test_markdown_emphasis_markers_are_stripped_but_text_kept():
    text = "The **best** option is *clearly* Sakura Sushi."
    result = to_spoken_text(text)
    assert "*" not in result
    assert "best" in result
    assert "clearly" in result
    assert "Sakura Sushi" in result


def test_markdown_headers_are_stripped():
    text = "## Results\nHere is what I found."
    result = to_spoken_text(text)
    assert "#" not in result
    assert "Results" in result


def test_ratings_and_review_counts_are_not_stripped():
    """The formatter must never remove substantive information -- only
    presentation syntax. Whether a rating/review count belongs in the
    answer at all is the model's decision (see personality policy), not
    something a regex should silently delete, since the user may have
    explicitly asked for exactly that detail."""
    text = "Sakura Sushi is rated 4.5 out of 5 with 320 reviews, Sir."
    assert to_spoken_text(text) == text


def test_phone_numbers_and_addresses_are_not_stripped():
    text = "The address is 123 Main Street and the phone number is 555-123-4567, Sir."
    assert to_spoken_text(text) == text


def test_restaurant_names_are_never_altered():
    text = "- The Grand Old House on the Waterfront (Seafood)\n- Nonna's Kitchen and Wine Bar (Italian)"
    result = to_spoken_text(text)
    assert "The Grand Old House on the Waterfront" in result
    assert "Nonna's Kitchen and Wine Bar" in result


def test_hebrew_and_emoji_pass_through_untouched():
    text = "שלום עולם \U0001f600 Sir, done."
    assert to_spoken_text(text) == text


def test_empty_and_whitespace_only_text_is_returned_unchanged():
    assert to_spoken_text("") == ""
    assert to_spoken_text("   ") == "   "


def test_text_that_is_only_a_url_falls_back_to_original_rather_than_going_blank():
    text = "https://example.com/only-a-url"
    assert to_spoken_text(text) == text


def test_short_action_confirmation_is_unaffected():
    for text in ("Done, Sir.", "Opening Chrome, Sir.", "Email sent, Sir."):
        assert to_spoken_text(text) == text


# ---------------------------------------------- natural TTS delivery fixes
def test_ascii_double_dash_becomes_a_pause_not_a_dropped_word_boundary():
    """Regression test: Piper's espeak-ng frontend silently drops an ASCII
    '--' used as a stylistic em-dash (as the personality prompt itself
    sometimes writes), running the two clauses together with no pause at
    all -- verified directly against piper.phonemize_espeak.EspeakPhonemizer.
    It must be normalized to a comma, which reliably produces a pause."""
    text = "I found it -- but it needs a restart."
    assert to_spoken_text(text) == "I found it, but it needs a restart."


def test_ascii_single_dash_with_spaces_becomes_a_pause():
    text = "I found it - but it needs a restart."
    assert to_spoken_text(text) == "I found it, but it needs a restart."


def test_ellipsis_becomes_a_period_not_a_merged_word_boundary():
    """Regression test: three or more dots don't just fail to pause -- they
    drop the word boundary entirely and merge the surrounding words into a
    garbled run when spoken. Collapsing to a single period fixes this and
    gives a clean sentence break."""
    text = "I found it... but it needs a restart."
    assert to_spoken_text(text) == "I found it. but it needs a restart."


def test_hyphenated_words_and_phone_numbers_are_not_treated_as_dashes():
    """A real compound-word hyphen or a phone number's single hyphens (no
    surrounding spaces, not doubled) must never be touched -- only a dash
    used as standalone punctuation is normalized."""
    text = "It's a well-known, state-of-the-art fix; call 555-123-4567."
    assert to_spoken_text(text) == text


# ------------------------------------------------ pronunciation overrides
@pytest.fixture
def name_override(monkeypatch):
    """The shipped table is empty (users add their own names); install one entry."""
    import app.voice.formatting as formatting

    monkeypatch.setattr(formatting, "_PRONUNCIATION_OVERRIDES", {re.compile(r"\bAlex\b", re.IGNORECASE): "Aleks"})


def test_name_is_respelled_for_correct_pronunciation(name_override):
    """A name in the override table reaches TTS as its respelling."""
    assert to_spoken_text("Your name is Alex, Sir.") == "Your name is Aleks, Sir."


def test_name_override_is_case_insensitive_and_word_bounded(name_override):
    assert to_spoken_text("alex") == "Aleks"
    # must not match inside a longer word
    assert to_spoken_text("Alexander") == "Alexander"


def test_no_overrides_ship_by_default():
    assert to_spoken_text("Your name is Alex, Sir.") == "Your name is Alex, Sir."
