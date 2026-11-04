"""Tests for the diagnostic content_fingerprint helper (app/core/logging.py),
used to log enough about an LLM/assistant response to diagnose duplicate-
output bugs without ever writing the full text to the log."""
from __future__ import annotations

import hashlib

from app.core.logging import content_fingerprint


def test_fingerprint_reports_exact_length():
    assert content_fingerprint("hello")["content_length"] == 5
    assert content_fingerprint("")["content_length"] == 0


def test_fingerprint_hash_is_deterministic_and_content_addressed():
    a = content_fingerprint("Your name is Alex, Sir.")
    b = content_fingerprint("Your name is Alex, Sir.")
    c = content_fingerprint("Something else entirely.")
    assert a["content_hash"] == b["content_hash"]
    assert a["content_hash"] != c["content_hash"]
    assert a["content_hash"] == hashlib.sha256(b"Your name is Alex, Sir.").hexdigest()[:12]


def test_fingerprint_excerpt_is_short_and_never_the_full_long_text():
    long_text = "word " * 100
    fp = content_fingerprint(long_text)
    assert len(fp["content_excerpt"]) <= 120
    assert fp["content_excerpt"] != long_text


def test_fingerprint_excerpt_flattens_whitespace():
    fp = content_fingerprint("line one\nline two\t\ttabbed")
    assert "\n" not in fp["content_excerpt"]
    assert "\t" not in fp["content_excerpt"]
