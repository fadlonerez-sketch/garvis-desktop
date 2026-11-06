"""Tests for the diagnostic content_fingerprint helper (app/core/logging.py),
used to log enough about an LLM/assistant response to diagnose duplicate-
output bugs without ever writing the full text to the log."""
from __future__ import annotations

import hashlib
import json
import logging

import pytest

from app.core import logging as core_logging
from app.core.logging import content_fingerprint


@pytest.fixture
def excerpts_enabled():
    core_logging.set_content_excerpt_logging(True)
    yield
    core_logging.set_content_excerpt_logging(False)


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


def test_fingerprint_excerpt_is_short_and_never_the_full_long_text(excerpts_enabled):
    long_text = "word " * 100
    fp = content_fingerprint(long_text)
    assert len(fp["content_excerpt"]) <= 120
    assert fp["content_excerpt"] != long_text


def test_fingerprint_excerpt_flattens_whitespace(excerpts_enabled):
    fp = content_fingerprint("line one\nline two\t\ttabbed")
    assert "\n" not in fp["content_excerpt"]
    assert "\t" not in fp["content_excerpt"]


def test_fingerprint_omits_excerpt_by_default():
    fp = content_fingerprint("Your appointment with the doctor is at 9.")
    assert "content_excerpt" not in fp
    assert fp["content_length"] == len("Your appointment with the doctor is at 9.")
    assert "content_hash" in fp


def test_configure_logging_excerpt_opt_in_and_back_off(tmp_path):
    root = logging.getLogger("garvis")
    saved = (root.level, list(root.handlers), root.propagate)
    try:
        core_logging.configure_logging("INFO", None, log_content_excerpts=True)
        assert content_fingerprint("hello there")["content_excerpt"] == "hello there"
        core_logging.configure_logging("INFO", None)
        assert "content_excerpt" not in content_fingerprint("hello there")
    finally:
        core_logging.set_content_excerpt_logging(False)
        root.setLevel(saved[0])
        root.handlers[:] = saved[1]
        root.propagate = saved[2]


def test_settings_default_keeps_excerpts_off(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("GARVIS_LOG_CONTENT_EXCERPTS", raising=False)
    assert Settings(_env_file=None).log_content_excerpts is False
    monkeypatch.setenv("GARVIS_LOG_CONTENT_EXCERPTS", "true")
    assert Settings(_env_file=None).log_content_excerpts is True


def test_formatted_log_line_has_no_reply_text_by_default():
    record = logging.LogRecord("garvis.llm", logging.INFO, __file__, 1, "llm.response", None, None)
    record.extra_fields = content_fingerprint("secret plans for tuesday")
    line = core_logging.JsonFormatter().format(record)
    assert "secret plans" not in line
    assert json.loads(line)["content_length"] == len("secret plans for tuesday")
