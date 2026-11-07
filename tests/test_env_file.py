"""Tests for scripts/env_file.py's .env writer -- it edits the user's real
config file, so it must replace exactly one key and preserve everything else."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "env_file.py"
_spec = importlib.util.spec_from_file_location("env_file", _SCRIPT)
env_file = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env_file)


def test_set_env_value_replaces_in_place_and_preserves_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nGARVIS_GROQ_API_KEY=abc\nGARVIS_LLM_PROVIDER=groq\n\nGARVIS_LOG_LEVEL=INFO\n", encoding="utf-8"
    )
    env_file.set_env_value("GARVIS_LLM_PROVIDER", "tiered", env)
    assert env.read_text(encoding="utf-8") == (
        "# comment\nGARVIS_GROQ_API_KEY=abc\nGARVIS_LLM_PROVIDER=tiered\n\nGARVIS_LOG_LEVEL=INFO\n"
    )


def test_set_env_value_appends_when_missing_and_creates_file(tmp_path):
    env = tmp_path / ".env"
    env_file.set_env_value("GARVIS_GEMINI_API_KEY", "k1", env)
    env_file.set_env_value("GARVIS_GEMINI_API_KEY", "k2", env)
    assert env.read_text(encoding="utf-8") == "GARVIS_GEMINI_API_KEY=k2\n"


def test_set_env_value_does_not_touch_similarly_named_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GARVIS_GEMINI_API_KEY_OLD=x\n# GARVIS_GEMINI_API_KEY=commented\n", encoding="utf-8")
    env_file.set_env_value("GARVIS_GEMINI_API_KEY", "new", env)
    assert env.read_text(encoding="utf-8").splitlines() == [
        "GARVIS_GEMINI_API_KEY_OLD=x",
        "# GARVIS_GEMINI_API_KEY=commented",
        "GARVIS_GEMINI_API_KEY=new",
    ]
