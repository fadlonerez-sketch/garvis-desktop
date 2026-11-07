"""Tests for scripts/check_anthropic.py -- no network, no real key."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

from app.core.config import Settings

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # the script imports its sibling env_file
_spec = importlib.util.spec_from_file_location("check_anthropic", _SCRIPTS / "check_anthropic.py")
check_anthropic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_anthropic)


async def test_missing_key_explains_where_it_goes_and_that_no_other_model_is_used(capsys):
    assert not await check_anthropic.verify(Settings(_env_file=None), live_test=True)
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out and ".env" in out and "--set-key" in out and "no other model" in out


def test_set_key_writes_only_that_line_and_never_prints_it(monkeypatch, tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text("GARVIS_LLM_PROVIDER=tiered\nGARVIS_OMNIROUTE_API_KEY=keep-me\n", encoding="utf-8")
    secret = "sk-ant-api03-test-secret"
    monkeypatch.setattr(check_anthropic, "ENV_PATH", env)
    monkeypatch.setattr(check_anthropic, "set_env_value", functools.partial(check_anthropic.set_env_value, path=env))
    monkeypatch.setattr(check_anthropic.getpass, "getpass", lambda prompt: secret)
    monkeypatch.setattr(sys, "argv", ["check_anthropic.py", "--set-key"])

    async def no_verify(settings, live_test):
        return True

    monkeypatch.setattr(check_anthropic, "verify", no_verify)
    assert check_anthropic.main() == 0
    assert env.read_text(encoding="utf-8").splitlines() == [
        "GARVIS_LLM_PROVIDER=tiered",
        "GARVIS_OMNIROUTE_API_KEY=keep-me",
        f"ANTHROPIC_API_KEY={secret}",
    ]
    assert secret not in capsys.readouterr().out


def test_set_key_rejects_something_that_is_not_a_claude_key(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setattr(check_anthropic, "ENV_PATH", env)
    monkeypatch.setattr(check_anthropic.getpass, "getpass", lambda prompt: "not-a-key")
    monkeypatch.setattr(sys, "argv", ["check_anthropic.py", "--set-key"])
    assert check_anthropic.main() == 1
    assert env.read_text(encoding="utf-8") == "A=1\n"
