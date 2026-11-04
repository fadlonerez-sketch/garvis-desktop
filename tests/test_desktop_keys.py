"""Tests for key-name parsing (app/desktop/keys.py) -- pure data, any platform."""
from __future__ import annotations

import pytest

from app.desktop.keys import key_to_vk, parse_hotkey


@pytest.mark.parametrize(
    "name, vk",
    [
        ("enter", 0x0D), ("Return", 0x0D), ("ESC", 0x1B), ("escape", 0x1B), ("tab", 0x09),
        ("a", 0x41), ("Z", 0x5A), ("7", 0x37), ("f5", 0x74), ("F12", 0x7B), ("page down", 0x22),
        ("ctrl", 0x11), ("Control", 0x11), ("win", 0x5B), ("volume_up", 0xAF),
    ],
)
def test_key_to_vk_known_names(name, vk):
    assert key_to_vk(name) == vk


@pytest.mark.parametrize("name", ["", "   ", "hyper", "f25", "ctrl-alt"])
def test_key_to_vk_rejects_unknown_names(name):
    with pytest.raises(ValueError):
        key_to_vk(name)


def test_parse_hotkey_orders_modifiers_first():
    assert [n for n, _ in parse_hotkey("t+shift+ctrl")] == ["shift", "ctrl", "t"]


def test_parse_hotkey_accepts_list_and_aliases():
    assert parse_hotkey(["Control", "Esc"]) == [("ctrl", 0x11), ("escape", 0x1B)]


def test_parse_hotkey_plus_key():
    assert [n for n, _ in parse_hotkey("ctrl++")] == ["ctrl", "plus"]


def test_parse_hotkey_single_modifier_is_allowed():
    # "win" alone opens the Start menu -- legitimate.
    assert parse_hotkey("win") == [("win", 0x5B)]


@pytest.mark.parametrize("combo", ["ctrl+shift", "ctrl+ctrl+c", "", "ctrl+bogus"])
def test_parse_hotkey_rejects_malformed(combo):
    with pytest.raises(ValueError):
        parse_hotkey(combo)


@pytest.mark.parametrize("combo", ["win+r", "R+Win", "win+x", "ctrl+shift+esc", "win+l"])
def test_parse_hotkey_blocks_command_surfaces(combo):
    with pytest.raises(ValueError, match="blocked"):
        parse_hotkey(combo)
