"""Key-name parsing for keyboard_press / keyboard_hotkey.

The LLM names keys as plain strings ("enter", "ctrl+shift+t", "f5"); this
module maps them to Windows virtual-key codes. Pure data + parsing, no
ctypes, so it's testable on any platform. Unknown names raise ValueError
rather than guessing -- a mistyped key must never turn into a different key.
"""
from __future__ import annotations

MODIFIER_KEYS: dict[str, int] = {
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "win": 0x5B,
    "windows": 0x5B,
    "meta": 0x5B,
    "cmd": 0x5B,
}

NAMED_KEYS: dict[str, int] = {
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "esc": 0x1B,
    "space": 0x20,
    "spacebar": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "capslock": 0x14,
    "printscreen": 0x2C,
    "menu": 0x5D,
    "apps": 0x5D,
    "volumeup": 0xAF,
    "volumedown": 0xAE,
    "volumemute": 0xAD,
    "mediaplaypause": 0xB3,
    "playpause": 0xB3,
    "medianext": 0xB0,
    "mediaprevious": 0xB1,
    "mediastop": 0xB2,
    "browserback": 0xA6,
    "browserforward": 0xA7,
    "browserrefresh": 0xA8,
    ";": 0xBA,
    "=": 0xBB,
    ",": 0xBC,
    "-": 0xBD,
    ".": 0xBE,
    "/": 0xBF,
    "`": 0xC0,
    "[": 0xDB,
    "\\": 0xDC,
    "]": 0xDD,
    "'": 0xDE,
    "plus": 0xBB,
    "minus": 0xBD,
    "comma": 0xBC,
    "period": 0xBE,
}
NAMED_KEYS.update({f"f{n}": 0x70 + n - 1 for n in range(1, 25)})
NAMED_KEYS.update({chr(c): c for c in range(ord("0"), ord("9") + 1)})
NAMED_KEYS.update({chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)})

EXTENDED_KEYS = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5D}
)
"""VKs that need KEYEVENTF_EXTENDEDKEY, or SendInput delivers the numpad
variant (e.g. arrow keys arriving as numpad 4/6 when NumLock is off)."""

BLOCKED_HOTKEYS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"win", "r"}),  # Run dialog -> arbitrary command line
        frozenset({"win", "x"}),  # power-user menu (terminal as admin, etc.)
        frozenset({"ctrl", "shift", "escape"}),  # Task Manager (kill processes)
        frozenset({"win", "l"}),  # locks the user out mid-task
    }
)
"""Hotkeys that open a command-execution surface or are otherwise
disproportionately destructive. Launching apps goes through
open_application instead; these are refused outright, not confirmed."""


def _canonical(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("_", "")
    aliases = {"control": "ctrl", "windows": "win", "meta": "win", "cmd": "win", "esc": "escape",
               "return": "enter", "del": "delete", "spacebar": "space", "apps": "menu"}
    return aliases.get(key, key)


def key_to_vk(name: str) -> int:
    """Map one key name (named key, letter, digit, F-key or modifier) to a VK code."""
    key = _canonical(name)
    if not key:
        raise ValueError("Empty key name.")
    if key in MODIFIER_KEYS:
        return MODIFIER_KEYS[key]
    if key in NAMED_KEYS:
        return NAMED_KEYS[key]
    raise ValueError(f"Unknown key name: {name!r}.")


def parse_hotkey(combo: str | list[str]) -> list[tuple[str, int]]:
    """Parse "ctrl+shift+t" (or ["ctrl", "shift", "t"]) into ordered
    (canonical_name, vk) pairs, modifiers first. Raises ValueError for an
    unknown key, a duplicate, a combo with no non-modifier key when more
    than one key is given, or a blocked combination."""
    if isinstance(combo, str):
        # "+" as the final key ("ctrl++") is the only way "+" can appear.
        parts = combo.split("+")
        if combo.endswith("++"):
            parts = [*combo[:-2].split("+"), "plus"]
    else:
        parts = list(combo)
    names = [_canonical(p) for p in parts if p.strip()]
    if not names:
        raise ValueError("Hotkey is empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"Hotkey repeats a key: {combo!r}.")

    pairs = [(n, key_to_vk(n)) for n in names]
    modifiers = [p for p in pairs if p[0] in MODIFIER_KEYS]
    others = [p for p in pairs if p[0] not in MODIFIER_KEYS]
    if len(pairs) > 1 and not others:
        raise ValueError(f"Hotkey {combo!r} has no non-modifier key.")
    if frozenset(names) in BLOCKED_HOTKEYS:
        raise ValueError(
            f"Hotkey {combo!r} is blocked for safety (it opens a command/system surface). "
            "Use open_application to launch programs instead."
        )
    return modifiers + others
