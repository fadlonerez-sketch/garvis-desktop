"""Shared helper for the setup scripts: edit one key in Garvis's .env.

The .env file is the user's real configuration, so set_env_value() replaces
exactly one `KEY=...` line (or appends it) and leaves every other line --
comments, blank lines, similarly named keys -- untouched. Values are never
printed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def set_env_value(key: str, value: str, path: Path = ENV_PATH) -> None:
    """Replace `key=...` in .env (or append it), leaving every other line intact."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replaced = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            replaced = True
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
