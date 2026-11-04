"""Create (or remove) the Start-menu shortcut "Garvis" with shortcut key
Ctrl+Alt+G (or GARVIS_GLOBAL_HOTKEY).

Explorer handles a Start-menu shortcut's key system-wide, so the key works
even when Garvis is NOT running: it starts Garvis normally (pythonw.exe
run_garvis.py, no console window). When Garvis IS running, the same launch
finds the running instance, brings its window to the front and exits --
never a second Garvis (see app/ui/single_instance.py). No background
process or listener is installed; nothing runs at logon.

Usage:
    python scripts/install_shortcut.py           # create / update
    python scripts/install_shortcut.py --remove  # delete it (frees the key)
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHORTCUT = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Garvis.lnk"


def _ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def shortcut_hotkey(spec: str) -> str:
    """'ctrl+alt+g' -> 'CTRL+ALT+G' (WScript.Shell's notation)."""
    names = {"ctrl": "CTRL", "control": "CTRL", "alt": "ALT", "shift": "SHIFT"}
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts or "win" in parts:
        raise SystemExit(f"'{spec}' can't be a shortcut key (Explorer supports Ctrl/Alt/Shift + one key).")
    return "+".join(names.get(p, p.upper()) for p in parts)


def pythonw() -> Path:
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return candidate if candidate.exists() else Path(sys.executable)


def create(hotkey: str) -> None:
    script = f"""
$ProgressPreference = 'SilentlyContinue'
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut({_ps(str(SHORTCUT))})
$link.TargetPath = {_ps(str(pythonw()))}
$link.Arguments = {_ps('"' + str(REPO / "run_garvis.py") + '"')}
$link.WorkingDirectory = {_ps(str(REPO))}
$link.Hotkey = {_ps(hotkey)}
$link.Description = 'Garvis personal assistant'
$link.Save()
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], check=True)
    print(f"Created {SHORTCUT}\n  starts: {pythonw()} \"{REPO / 'run_garvis.py'}\"\n  shortcut key: {hotkey}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remove", action="store_true", help="delete the shortcut")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Windows only.")
    if args.remove:
        if SHORTCUT.exists():
            SHORTCUT.unlink()
            print(f"Removed {SHORTCUT}")
        else:
            print("No Garvis shortcut to remove.")
        return 0
    sys.path.insert(0, str(REPO))
    from app.core.config import get_settings

    spec = get_settings().global_hotkey
    if not spec:
        raise SystemExit("GARVIS_GLOBAL_HOTKEY is empty; set it (e.g. ctrl+alt+g) first.")
    create(shortcut_hotkey(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
