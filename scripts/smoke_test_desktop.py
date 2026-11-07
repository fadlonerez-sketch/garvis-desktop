"""Manual smoke test for Desktop Control -- real mouse/keyboard, real windows.

    python scripts/smoke_test_desktop.py

Spawns a small throwaway target window in a *separate process* (a Qt line
edit), then drives it only through DesktopController -- focus_window, a
mouse click on the field, keyboard_hotkey, keyboard_type (Unicode, incl.
Hebrew), keyboard_press(Enter) -- and checks what the window actually
received. Your own windows are not typed into; the mouse pointer is
returned to where it started.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.desktop.build import build_desktop_controller  # noqa: E402

_TARGET_TITLE = "Garvis Desktop Smoke Target"
_EXPECTED = "Hello, Garvis! שלום 123"

_TARGET_SCRIPT = textwrap.dedent(
    f"""
    import json, sys
    from PySide6.QtWidgets import QApplication, QLineEdit
    out = sys.argv[1]
    app = QApplication([])
    edit = QLineEdit()
    edit.setWindowTitle({_TARGET_TITLE!r})
    edit.setText("text that ctrl+a should replace")
    edit.resize(520, 60)
    edit.move(200, 200)
    def done():
        open(out, "w", encoding="utf-8").write(json.dumps({{"text": edit.text()}}))
        app.quit()
    edit.returnPressed.connect(done)
    edit.show()
    app.exec()
    """
)


async def main() -> int:
    if sys.platform != "win32":
        print("Desktop control smoke test only runs on Windows.")
        return 1
    controller = build_desktop_controller(get_settings())
    info = await controller.screen_info()
    start_cursor = (info["cursor"]["x"], info["cursor"]["y"])
    print("screen:", info["virtual_screen"], "cursor:", start_cursor)
    print("focused:", (info["foreground_window"] or {}).get("title"))

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        script = Path(tmp) / "target.py"
        script.write_text(_TARGET_SCRIPT, encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(script), str(result_path)])
        try:
            for _ in range(100):
                windows = await controller.list_windows()
                target = next((w for w in windows if w.title == _TARGET_TITLE), None)
                if target:
                    break
                time.sleep(0.1)
            else:
                print("FAIL: target window never appeared")
                return 1
            print("target window:", target.title, target.rect)

            await controller.focus_window(_TARGET_TITLE)
            rect = target.rect
            await controller.click((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
            await controller.hotkey("ctrl+a", window=_TARGET_TITLE)
            await controller.type_text(_EXPECTED, window=_TARGET_TITLE)
            await controller.press_key("enter", window=_TARGET_TITLE)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
            await controller.move_mouse(*start_cursor)

        received = json.loads(result_path.read_text(encoding="utf-8"))["text"] if result_path.exists() else None

    shot = await controller.capture_screen()
    print(f"screenshot: {shot['width']}x{shot['height']} -> {shot['path']}")
    Path(shot["path"]).unlink(missing_ok=True)
    ok = received == _EXPECTED
    print("typed text received:", repr(received))
    print("PASS" if ok else f"FAIL: expected {_EXPECTED!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
