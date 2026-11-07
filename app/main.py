"""Application entry point: boots config/logging and launches the Qt UI.

On Windows, a second launch never starts a second Garvis: it hands over to
the running one (app/ui/single_instance.py) before anything heavy is
imported. The running instance registers the global hotkey
(GARVIS_GLOBAL_HOTKEY, default Ctrl+Alt+G) that brings its window to the
front (app/ui/global_hotkey.py), expanding it from the floating orb when
collapsed (app/ui/presence.py).
"""

from __future__ import annotations

import sys


def _report_forward(message: str) -> None:
    """A second launch has no window of its own; with pythonw.exe (the
    Start-menu shortcut) there's no console either, so show a message box."""
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    else:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Garvis", 0x40)


def _ensure_std_streams() -> None:
    """pythonw.exe (the Start-menu shortcut) has no console: sys.stdout and
    sys.stderr are None, and libraries that write to them would fail. The
    log file (data/logs/garvis.log) is unaffected."""
    import os

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 (lives for the process)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def main() -> int:
    lock = None
    if sys.platform == "win32":
        from app.ui.single_instance import InstanceLock, forward_activation

        lock = InstanceLock()
        if not lock.acquire():
            result = forward_activation()
            if result.status != "activated":
                _report_forward(result.message)
                return 1
            return 0

    _ensure_std_streams()
    from PySide6.QtWidgets import QApplication

    from app.core.config import get_settings
    from app.core.logging import configure_logging, get_logger
    from app.ui.main_window import MainWindow

    logger = get_logger("main")
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir, settings.log_content_excerpts)
    logger.info("Starting Garvis")

    app = QApplication(sys.argv)
    window = MainWindow()
    # The floating orb, the orb <-> dashboard transition and the tray icon;
    # also restores the remembered window placement (data/ui_state.json).
    window.install_desktop_presence(app)
    window.show()
    hotkey = window.install_global_hotkey(app, settings.global_hotkey) if sys.platform == "win32" else None
    try:
        return app.exec()
    finally:
        if hotkey is not None:
            hotkey.unregister()
            app.removeNativeEventFilter(hotkey)
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    sys.exit(main())
