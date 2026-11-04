"""Manual smoke test: constructs the main window and closes it after 1s.

Verifies the UI builds and wires to the agent bridge without crashing.
Run with: python scripts/smoke_test_ui.py
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ui.main_window import MainWindow


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("MainWindow constructed and shown successfully.")

    QTimer.singleShot(1000, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
