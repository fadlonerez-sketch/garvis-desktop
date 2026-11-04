"""Where the orb and the dashboard go, and the small file that remembers it.

Positions are Qt logical (DPI-independent) global coordinates, which stay
valid across Windows display-scaling changes. A remembered position is
reused only while it is still mostly on a connected screen (monitors get
unplugged, resolutions change); otherwise the orb goes to its configured
corner of the primary screen and the dashboard is centred there.

Preferences (orb on/off, corner, animations) live in the same JSON file:
data/ui_state.json. It holds no secrets and can be deleted at any time.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize

CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")
CORNER_LABELS = {"bottom-right": "Bottom right", "bottom-left": "Bottom left",
                 "top-right": "Top right", "top-left": "Top left"}
ORB_MARGIN = 28
_MIN_VISIBLE_FRACTION = 0.6


def corner_position(available: QRect, size: QSize, corner: str, margin: int = ORB_MARGIN) -> QPoint:
    right = available.right() - size.width() - margin + 1
    bottom = available.bottom() - size.height() - margin + 1
    left = available.left() + margin
    top = available.top() + margin
    x = left if corner.endswith("left") else right
    y = top if corner.startswith("top") else bottom
    return QPoint(x, y)


def visible_fraction(rect: QRect, screens: list[QRect]) -> float:
    """The largest share of `rect` inside any single screen."""
    area = rect.width() * rect.height()
    if area <= 0:
        return 0.0
    best = 0
    for screen in screens:
        overlap = rect.intersected(screen)
        best = max(best, overlap.width() * overlap.height())
    return best / area


def restore_position(saved: tuple[int, int] | None, size: QSize, screens: list[QRect]) -> QPoint | None:
    """`saved` if a window of `size` there would be mostly on one screen."""
    if saved is None:
        return None
    rect = QRect(QPoint(*saved), size)
    return rect.topLeft() if visible_fraction(rect, screens) >= _MIN_VISIBLE_FRACTION else None


def fit_rect(rect: QRect, available: QRect) -> QRect:
    """`rect` shrunk to fit `available` and moved fully inside it."""
    width = min(rect.width(), available.width())
    height = min(rect.height(), available.height())
    x = min(max(rect.x(), available.left()), available.right() - width + 1)
    y = min(max(rect.y(), available.top()), available.bottom() - height + 1)
    return QRect(x, y, width, height)


def default_dashboard_rect(available: QRect) -> QRect:
    width = min(1440, int(available.width() * 0.86))
    height = min(900, int(available.height() * 0.86))
    return QRect(available.center().x() - width // 2, available.center().y() - height // 2, width, height)


@dataclass
class UiPrefs:
    orb_enabled: bool = True
    orb_corner: str = "bottom-right"
    animations: bool = True
    orb_pos: list[int] | None = None
    dashboard_rect: list[int] | None = None
    dashboard_maximized: bool = False
    extra: dict = field(default_factory=dict)
    """Unknown keys from a newer version, written back untouched."""


class UiStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UiPrefs:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return UiPrefs()
        if not isinstance(raw, dict):
            return UiPrefs()
        prefs = UiPrefs()
        known = set(asdict(prefs)) - {"extra"}
        for key, value in raw.items():
            if key in known:
                setattr(prefs, key, value)
            else:
                prefs.extra[key] = value
        if prefs.orb_corner not in CORNERS:
            prefs.orb_corner = "bottom-right"
        if not _int_list(prefs.orb_pos, 2):
            prefs.orb_pos = None
        if not _int_list(prefs.dashboard_rect, 4):
            prefs.dashboard_rect = None
        prefs.orb_enabled = bool(prefs.orb_enabled)
        prefs.animations = bool(prefs.animations)
        prefs.dashboard_maximized = bool(prefs.dashboard_maximized)
        return prefs

    def save(self, prefs: UiPrefs) -> None:
        data = {key: value for key, value in asdict(prefs).items() if key != "extra"}
        data.update(prefs.extra)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass  # a preference that can't be saved isn't worth interrupting the user over


def _int_list(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(v, int) for v in value)
