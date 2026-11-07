"""The Garvis command center: building blocks and pages.

MainWindow (app/ui/main_window.py) assembles the shell and owns the
controls and their gating logic; the pages here only *render* GarvisState
(app/ui/state.py), the bridge's static system info and the usage ledger
summary. Nothing on these pages is sample data: every empty state says so.

Pages rebuild lazily -- a change while a page is hidden only marks it dirty,
and it is rebuilt (debounced) the next time it is shown.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from itertools import pairwise

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui import fonts, motion, theme
from app.ui.placement import CORNER_LABELS, CORNERS
from app.ui.state import MODE_LABELS, GarvisState, Mode, Turn
from app.ui.visuals import CoreVisual, Dynamics

PAGES = ("overview", "activity", "protocols", "system", "sources", "voice", "usage", "tools", "settings")
PAGE_TITLES = {
    "overview": "Overview",
    "activity": "Activity",
    "protocols": "Protocols",
    "system": "System",
    "sources": "Sources",
    "voice": "Voice",
    "usage": "Usage",
    "tools": "Tools",
    "settings": "Settings",
}
PAGE_ICONS = {
    "overview": "space_dashboard",
    "activity": "monitoring",
    "protocols": "account_tree",
    "system": "memory",
    "sources": "library_books",
    "voice": "graphic_eq",
    "usage": "bar_chart",
    "tools": "widgets",
    "settings": "settings",
}
"""Material Symbols for the navigation rail (approved mockup)."""

MODE_DESCRIPTIONS = {
    Mode.IDLE: "Ready for your next request.",
    Mode.LISTENING: "Microphone open — speak now.",
    Mode.THINKING: "Working on your request.",
    Mode.SPEAKING: "Replying through the speakers.",
    Mode.ERROR: "Something needs your attention — see Alerts.",
}
STANDBY_DESCRIPTION = "Standing by — say “Hey Garvis” to wake me."

GiB = 1024**3
MiB = 1024**2


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def label(
    text: str = "",
    role: str | None = None,
    parent: QWidget | None = None,
    wrap: bool = False,
    spacing: float | None = None,
) -> QLabel:
    widget = QLabel(text, parent)
    if role:
        widget.setProperty("role", role)
    if wrap:
        widget.setWordWrap(True)
    if spacing is not None:
        theme.tracked(widget, spacing)
    return widget


def set_tone(widget: QWidget, tone: str | None) -> None:
    if widget.property("tone") == tone:
        return
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_text_color(widget: QWidget, color: str) -> None:
    """Re-style only on a real change: setStyleSheet re-polishes the widget,
    which is far too costly to repeat on every refresh."""
    if widget.property("textColor") != color:
        widget.setProperty("textColor", color)
        widget.setStyleSheet(f"color: {color};" if color else "")


def chip(text: str, tone: str | None = None) -> QLabel:
    """A small status pill (sentence case, tinted by tone)."""
    widget = label(text, "chip")
    if tone:
        widget.setProperty("tone", tone)
    return widget


def icon_label(name: str, pixel_size: int = 18, color: str = theme.ACCENT_TEXT, filled: bool = False) -> QLabel:
    """A Material Symbols glyph as a label (empty and hidden if the icon font is missing)."""
    widget = QLabel()
    widget.setProperty("role", "cardIcon")
    char = fonts.icon_char(name)
    if char and fonts.icons_available(filled):
        # The style sheet (not setFont): the window-wide QWidget rule would override a plain font.
        family = fonts.ICONS_FILLED if filled else fonts.ICONS
        widget.setText(char)
        widget.setStyleSheet(f"color: {color}; font-family: '{family}'; font-size: {pixel_size}px; font-weight: 400;")
        widget.setFixedSize(pixel_size + 2, pixel_size + 2)
        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    else:
        widget.hide()
    return widget


def set_button_icon(button: QAbstractButton, name: str, color: str = theme.TEXT_1, size: int = 19) -> None:
    icon = fonts.icon(name, color, size, disabled_color="#5D7290")
    if not icon.isNull():
        button.setIcon(icon)
        button.setIconSize(QSize(size, size))


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def clock_time(at: float) -> str:
    return datetime.fromtimestamp(at).strftime("%H:%M:%S")


def ago(at: float, now: float | None = None) -> str:
    seconds = max(0, int((now or time.time()) - at))
    if seconds < 60:
        return "just now" if seconds < 5 else f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def elide(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def tokens(value: int | None) -> str:
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,}"


def money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def fmt_bytes(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    if value >= GiB:
        return f"{value / GiB:.{digits}f} GB"
    if value >= MiB:
        return f"{value / MiB:.0f} MB"
    return f"{value / 1024:.0f} KB"


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= MiB:
        return f"{value / MiB:.1f} MB/s"
    if value >= 1024:
        return f"{value / 1024:.0f} KB/s"
    return f"{value:.0f} B/s"


def fmt_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}%"


def fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days} d {hours} h"
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def load_tone(percent: float | None, warn: float = 75.0, error: float = 90.0) -> str | None:
    if percent is None:
        return None
    return "error" if percent >= error else "warn" if percent >= warn else None


TONE_COLORS = {"ok": theme.OK, "warn": theme.WARN_TEXT, "error": theme.ERROR, "live": theme.NAV_ACTIVE}


def open_url(url: str) -> bool:
    qurl = QUrl(url)
    if qurl.scheme() not in ("http", "https"):
        return False
    return QDesktopServices.openUrl(qurl)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
CARD_RADIUS = 18.0

_PANEL_ICONS = {
    "status": "bolt",
    "this computer": "memory",
    "ai usage · estimated": "bar_chart",
    "alerts": "notifications",
    "current task": "task_alt",
    "conversation": "forum",
    "tool log": "receipt_long",
    "tasks": "checklist",
    "requests": "history",
    "sources": "library_books",
    "live state": "graphic_eq",
    "controls": "tune",
    "engine": "settings_voice",
    "timeline": "timeline",
    "estimated spend": "payments",
    "today": "today",
    "estimated cost per day · last 14 days": "bar_chart",
    "recent requests": "list",
    "model": "neurology",
    "voice": "record_voice_over",
    "desktop & browser": "desktop_windows",
    "phone": "smartphone",
    "interface": "palette",
    "system monitor": "monitor_heart",
    "processor": "memory",
    "memory": "memory_alt",
    "garvis process": "deployed_code",
    "storage": "hard_drive",
    "graphics": "developer_board",
    "machine & monitoring": "computer",
    "integrations": "hub",
    "library": "folder_open",
    "protocol": "account_tree",
    "steps": "format_list_numbered",
    "checks": "fact_check",
    "latest run": "play_circle",
    "planned — not implemented yet": "schedule",
    "web & research": "travel_explore",
    "browser automation": "web",
    "desktop control": "mouse",
    "applications & system": "apps",
    "files": "folder",
    "communication": "mail",
    "memory & utilities": "lightbulb",
    "other tools": "extension",
}
"""Card-title icons (approved mockup style), by title; titles not listed have none."""


def panel_icon(title: str) -> str | None:
    key = title.lower()
    return _PANEL_ICONS.get(key) or _PANEL_ICONS.get(key.split(" · ")[0])


def card_path(rect: QRectF, radius: float = CARD_RADIUS) -> QPainterPath:
    path = QPainterPath()
    radius = min(radius, rect.height() / 2, rect.width() / 2)
    path.addRoundedRect(rect, radius, radius)
    return path


def paint_card(
    painter: QPainter,
    rect: QRectF,
    radius: float = CARD_RADIUS,
    alpha: float = 0.78,
    top: QColor | None = None,
    bottom: QColor | None = None,
    border: QColor | None = None,
) -> None:
    """The approved card: a soft vertical gradient around #102440 -> #0B1A2F,
    a hairline border and a faint top highlight."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = card_path(rect, radius)
    gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gradient.setColorAt(0.0, top or theme.qcolor(theme.CARD_TOP, alpha))
    gradient.setColorAt(1.0, bottom or theme.qcolor(theme.CARD_BOTTOM, alpha))
    painter.fillPath(path, QBrush(gradient))
    painter.setPen(QPen(border or QColor(160, 200, 245, 22), 1))
    painter.drawPath(path)
    painter.setClipPath(path)
    painter.setPen(QPen(QColor(255, 255, 255, 11), 1))
    painter.drawLine(
        QPointF(rect.left() + radius * 0.6, rect.top() + 1.0), QPointF(rect.right() - radius * 0.6, rect.top() + 1.0)
    )
    painter.restore()


class Panel(QFrame):
    """A card: rounded gradient surface with an icon + title row (and an
    optional link at the right). Content goes in `body`. `index` is kept for
    callers but no longer drawn (the redesign has no panel numbers)."""

    def __init__(self, title: str, index: str = "", parent: QWidget | None = None, icon: str | None = None) -> None:
        super().__init__(parent)
        self.index = index
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 13, 16, 14)
        outer.setSpacing(10)
        header = QHBoxLayout()
        header.setSpacing(8)
        name = icon if icon is not None else panel_icon(title)
        self.icon_label = icon_label(name) if name and fonts.icon_char(name) and fonts.icons_available() else None
        if self.icon_label is not None:
            header.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.title_label = label(title, "panelTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.header = header
        outer.addLayout(header)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        outer.addLayout(self.body, 1)

    def add_header_widget(self, widget: QWidget) -> None:
        self.header.addWidget(widget)

    def add_header_link(self, text: str) -> QPushButton:
        """A small text link at the right of the title row (e.g. "Details")."""
        button = QPushButton(text)
        button.setProperty("flat", True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        return button

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        paint_card(painter, QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
        painter.end()


def paint_inner_card(widget: QWidget, accent: QColor | None = None, dim: bool = False, radius: float = 14.0) -> None:
    """A row card inside a page (requests, sources, steps): a quieter surface
    than a Panel, with an optional accent bar at the left."""
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    rect = QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
    path = card_path(rect, radius)
    painter.fillPath(path, QColor(16, 43, 76, 70 if dim else 120))
    painter.setPen(QPen(QColor(160, 200, 245, 20), 1))
    painter.drawPath(path)
    if accent is not None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(rect.left() + 5, rect.top() + 12, 3, max(0.0, rect.height() - 24)), 1.5, 1.5)
    painter.end()


class KeyValueGrid(QWidget):
    """Aligned key  value rows, updatable by key."""

    def __init__(self, parent: QWidget | None = None, key_width: int = 82) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(7)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnMinimumWidth(0, key_width)
        self._values: dict[str, QLabel] = {}

    def set(self, key: str, value: str, tone: str | None = None) -> QLabel:
        widget = self._values.get(key)
        if widget is None:
            row = len(self._values)
            self._grid.addWidget(label(key, "key"), row, 0, Qt.AlignmentFlag.AlignTop)
            widget = label(value, "value", wrap=True)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(widget, row, 1)
            self._values[key] = widget
        if widget.text() != value:
            widget.setText(value)
        set_text_color(widget, TONE_COLORS.get(tone or "", ""))
        return widget

    def value(self, key: str) -> str:
        widget = self._values.get(key)
        return widget.text() if widget is not None else ""


class StatTile(QWidget):
    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)
        self.value_label = label("—", "stat")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label = label(caption, "statLabel")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)

    def set_value(self, text: str, hint: str = "") -> None:
        self.value_label.setText(text)
        self.setToolTip(hint)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        paint_inner_card(self, radius=12.0)


def _bar_fill(rect: QRectF, color: QColor | None) -> QBrush:
    if color is not None:
        return QBrush(color)
    gradient = QLinearGradient(rect.topLeft(), rect.topRight())
    gradient.setColorAt(0.0, QColor(theme.ELECTRIC))
    gradient.setColorAt(1.0, QColor(theme.ELECTRIC_HI))
    return QBrush(gradient)


def paint_bar(
    painter: QPainter, width: float, height: float, ratio: float, color: QColor | None, thickness: float = 6.0
) -> None:
    """A rounded track with a filled share (Electric Blue gradient unless a tone colour is given)."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    thickness = min(thickness, height)
    y = (height - thickness) / 2
    radius = thickness / 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255, 16))
    painter.drawRoundedRect(QRectF(0, y, width, thickness), radius, radius)
    filled = width * max(0.0, min(1.0, ratio))
    if filled > 0.5:
        rect = QRectF(0, y, max(filled, thickness), thickness)
        painter.setBrush(_bar_fill(QRectF(0, y, width, thickness), color))
        painter.drawRoundedRect(rect, radius, radius)


class BudgetBar(QWidget):
    """Spend vs. limit, as a rounded meter (amber from `warn`, red from `error`)."""

    def __init__(
        self, parent: QWidget | None = None, warn: float = 0.75, error: float = 1.0, segments: int = 40
    ) -> None:
        super().__init__(parent)
        self._ratio = 0.0
        self._warn, self._error = warn, error
        self.setMinimumHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def ratio(self) -> float:
        return self._ratio

    def set_ratio(self, ratio: float) -> None:
        self._ratio = max(0.0, ratio)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        color = None if self._ratio < self._warn else QColor(theme.WARN if self._ratio < self._error else theme.ERROR)
        paint_bar(painter, self.width(), self.height(), self._ratio, color)
        painter.end()


class Readout(QWidget):
    """A compact figure: the value over a small caption, with an optional detail line."""

    def __init__(self, caption: str, parent: QWidget | None = None, big: bool = False) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.value_label = label("—", "metric" if big else "stat")
        self.caption_label = label(caption, "statLabel")
        self.hint_label = label("", "faint")
        self.hint_label.hide()
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.hint_label)

    def set_value(self, text: str, hint: str = "", tone: str | None = None, detail: str = "") -> None:
        self.value_label.setText(text)
        set_text_color(self.value_label, TONE_COLORS.get(tone or "", ""))
        self.setToolTip(hint)
        self.hint_label.setVisible(bool(detail))
        self.hint_label.setText(detail)


class Sparkline(QWidget):
    """A real time series (app/ui/telemetry.py Series) over the last
    `span` seconds. Gaps in sampling stay gaps; nothing is interpolated."""

    def __init__(
        self,
        color: str,
        ceiling: float | None = 100.0,
        span: float = 120.0,
        height: int = 34,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._ceiling = ceiling
        self._span = span
        self._series = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(height)

    def set_series(self, series) -> None:
        self._series = series
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        painter.setPen(QPen(QColor(160, 200, 245, 12), 1))
        for frac in (0.25, 0.5, 0.75):
            painter.drawLine(QPointF(0, h * frac), QPointF(w, h * frac))
        painter.setPen(QPen(QColor(160, 200, 245, 26), 1))
        painter.drawLine(QPointF(0, h - 0.5), QPointF(w, h - 0.5))
        now = time.time()
        points = self._series.since(now - self._span) if self._series is not None else []
        if len(points) < 2:
            painter.end()
            return
        ceiling = self._ceiling or (max(v for _, v in points) * 1.2) or 1.0
        # The axis begins at the oldest sample shown (up to `span` ago), so a
        # fresh series fills the width instead of huddling at the right edge.
        start = max(now - self._span, points[0][0])
        span = max(now - start, 1.0)
        steps = sorted(b[0] - a[0] for a, b in pairwise(points))
        gap = max(5.0, 3.0 * steps[len(steps) // 2])
        runs: list[list[QPointF]] = [[]]
        previous = None
        for t, v in points:
            if previous is not None and t - previous > gap:
                runs.append([])
            previous = t
            runs[-1].append(QPointF(w * (t - start) / span, h - 1.5 - (h - 4) * min(1.0, v / ceiling)))
        top, bottom = QColor(self._color), QColor(self._color)
        top.setAlpha(60)
        bottom.setAlpha(0)
        for run in runs:
            if len(run) < 2:
                continue
            area = QPainterPath(QPointF(run[0].x(), h))
            for point in run:
                area.lineTo(point)
            area.lineTo(QPointF(run[-1].x(), h))
            gradient = QLinearGradient(QPointF(0, 0), QPointF(0, h))
            gradient.setColorAt(0.0, top)
            gradient.setColorAt(1.0, bottom)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawPath(area)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self._color, 1.5))
            painter.drawPolyline(run)
        if runs[-1]:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.TEXT_1))
            painter.drawEllipse(runs[-1][-1], 2.2, 2.2)
        painter.end()


class ArcGauge(QWidget):
    """A 240-degree gauge for a real percentage; None draws an empty dial."""

    def __init__(self, caption: str, color: str = theme.CYAN, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._caption = caption
        self._color = color
        self._value: float | None = None
        self._sub = ""
        self.setFixedSize(140, 140)

    @property
    def value(self) -> float | None:
        return self._value

    def set_value(self, percent: float | None, sub: str = "") -> None:
        self._value = None if percent is None else max(0.0, min(100.0, percent))
        self._sub = sub
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width(), self.height())
        c = QPointF(self.width() / 2, self.height() / 2)
        r = side / 2 - 9
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        start, sweep = 210.0, -240.0
        track = QPen(QColor(255, 255, 255, 16), 8)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track)
        painter.drawArc(rect, int(start * 16), int(sweep * 16))
        value = self._value
        tone = load_tone(value)
        color = QColor(TONE_COLORS[tone] if tone else self._color)
        if value is not None and value > 0:
            pen = QPen(color, 8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, int(start * 16), int(sweep * value / 100 * 16))
        painter.setPen(QColor(theme.TEXT_1) if value is not None else QColor(theme.TEXT_3))
        painter.setFont(theme.display_font(17, QFont.Weight.Bold))
        painter.drawText(
            QRectF(0, c.y() - 24, self.width(), 32),
            Qt.AlignmentFlag.AlignCenter,
            "—" if value is None else f"{value:.0f}%",
        )
        painter.setPen(QColor(theme.TEXT_2))
        painter.setFont(theme.text_font(11, QFont.Weight.Medium))
        painter.drawText(QRectF(0, c.y() + 8, self.width(), 15), Qt.AlignmentFlag.AlignCenter, self._caption)
        if self._sub:
            painter.setPen(QColor(theme.TEXT_3))
            painter.setFont(theme.text_font(10))
            painter.drawText(QRectF(0, c.y() + 24, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self._sub)
        painter.end()


class CoreGrid(QWidget):
    """Per-logical-processor load as a row of fine bars."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loads: tuple[float, ...] | None = None
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def loads(self) -> tuple[float, ...] | None:
        return self._loads

    def set_loads(self, loads: tuple[float, ...] | None) -> None:
        self._loads = loads
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        loads = self._loads
        if not loads:
            painter.setPen(QColor(theme.TEXT_3))
            painter.setFont(theme.text_font(12))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                "Per-core load appears after the second sample.",
            )
            painter.end()
            return
        n = len(loads)
        gap = 4.0 if n <= 16 else 2.0
        width = (self.width() - gap * (n - 1)) / n
        bar_h = self.height() - 16
        radius = min(3.0, width / 2)
        painter.setFont(theme.text_font(9))
        for i, load in enumerate(loads):
            x = i * (width + gap)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 14))
            painter.drawRoundedRect(QRectF(x, 0, width, bar_h), radius, radius)
            tone = load_tone(load)
            filled = bar_h * load / 100.0
            if filled > 0.5:
                if tone:
                    painter.setBrush(QColor(TONE_COLORS[tone]))
                else:
                    gradient = QLinearGradient(QPointF(0, bar_h - filled), QPointF(0, bar_h))
                    gradient.setColorAt(0.0, QColor(theme.ELECTRIC_HI))
                    gradient.setColorAt(1.0, QColor(theme.ELECTRIC))
                    painter.setBrush(QBrush(gradient))
                painter.drawRoundedRect(QRectF(x, bar_h - filled, width, filled), radius, radius)
            if n <= 32:
                painter.setPen(QColor(theme.TEXT_3))
                painter.drawText(QRectF(x - 4, bar_h + 2, width + 8, 12), Qt.AlignmentFlag.AlignCenter, str(i))
        painter.end()


class UsageTrend(QWidget):
    """Estimated cost per day from the local ledger (UsagePage)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._days: list[dict] = []
        self.setMinimumHeight(120)

    @property
    def days(self) -> list[dict]:
        return self._days

    def set_days(self, days: list[dict]) -> None:
        self._days = list(days or [])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        days = self._days
        w, h = float(self.width()), float(self.height())
        chart_h = h - 18
        painter.setPen(QPen(QColor(160, 200, 245, 14), 1))
        for frac in (0.25, 0.5, 0.75):
            painter.drawLine(QPointF(46, chart_h * frac), QPointF(w, chart_h * frac))
        painter.setFont(theme.text_font(10))
        if not days:
            painter.setPen(QColor(theme.TEXT_3))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No daily history yet.")
            painter.end()
            return
        peak = max(d["cost_usd"] for d in days) or 1.0
        n = len(days)
        gap = 5.0
        width = max(2.0, (w - 46 - gap * (n - 1)) / n)
        radius = min(4.0, width / 2)
        for i, day in enumerate(days):
            x = 46 + i * (width + gap)
            value = day["cost_usd"]
            bar = (chart_h - 4) * value / peak
            painter.setPen(Qt.PenStyle.NoPen)
            if value > 0:
                gradient = QLinearGradient(QPointF(0, chart_h - bar), QPointF(0, chart_h))
                gradient.setColorAt(0.0, QColor(theme.ELECTRIC_HI))
                gradient.setColorAt(1.0, QColor(theme.ELECTRIC))
                painter.setBrush(QBrush(gradient))
                painter.drawRoundedRect(QRectF(x, chart_h - bar, width, bar), radius, radius)
            else:
                painter.setBrush(QColor(255, 255, 255, 16))
                painter.drawRoundedRect(QRectF(x, chart_h - 2, width, 2), 1, 1)
            if i in (0, n - 1) or (n > 7 and i == n // 2):
                painter.setPen(QColor(theme.TEXT_3))
                painter.drawText(
                    QRectF(x - 20, chart_h + 3, width + 40, 13), Qt.AlignmentFlag.AlignCenter, day["day"][5:]
                )
        painter.setPen(QColor(theme.TEXT_2))
        painter.drawText(QRectF(0, 0, 44, 14), Qt.AlignmentFlag.AlignLeft, money(peak))
        painter.drawText(QRectF(0, chart_h - 14, 44, 14), Qt.AlignmentFlag.AlignLeft, "$0")
        painter.end()


class LevelScope(QWidget):
    """Scrolling trace of Garvis's real playback level (Voice page). Runs a
    timer only while visible and speaking."""

    def __init__(self, level_source: Callable[[], float] | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level_source = level_source
        self._history: deque[float] = deque([0.0] * 120, maxlen=120)
        self._dyn = Dynamics()
        self._mode = Mode.IDLE
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.setMinimumHeight(110)

    @property
    def is_animating(self) -> bool:
        return self._timer.isActive()

    def set_mode(self, mode: str | Mode) -> None:
        self._mode = Mode(mode)
        self._sync_timer()

    def _sync_timer(self) -> None:
        wanted = self.isVisible() and (self._mode == Mode.SPEAKING or self._dyn.level > 0.01)
        if wanted and not self._timer.isActive():
            self._timer.start()
        elif not wanted and self._timer.isActive():
            self._timer.stop()
            self.update()

    def _tick(self) -> None:
        raw = self._level_source() if self._mode == Mode.SPEAKING and self._level_source else 0.0
        self._dyn.step(0.033, self._mode, raw)
        self._history.append(self._dyn.level)
        if self._mode != Mode.SPEAKING and self._dyn.level <= 0.01:
            self._history.extend([0.0] * 4)
            self._sync_timer()
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        mid = h / 2
        painter.setPen(QPen(QColor(160, 200, 245, 14), 1))
        for i in range(1, 6):
            painter.drawLine(QPointF(0, h * i / 6), QPointF(w, h * i / 6))
        count = len(self._history)
        step = w / max(1, count - 1)
        path_top = QPainterPath(QPointF(0, mid))
        for i, value in enumerate(self._history):
            wobble = 0.55 + 0.45 * math.sin(i * 0.8)
            path_top.lineTo(QPointF(i * step, mid - value * wobble * (mid - 6)))
        path_bottom = QPainterPath(QPointF(0, mid))
        for i, value in enumerate(self._history):
            wobble = 0.55 + 0.45 * math.sin(i * 0.8)
            path_bottom.lineTo(QPointF(i * step, mid + value * wobble * (mid - 6)))
        painter.setPen(QPen(QColor(58, 146, 230, 235), 1.7))
        painter.drawPath(path_top)
        painter.setPen(QPen(QColor(58, 146, 230, 120), 1.2))
        painter.drawPath(path_bottom)
        painter.setPen(QPen(QColor(234, 242, 251, 40), 1))
        painter.drawLine(QPointF(0, mid), QPointF(w, mid))
        painter.end()


_NAV_ICONS = PAGES


def _mix(a: str | QColor, b: str | QColor, t: float) -> QColor:
    ca, cb = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor(
        int(ca.red() + (cb.red() - ca.red()) * t),
        int(ca.green() + (cb.green() - ca.green()) * t),
        int(ca.blue() + (cb.blue() - ca.blue()) * t),
        int(ca.alpha() + (cb.alpha() - ca.alpha()) * t),
    )


def _paint_icon(painter: QPainter, kind: str, center: QPointF, size: float, color: QColor) -> None:
    """Hand-drawn page icons: the fallback when the Material Symbols font is unavailable."""
    painter.save()
    painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    s = size / 2
    x, y = center.x(), center.y()
    if kind == "overview":
        painter.drawEllipse(center, s, s)
        painter.drawEllipse(center, s * 0.45, s * 0.45)
        painter.setBrush(color)
        painter.drawEllipse(center, s * 0.12, s * 0.12)
    elif kind == "activity":
        path = QPainterPath(QPointF(x - s, y))
        for px, py in ((-0.5, 0), (-0.25, -0.8), (0.1, 0.8), (0.4, -0.3), (0.6, 0), (1.0, 0)):
            path.lineTo(QPointF(x + px * s, y + py * s))
        painter.drawPath(path)
    elif kind == "sources":
        for k, offset in enumerate((-0.55, 0.0, 0.55)):
            painter.drawLine(
                QPointF(x - s * 0.9 + (0.3 if k == 0 else 0) * s, y + offset * s), QPointF(x + s * 0.9, y + offset * s)
            )
        painter.drawEllipse(QPointF(x - s * 0.9, y - 0.55 * s), s * 0.12, s * 0.12)
    elif kind == "voice":
        for k, height in enumerate((0.35, 0.75, 1.0, 0.6, 0.3)):
            px = x + (k - 2) * s * 0.42
            painter.drawLine(QPointF(px, y - height * s), QPointF(px, y + height * s))
    elif kind == "usage":
        for k, height in enumerate((0.45, 0.8, 0.6, 1.0)):
            px = x + (k - 1.5) * s * 0.5
            painter.drawLine(QPointF(px, y + s), QPointF(px, y + s - height * 2 * s))
    elif kind == "system":
        painter.drawRect(QRectF(x - s * 0.55, y - s * 0.55, s * 1.1, s * 1.1))
        painter.drawRect(QRectF(x - s * 0.2, y - s * 0.2, s * 0.4, s * 0.4))
        for k in (-0.3, 0.0, 0.3):
            for a, b in (
                ((x + k * s, y - s * 0.55), (x + k * s, y - s * 0.85)),
                ((x + k * s, y + s * 0.55), (x + k * s, y + s * 0.85)),
                ((x - s * 0.55, y + k * s), (x - s * 0.85, y + k * s)),
                ((x + s * 0.55, y + k * s), (x + s * 0.85, y + k * s)),
            ):
                painter.drawLine(QPointF(*a), QPointF(*b))
    elif kind == "protocols":
        nodes = (QPointF(x - s * 0.6, y - s * 0.6), QPointF(x + s * 0.6, y), QPointF(x - s * 0.6, y + s * 0.6))
        painter.drawLine(nodes[0], nodes[1])
        painter.drawLine(nodes[1], nodes[2])
        for node in nodes:
            painter.drawEllipse(node, s * 0.24, s * 0.24)
    elif kind == "tools":
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cell = QRectF(x + dx * s * 0.5 - s * 0.36, y + dy * s * 0.5 - s * 0.36, s * 0.72, s * 0.72)
            if dx == 1 and dy == 1:
                painter.drawEllipse(cell.center(), s * 0.36, s * 0.36)
            else:
                painter.drawRoundedRect(cell, 1.5, 1.5)
    elif kind == "settings":
        painter.drawEllipse(center, s * 0.45, s * 0.45)
        for k in range(8):
            angle = k * math.pi / 4
            painter.drawLine(
                QPointF(x + math.cos(angle) * s * 0.68, y + math.sin(angle) * s * 0.68),
                QPointF(x + math.cos(angle) * s, y + math.sin(angle) * s),
            )
    painter.restore()


class _HoverMixin:
    """Fading hover level for custom-painted buttons (motion.Fader)."""

    def _init_hover(self) -> None:
        self._hover = 0.0
        self._hover_fader = motion.Fader(self._set_hover, parent=self)

    def _set_hover(self, value: float) -> None:
        self._hover = value
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.isEnabled():
            self._hover_fader.to(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover_fader.to(0.0)
        super().leaveEvent(event)


class NavButton(_HoverMixin, QAbstractButton):
    """A rail item: Material Symbols icon (filled when selected) over its label."""

    FULL_HEIGHT = 64
    COMPACT_HEIGHT = 46

    def __init__(self, key: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.icon_name = PAGE_ICONS.get(key, "")
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.compact = False
        self.setFixedSize(68, self.FULL_HEIGHT)
        self._init_hover()
        self.pressed.connect(self.update)
        self.released.connect(self.update)

    def set_compact(self, compact: bool) -> None:
        if compact != self.compact:
            self.compact = compact
            self.setFixedSize(68, self.COMPACT_HEIGHT if compact else self.FULL_HEIGHT)
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(68, self.COMPACT_HEIGHT if self.compact else self.FULL_HEIGHT)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        rect = QRectF(self.rect())
        if self.isDown():  # press feedback: settle to 98 %
            painter.translate(rect.center())
            painter.scale(0.98, 0.98)
            painter.translate(-rect.center())
        checked = self.isChecked()
        if self._hover > 0.001 and not checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(58, 146, 230, int(26 * self._hover)))
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
        pill_y = (rect.height() - 30) / 2 if self.compact else 8.0
        pill = QRectF((rect.width() - 44) / 2, pill_y, 44, 30)
        if checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(22, 119, 210, 46))
            painter.drawRoundedRect(pill, 10, 10)
        color = QColor(theme.NAV_ACTIVE) if checked else _mix(theme.TEXT_3, theme.TEXT_2, self._hover)
        if not fonts.draw_icon(painter, pill, self.icon_name, color, 21, filled=checked):
            _paint_icon(painter, self.key, pill.center(), 16, color)
        if not self.compact:
            painter.setPen(QColor(theme.TEXT_1) if checked else _mix(theme.TEXT_3, theme.TEXT_2, self._hover))
            painter.setFont(theme.text_font(11, QFont.Weight.Medium))
            painter.drawText(QRectF(0, pill.bottom() + 3, rect.width(), 16), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class NavRail(QWidget):
    page_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(84)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 8)
        layout.setSpacing(4)
        self.buttons: dict[str, NavButton] = {}
        for key in PAGES:
            button = NavButton(key, PAGE_TITLES[key], self)
            button.clicked.connect(lambda _checked=False, k=key: self.select(k))
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
            self.buttons[key] = button
        layout.addStretch(1)
        self.select("overview", emit=False)

    def select(self, key: str, emit: bool = True) -> None:
        for name, button in self.buttons.items():
            button.setChecked(name == key)
        if emit:
            self.page_selected.emit(key)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # A short window: icons only (labels stay in the tooltips).
        needed = len(self.buttons) * (NavButton.FULL_HEIGHT + 4) + 22
        compact = self.height() < needed
        for button in self.buttons.values():
            button.set_compact(compact)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(5, 11, 22, 90))
        painter.setPen(QPen(QColor(160, 200, 245, 20), 1))
        painter.drawLine(QPointF(self.width() - 0.5, 0), QPointF(self.width() - 0.5, self.height()))
        painter.end()


class GlyphButton(_HoverMixin, QAbstractButton):
    """Title-bar button: 'orb' (collapse to the orb), 'max', 'restore', 'quit'."""

    ICONS = {"orb": "remove", "max": "crop_square", "restore": "filter_none", "quit": "close"}

    def __init__(self, kind: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setFixedSize(40, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_hover()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        if self.isDown():
            painter.translate(rect.center())
            painter.scale(0.98, 0.98)
            painter.translate(-rect.center())
        if self._hover > 0.001:
            fill = (
                QColor(240, 106, 106, int(60 * self._hover))
                if self.kind == "quit"
                else QColor(58, 146, 230, int(30 * self._hover))
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)
        color = _mix(theme.TEXT_3, theme.TEXT_1, self._hover)
        size = 18 if self.kind in ("max", "restore") else 21
        if fonts.draw_icon(painter, rect, self.ICONS.get(self.kind, ""), color, size):
            painter.end()
            return
        painter.setPen(QPen(color, 1.3))  # fallback: the painted glyphs
        c = rect.center()
        if self.kind == "orb":
            painter.drawLine(QPointF(c.x() - 6, c.y()), QPointF(c.x() + 6, c.y()))
        elif self.kind == "max":
            painter.drawRect(QRectF(c.x() - 5, c.y() - 5, 10, 10))
        elif self.kind == "restore":
            painter.drawRect(QRectF(c.x() - 5, c.y() - 3, 8, 8))
            painter.drawPolyline(
                [QPointF(c.x() - 3, c.y() - 5), QPointF(c.x() + 5, c.y() - 5), QPointF(c.x() + 5, c.y() + 3)]
            )
        elif self.kind == "quit":
            painter.drawLine(QPointF(c.x() - 5, c.y() - 5), QPointF(c.x() + 5, c.y() + 5))
            painter.drawLine(QPointF(c.x() + 5, c.y() - 5), QPointF(c.x() - 5, c.y() + 5))
        painter.end()


def _dot_text(text: str, color: str) -> str:
    return f'<span style="color:{color}; font-size:10px;">&#9679;</span>&nbsp;&nbsp;{html_escape(text)}'


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def display_model_name(model: str) -> str:
    """'claude-haiku-5-5' -> 'Claude Haiku 5.5' (the id stays in the tooltip)."""
    words: list[str] = []
    numbers: list[str] = []
    for part in model.replace("_", "-").split("-"):
        if part.isdigit() and len(part) <= 2:
            numbers.append(part)
        elif part:
            if numbers:
                words.append(".".join(numbers))
                numbers = []
            words.append(part[:1].upper() + part[1:])
    if numbers:
        words.append(".".join(numbers))
    return " ".join(words) or model


class TitleBar(QWidget):
    """The original logo lockup, live mode chip, status text, key/model chips,
    clock and window buttons. Dragging it moves the frameless window;
    double-click maximizes."""

    collapse_clicked = Signal()
    maximize_clicked = Signal()
    quit_clicked = Signal()

    MODE_DOTS = {"ok": theme.OK, "live": theme.NAV_ACTIVE, "error": theme.ERROR}

    def __init__(self, status_label: QLabel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(17, 0, 8, 0)
        layout.setSpacing(14)
        from app.ui.logo import render_logo_pixmap

        self.logo = QLabel(self)
        self.logo.setPixmap(render_logo_pixmap(34, theme.ORANGE, self.devicePixelRatioF()))
        layout.addWidget(self.logo)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addStretch(1)
        for text, role, spacing in (("GARVIS", "wordmark", 6.0), ("PERSONAL AI SYSTEM", "tagline", 3.0)):
            widget = label(text, role, spacing=spacing)
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)  # never clipped
            brand.addWidget(widget)
        brand.addStretch(1)
        layout.addLayout(brand)
        layout.addSpacing(18)
        self.mode_chip = label("", "titleChip")
        self.mode_chip.setTextFormat(Qt.TextFormat.RichText)
        self._set_mode_chip("Online", "ok")
        layout.addWidget(self.mode_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        self.status_label = status_label
        status_label.setProperty("role", "titleStatus")
        status_label.setMinimumWidth(40)
        status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(status_label, 1)
        icon = fonts.icon_html("key", theme.WARN_TEXT, 17)
        self.key_chip = label(f"{icon}&nbsp; API key missing" if icon else "API key missing", "titleChip")
        self.key_chip.setTextFormat(Qt.TextFormat.RichText)
        self.key_chip.setProperty("tone", "warn")
        self.key_chip.setToolTip(
            "ANTHROPIC_API_KEY is not set in Garvis's .env: every model request fails until "
            "it is added (python scripts/check_anthropic.py --set-key)."
        )
        self.key_chip.hide()
        layout.addWidget(self.key_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        self.model_chip = label("—", "titleChip")
        self.model_chip.setTextFormat(Qt.TextFormat.PlainText)
        self._model_known = True
        for chip_label in (self.mode_chip, self.key_chip, self.model_chip):
            chip_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.model_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        self.clock = label("", "clock")
        layout.addWidget(self.clock)
        layout.addSpacing(2)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        self.collapse_button = GlyphButton("orb", "Collapse to the orb", self)
        self.max_button = GlyphButton("max", "Maximize", self)
        self.quit_button = GlyphButton("quit", "Quit Garvis", self)
        for button in (self.collapse_button, self.max_button, self.quit_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.collapse_button.clicked.connect(self.collapse_clicked)
        self.max_button.clicked.connect(self.maximize_clicked)
        self.quit_button.clicked.connect(self.quit_clicked)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(10_000)
        self._tick_clock()

    def _tick_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%H:%M"))

    def _set_mode_chip(self, text: str, tone: str) -> None:
        self.mode_text = text
        self.mode_chip.setText(_dot_text(text, self.MODE_DOTS.get(tone, theme.NAV_ACTIVE)))
        set_tone(self.mode_chip, None if tone == "ok" else tone)

    def set_model(self, model: str, provider: str) -> None:
        self.model_chip.setText(f"{display_model_name(model)} · {provider}" if provider else display_model_name(model))
        self.set_model_known(True)

    def set_model_known(self, known: bool) -> None:
        self._model_known = known
        self._fit()

    NARROW = 1040
    """Below this width the model chip yields its place (the model is on the Status card too)."""

    def _fit(self) -> None:
        self.model_chip.setVisible(self._model_known and self.width() >= self.NARROW)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def set_maximized(self, maximized: bool) -> None:
        self.max_button.kind = "restore" if maximized else "max"
        self.max_button.setToolTip("Restore" if maximized else "Maximize")
        self.max_button.update()

    def set_mode(self, mode: Mode, standby: bool) -> None:
        text = "Standby" if mode == Mode.IDLE and standby else MODE_LABELS[mode]
        tone = {Mode.IDLE: "ok", Mode.ERROR: "error"}.get(mode, "live")
        self._set_mode_chip(text, tone)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton and self.window().windowHandle() is not None:
            if not getattr(self.window(), "is_maximized_custom", False):
                self.window().windowHandle().startSystemMove()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_clicked.emit()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(160, 200, 245, 20), 1))
        y = self.height() - 0.5
        painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
        painter.end()


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 4)
        layout.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(3)
        self.title = label(title, "pageTitle")
        text.addWidget(self.title)
        self.subtitle = label(subtitle, "pageSub", wrap=True)
        text.addWidget(self.subtitle)
        layout.addLayout(text, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        layout.addLayout(self.actions)


class Page(QWidget):
    """Base page: lazy, debounced refresh while visible."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dirty = True
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)

    def _on_refresh_timer(self) -> None:
        # Hidden since the timer started: stay dirty, showEvent refreshes.
        if self.isVisible():
            self._do_refresh()

    def mark_dirty(self) -> None:
        self._dirty = True
        if self.isVisible():
            self._refresh_timer.start()

    def refresh_now(self) -> None:
        self._dirty = True
        self._do_refresh()

    def flush(self) -> None:
        """Apply a pending refresh now, even while hidden (before a snapshot)."""
        self._refresh_timer.stop()
        self._do_refresh()

    def _do_refresh(self) -> None:
        if self._dirty:
            self._dirty = False
            self.refresh()

    def refresh(self) -> None:  # pragma: no cover - overridden
        pass

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self._refresh_timer.start(0)


class _FitWidth(QWidget):
    """Scroll-area content that always takes the viewport's width (its
    labels wrap) and scrolls only vertically."""

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(0, super().sizeHint().height())


class ResponsiveGrid(QWidget):
    """Lays its widgets out in two columns, or one when narrower than
    `breakpoint` -- panels reflow instead of being clipped."""

    def __init__(self, breakpoint: int = 1000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._breakpoint = breakpoint
        self._items: list[QWidget] = []
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(12)

    @property
    def columns(self) -> int:
        return self._columns

    def add(self, widget: QWidget) -> None:
        self._items.append(widget)
        self._columns = 0
        self._reflow()

    def clear(self) -> None:
        for widget in self._items:
            self._grid.removeWidget(widget)
            widget.deleteLater()
        self._items = []
        self._columns = 0

    def _reflow(self) -> None:
        columns = 2 if self.width() >= self._breakpoint else 1
        if columns == self._columns:
            return
        self._columns = columns
        for widget in self._items:
            self._grid.removeWidget(widget)
        for i, widget in enumerate(self._items):
            self._grid.addWidget(widget, i // columns, i % columns)
        for column in range(2):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._reflow()


def scroll_column(parent: QWidget | None = None) -> tuple[QScrollArea, QVBoxLayout]:
    area = QScrollArea(parent)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inner = QWidget()
    column = QVBoxLayout(inner)
    column.setContentsMargins(0, 0, 6, 0)
    column.setSpacing(10)
    area.setWidget(inner)
    return area, column


class StatusMark(QWidget):
    """Painted running / done / failed mark (no dependency on symbol fonts)."""

    def __init__(self, success: bool | str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.success = success  # True done, False failed, None running, "warn" a warning
        self.setFixedSize(14, 14)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.success == "warn":
            painter.setPen(
                QPen(QColor(theme.WARN), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            )
            painter.drawPolygon([QPointF(7, 1.8), QPointF(12.6, 12), QPointF(1.4, 12)])
            painter.drawLine(QPointF(7, 5.2), QPointF(7, 8.4))
            painter.drawPoint(QPointF(7, 10.4))
            painter.end()
            return
        color = QColor(theme.CYAN if self.success is None else theme.OK if self.success else theme.ERROR)
        painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if self.success is None:
            painter.drawArc(QRectF(2, 2, 10, 10), 90 * 16, 270 * 16)
        elif self.success:
            painter.drawPolyline([QPointF(3, 7.5), QPointF(6, 10.5), QPointF(11.5, 3.5)])
        else:
            painter.drawLine(QPointF(3.5, 3.5), QPointF(10.5, 10.5))
            painter.drawLine(QPointF(10.5, 3.5), QPointF(3.5, 10.5))
        painter.end()


class ToggleSwitch(QCheckBox):
    """A check box drawn as a switch (the approved "Speak replies" toggle).
    Same API and signals as QCheckBox; the knob slides in MOTION_MS (jumps
    under reduced motion)."""

    TRACK_W, TRACK_H, KNOB = 36.0, 22.0, 16.0

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.compact = False
        """Compact: the switch alone, its caption in the tooltip (narrow windows)."""
        self._pos = 0.0
        self._fader = motion.Fader(self._set_pos, parent=self)
        self.toggled.connect(lambda on: self._fader.to(1.0 if on else 0.0))

    def _set_pos(self, value: float) -> None:
        self._pos = value
        self.update()

    def set_compact(self, compact: bool) -> None:
        if compact != self.compact:
            self.compact = compact
            self.setToolTip(self.text() if compact else "")
            self.updateGeometry()
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        shown = self.text() and not self.compact
        text_width = QFontMetrics(self.font()).horizontalAdvance(self.text()) + 2 if shown else 0
        return QSize(
            int(self.TRACK_W) + (10 + text_width if text_width else 0) + 2,
            max(int(self.TRACK_H) + 2, QFontMetrics(self.font()).height() + 4),
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return self.sizeHint()

    def hitButton(self, pos) -> bool:  # noqa: N802 (Qt override)
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        target = 1.0 if self.isChecked() else 0.0
        if self._fader.target != target:  # changed with signals blocked: no animation owed
            self._fader.to(target)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        enabled = self.isEnabled()
        h = float(self.height())
        track = QRectF(1, (h - self.TRACK_H) / 2, self.TRACK_W, self.TRACK_H)
        off, on = QColor(255, 255, 255, 30), QColor(theme.ELECTRIC)
        color = _mix(off, on, self._pos)
        if not enabled:
            color.setAlpha(int(color.alpha() * 0.45))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        if self.hasFocus():
            painter.setPen(QPen(QColor(22, 119, 210, 120), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track.adjusted(-2, -2, 2, 2), self.TRACK_H / 2 + 2, self.TRACK_H / 2 + 2)
            painter.setPen(Qt.PenStyle.NoPen)
        travel = self.TRACK_W - self.KNOB - 6
        knob = QRectF(track.left() + 3 + travel * self._pos, track.top() + 3, self.KNOB, self.KNOB)
        knob_color = _mix(QColor("#C7D5E6"), QColor("#FFFFFF"), self._pos)
        if not enabled:
            knob_color.setAlpha(140)
        painter.setBrush(knob_color)
        painter.drawEllipse(knob)
        if self.text() and not self.compact:
            painter.setPen(QColor(theme.TEXT_2 if enabled else theme.TEXT_3))
            painter.setFont(self.font())
            painter.drawText(
                QRectF(track.right() + 10, 0, self.width() - track.right() - 10, h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.text(),
            )
        painter.end()


def tool_line(tool) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(StatusMark(tool.success))
    name = label(tool.name, "value")
    layout.addWidget(name)
    detail = tool.error if tool.success is False and tool.error else tool.arguments
    detail_label = label(elide(detail, 120), "faint")
    detail_label.setToolTip(detail)
    layout.addWidget(detail_label, 1)
    if tool.sources:
        layout.addWidget(chip(f"{len(tool.sources)} sources", "live"))
    if tool.duration is not None:
        layout.addWidget(label(f"{tool.duration:.1f}s", "faint"))
    return row


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
class ResourceLine(QWidget):
    """Caption ........ value  (detail), with the real history underneath."""

    def __init__(self, caption: str, color: str, ceiling: float | None = 100.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(8)
        swatch = QFrame()
        swatch.setFixedSize(8, 8)
        swatch.setStyleSheet(f"background: {color}; border-radius: 4px;")
        top.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(label(caption, "dim"))
        self.detail_label = label("", "faint")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.detail_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        top.addWidget(self.detail_label, 1)
        self.value_label = label("—", "metricSmall")
        top.addWidget(self.value_label)
        layout.addLayout(top)
        self.spark = Sparkline(color, ceiling=ceiling, height=30)
        layout.addWidget(self.spark)

    def set(self, value: str, detail: str = "", series=None, tone: str | None = None) -> None:
        self.value_label.setText(value)
        set_text_color(self.value_label, TONE_COLORS.get(tone or "", ""))
        self.detail_label.setText(detail)
        self.spark.set_series(series)


class _MeterBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ratio = 0.0
        self.tone: str | None = None
        self.setFixedHeight(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, percent: float | None, tone: str | None) -> None:
        self.ratio = 0.0 if percent is None else max(0.0, min(1.0, percent / 100.0))
        self.tone = tone
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        paint_bar(
            painter,
            self.width(),
            self.height(),
            self.ratio,
            QColor(TONE_COLORS[self.tone]) if self.tone in ("warn", "error") else None,
        )
        painter.end()


class MeterLine(QWidget):
    """Overview meter (approved mockup): caption | rounded bar | value, with
    an optional detail line under the bar. Same `set` as ResourceLine."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(0)
        self.caption_label = label(caption, "meterLabel")
        self.caption_label.setFixedWidth(62)
        grid.addWidget(self.caption_label, 0, 0)
        self.bar = _MeterBar()
        grid.addWidget(self.bar, 0, 1)
        self.value_label = label("—", "metricSmall")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(42)
        grid.addWidget(self.value_label, 0, 2)
        self.detail_label = label("", "faint")
        self.detail_label.hide()
        grid.addWidget(self.detail_label, 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        self.series = None

    def set(
        self, value: str, detail: str = "", series=None, tone: str | None = None, percent: float | None = None
    ) -> None:
        self.value_label.setText(value)
        set_text_color(self.value_label, TONE_COLORS.get(tone or "", ""))
        self.detail_label.setText(detail)
        self.detail_label.setVisible(bool(detail))
        self.bar.set(percent, tone)
        self.series = series


def link_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("flat", True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def chip_button(text: str) -> QPushButton:
    """A pill-shaped shortcut (the Overview's page links)."""
    button = QPushButton(text)
    button.setProperty("chip", True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class OverviewPage(Page):
    """The at-a-glance page: is Garvis operational, what is it doing, what
    does it cost the computer and the API, and where to go next. Detail
    (tool logs, full histories, hardware) lives on its own pages.

    Layout (approved mockup): status cards on the left, the orb with its
    state in the centre, the conversation on the right."""

    open_sources = Signal(int)
    open_page = Signal(str)

    LEFT_WIDTH, RIGHT_WIDTH = 300, 344
    LEFT_MIN, RIGHT_MIN, CENTER_MIN = 232, 250, 300
    GAP = 16

    def __init__(
        self, state: GarvisState, info: dict, level_source, conversation_view: QTextEdit, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._info = info
        self._usage: dict | None = None
        self._monitor = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self.GAP)

        # Left column: status, this computer, AI usage, alerts
        left = QVBoxLayout()
        left.setSpacing(12)
        self.status_panel = Panel("Status")
        self.status = KeyValueGrid()
        self.status_panel.body.addWidget(self.status)
        left.addWidget(self.status_panel, 0)

        self.resources_panel = Panel("This computer")
        self.system_link = self.resources_panel.add_header_link("Details")
        self.system_link.setToolTip("System details")
        self.system_link.clicked.connect(lambda: self.open_page.emit("system"))
        self.cpu_line = MeterLine("CPU")
        self.ram_line = MeterLine("Memory")
        self.gpu_line = MeterLine("GPU")
        for line in (self.cpu_line, self.ram_line, self.gpu_line):
            self.resources_panel.body.addWidget(line)
        self.process_label = label("", "faint", wrap=True)
        self.resources_panel.body.addWidget(self.process_label)
        left.addWidget(self.resources_panel, 0)

        self.usage_panel = Panel("AI usage · estimated")
        self.usage_link = self.usage_panel.add_header_link("Usage")
        self.usage_link.clicked.connect(lambda: self.open_page.emit("usage"))
        usage_row = QGridLayout()
        usage_row.setHorizontalSpacing(8)
        self.tile_cost = Readout("Today")
        self.tile_requests = Readout("Requests")
        self.tile_month = Readout("This month")
        for column, readout in enumerate((self.tile_cost, self.tile_requests, self.tile_month)):
            usage_row.addWidget(readout, 0, column)
            usage_row.setColumnStretch(column, 1)
        self.usage_panel.body.addLayout(usage_row)
        self.budget_bar = BudgetBar()
        self.budget_bar.setFixedHeight(10)
        self.usage_panel.body.addWidget(self.budget_bar)
        self.budget_caption = label("", "faint", wrap=True)
        self.usage_panel.body.addWidget(self.budget_caption)
        left.addWidget(self.usage_panel, 0)

        self.alerts_panel = Panel("Alerts")
        self.alerts_body = QVBoxLayout()
        self.alerts_body.setSpacing(6)
        self.alerts_panel.body.addLayout(self.alerts_body)
        self.alerts_panel.body.addStretch(1)
        left.addWidget(self.alerts_panel, 1)

        # Centre: the orb, what Garvis is doing, and the way to everything else
        center = QVBoxLayout()
        center.setSpacing(0)
        self.core = CoreVisual(theme.BLUE, level_source=level_source)
        self.core.setMinimumSize(120, 120)
        self.core.setMaximumHeight(470)
        center.addWidget(self.core, 1)
        self.mode_label = label("Online", "stateTitle")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.mode_label)
        self.mode_detail = label("", "stateHint", wrap=True)
        self.mode_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.mode_detail)
        center.addSpacing(16)
        self.task_panel = Panel("Current task")
        self.task_panel.setMaximumWidth(470)
        self.task_body = QVBoxLayout()
        self.task_body.setSpacing(5)
        self.task_panel.body.addLayout(self.task_body)
        task_row = QHBoxLayout()
        task_row.addStretch(1)
        task_row.addWidget(self.task_panel, 100)
        task_row.addStretch(1)
        center.addLayout(task_row)
        center.addSpacing(14)
        links = QHBoxLayout()
        links.setSpacing(8)
        links.addStretch(1)
        self._links_row = links
        self.quick_links: dict[str, QPushButton] = {}
        for key, text in (("activity", "Activity"), ("sources", "Sources"), ("tools", "Tools"), ("voice", "Voice")):
            button = chip_button(text)
            button.clicked.connect(lambda _checked=False, k=key: self.open_page.emit(k))
            links.addWidget(button)
            self.quick_links[key] = button
        links.addStretch(1)
        center.addLayout(links)
        center.addStretch(0)

        # Right column: the conversation
        right = QVBoxLayout()
        right.setSpacing(12)
        self.conversation_panel = Panel("Conversation")
        self.conversation_panel.body.addWidget(conversation_view)
        right.addWidget(self.conversation_panel, 1)

        # Side columns keep the mockup's widths and give way (to a minimum)
        # on a narrower window; long wrapped text never widens a column.
        left_inner = _FitWidth()
        left_inner.setLayout(left)
        left.setContentsMargins(0, 0, 0, 0)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setWidget(left_inner)
        self._left_holder = scroller
        self._center_holder = QWidget()
        self._center_holder.setLayout(center)
        center.setContentsMargins(0, 0, 0, 0)
        self._right_holder = QWidget()
        self._right_holder.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)
        for holder in (self._left_holder, self._right_holder):
            holder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._center_holder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._center_holder.setMinimumWidth(200)
        row.addWidget(self._left_holder)
        row.addWidget(self._center_holder, 1)
        row.addWidget(self._right_holder)
        self._fit_columns()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._render_task)
        for signal in (
            state.mode_changed,
            state.status_changed,
            state.activity_changed,
            state.errors_changed,
            state.voice_changed,
            state.sources_changed,
        ):
            signal.connect(self.mark_dirty)
        state.mode_changed.connect(self.core.set_mode)
        state.voice_changed.connect(lambda: self.core.set_standby(state.standby))
        self.core.set_mode(state.mode)
        self.core.set_standby(state.standby)
        self._render_resources()

    def column_widths(self, width: int) -> tuple[int, int]:
        """Left and right column widths for a page `width` px wide."""
        sides = self.LEFT_WIDTH + self.RIGHT_WIDTH
        room = width - 2 * self.GAP - self.CENTER_MIN
        scale = min(1.0, max(0.0, room / sides))
        return (max(self.LEFT_MIN, int(self.LEFT_WIDTH * scale)), max(self.RIGHT_MIN, int(self.RIGHT_WIDTH * scale)))

    def _fit_columns(self) -> None:
        left, right = self.column_widths(self.width())
        self._left_holder.setFixedWidth(left)
        self._right_holder.setFixedWidth(right)
        # Every page is in the navigation rail too; on a narrow window keep
        # only the links that matter most here.
        wide = self.width() - left - right - 2 * self.GAP >= 330
        tall = self.height() >= 470  # a very short window keeps the orb and the task, not the shortcuts
        self.quick_links["activity"].setVisible(tall)
        self.quick_links["sources"].setVisible(tall)
        self.quick_links["tools"].setVisible(wide and tall)
        self.quick_links["voice"].setVisible(wide and tall)

    def set_info(self, info: dict) -> None:
        self._info = info
        self.mark_dirty()

    def set_usage(self, usage: dict) -> None:
        self._usage = usage
        self.mark_dirty()

    def set_telemetry(self, monitor) -> None:
        """app/ui/telemetry.py TelemetryMonitor; its samples update the
        resource lines in place (no page rebuild)."""
        self._monitor = monitor
        monitor.updated.connect(self._on_telemetry)
        self._render_resources()

    def _on_telemetry(self) -> None:
        if self.isVisible():
            self._render_resources()
            self._render_alerts()

    # ------------------------------------------------------------ render
    def refresh(self) -> None:
        state, info = self._state, self._info
        mode = state.mode
        standby = mode == Mode.IDLE and state.standby
        self.mode_label.setText("Standby" if standby else MODE_LABELS[mode])
        self.mode_detail.setText(STANDBY_DESCRIPTION if standby else MODE_DESCRIPTIONS[mode])

        operational = "Standby" if standby else MODE_LABELS[mode]
        self.status.set("State", operational, "error" if mode == Mode.ERROR else "live" if mode != Mode.IDLE else "ok")
        if state.session_active:
            voice = f"Session · {state.session_state}"
        elif state.push_to_talk:
            voice = "Push-to-talk recording"
        else:
            voice = "Listening for wake word" if state.standby else "Idle"
        if state.playing:
            voice += " · speaking"
        self.status.set("Voice", voice, "live" if state.playing or state.push_to_talk or state.session_active else None)
        self.status.set("Model", info.get("model", "—"))
        if "api_key_configured" in info:
            configured = info["api_key_configured"]
            self.status.set(
                "API key",
                "Configured" if configured else "Not set — requests will fail",
                "ok" if configured else "warn",
            )
        self.status.set("Requests", f"{plural(len(state.turns), 'request')} · {plural(state.tool_calls, 'tool step')}")

        sources = self.quick_links["sources"]
        sources.setText(f"Sources ({state.source_count})" if state.source_count else "Sources")
        self._render_task()
        self._render_usage()
        self._render_alerts()

    def _render_usage(self) -> None:
        usage, info = self._usage, self._info
        if usage and usage.get("available"):
            today, month = usage["today"], usage["month"]
            hint = (
                "Estimated from provider-reported tokens × config/models.json prices (local ledger). "
                "Not a bill — the Anthropic Console is authoritative."
            )
            self.tile_cost.set_value(money(today["cost_usd"]), hint)
            self.tile_requests.set_value(
                str(today["calls"]), "Model requests today, from the local ledger.", "warn" if today["errors"] else None
            )
            self.tile_month.set_value(money(month["cost_usd"]), hint)
            daily = info.get("daily_budget_usd")
            if daily and info.get("budget_enabled", True):
                self.budget_bar.show()
                self.budget_bar.set_ratio(today["cost_usd"] / daily)
                self.budget_caption.setText(f"{100 * today['cost_usd'] / daily:.0f}% of the {money(daily)} daily limit")
            else:
                self.budget_bar.hide()
                self.budget_caption.setText("No local daily limit set.")
        else:
            reason = (usage or {}).get("reason", "Usage not loaded yet.")
            for readout in (self.tile_cost, self.tile_requests, self.tile_month):
                readout.set_value("—", reason)
            self.budget_bar.hide()
            self.budget_caption.setText(reason)

    _GPU_NOTE = "GPU load is measured while the System page is open (the last reading is shown for 15 s)."

    def _render_resources(self) -> None:
        monitor = self._monitor
        snap = monitor.history.latest if monitor is not None else None
        if snap is None:
            waiting = "Waiting for the first sample…" if monitor is not None else "Telemetry is not running."
            for line in (self.cpu_line, self.ram_line, self.gpu_line):
                line.set("—")
            self.gpu_line.setToolTip(self._GPU_NOTE)
            self.process_label.setText(waiting)
            return
        series = monitor.history.series
        cpu = snap.cpu_percent
        self.cpu_line.set(fmt_percent(cpu), "", series["cpu"], load_tone(cpu), percent=cpu)
        ram = snap.ram_percent
        detail = f"{fmt_bytes(snap.ram_used)} / {fmt_bytes(snap.ram_total)}" if snap.ram_used is not None else ""
        self.ram_line.set(fmt_percent(ram), detail, series["ram"], load_tone(ram, 80.0, 92.0), percent=ram)
        # The GPU is sampled only in detail (System page); show that reading while it is recent.
        latest_detail = monitor.history.latest_detail
        gpu = None
        if latest_detail is not None and snap.at - latest_detail.at <= 15.0:
            gpu = latest_detail.gpu_percent
        self.gpu_line.set(
            fmt_percent(gpu),
            "" if gpu is not None else "Live on the System page",
            series.get("gpu"),
            load_tone(gpu),
            percent=gpu,
        )
        self.gpu_line.setToolTip(self._GPU_NOTE)
        process = []
        if snap.process_memory is not None:
            process.append(f"{fmt_bytes(snap.process_memory)} memory")
        if snap.process_cpu_percent is not None:
            process.append(f"{snap.process_cpu_percent:.1f}% CPU")
        self.process_label.setText("Garvis itself: " + " · ".join(process) if process else "")

    def _render_task(self) -> None:
        clear_layout(self.task_body)
        state = self._state
        turn = state.current_turn
        if turn is None:
            self._elapsed_timer.stop()
            last = state.turns[-1] if state.turns else None
            if last is None:
                self.task_body.addWidget(
                    label("No task yet. Type below, or start a voice session.", "empty", wrap=True)
                )
                return
            outcome = "failed" if last.error else "done"
            head = QHBoxLayout()
            head.addWidget(label(f"Last task · {outcome}", "key"))
            head.addStretch(1)
            head.addWidget(label(ago(last.finished_at or last.started_at), "faint"))
            self.task_body.addLayout(head)
            self.task_body.addWidget(label(elide(last.request, 140), wrap=True))
            if last.error:
                error = label(elide(last.error, 160), wrap=True)
                error.setStyleSheet(f"color: {theme.ERROR};")
                self.task_body.addWidget(error)
            elif last.tools:
                self.task_body.addWidget(
                    label(f"{plural(len(last.tools), 'tool step')} · details in Activity", "faint")
                )
            self._add_sources_link(last)
            return
        if not self._elapsed_timer.isActive() and self.isVisible():
            self._elapsed_timer.start()
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(chip("Running", "live"))
        header.addWidget(chip("Phone" if turn.source == "mobile" else turn.source))
        header.addStretch(1)
        header.addWidget(label(f"{time.time() - turn.started_at:.0f}s", "metricSmall"))
        self.task_body.addLayout(header)
        self.task_body.addWidget(label(elide(turn.request, 140), wrap=True))
        if not turn.tools:
            self.task_body.addWidget(label("Thinking — no tool steps yet.", "faint"))
        for tool in turn.tools[-2:]:
            self.task_body.addWidget(tool_line(tool))
        self._add_sources_link(turn)

    def _add_sources_link(self, turn: Turn) -> None:
        if turn.sources:
            button = link_button(f"View {len(turn.sources)} sources")
            button.clicked.connect(lambda: self.open_sources.emit(turn.id))
            self.task_body.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)

    def _render_alerts(self) -> None:
        from app.ui.telemetry import resource_alerts

        clear_layout(self.alerts_body)
        rows: list[tuple[str, str, str]] = []
        if self._monitor is not None:
            rows += [(alert.level, alert.text, "now") for alert in resource_alerts(self._monitor.history)]
        rows += [("error", record.message, ago(record.at)) for record in reversed(list(self._state.errors)[-3:])]
        if not rows:
            self.alerts_body.addWidget(label("All clear — no errors or resource warnings.", "empty", wrap=True))
            return
        for level, message, when in rows[:4]:
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(StatusMark(False if level == "error" else "warn"), 0, Qt.AlignmentFlag.AlignTop)
            text = label(elide(message, 150), wrap=True)
            text.setToolTip(message)
            if level == "warn":
                text.setStyleSheet(f"color: {theme.WARN_TEXT};")
            row.addWidget(text, 1)
            row.addWidget(label(when, "faint"), 0, Qt.AlignmentFlag.AlignTop)
            self.alerts_body.addLayout(row)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._render_resources()
        if self._state.current_turn is not None:
            self._render_task()  # restarts the elapsed-time ticker

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._elapsed_timer.stop()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit_columns()


class TurnCard(QFrame):
    open_sources = Signal(int)

    def __init__(self, turn: Turn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.turn_id = turn.id
        self._running = turn.running
        self._failed = bool(turn.error)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 14, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(label(f"#{turn.id:03d}", "panelIndex"))
        header.addWidget(label(clock_time(turn.started_at), "value"))
        header.addWidget(chip("Phone" if turn.source == "mobile" else turn.source))
        if turn.running:
            header.addWidget(chip("Running", "live"))
        elif turn.error:
            header.addWidget(chip("Error", "error"))
        else:
            header.addWidget(chip("Done", "ok"))
        header.addStretch(1)
        facts = []
        if turn.tools:
            facts.append(f"{len(turn.tools)} step{'s' if len(turn.tools) != 1 else ''}")
        if turn.finished_at is not None:
            facts.append(f"{turn.finished_at - turn.started_at:.1f}s")
        if facts:
            header.addWidget(label(" · ".join(facts), "faint"))
        layout.addLayout(header)
        request = label(turn.request, wrap=True)
        request.setStyleSheet(f"color: {theme.TEXT_1}; font-weight: 600;")
        request.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(request)
        if turn.tools:
            steps = QVBoxLayout()
            steps.setContentsMargins(4, 2, 0, 2)
            steps.setSpacing(3)
            for index, tool in enumerate(turn.tools, start=1):
                row = QHBoxLayout()
                row.setSpacing(8)
                row.addWidget(label(f"{index:02d}", "panelIndex"), 0, Qt.AlignmentFlag.AlignVCenter)
                row.addWidget(tool_line(tool), 1)
                steps.addLayout(row)
            layout.addLayout(steps)
        if turn.error:
            error = label(turn.error, wrap=True)
            error.setStyleSheet(f"color: {theme.ERROR};")
            layout.addWidget(error)
        if turn.reply:
            reply = label(elide(turn.reply, 420), "dim", wrap=True)
            reply.setToolTip(elide(turn.reply, 2000))
            layout.addWidget(reply)
        if turn.sources:
            button = QPushButton(f"View {len(turn.sources)} sources")
            button.setProperty("flat", True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda: self.open_sources.emit(turn.id))
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        accent = None
        if self._failed:
            accent = QColor(theme.ERROR)
        elif self._running:
            accent = QColor(theme.ELECTRIC_HI)
        paint_inner_card(self, accent)


class ActivityPage(Page):
    """What Garvis did: every request as a card with its numbered tool steps,
    results, errors and sources (newest first), plus the raw tool log."""

    open_sources = Signal(int)
    _MAX_CARDS = 60

    def __init__(
        self, state: GarvisState, activity_list: QListWidget | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(
            PageHeader(
                "Activity",
                "Every request this session — desktop, voice and phone — with "
                "its tool steps, results and sources. Newest first.",
            )
        )
        summary = QHBoxLayout()
        summary.setSpacing(28)
        self.stat_requests = Readout("Requests")
        self.stat_steps = Readout("Tool steps")
        self.stat_failed = Readout("Failed steps")
        self.stat_duration = Readout("Avg. request")
        self.stat_sources = Readout("Sources")
        for readout in (self.stat_requests, self.stat_steps, self.stat_failed, self.stat_duration, self.stat_sources):
            summary.addWidget(readout)
        summary.addStretch(1)
        layout.addLayout(summary)

        row = QHBoxLayout()
        row.setSpacing(14)
        area, self._column = scroll_column(self)
        row.addWidget(area, 3)
        side = QVBoxLayout()
        side.setSpacing(12)
        self.tasks_panel = TasksPanel()
        side.addWidget(self.tasks_panel, 3)
        self.log_panel = Panel("Tool log", "")
        if activity_list is not None:
            self.log_panel.body.addWidget(activity_list, 1)
            self.log_panel.body.addWidget(
                label("Each tool call as it started and finished, in order.", "faint", wrap=True)
            )
        self.log_panel.setMinimumWidth(240)
        side.addWidget(self.log_panel, 2)
        row.addLayout(side, 2)
        layout.addLayout(row, 1)
        state.activity_changed.connect(self.mark_dirty)
        state.sources_changed.connect(self.mark_dirty)

    def refresh(self) -> None:
        state = self._state
        turns_all = list(state.turns)
        tools = [tool for turn in turns_all for tool in turn.tools]
        failed = sum(1 for tool in tools if tool.success is False)
        durations = [t.finished_at - t.started_at for t in turns_all if t.finished_at is not None]
        self.stat_requests.set_value(str(len(turns_all)))
        self.stat_steps.set_value(str(len(tools)))
        self.stat_failed.set_value(str(failed), tone="error" if failed else None)
        self.stat_duration.set_value(f"{sum(durations) / len(durations):.1f}s" if durations else "—")
        self.stat_sources.set_value(str(state.source_count))

        clear_layout(self._column)
        turns = turns_all[-self._MAX_CARDS :]
        if not turns:
            self._column.addWidget(
                label("Nothing yet. Requests appear here as soon as Garvis receives them.", "empty", wrap=True)
            )
        for turn in reversed(turns):
            card = TurnCard(turn)
            card.open_sources.connect(self.open_sources)
            self._column.addWidget(card)
        self._column.addStretch(1)


class SourceCard(QFrame):
    def __init__(self, index: int, source: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.url = source["url"]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 12, 12)
        layout.setSpacing(12)
        number = label(f"{index:02d}", "metricSmall")
        number.setStyleSheet(f"color: {theme.ACCENT_TEXT};")
        layout.addWidget(number, 0, Qt.AlignmentFlag.AlignTop)
        body = QVBoxLayout()
        body.setSpacing(4)
        title = label(source.get("title") or source.get("publisher") or self.url, "link", wrap=True)
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(title)
        meta = QHBoxLayout()
        meta.setSpacing(6)
        if source.get("publisher"):
            meta.addWidget(chip(source["publisher"]))
        read = source.get("kind") == "read"
        kind_chip = chip("Read by Garvis" if read else "Search result", "ok" if read else None)
        kind_chip.setToolTip(
            "Garvis opened this page and read its text."
            if read
            else "Listed by the search engine; Garvis saw only this snippet."
        )
        meta.addWidget(kind_chip)
        meta.addWidget(label(source.get("tool", ""), "faint"))
        meta.addStretch(1)
        body.addLayout(meta)
        url = label(elide(self.url, 110), "url")
        url.setToolTip(self.url)
        url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(url)
        if source.get("snippet"):
            body.addWidget(label(source["snippet"], "dim", wrap=True))
        layout.addLayout(body, 1)
        self.open_button = QPushButton("Open")
        set_button_icon(self.open_button, "open_in_new", theme.TEXT_1, 16)
        self.open_button.setToolTip(f"Open in your browser: {self.url}")
        self.open_button.clicked.connect(lambda: open_url(self.url))
        layout.addWidget(self.open_button, 0, Qt.AlignmentFlag.AlignTop)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        paint_inner_card(self)


class SourcesPage(Page):
    def __init__(self, state: GarvisState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._selected_turn: int | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.header = PageHeader(
            "Sources",
            "Exactly what Garvis's search and page-reading tools returned — real titles and links, grouped by request.",
        )
        layout.addWidget(self.header)
        row = QHBoxLayout()
        row.setSpacing(14)
        self.turn_panel = Panel("Requests", "")
        self.turn_list = QListWidget()
        self.turn_list.currentItemChanged.connect(self._on_turn_selected)
        self.turn_panel.body.addWidget(self.turn_list)
        self.turn_panel.setFixedWidth(300)
        row.addWidget(self.turn_panel)
        self.detail_panel = Panel("Sources", "")
        self.detail_heading = label("", wrap=True)
        self.detail_heading.setStyleSheet(
            f"color: {theme.TEXT_1}; font-family: '{theme.display_family()}'; font-size: 15px; font-weight: 600;"
        )
        self.detail_panel.body.addWidget(self.detail_heading)
        self.detail_summary = label("", "faint")
        self.detail_panel.body.addWidget(self.detail_summary)
        area, self._cards = scroll_column(self)
        self.detail_panel.body.addWidget(area, 1)
        row.addWidget(self.detail_panel, 1)
        layout.addLayout(row, 1)
        state.sources_changed.connect(self.mark_dirty)
        state.activity_changed.connect(self.mark_dirty)

    @property
    def selected_turn(self) -> int | None:
        return self._selected_turn

    def show_turn(self, turn_id: int | None) -> None:
        self._selected_turn = turn_id
        self.refresh_now()

    def _turns(self) -> list[Turn]:
        return [turn for turn in reversed(self._state.turns) if turn.sources]

    def refresh(self) -> None:
        turns = self._turns()
        if self._selected_turn is None or self._state.turn(self._selected_turn) is None:
            self._selected_turn = turns[0].id if turns else None
        self.turn_list.blockSignals(True)
        self.turn_list.clear()
        for turn in turns:
            item = QListWidgetItem(
                f"{clock_time(turn.started_at)}  ·  {elide(turn.request, 60)}\n{len(turn.sources)} sources"
            )
            item.setData(Qt.ItemDataRole.UserRole, turn.id)
            item.setToolTip(turn.request)
            self.turn_list.addItem(item)
            if turn.id == self._selected_turn:
                self.turn_list.setCurrentItem(item)
        self.turn_list.blockSignals(False)
        self._render_cards()

    def _on_turn_selected(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self._selected_turn = current.data(Qt.ItemDataRole.UserRole)
            self._render_cards()

    def _render_cards(self) -> None:
        clear_layout(self._cards)
        turn = self._state.turn(self._selected_turn) if self._selected_turn is not None else None
        if turn is None or not turn.sources:
            self.detail_heading.setText("No sources yet")
            self.detail_summary.setText("")
            message = (
                "When Garvis searches the web or reads a page, the exact titles and links it used "
                "appear here, linked to the request they answered. Answers that needed no lookup "
                "have no sources."
            )
            if turn is not None:
                message = "This request used no web sources."
            self._cards.addWidget(label(message, "empty", wrap=True))
            self._cards.addStretch(1)
            return
        self.detail_heading.setText(turn.request)
        read = sum(1 for s in turn.sources if s.get("kind") == "read")
        listed = len(turn.sources) - read
        self.detail_summary.setText(
            f"{clock_time(turn.started_at)}  ·  {read} read by Garvis  ·  {listed} search results"
        )
        for index, source in enumerate(turn.sources, start=1):
            self._cards.addWidget(SourceCard(index, source))
        self._cards.addStretch(1)


class VoicePage(Page):
    def __init__(
        self,
        state: GarvisState,
        info: dict,
        level_source,
        standby_setter: Callable[[bool], None] | None,
        speak_replies: QCheckBox,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._info = info
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(
            PageHeader(
                "Voice",
                "Local speech: faster-whisper listens, Piper speaks. This page "
                "shows the live state of that pipeline; it does not run one of its own.",
            )
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        self.state_panel = Panel("Live state", "01")
        self.mode_label = label("", "pageTitle")
        self.state_panel.body.addWidget(self.mode_label)
        self.scope = LevelScope(level_source)
        self.state_panel.body.addWidget(self.scope)
        self.scope_caption = label("Output level of Garvis's voice, from the audio being played.", "faint")
        self.state_panel.body.addWidget(self.scope_caption)
        self.state = KeyValueGrid()
        self.state_panel.body.addWidget(self.state)
        grid.addWidget(self.state_panel, 0, 0)

        self.controls_panel = Panel("Controls", "02")
        self.standby_checkbox = ToggleSwitch("Wake word standby (“Hey Garvis”)")
        self.standby_checkbox.setEnabled(standby_setter is not None)
        if standby_setter is not None:
            self.standby_checkbox.toggled.connect(standby_setter)
        self.controls_panel.body.addWidget(self.standby_checkbox)
        self.controls_panel.body.addWidget(
            label(
                "Listens locally for the wake phrase whenever no session or "
                "push-to-talk turn holds the microphone -- and only while no other "
                "Garvis device (phone, Eye Station) has the voice. Off at every "
                "launch: tick it to ask for standby.",
                "faint",
                wrap=True,
            )
        )
        self.voice_owner_label = label("No device is listening for “Hey Garvis”.", "faint", wrap=True)
        self.voice_owner_label.setObjectName("voiceOwner")
        self.controls_panel.body.addWidget(self.voice_owner_label)
        self.speak_checkbox = ToggleSwitch("Speak replies to typed requests")
        self.speak_checkbox.setChecked(speak_replies.isChecked())
        self.speak_checkbox.toggled.connect(speak_replies.setChecked)
        speak_replies.toggled.connect(self.speak_checkbox.setChecked)
        self.controls_panel.body.addWidget(self.speak_checkbox)
        self.controls_panel.body.addStretch(1)
        grid.addWidget(self.controls_panel, 0, 1)

        self.engine_panel = Panel("Engine", "03")
        self.engine = KeyValueGrid()
        self.engine_panel.body.addWidget(self.engine)
        self.engine_panel.body.addStretch(1)
        grid.addWidget(self.engine_panel, 1, 1)

        self.timeline_panel = Panel("Timeline", "04")
        self.timeline = QListWidget()
        self.timeline_panel.body.addWidget(self.timeline)
        grid.addWidget(self.timeline_panel, 1, 0)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, 1)

        state.mode_changed.connect(self.scope.set_mode)
        state.voice_changed.connect(self.mark_dirty)
        state.mode_changed.connect(self.mark_dirty)

    def set_info(self, info: dict) -> None:
        self._info = info
        self.mark_dirty()

    def set_voice_owner(self, text: str) -> None:
        """Who holds Garvis's voice now (app/core/voice_ownership.py), from the bridge."""
        self.voice_owner_label.setText(text)

    def set_standby_checked(self, active: bool) -> None:
        self.standby_checkbox.blockSignals(True)
        self.standby_checkbox.setChecked(active)
        self.standby_checkbox.blockSignals(False)

    def refresh(self) -> None:
        state, info = self._state, self._info
        mode = state.mode
        self.mode_label.setText("Standby" if mode == Mode.IDLE and state.standby else MODE_LABELS[mode])
        self.state.set(
            "Playback", "Speaking on the PC speakers" if state.playing else "Silent", "live" if state.playing else None
        )
        self.state.set(
            "Push-to-talk", "Recording" if state.push_to_talk else "Off", "live" if state.push_to_talk else None
        )
        self.state.set(
            "Session",
            state.session_state if state.session_active else "Not running",
            "live" if state.session_active else None,
        )
        self.state.set("Wake word", "Armed" if state.standby else "Not listening")
        self.engine.set("Speech to text", info.get("stt", "—"))
        self.engine.set("Language", info.get("stt_language", "—"))
        self.engine.set("Text to speech", info.get("tts", "—"))
        self.engine.set("Playback", info.get("playback_backend", "—"))
        if "echo_guard" in info:
            self.engine.set("Echo guard", "On (barge-in safe)" if info["echo_guard"] else "Off")
        self.timeline.clear()
        events = list(state.timeline)
        if not events:
            self.timeline.addItem("No voice activity yet.")
        for event in reversed(events):
            self.timeline.addItem(f"{clock_time(event.at)}   {event.label}")


class UsagePage(Page):
    refresh_requested = Signal()

    def __init__(self, info: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = info
        self._usage: dict | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.header = PageHeader(
            "Usage",
            "Claude API requests from Garvis's local ledger. Costs are estimates "
            "(provider-reported tokens × config/models.json prices), not a bill — the "
            "Anthropic Console is authoritative.",
        )
        self.refresh_button = QPushButton("Refresh")
        set_button_icon(self.refresh_button, "refresh", theme.TEXT_1, 18)
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.header.actions.addWidget(self.refresh_button)
        layout.addWidget(self.header)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        self.spend_panel = Panel("Estimated spend", "01")
        self.day_caption = label("", "dim")
        self.day_bar = BudgetBar()
        self.day_bar.setFixedHeight(12)
        self.month_caption = label("", "dim")
        self.month_bar = BudgetBar()
        self.month_bar.setFixedHeight(12)
        for widget in (self.day_caption, self.day_bar, self.month_caption, self.month_bar):
            self.spend_panel.body.addWidget(widget)
        self.budget_note = label("", "faint", wrap=True)
        self.spend_panel.body.addWidget(self.budget_note)
        self.spend_panel.body.addStretch(1)
        grid.addWidget(self.spend_panel, 0, 0)
        self.today_panel = Panel("Today", "02")
        self.today = KeyValueGrid()
        self.today_panel.body.addWidget(self.today)
        self.today_panel.body.addStretch(1)
        grid.addWidget(self.today_panel, 0, 1)
        self.trend_panel = Panel("Estimated cost per day · last 14 days", "03")
        self.trend = UsageTrend()
        self.trend_panel.body.addWidget(self.trend)
        self.trend_caption = label("", "faint")
        self.trend_panel.body.addWidget(self.trend_caption)
        grid.addWidget(self.trend_panel, 1, 0, 1, 2)
        self.recent_panel = Panel("Recent requests", "04")
        self.recent = QListWidget()
        self.recent.setFont(theme.mono_font(8.5))
        self.recent_panel.body.addWidget(self.recent)
        grid.addWidget(self.recent_panel, 2, 0, 1, 2)
        grid.setRowStretch(2, 1)
        layout.addLayout(grid, 1)

    def set_info(self, info: dict) -> None:
        self._info = info
        self.mark_dirty()

    def set_usage(self, usage: dict) -> None:
        self._usage = usage
        self.mark_dirty()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.refresh_requested.emit()

    def refresh(self) -> None:
        usage, info = self._usage, self._info
        self.recent.clear()
        if not usage or not usage.get("available"):
            reason = (usage or {}).get("reason", "Loading…")
            self.day_caption.setText("Today  —")
            self.month_caption.setText("This month  —")
            self.day_bar.set_ratio(0)
            self.month_bar.set_ratio(0)
            self.budget_note.setText(reason)
            self.recent.addItem(reason)
            self.trend.set_days([])
            self.trend_caption.setText("")
            return
        daily_rows = usage.get("daily") or []
        self.trend.set_days(daily_rows)
        if daily_rows:
            total = sum(d["cost_usd"] for d in daily_rows)
            calls = sum(d["calls"] for d in daily_rows)
            self.trend_caption.setText(
                f"{money(total)} over {len(daily_rows)} days · {calls} requests · "
                "estimates from the local ledger, not a bill"
            )
        today, month = usage["today"], usage["month"]
        daily, monthly = info.get("daily_budget_usd"), info.get("monthly_budget_usd")
        self.day_caption.setText(f"Today  {money(today['cost_usd'])}" + (f"  /  {money(daily)}" if daily else ""))
        self.month_caption.setText(
            f"This month  {money(month['cost_usd'])}" + (f"  /  {money(monthly)}" if monthly else "")
        )
        self.day_bar.set_ratio(today["cost_usd"] / daily if daily else 0)
        self.month_bar.set_ratio(month["cost_usd"] / monthly if monthly else 0)
        if info.get("budget_enabled") is False:
            self.budget_note.setText("Local spending limits are off (GARVIS_LLM_BUDGET_ENABLED=false).")
        elif daily or monthly:
            self.budget_note.setText(
                f"Garvis stops sending requests at its local limits (per task "
                f"{money(info.get('max_cost_per_task_usd'))}). Set a hard limit in the "
                "Anthropic Console too."
            )
        self.today.set(
            "Requests",
            f"{today['calls']}  ({today['ok']} ok · {today['errors']} failed)",
            "warn" if today["errors"] else None,
        )
        self.today.set("Tasks", str(today["tasks"]))
        self.today.set("Input tokens", tokens(today["input_tokens"]))
        self.today.set("Output tokens", tokens(today["output_tokens"]))
        self.today.set("Cache read", tokens(today["cache_read_tokens"]))
        self.today.set("Cache write", tokens(today["cache_write_tokens"]))
        self.today.set("Month requests", str(month["calls"]))
        if not usage["recent"]:
            self.recent.addItem("No requests recorded yet.")
        for row in usage["recent"]:
            stamp = (row["ts"] or "")[5:16].replace("T", " ")
            status = row["status"] if row["status"] == "ok" else f"{row['status']}:{row['error_type'] or '?'}"
            latency = f"{row['latency_ms'] / 1000:.1f}s" if row["latency_ms"] else "—"
            item = QListWidgetItem(
                f"{stamp}  {(row['tier'] or '-'):<6} {status:<16} in {tokens(row['input_tokens']):>7}  "
                f"out {tokens(row['output_tokens']):>6}  cache {tokens(row['cache_read_tokens']):>7}  "
                f"{money(row['cost_usd']):>8}  {latency:>6}  {row['model'] or ''}"
            )
            if row["status"] != "ok":
                item.setForeground(QColor(theme.ERROR))
            self.recent.addItem(item)


class SettingsPage(Page):
    """Current configuration (read from .env at startup -- edit it and restart
    to change those) plus interface preferences that apply immediately."""

    orb_enabled_changed = Signal(bool)
    orb_corner_changed = Signal(str)
    animations_changed = Signal(bool)

    def __init__(self, info: dict, mobile_button: QPushButton, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = info
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(
            PageHeader(
                "Settings",
                "Values below come from Garvis's .env (secrets are never shown). "
                "Edit .env and restart Garvis to change them. Interface preferences apply "
                "immediately.",
            )
        )
        area, column = scroll_column(self)
        layout.addWidget(area, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        self.model_panel = Panel("Model", "01")
        self.model = KeyValueGrid()
        self.model_panel.body.addWidget(self.model)
        self.model_panel.body.addStretch(1)
        grid.addWidget(self.model_panel, 0, 0)
        self.voice_panel = Panel("Voice", "02")
        self.voice = KeyValueGrid()
        self.voice_panel.body.addWidget(self.voice)
        self.voice_panel.body.addStretch(1)
        grid.addWidget(self.voice_panel, 0, 1)
        self.desktop_panel = Panel("Desktop & browser", "03")
        self.desktop = KeyValueGrid()
        self.desktop_panel.body.addWidget(self.desktop)
        self.desktop_panel.body.addStretch(1)
        grid.addWidget(self.desktop_panel, 1, 0)
        self.mobile_panel = Panel("Phone", "04")
        self.mobile = KeyValueGrid()
        self.mobile_panel.body.addWidget(self.mobile)
        self.mobile_panel.body.addWidget(mobile_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.mobile_panel.body.addStretch(1)
        grid.addWidget(self.mobile_panel, 1, 1)

        self.interface_panel = Panel("Interface", "05")
        self.orb_checkbox = ToggleSwitch("Floating orb when collapsed")
        self.orb_checkbox.setChecked(True)
        self.orb_checkbox.toggled.connect(self.orb_enabled_changed)
        self.interface_panel.body.addWidget(self.orb_checkbox)
        corner_row = QHBoxLayout()
        corner_row.addWidget(label("Orb corner", "key"))
        self.corner_combo = QComboBox()
        for corner in CORNERS:
            self.corner_combo.addItem(CORNER_LABELS[corner], corner)
        self.corner_combo.activated.connect(
            lambda index: self.orb_corner_changed.emit(self.corner_combo.itemData(index))
        )
        corner_row.addWidget(self.corner_combo)
        corner_row.addStretch(1)
        self.interface_panel.body.addLayout(corner_row)
        self.interface_panel.body.addWidget(
            label(
                "Choosing a corner moves the orb there (on the primary screen); "
                "drag the orb anywhere, on any monitor, and it stays put.",
                "faint",
                wrap=True,
            )
        )
        self.animations_checkbox = ToggleSwitch("Orb ↔ dashboard animation")
        self.animations_checkbox.setChecked(True)
        self.animations_checkbox.toggled.connect(self.animations_changed)
        self.interface_panel.body.addWidget(self.animations_checkbox)
        self.interface_note = label("", "faint", wrap=True)
        self.interface_panel.body.addWidget(self.interface_note)
        self.set_interface_available(False)
        grid.addWidget(self.interface_panel, 2, 0)

        self.telemetry_panel = Panel("System monitor", "06")
        self.telemetry = KeyValueGrid()
        self.telemetry.set("Source", "Windows performance APIs (no extra software)")
        self.telemetry.set("Sampling", "Background thread · 1 s on System · 2 s elsewhere · off when collapsed")
        self.telemetry.set("Model calls", "None — telemetry never uses the API")
        self.telemetry_panel.body.addWidget(self.telemetry)
        self.telemetry_panel.body.addWidget(label("Planned features are listed on the Tools page.", "faint", wrap=True))
        self.telemetry_panel.body.addStretch(1)
        grid.addWidget(self.telemetry_panel, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        column.addLayout(grid)
        column.addStretch(1)

    def set_info(self, info: dict) -> None:
        self._info = info
        self.mark_dirty()

    def set_interface_available(self, available: bool, prefs=None, state_path: str = "") -> None:
        for widget in (self.orb_checkbox, self.corner_combo, self.animations_checkbox):
            widget.setEnabled(available)
        if prefs is not None:
            for widget, value in ((self.orb_checkbox, prefs.orb_enabled), (self.animations_checkbox, prefs.animations)):
                widget.blockSignals(True)
                widget.setChecked(value)
                widget.blockSignals(False)
            self.corner_combo.setCurrentIndex(max(0, self.corner_combo.findData(prefs.orb_corner)))
        self.interface_note.setText(
            f"Saved in {state_path}" if available and state_path else "Available when Garvis runs with its desktop orb."
        )

    def refresh(self) -> None:
        info = self._info

        def yes_no(key: str) -> str:
            return "—" if key not in info else ("On" if info[key] else "Off")

        self.model.set("Provider", info.get("provider", "—"))
        self.model.set("Model", info.get("model", "—"))
        if "api_key_configured" in info:
            self.model.set(
                "API key",
                "Configured" if info["api_key_configured"] else "Not set",
                "ok" if info["api_key_configured"] else "warn",
            )
        self.model.set("Spending limits", yes_no("budget_enabled"))
        if "daily_budget_usd" in info:
            self.model.set(
                "Limits",
                f"{money(info['max_cost_per_task_usd'])} / task · "
                f"{money(info['daily_budget_usd'])} / day · {money(info['monthly_budget_usd'])} / month",
            )
        self.model.set("Usage ledger", info.get("usage_db", "—"))
        self.voice.set("Speech to text", info.get("stt", "—"))
        self.voice.set("Language", info.get("stt_language", "—"))
        self.voice.set("Text to speech", info.get("tts", "—"))
        self.voice.set("Playback", info.get("playback_backend", "—"))
        self.voice.set("Echo guard", yes_no("echo_guard"))
        self.voice.set("Wake word", yes_no("wake_word_enabled"))
        self.desktop.set("Desktop control", yes_no("desktop_control_enabled"))
        self.desktop.set("Garvis workspace", yes_no("workspace_enabled"))
        self.desktop.set("Research engine", info.get("research_engine", "—"))
        self.desktop.set("Global hotkey", info.get("global_hotkey") or "—")
        self.mobile.set("Phone interface", yes_no("mobile_enabled"))
        self.mobile.set("Works with", "iPhone (Safari) · Android (Chrome)")


# --------------------------------------------------------------------------
# System page: live telemetry of this computer (app/ui/telemetry.py)
# --------------------------------------------------------------------------
class SystemPage(Page):
    """Real measurements from Windows' own interfaces, sampled on a
    background thread only while the dashboard is open. Estimates say so;
    anything Windows doesn't expose is shown as unavailable."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._monitor = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.header = PageHeader(
            "System",
            "Live measurements of this computer from Windows' own performance "
            "interfaces — sampled in the background, once a second while this page is open.",
        )
        self.live_chip = chip("Paused")
        self.live_chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.refreshed_label = label("", "faint")
        self.header.actions.addWidget(self.refreshed_label)
        self.header.actions.addWidget(self.live_chip)
        layout.addWidget(self.header)
        area, column = scroll_column(self)
        layout.addWidget(area, 1)
        self.grid = ResponsiveGrid(1040)

        # Processor
        self.cpu_panel = Panel("Processor", "01")
        top = QHBoxLayout()
        top.setSpacing(16)
        self.cpu_gauge = ArcGauge("CPU load", theme.CHART_CPU)
        top.addWidget(self.cpu_gauge, 0, Qt.AlignmentFlag.AlignTop)
        side = QVBoxLayout()
        side.setSpacing(6)
        self.cpu_spark = Sparkline(theme.CHART_CPU, span=180.0, height=56)
        side.addWidget(label("Last 3 minutes · system-wide", "section"))
        side.addWidget(self.cpu_spark)
        self.cpu_info = KeyValueGrid()
        side.addWidget(self.cpu_info)
        top.addLayout(side, 1)
        self.cpu_panel.body.addLayout(top)
        self.cpu_panel.body.addWidget(label("Per logical processor", "section"))
        self.core_grid = CoreGrid()
        self.cpu_panel.body.addWidget(self.core_grid)
        self.grid.add(self.cpu_panel)

        # Memory
        self.ram_panel = Panel("Memory", "02")
        top = QHBoxLayout()
        top.setSpacing(16)
        self.ram_gauge = ArcGauge("In use", theme.CHART_MEMORY)
        top.addWidget(self.ram_gauge, 0, Qt.AlignmentFlag.AlignTop)
        side = QVBoxLayout()
        side.setSpacing(6)
        self.ram_spark = Sparkline(theme.CHART_MEMORY, span=180.0, height=56)
        side.addWidget(label("Last 3 minutes · system-wide", "section"))
        side.addWidget(self.ram_spark)
        self.ram_info = KeyValueGrid()
        side.addWidget(self.ram_info)
        top.addLayout(side, 1)
        self.ram_panel.body.addLayout(top)
        self.ram_panel.body.addStretch(1)
        self.grid.add(self.ram_panel)

        # Garvis's own process
        self.process_panel = Panel("Garvis process", "03")
        self.process_cpu = ResourceLine("CPU (share of all cores)", theme.CHART_PROCESS)
        self.process_memory = ResourceLine("Working set", theme.CHART_PROCESS, ceiling=None)
        self.process_panel.body.addWidget(self.process_cpu)
        self.process_panel.body.addWidget(self.process_memory)
        self.process_info = KeyValueGrid()
        self.process_panel.body.addWidget(self.process_info)
        self.process_panel.body.addStretch(1)
        self.grid.add(self.process_panel)

        # Storage
        self.storage_panel = Panel("Storage", "04")
        self.disk_rows = QVBoxLayout()
        self.disk_rows.setSpacing(8)
        self.storage_panel.body.addLayout(self.disk_rows)
        self.disk_read = ResourceLine("Disk read", theme.CHART_DISK_READ, ceiling=None)
        self.disk_write = ResourceLine("Disk write", theme.CHART_DISK_WRITE, ceiling=None)
        self.storage_panel.body.addWidget(self.disk_read)
        self.storage_panel.body.addWidget(self.disk_write)
        self.storage_panel.body.addStretch(1)
        self._disk_widgets: dict[str, tuple[QLabel, BudgetBar, QLabel]] = {}
        self.grid.add(self.storage_panel)

        # Graphics
        self.gpu_panel = Panel("Graphics", "05")
        self.gpu_line = ResourceLine("GPU load (busiest engine)", theme.CHART_GPU)
        self.gpu_panel.body.addWidget(self.gpu_line)
        self.gpu_info = KeyValueGrid()
        self.gpu_panel.body.addWidget(self.gpu_info)
        self.gpu_panel.body.addStretch(1)
        self.grid.add(self.gpu_panel)

        # This machine / the monitor itself
        self.machine_panel = Panel("Machine & monitoring", "06")
        self.machine_info = KeyValueGrid()
        self.machine_panel.body.addWidget(self.machine_info)
        self.machine_panel.body.addStretch(1)
        self.grid.add(self.machine_panel)
        column.addWidget(self.grid)
        column.addStretch(1)

    def set_monitor(self, monitor) -> None:
        self._monitor = monitor
        monitor.updated.connect(self.mark_dirty)
        monitor.static_ready.connect(lambda _info: self.mark_dirty())
        self.mark_dirty()

    def refresh(self) -> None:
        monitor = self._monitor
        static = monitor.static if monitor is not None else None
        snap = monitor.history.latest_detail if monitor is not None else None
        latest = monitor.history.latest if monitor is not None else None
        series = monitor.history.series if monitor is not None else {}

        if monitor is None:
            set_tone(self.live_chip, None)
            self.live_chip.setText("Unavailable")
        elif monitor.demand == "detail" and latest is not None:
            set_tone(self.live_chip, "live")
            self.live_chip.setText(f"● Live · {monitor.interval:.0f} s")
        else:
            set_tone(self.live_chip, None)
            self.live_chip.setText("Paused" if latest is not None else "Starting")
        self.refreshed_label.setText(f"Last sample {clock_time(latest.at)}" if latest is not None else "")
        snap = snap or latest

        # Processor
        self.cpu_gauge.set_value(
            snap.cpu_percent if snap else None,
            f"{static.logical_cpus} threads" if static and static.logical_cpus else "",
        )
        self.cpu_spark.set_series(series.get("cpu"))
        self.cpu_info.set("Model", (static.cpu_model if static else None) or "Unavailable")
        self.cpu_info.set("Logical processors", str(static.logical_cpus) if static and static.logical_cpus else "—")
        if snap is not None and snap.cpu_mhz:
            self.cpu_info.set("Speed", f"≈ {snap.cpu_mhz / 1000:.2f} GHz (Windows estimate)")
        else:
            self.cpu_info.set("Speed", "Unavailable")
        self.cpu_info.set("Base clock", f"{static.base_mhz / 1000:.2f} GHz" if static and static.base_mhz else "—")
        self.core_grid.set_loads(snap.per_core if snap else None)

        # Memory
        self.ram_gauge.set_value(
            snap.ram_percent if snap else None, fmt_bytes(snap.ram_total, 0) if snap and snap.ram_total else ""
        )
        self.ram_spark.set_series(series.get("ram"))
        self.ram_info.set("In use", fmt_bytes(snap.ram_used) if snap else "—")
        self.ram_info.set("Available", fmt_bytes(snap.ram_available) if snap else "—")
        self.ram_info.set("Total", fmt_bytes(snap.ram_total) if snap else "—")
        if snap is not None and snap.process_memory is not None and snap.ram_total:
            share = 100.0 * snap.process_memory / snap.ram_total
            self.ram_info.set("Garvis's share", f"{fmt_bytes(snap.process_memory)} ({share:.1f}% of total)")

        # Garvis process
        proc_cpu = snap.process_cpu_percent if snap else None
        self.process_cpu.set(
            f"{proc_cpu:.1f}%" if proc_cpu is not None else "—",
            "",
            series.get("process_cpu"),
            load_tone(proc_cpu, 50.0, 80.0),
        )
        self.process_memory.set(fmt_bytes(snap.process_memory) if snap else "—", "", series.get("process_memory"))
        self.process_info.set("Private memory", fmt_bytes(snap.process_private) if snap else "—")
        self.process_info.set("Measured", "This process only — the app, its agent loop and voice pipeline")

        # Storage
        disks = snap.disks if snap is not None else None
        if not disks:
            if not self._disk_widgets and self.disk_rows.count() == 0:
                self.disk_rows.addWidget(
                    label("Drive capacity appears with the first detailed sample.", "empty", wrap=True)
                )
        else:
            if not self._disk_widgets:
                clear_layout(self.disk_rows)
            for disk in disks:
                widgets = self._disk_widgets.get(disk.root)
                if widgets is None:
                    row = QHBoxLayout()
                    row.setSpacing(10)
                    name = label(disk.root, "metricSmall")
                    name.setFixedWidth(34)
                    bar = BudgetBar(warn=0.85, error=0.95)
                    bar.setFixedHeight(12)
                    text = label("", "faint")
                    row.addWidget(name)
                    row.addWidget(bar, 1)
                    row.addWidget(text)
                    self.disk_rows.addLayout(row)
                    widgets = self._disk_widgets[disk.root] = (name, bar, text)
                widgets[1].set_ratio(disk.used / disk.total if disk.total else 0)
                widgets[2].setText(
                    f"{fmt_bytes(disk.used, 0)} / {fmt_bytes(disk.total, 0)} · {fmt_bytes(disk.free, 0)} free"
                )
        self.disk_read.set(fmt_rate(snap.disk_read_bps) if snap else "—", "", series.get("disk_read"))
        self.disk_write.set(fmt_rate(snap.disk_write_bps) if snap else "—", "", series.get("disk_write"))

        # Graphics
        gpus = static.gpus if static else ()
        gpu = snap.gpu_percent if snap else None
        self.gpu_line.set(fmt_percent(gpu) if gpu is not None else "Unavailable", "", series.get("gpu"), load_tone(gpu))
        self.gpu_info.set("Adapter", "  ·  ".join(g.name for g in gpus) if gpus else "Unavailable")
        dedicated_total = next((g.dedicated_bytes for g in gpus if g.dedicated_bytes), None)
        used = snap.gpu_memory_used if snap else None
        if used is not None:
            self.gpu_info.set(
                "Dedicated memory",
                f"{fmt_bytes(used)} in use" + (f" of {fmt_bytes(dedicated_total)}" if dedicated_total else ""),
            )
        else:
            self.gpu_info.set("Dedicated memory", "Unavailable")
        shared = snap.gpu_shared_used if snap else None
        self.gpu_info.set("Shared memory", f"{fmt_bytes(shared)} in use" if shared is not None else "Unavailable")
        self.gpu_info.set("Temperature", "Not exposed reliably by Windows — not shown")

        # Machine & monitoring
        self.machine_info.set("Windows", (static.os_name if static else None) or "—")
        self.machine_info.set("Uptime", fmt_uptime(latest.uptime_seconds) if latest else "—")
        self.machine_info.set("Source", "kernel32 · ntdll · psapi · PDH counters (no extra software)")
        cadence = "1 s on this page · 2 s elsewhere in the dashboard · stopped while collapsed"
        self.machine_info.set("Sampling", cadence)
        errors = latest.errors if latest is not None else ()
        self.machine_info.set("Sampling errors", "; ".join(errors) if errors else "None", "warn" if errors else None)


# --------------------------------------------------------------------------
# Tools page: what Garvis can actually do (the live tool registry)
# --------------------------------------------------------------------------
TOOL_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("web", "Web & research", ("web_research", "web_search", "fetch_webpage_text", "open_website")),
    ("browser", "Browser automation", ("browser_",)),
    (
        "desktop",
        "Desktop control",
        (
            "mouse_",
            "keyboard_",
            "scroll",
            "screen_",
            "list_windows",
            "focus_window",
            "switch_window",
            "window_action",
            "wait_for_window",
            "read_screen_text",
            "inspect_ui",
            "click_ui_element",
            "switch_workspace",
        ),
    ),
    (
        "apps",
        "Applications & system",
        (
            "open_application",
            "list_installed_apps",
            "list_processes",
            "system_status",
            "recent_system_errors",
            "run_powershell",
        ),
    ),
    (
        "files",
        "Files",
        ("open_path", "list_directory", "read_text_file", "find_files", "write_text_file", "edit_text_file"),
    ),
    ("mail", "Communication", ("gmail_",)),
    ("memory", "Memory & utilities", ("remember_fact", "recall_facts", "calculator", "get_current_datetime")),
)

PLANNED = (
    ("Persistent memory and context efficiency", "Fewer repeated instructions per request."),
    ("Camera / vision from the phone", "Not implemented."),
    ("Per-sentence citations inside answers", "Today sources are linked per request."),
)


def tool_group(name: str) -> str:
    for key, _title, patterns in TOOL_GROUPS:
        if any(name == p or (p.endswith("_") and name.startswith(p)) for p in patterns):
            return key
    return "other"


def humanize_tool(name: str) -> str:
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def first_sentence(text: str, limit: int = 170) -> str:
    text = " ".join((text or "").split())
    for end in (". ", "; "):
        if end in text:
            text = text.split(end)[0] + "."
            break
    return elide(text, limit)


def integration_rows(info: dict, telemetry_supported: bool = True) -> list[dict]:
    """Each integration's genuine status, derived only from the running
    configuration (AgentBridge.system_info). `action` is an existing,
    working control or None."""
    rows: list[dict] = []

    def add(name, status, tone, detail, action=None):
        rows.append({"name": name, "status": status, "tone": tone, "detail": detail, "action": action})

    if "api_key_configured" in info:
        ok = info["api_key_configured"]
        add(
            f"{info.get('model', 'Claude')} · Anthropic API",
            "Configured" if ok else "API key missing",
            "ok" if ok else "warn",
            "The only model Garvis uses — no other model or provider.",
        )
    if "stt" in info:
        add(
            "Local speech",
            "Available",
            "ok",
            f"Listens with {info['stt']} · speaks with {info.get('tts', '—')}",
            ("Voice", "voice"),
        )
    if "wake_word_enabled" in info:
        add(
            "Wake word “Hey Garvis”",
            "Tick to listen",
            None,
            "Never claimed at startup: tick it on the Voice page. Only one Garvis device listens at a time.",
            ("Voice", "voice"),
        )
    if "research_engine" in info:
        add(
            "Web research",
            "Available",
            "ok",
            f"Searches with {info['research_engine']}, reads pages in Chrome.",
            ("Sources", "sources"),
        )
        add("Browser automation", "Available", "ok", "Chrome through Playwright, opened on first use.")
    if "desktop_control_enabled" in info:
        on = info["desktop_control_enabled"]
        add(
            "Desktop control",
            "Available" if on else "Disabled",
            "ok" if on else None,
            "Mouse, keyboard, windows and screen reading — risky steps ask first."
            if on
            else "Turned off in Garvis's .env (desktop_control_enabled).",
        )
    if "workspace_enabled" in info:
        on = info["workspace_enabled"]
        add(
            "Garvis workspace",
            "Available" if on else "Disabled",
            "ok" if on else None,
            "A separate virtual desktop for Garvis's own windows." if on else "Turned off in Garvis's .env.",
        )
    if "gmail_status" in info:
        status = info["gmail_status"]
        label_text, tone, detail = {
            "signed_in": ("Signed in", "ok", "Google sign-in saved on this PC. Sending always asks first."),
            "sign_in_needed": (
                "Sign-in on first use",
                "warn",
                "Client file present; Google asks you to sign in the first time Garvis uses Gmail.",
            ),
        }.get(status, ("Not configured", None, "No Google OAuth client file — see the README to set Gmail up."))
        add("Gmail", label_text, tone, detail)
    if "mobile_enabled" in info:
        on = info["mobile_enabled"]
        add(
            "iPhone interface",
            "Available" if on else "Disabled",
            "ok" if on else None,
            "The same Garvis from Safari on your iPhone." if on else "Turned off in Garvis's .env (mobile_enabled).",
            ("Pair iPhone…", "pair") if on else None,
        )
        add(
            "Android interface",
            "Available" if on else "Disabled",
            "ok" if on else None,
            "The same Garvis from Chrome on your Android phone (e.g. Samsung Galaxy)."
            if on
            else "Turned off in Garvis's .env (mobile_enabled).",
            ("Pair Android phone…", "pair") if on else None,
        )
    if "budget_enabled" in info:
        on = info["budget_enabled"]
        add(
            "Spending limits",
            "On" if on else "Off",
            "ok" if on else "warn",
            f"Local limits: {money(info.get('max_cost_per_task_usd'))} per task · "
            f"{money(info.get('daily_budget_usd'))} per day."
            if on
            else "Garvis will not stop at a local limit.",
            ("Usage", "usage"),
        )
    add(
        "System telemetry",
        "Live" if telemetry_supported else "Unavailable",
        "ok" if telemetry_supported else None,
        "CPU, memory, disks and GPU from Windows' own interfaces."
        if telemetry_supported
        else "Live telemetry is only implemented on Windows.",
        ("System", "system"),
    )
    return rows


class _SearchableRow(QWidget):
    def __init__(self, haystack: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._haystack = haystack.lower()

    def matches(self, query: str) -> bool:
        return all(word in self._haystack for word in query.lower().split())


class IntegrationRow(_SearchableRow):
    action_requested = Signal(str)

    def __init__(self, row: dict, parent: QWidget | None = None) -> None:
        super().__init__(f"{row['name']} {row['status']} {row['detail']}", parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        text = QVBoxLayout()
        text.setSpacing(2)
        title = QHBoxLayout()
        title.setSpacing(8)
        name = label(row["name"])
        name.setStyleSheet(f"color: {theme.TEXT_1}; font-weight: 600;")
        title.addWidget(name)
        self.status_chip = chip(row["status"], row["tone"])
        title.addWidget(self.status_chip)
        title.addStretch(1)
        text.addLayout(title)
        text.addWidget(label(row["detail"], "faint", wrap=True))
        layout.addLayout(text, 1)
        self.action_button = None
        if row.get("action"):
            caption, key = row["action"]
            self.action_button = QPushButton(caption)
            self.action_button.setProperty("flat", True)
            self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.action_button.clicked.connect(lambda: self.action_requested.emit(key))
            layout.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignVCenter)


class ToolRow(_SearchableRow):
    def __init__(self, tool: dict, parent: QWidget | None = None) -> None:
        description = first_sentence(tool.get("description", ""))
        super().__init__(f"{tool['name']} {humanize_tool(tool['name'])} {tool.get('description', '')}", parent)
        self.tool_name = tool["name"]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(10)
        text = QVBoxLayout()
        text.setSpacing(1)
        title = QHBoxLayout()
        title.setSpacing(8)
        name = label(humanize_tool(tool["name"]))
        name.setStyleSheet(f"color: {theme.TEXT_1}; font-weight: 500;")
        title.addWidget(name)
        title.addWidget(label(tool["name"], "url"))
        title.addStretch(1)
        text.addLayout(title)
        detail = label(description, "faint", wrap=True)
        detail.setToolTip(tool.get("description", ""))
        text.addWidget(detail)
        layout.addLayout(text, 1)
        if tool.get("confirm"):
            ask = chip("Asks first", "warn")
            ask.setToolTip("Garvis asks for your confirmation before running this tool.")
            layout.addWidget(ask, 0, Qt.AlignmentFlag.AlignTop)
        elif tool.get("conditional"):
            ask = chip("Asks when risky")
            ask.setToolTip("Garvis decides per call: anything that could change your system asks first.")
            layout.addWidget(ask, 0, Qt.AlignmentFlag.AlignTop)
        risk = tool.get("risk", "low")
        if risk in ("medium", "high") and not tool.get("confirm") and not tool.get("conditional"):
            layout.addWidget(chip(f"{risk} risk", "warn" if risk == "high" else None), 0, Qt.AlignmentFlag.AlignTop)


class ToolsPage(Page):
    """The integrations and tools Garvis really has: read from the running
    Agent's tool registry and configuration, never from a wish list (the
    planned items are listed separately, as planned)."""

    open_page = Signal(str)
    pair_requested = Signal()

    def __init__(self, info: dict, catalog: list[dict] | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = info
        self._catalog = catalog
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.header = PageHeader(
            "Tools",
            "Everything Garvis can actually do, read from its live tool registry and "
            "configuration. Tools marked “Asks first” never run without your confirmation.",
        )
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter tools…")
        search_icon = fonts.icon("search", theme.TEXT_3, 18)
        if not search_icon.isNull():
            self.search.addAction(search_icon, QLineEdit.ActionPosition.LeadingPosition)
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self._apply_filter)
        self.header.actions.addWidget(self.search)
        layout.addWidget(self.header)
        area, self._column = scroll_column(self)
        layout.addWidget(area, 1)
        self._rows: list[tuple[Panel, list[_SearchableRow]]] = []
        self.no_match: QLabel | None = None

    @property
    def tool_names(self) -> list[str]:
        return [row.tool_name for _panel, rows in self._rows for row in rows if isinstance(row, ToolRow)]

    def set_info(self, info: dict) -> None:
        self._info = info
        self.mark_dirty()

    def refresh(self) -> None:
        self._build()  # built on first show (and when the configuration changes), not at startup

    def _build(self) -> None:
        clear_layout(self._column)
        self._rows = []
        integrations = Panel("Integrations", "01")
        grid = ResponsiveGrid(900)
        rows: list[_SearchableRow] = []
        for row in integration_rows(self._info, telemetry_supported=sys.platform == "win32"):
            widget = IntegrationRow(row)
            widget.action_requested.connect(self._on_action)
            grid.add(widget)
            rows.append(widget)
        integrations.body.addWidget(grid)
        self._column.addWidget(integrations)
        self._rows.append((integrations, rows))

        catalog = self._catalog
        if catalog is None:
            panel = Panel("Tools", "02")
            panel.body.addWidget(
                label("The tool list comes from the running Agent and isn't available here.", "empty", wrap=True)
            )
            self._column.addWidget(panel)
        else:
            grouped: dict[str, list[dict]] = {}
            for tool in sorted(catalog, key=lambda t: t["name"]):
                grouped.setdefault(tool_group(tool["name"]), []).append(tool)
            titles = {key: title for key, title, _patterns in TOOL_GROUPS} | {"other": "Other tools"}
            grid = ResponsiveGrid(1000)
            order = [key for key, _t, _p in TOOL_GROUPS] + ["other"]
            index = 0
            for key in order:
                tools = grouped.get(key)
                if not tools:
                    continue
                panel = Panel(f"{titles[key]} · {len(tools)}", f"{index + 2:02d}")
                tool_rows = [ToolRow(tool) for tool in tools]
                for row in tool_rows:
                    panel.body.addWidget(row)
                panel.body.addStretch(1)
                grid.add(panel)
                self._rows.append((panel, tool_rows))
                index += 1
            self._column.addWidget(grid)

        planned = Panel("Planned — not implemented yet", "")
        for title, note in PLANNED:
            row = label(f"•  {title} — {note}", "dim", wrap=True)
            planned.body.addWidget(row)
        self._column.addWidget(planned)
        self.no_match = label("No tools or integrations match.", "empty")
        self.no_match.hide()
        self._column.addWidget(self.no_match)
        self._column.addStretch(1)
        self._apply_filter(self.search.text())

    def _apply_filter(self, query: str) -> None:
        any_visible = False
        for panel, rows in self._rows:
            visible = 0
            for row in rows:
                shown = row.matches(query) if query.strip() else True
                row.setVisible(shown)
                visible += shown
            panel.setVisible(visible > 0)
            any_visible = any_visible or visible > 0
        if self.no_match is not None:
            self.no_match.setVisible(bool(query.strip()) and not any_visible)

    def _on_action(self, key: str) -> None:
        if key == "pair":
            self.pair_requested.emit()
        else:
            self.open_page.emit(key)


# --------------------------------------------------------------------------
# Tasks panel (Activity page): the TaskManager's records, live
# --------------------------------------------------------------------------
TASK_STATUS_TONE = {
    "queued": None,
    "running": "live",
    "waiting_confirmation": "warn",
    "cancelling": "live",
    "completed": "ok",
    "failed": "error",
    "cancelled": None,
    "interrupted": "error",
}
TASK_STATUS_TEXT = {
    "queued": "Queued",
    "running": "Running",
    "waiting_confirmation": "Needs approval",
    "cancelling": "Stopping",
    "completed": "Done",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "interrupted": "Interrupted",
}


class TaskRow(QFrame):
    cancel_requested = Signal(str)

    def __init__(self, task: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.task_id = task["id"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(6)
        status = task.get("status", "")
        head.addWidget(chip(TASK_STATUS_TEXT.get(status, status), TASK_STATUS_TONE.get(status)))
        kind = {"protocol": "Protocol", "chat": "Chat", "instruction": "Task"}.get(task.get("kind"), "Task")
        origin = (task.get("device") or "Phone") if task.get("origin") == "mobile" else "PC"
        head.addWidget(label(f"{kind} · {origin}", "faint"))
        head.addStretch(1)
        if task.get("cancellable"):
            self.cancel_button = QPushButton("Cancel" if status == "queued" else "Stop")
            self.cancel_button.setProperty("flat", True)
            self.cancel_button.setProperty("danger", True)
            self.cancel_button.clicked.connect(lambda: self.cancel_requested.emit(self.task_id))
            head.addWidget(self.cancel_button)
        layout.addLayout(head)
        title = label(elide(task.get("title", ""), 90))
        title.setStyleSheet(f"color: {theme.TEXT_1};")
        title.setToolTip(task.get("title", ""))
        layout.addWidget(title)
        progress = task.get("progress") or {}
        line = task.get("note") or ""
        if not line and progress.get("total") and status in ("running", "waiting_confirmation", "cancelling"):
            line = f"Step {progress.get('step', 0)} of {progress['total']}" + (
                f" — {progress['label']}" if progress.get("label") else ""
            )
        if not line:
            line = task.get("error") if status in ("failed", "interrupted") else task.get("result_preview", "")
        if line:
            detail = label(elide(line, 140), "faint", wrap=True)
            detail.setToolTip(line)
            layout.addWidget(detail)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(160, 200, 245, 16), 1))
        painter.drawLine(QPointF(0, self.height() - 0.5), QPointF(self.width(), self.height() - 0.5))
        painter.end()


class TasksPanel(Panel):
    """Phone instructions, protocol runs and PC chat turns, newest first."""

    cancel_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tasks", "", parent)
        self._tasks: dict[str, dict] = {}
        self._order: list[str] = []
        self.rows = QVBoxLayout()
        self.rows.setSpacing(0)
        self.body.addLayout(self.rows)
        self.body.addStretch(1)
        self._render()

    @property
    def tasks(self) -> list[dict]:
        return [self._tasks[i] for i in self._order]

    def set_tasks(self, tasks: list[dict]) -> None:
        self._tasks = {t["id"]: t for t in tasks if t.get("id")}
        self._order = [t["id"] for t in tasks if t.get("id")]
        self._render()

    def update_task(self, task: dict) -> None:
        if not task.get("id"):
            return
        if task["id"] not in self._tasks:
            self._order.insert(0, task["id"])
            del self._order[30:]
        self._tasks[task["id"]] = task
        self._render()

    def _render(self) -> None:
        clear_layout(self.rows)
        if not self._order:
            self.rows.addWidget(
                label("No tasks yet. Phone instructions and protocol runs appear here.", "empty", wrap=True)
            )
            return
        for task_id in self._order[:12]:
            row = TaskRow(self._tasks[task_id])
            row.cancel_requested.connect(self.cancel_requested)
            self.rows.addWidget(row)
