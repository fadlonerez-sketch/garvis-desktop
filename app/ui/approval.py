"""The approval dialog Garvis shows before a step that needs your OK.

Approved design (2026-10-31): a rounded Midnight Blue card over the dimmed,
blurred dashboard -- shield badge, title, subtitle, the step's details, then
Deny / Allow. Behaviour is the old QMessageBox's: modal, Deny is the default
(Enter and Escape both deny), and `button(QMessageBox.StandardButton.Yes /
.No)` returns Allow / Deny, so callers and tests drive it the same way.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import fonts, theme

CARD_WIDTH = 460
_SHADOW = 36
_VALUE_LIMIT = 160


def _humanize(name: str) -> str:
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _short(value) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= _VALUE_LIMIT else text[: _VALUE_LIMIT - 1].rstrip() + "…"


def detail_rows(tool_name: str, arguments: dict, reason: str) -> list[tuple[str, str, str]]:
    """(key, shown value, full value) for every fact the old dialog showed:
    the tool, each argument and the reason it asks."""
    rows = [("Tool", tool_name, tool_name)]
    if isinstance(arguments, dict) and arguments:
        for key, value in arguments.items():
            rows.append((_humanize(str(key)), _short(value), str(value)))
    elif arguments:
        rows.append(("Arguments", _short(arguments), str(arguments)))
    else:
        rows.append(("Arguments", "None", ""))
    if reason:
        rows.append(("Why", _short(reason), reason))
    return rows


class _Badge(QWidget):
    """The amber shield badge."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(242, 182, 82, 36))
        painter.drawRoundedRect(QRectF(self.rect()), 14, 14)
        if not fonts.draw_icon(painter, QRectF(self.rect()), "shield", theme.WARN_TEXT, 24):
            painter.setPen(QPen(QColor(theme.WARN_TEXT), 1.8))
            c = QRectF(self.rect()).center()
            path = QPainterPath(QPointF(c.x(), c.y() - 9))
            path.lineTo(c.x() + 8, c.y() - 6)
            path.quadTo(c.x() + 8, c.y() + 6, c.x(), c.y() + 10)
            path.quadTo(c.x() - 8, c.y() + 6, c.x() - 8, c.y() - 6)
            path.closeSubpath()
            painter.drawPath(path)
        painter.end()


class _Card(QFrame):
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 22, 22)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QColor("#13305A"))
        gradient.setColorAt(1.0, QColor("#0E2442"))
        painter.fillPath(path, gradient)
        painter.setPen(QPen(QColor(160, 200, 245, 31), 1))
        painter.drawPath(path)
        painter.setClipPath(path)
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        painter.drawLine(QPointF(rect.left() + 16, rect.top() + 1), QPointF(rect.right() - 16, rect.top() + 1))
        painter.end()


class _Details(QFrame):
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        painter.fillPath(path, QColor(5, 11, 22, 115))
        painter.setPen(QPen(QColor(160, 200, 245, 20), 1))
        painter.drawPath(path)
        painter.end()


class ApprovalDialog(QDialog):
    """Modal confirmation for one tool call. `approved` after exec()."""

    def __init__(
        self,
        tool_name: str,
        arguments: dict,
        reason: str,
        parent: QWidget | None = None,
        title: str | None = None,
        subtitle: str = "Approve this step to continue the task.",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Garvis needs your confirmation")
        self.setObjectName("ApprovalDialog")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(theme.stylesheet())
        self.tool_name = tool_name
        self._backdrop: QPixmap | None = None
        self._covering = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(_SHADOW, _SHADOW, _SHADOW, _SHADOW)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.card = _Card(self)
        self.card.setFixedWidth(CARD_WIDTH)
        row.addWidget(self.card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        body = QVBoxLayout(self.card)
        body.setContentsMargins(22, 22, 22, 22)
        body.setSpacing(0)
        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(_Badge(self.card), 0, Qt.AlignmentFlag.AlignVCenter)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel(title or f"Garvis wants to run “{_humanize(tool_name)}”", self.card)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            f"color: {theme.TEXT_1}; font-family: '{theme.display_family()}'; font-size: 18px; font-weight: 700;"
        )
        self.subtitle_label = QLabel(subtitle, self.card)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 13px;")
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        head.addLayout(titles, 1)
        body.addLayout(head)
        body.addSpacing(16)

        self.details = _Details(self.card)
        grid = QGridLayout(self.details)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(7)
        grid.setColumnMinimumWidth(0, 70)
        grid.setColumnStretch(1, 1)
        self.rows = detail_rows(tool_name, arguments, reason)
        for index, (key, shown, full) in enumerate(self.rows):
            key_label = QLabel(key, self.details)
            key_label.setStyleSheet(f"color: {theme.TEXT_3}; font-size: 13px;")
            value_label = QLabel(shown, self.details)
            value_label.setWordWrap(True)
            value_label.setTextFormat(Qt.TextFormat.PlainText)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setStyleSheet(f"color: {theme.TEXT_1}; font-size: 13px;")
            if full and full != shown:
                value_label.setToolTip(full)
            grid.addWidget(key_label, index, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value_label, index, 1)
        body.addWidget(self.details)
        body.addSpacing(18)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addStretch(1)
        self.deny_button = QPushButton("Deny", self.card)
        self.allow_button = QPushButton("Allow", self.card)
        self.allow_button.setProperty("accent", True)
        for button in (self.deny_button, self.allow_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(42)
            button.setMinimumWidth(84)
            actions.addWidget(button)
        # As before: the safe answer is the default -- Enter (and Escape) deny.
        self.allow_button.setAutoDefault(False)
        self.deny_button.setDefault(True)
        self.deny_button.setFocus()
        self.allow_button.clicked.connect(self.accept)
        self.deny_button.clicked.connect(self.reject)
        body.addLayout(actions)
        self._buttons = {
            QMessageBox.StandardButton.Yes: self.allow_button,
            QMessageBox.StandardButton.No: self.deny_button,
        }
        self._place()

    # ------------------------------------------------------------- API
    def button(self, which) -> QPushButton | None:
        """Allow for QMessageBox.StandardButton.Yes, Deny for .No (QMessageBox's API)."""
        return self._buttons.get(which)

    @property
    def approved(self) -> bool:
        return int(self.result()) == 1  # QDialog.Accepted

    @property
    def covering(self) -> bool:
        """True when the dialog sits over its (visible) parent window as a scrim."""
        return self._covering

    # -------------------------------------------------------- placement
    def _place(self) -> None:
        parent = self.parentWidget()
        window = parent.window() if parent is not None else None
        hint = self.sizeHint()
        size = QSize(max(hint.width(), CARD_WIDTH + 2 * _SHADOW), hint.height())
        if window is not None and window.isVisible() and not window.isMinimized():
            # Over the dashboard: cover it with a dimmed, blurred picture of itself.
            self._covering = True
            self._backdrop = self._blurred(window)
            top_left = window.mapToGlobal(QPoint(0, 0))
            self.setGeometry(QRect(top_left, window.size()))
            return
        screen = (window.screen() if window is not None else None) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)
        anchor = window.frameGeometry() if window is not None and window.isVisible() else area
        rect = QRect(QPoint(0, 0), size)
        rect.moveCenter(anchor.center())
        self.setGeometry(rect)

    @staticmethod
    def _blurred(window: QWidget) -> QPixmap | None:
        try:
            shot = window.grab()
        except Exception:  # noqa: BLE001 -- the scrim alone is fine
            return None
        if shot.isNull():
            return None
        small = shot.scaled(
            max(1, shot.width() // 5),
            max(1, shot.height() // 5),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return small.scaled(
            shot.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._covering:
            if self._backdrop is not None:
                painter.drawPixmap(self.rect(), self._backdrop)
            painter.fillRect(self.rect(), QColor(2, 6, 12, 158))
        # Soft shadow under the card
        card = QRectF(self.card.geometry())
        for i in range(12, 0, -1):
            spread = i * 2.6
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, int(9 * (1 - i / 13))))
            painter.drawRoundedRect(card.adjusted(-spread, -spread + 10, spread, spread + 14), 22 + spread, 22 + spread)
        painter.end()
