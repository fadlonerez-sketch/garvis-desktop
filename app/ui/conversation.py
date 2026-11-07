"""The Overview's conversation, drawn as chat bubbles (approved mockup).

Still a read-only QTextEdit -- MainWindow appends to it exactly as before and
its plain text stays the transcript -- but each message is a one-cell table:
your messages right-aligned on Electric Blue, Garvis's left-aligned on a
raised surface, tool results as small chips. Qt's rich text can't round a
cell, so the rounded bubbles are painted behind the text from the document
layout (`paintEvent`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextFrameFormat,
    QTextTableCellFormat,
    QTextTableFormat,
)
from PySide6.QtWidgets import QTextEdit, QWidget

from app.ui import fonts, theme

USER = "user"
GARVIS = "garvis"
TOOL = "tool"


@dataclass
class _Bubble:
    table: object  # QTextTable
    kind: str


def _speaker_kind(speaker: str) -> str:
    return USER if speaker.startswith("You") else GARVIS


class ConversationView(QTextEdit):
    """Read-only transcript with bubble styling. `add_message(speaker, html)`
    replaces QTextEdit.append for the conversation; the speaker is shown in
    the bubble when it says more than its side does ("You (iPhone)")."""

    SIDE_MARGIN = 34

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("ConversationView")
        self.setFrameShape(QTextEdit.Shape.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(2)
        root = self.document().rootFrame().frameFormat()
        root.setPadding(0)
        self.document().rootFrame().setFrameFormat(root)
        self._bubbles: list[_Bubble] = []
        self._last_day: date | None = None

    # ----------------------------------------------------------- content
    def add_message(self, speaker: str, html_text: str) -> None:
        kind = _speaker_kind(speaker)
        self._maybe_day_marker()
        label = ""
        if speaker not in ("You", "Garvis"):
            color = "rgba(255,255,255,0.78)" if kind == USER else theme.TEXT_2
            label = f'<span style="color:{color}; font-weight:600;">{speaker}:</span> '
        text_color = "#FFFFFF" if kind == USER else theme.TEXT_1
        self._insert_bubble(kind, f'<span style="color:{text_color};">{label}{html_text}</span>')

    def add_tool(self, name: str, success: bool, detail: str = "") -> None:
        """A small chip for a finished tool step (the full log is on Activity)."""
        import html

        icon = fonts.icon_html("check_circle" if success else "error", theme.OK if success else theme.ERROR, 15)
        mark = icon or ("✓" if success else "✕")
        text = html.escape(name) + (f" · {html.escape(detail)}" if detail else "")
        self._insert_bubble(
            TOOL, f'<span style="color:{theme.TEXT_2}; font-size:12px;">{mark}&nbsp;&nbsp;{text}</span>'
        )

    def _maybe_day_marker(self) -> None:
        today = date.today()
        if self._last_day == today:
            return
        first = self._last_day is None
        self._last_day = today
        cursor = self._end_cursor()
        block = QTextBlockFormat()
        block.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        block.setTopMargin(0 if first else 8)
        block.setBottomMargin(2)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(theme.TEXT_3))
        fmt.setFontPointSize(8.5)
        if not self.document().isEmpty():
            cursor.insertBlock(block, fmt)
        else:
            cursor.setBlockFormat(block)
            cursor.setCharFormat(fmt)
        cursor.insertText("Today" if first else today.strftime("%A %d %B"), fmt)

    def _end_cursor(self) -> QTextCursor:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        return cursor

    def _insert_bubble(self, kind: str, html_text: str) -> None:
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        cursor = self._end_cursor()
        fmt = QTextTableFormat()
        fmt.setBorder(0)
        fmt.setBorderStyle(QTextFrameFormat.BorderStyle.BorderStyle_None)
        fmt.setCellSpacing(0)
        fmt.setCellPadding(0)
        fmt.setTopMargin(4 if kind == TOOL else 5)
        fmt.setBottomMargin(4 if kind == TOOL else 5)
        if kind == USER:
            fmt.setAlignment(Qt.AlignmentFlag.AlignRight)
            fmt.setLeftMargin(self.SIDE_MARGIN)
        else:
            fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
            fmt.setRightMargin(self.SIDE_MARGIN)
        table = cursor.insertTable(1, 1, fmt)
        cell = table.cellAt(0, 0)
        cell_format = QTextTableCellFormat()
        vertical, horizontal = self.padding(kind)
        cell_format.setTopPadding(vertical)
        cell_format.setBottomPadding(vertical)
        cell_format.setLeftPadding(horizontal)
        cell_format.setRightPadding(horizontal)
        cell.setFormat(cell_format)
        cell.firstCursorPosition().insertHtml(html_text)
        self._bubbles.append(_Bubble(table, kind))
        if at_bottom:
            bar.setValue(bar.maximum())

    def append(self, text: str) -> None:  # noqa: D401 (Qt override)
        """Plain QTextEdit.append still works (shown as a Garvis bubble)."""
        self.add_message("Garvis", text)

    @property
    def bubble_count(self) -> int:
        return len(self._bubbles)

    # ---------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        layout = self.document().documentLayout()
        dx, dy = -self.horizontalScrollBar().value(), -self.verticalScrollBar().value()
        visible = QRectF(event.rect())
        for bubble in self._bubbles:
            rect = self.bubble_rect(bubble.table, bubble.kind, layout).translated(dx, dy)
            if not rect.isValid() or not rect.intersects(visible):
                continue
            self._paint_bubble(painter, rect.adjusted(0.5, 0.5, -0.5, -0.5), bubble.kind)
        painter.end()
        super().paintEvent(event)

    @staticmethod
    def padding(kind: str) -> tuple[int, int]:
        """(vertical, horizontal) cell padding of a bubble."""
        return (6, 10) if kind == TOOL else (9, 12)

    def bubble_rect(self, table, kind: str, layout=None) -> QRectF:
        """The bubble around a message: its text blocks plus the cell padding
        (document coordinates)."""
        layout = layout or self.document().documentLayout()
        cell = table.cellAt(0, 0)
        block, last = cell.firstCursorPosition().block(), cell.lastCursorPosition().block()
        rect = QRectF()
        while block.isValid():
            box = layout.blockBoundingRect(block)
            rect = box if rect.isNull() else rect.united(box)
            if block == last:
                break
            block = block.next()
        vertical, horizontal = self.padding(kind)
        return rect.adjusted(-horizontal, -vertical, horizontal, vertical) if not rect.isNull() else rect

    @staticmethod
    def _paint_bubble(painter: QPainter, rect: QRectF, kind: str) -> None:
        radius, tail = (10.0, 10.0) if kind == TOOL else (14.0, 5.0)
        path = _rounded(rect, radius, tail_right=kind == USER, tail_left=kind == GARVIS, tail=tail)
        if kind == USER:
            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0.0, QColor(theme.ELECTRIC))
            gradient.setColorAt(1.0, QColor(theme.ELECTRIC_GRADIENT_END))
            painter.fillPath(path, gradient)
            return
        painter.fillPath(path, QColor(theme.SURFACE_2) if kind == GARVIS else QColor(16, 43, 76, 128))
        painter.setPen(QPen(QColor(160, 200, 245, 22), 1))
        painter.drawPath(path)


def _rounded(rect: QRectF, radius: float, tail_right: bool, tail_left: bool, tail: float) -> QPainterPath:
    """A rounded rect whose bottom corner on the speaker's side is tighter."""
    r = min(radius, rect.height() / 2, rect.width() / 2)
    bl = min(tail, r) if tail_left else r
    br = min(tail, r) if tail_right else r
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    path = QPainterPath(QPointF(x0 + r, y0))
    path.lineTo(x1 - r, y0)
    path.quadTo(x1, y0, x1, y0 + r)
    path.lineTo(x1, y1 - br)
    path.quadTo(x1, y1, x1 - br, y1)
    path.lineTo(x0 + bl, y1)
    path.quadTo(x0, y1, x0, y1 - bl)
    path.lineTo(x0, y0 + r)
    path.quadTo(x0, y0, x0 + r, y0)
    path.closeSubpath()
    return path
