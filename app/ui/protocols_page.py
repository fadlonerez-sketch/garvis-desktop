"""The Protocols page: build, test, save and run reusable desktop workflows.

A protocol is edited as a visual sequence of step cards; each card's form is
generated from its action's parameter spec (app/tasks/protocols.py), so the
editor can only produce steps the runner really implements. Actions whose
tool isn't available in this configuration are listed but disabled, with
the reason. "Test" is a dry run: it validates and shows exactly what each
step would do, without doing any of it. "Run" hands the saved protocol to
the TaskManager -- the same path a phone request takes -- and the latest
run's step-by-step results appear below the editor.

File operations go through ProtocolStore (thread-safe); running and
cancelling go through the bridge (the agent loop). Dialogs that need an
answer go through `ask`/`inform`, which tests replace.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.tasks.protocols import (
    ACTIONS,
    Protocol,
    Step,
    action_available,
    blocking,
    clean_params,
    describe_step,
    validate,
)
from app.ui import fonts, theme
from app.ui.dashboard import (
    Page,
    PageHeader,
    Panel,
    StatusMark,
    ToggleSwitch,
    chip,
    clear_layout,
    label,
    link_button,
    paint_inner_card,
    scroll_column,
    set_button_icon,
)

PRESENT_CHOICES = (
    ("default", "Default for this action"),
    ("none", "Don't show me"),
    ("subtle", "Small card"),
    ("prominent", "Animation + card"),
)
FAILURE_CHOICES = (("stop", "Stop the protocol"), ("continue", "Continue with the next step"))
_STEP_ICON = {
    "app": "▣",
    "web": "◎",
    "folder": "▤",
    "window": "▢",
    "wait": "◷",
    "notify": "◈",
    "confirm": "?",
    "ai": "✦",
    "step": "•",
}
_STEP_SYMBOL = {
    "app": "apps",
    "web": "language",
    "folder": "folder_open",
    "window": "select_window",
    "wait": "hourglass_empty",
    "notify": "notifications",
    "confirm": "help",
    "ai": "auto_awesome",
    "step": "radio_button_checked",
}
"""Material Symbols for the step kinds (the characters above are the fallback)."""


def _button(text: str, tooltip: str = "", flat: bool = True) -> QPushButton:
    button = QPushButton(text)
    if flat:
        button.setProperty("flat", True)
    if tooltip:
        button.setToolTip(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class StepCard(QFrame):
    """One step: number, action, live summary, controls and its form."""

    changed = Signal()
    move_requested = Signal(int)  # -1 up, +1 down
    duplicate_requested = Signal()
    remove_requested = Signal()

    def __init__(self, step: Step, index: int, available: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.step = step
        self.index = index
        self._available = available
        self._fields: dict[str, Callable[[], object]] = {}
        action = ACTIONS.get(step.type)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 10, 12, 10)
        outer.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.number = label(f"{index + 1:02d}", "metricSmall")
        self.number.setStyleSheet(f"color: {theme.ACCENT_TEXT};")
        head.addWidget(self.number)
        kind = action.icon if action else "step"
        symbol = _STEP_SYMBOL.get(kind, "")
        icon = label()
        if fonts.icon_char(symbol) and fonts.icons_available():
            icon.setText(fonts.icon_char(symbol))
            icon.setStyleSheet(f"color: {theme.ACCENT_TEXT}; font-family: '{fonts.ICONS}'; font-size: 20px;")
        else:
            icon.setText(_STEP_ICON.get(kind, "•"))
            icon.setStyleSheet(f"color: {theme.ACCENT_TEXT}; font-size: 12pt;")
        head.addWidget(icon)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        self.kind_label = label(action.label if action else f"Unknown action '{step.type}'", "key")
        title_box.addWidget(self.kind_label)
        self.summary = label(describe_step(step))
        self.summary.setStyleSheet(f"color: {theme.TEXT_1}; font-weight: 600;")
        title_box.addWidget(self.summary)
        head.addLayout(title_box, 1)
        if not available and action is not None:
            unavailable = chip("Unavailable", "warn")
            unavailable.setToolTip(action.unavailable_reason)
            head.addWidget(unavailable)
        self.enabled_box = ToggleSwitch("On")
        self.enabled_box.setToolTip("Off: the step is kept but skipped")
        self.enabled_box.setChecked(step.enabled)
        self.enabled_box.toggled.connect(self._on_change)
        head.addWidget(self.enabled_box)
        for text, tip, slot in (
            ("▲", "Move up", lambda: self.move_requested.emit(-1)),
            ("▼", "Move down", lambda: self.move_requested.emit(1)),
            ("Copy", "Duplicate this step", self.duplicate_requested.emit),
            ("×", "Remove this step", self.remove_requested.emit),
        ):
            button = _button(text, tip)
            button.setFixedWidth(46 if len(text) > 1 else 30)
            button.clicked.connect(slot)
            head.addWidget(button)
        outer.addLayout(head)

        if action is not None:
            form = QVBoxLayout()
            form.setContentsMargins(44, 0, 0, 0)
            form.setSpacing(6)
            for param in action.params:
                form.addLayout(self._field(param))
            options = QHBoxLayout()
            options.setSpacing(8)
            options.addWidget(label("Show me", "key"))
            self.present_combo = QComboBox()
            for key, text in PRESENT_CHOICES:
                self.present_combo.addItem(text, key)
            self.present_combo.setCurrentIndex(max(0, self.present_combo.findData(step.present)))
            self.present_combo.currentIndexChanged.connect(self._on_change)
            options.addWidget(self.present_combo)
            options.addSpacing(10)
            options.addWidget(label("If it fails", "key"))
            self.failure_combo = QComboBox()
            for key, text in FAILURE_CHOICES:
                self.failure_combo.addItem(text, key)
            self.failure_combo.setCurrentIndex(max(0, self.failure_combo.findData(step.on_failure)))
            self.failure_combo.currentIndexChanged.connect(self._on_change)
            options.addWidget(self.failure_combo)
            options.addStretch(1)
            form.addLayout(options)
            self.problem = label("", "faint", wrap=True)
            self.problem.setStyleSheet(f"color: {theme.WARN};")
            self.problem.hide()
            form.addWidget(self.problem)
            outer.addLayout(form)

    def _field(self, param) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        caption = label(param.label, "key")
        caption.setFixedWidth(130)
        row.addWidget(caption, 0, Qt.AlignmentFlag.AlignTop)
        value = self.step.params.get(param.name, param.default)
        if param.kind in ("longtext", "urls"):
            widget = QPlainTextEdit()
            text = "\n".join(value) if isinstance(value, list) else str(value or "")
            widget.setPlainText(text)
            widget.setFixedHeight(64 if param.kind == "longtext" else 86)
            widget.textChanged.connect(self._on_change)
            getter = (
                (lambda w=widget: [ln.strip() for ln in w.toPlainText().splitlines() if ln.strip()])
                if param.kind == "urls"
                else (lambda w=widget: w.toPlainText())
            )
        elif param.kind == "seconds":
            widget = QDoubleSpinBox()
            widget.setRange(param.minimum or 0, param.maximum or 3600)
            widget.setDecimals(0 if (param.minimum or 0) >= 1 else 1)
            widget.setSuffix(" s")
            try:
                widget.setValue(float(value if value not in (None, "") else param.default or 0))
            except TypeError, ValueError:
                widget.setValue(float(param.default or 0))
            widget.valueChanged.connect(self._on_change)
            getter = widget.value
        elif param.kind == "choice":
            widget = QComboBox()
            for key, text in param.choices:
                widget.addItem(text, key)
            widget.setCurrentIndex(max(0, widget.findData(value)))
            widget.currentIndexChanged.connect(self._on_change)
            getter = widget.currentData
        else:
            widget = QLineEdit(str(value or ""))
            widget.setPlaceholderText(
                {
                    "url": "https://…",
                    "path": "C:\\Users\\…",
                    "app": "e.g. Visual Studio Code",
                    "window": "e.g. chrome",
                }.get(param.kind, "")
            )
            widget.textChanged.connect(self._on_change)
            getter = widget.text
        if param.help:
            widget.setToolTip(param.help)
        box = QVBoxLayout()
        box.setSpacing(2)
        box.addWidget(widget)
        if param.help and param.kind in ("urls", "path", "app"):
            box.addWidget(label(param.help, "faint", wrap=True))
        row.addLayout(box, 1)
        if param.kind == "path":
            browse = _button("Browse…", "Pick a folder", flat=False)
            browse.clicked.connect(lambda: self._browse(widget))
            row.addWidget(browse, 0, Qt.AlignmentFlag.AlignTop)
        self._fields[param.name] = getter
        return row

    def _browse(self, widget: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder", widget.text() or "")
        if path:
            widget.setText(path.replace("/", "\\"))

    def _on_change(self, *_args) -> None:
        params = {name: getter() for name, getter in self._fields.items()}
        update = {"params": params, "enabled": self.enabled_box.isChecked()}
        if hasattr(self, "present_combo"):
            update.update(present=self.present_combo.currentData(), on_failure=self.failure_combo.currentData())
        self.step = self.step.model_copy(update=update)
        self.summary.setText(describe_step(self.step))
        self.changed.emit()

    def set_problem(self, text: str) -> None:
        if hasattr(self, "problem"):
            self.problem.setText(text)
            self.problem.setVisible(bool(text))

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        accent = QColor(theme.ELECTRIC_HI if self.step.enabled and self._available else theme.TEXT_FAINT)
        paint_inner_card(self, accent, dim=not self.step.enabled)


class _FlowRail(QWidget):
    """The connector drawn between step cards: the visual sequence."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(16)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x = 30.0
        painter.setPen(QPen(QColor(58, 146, 230, 110), 1.4))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.setBrush(QColor(theme.ELECTRIC_HI))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(x, self.height() / 2), 2.2, 2.2)
        painter.end()


class ProtocolsPage(Page):
    run_requested = Signal(str)
    cancel_requested = Signal(str)
    open_activity = Signal()

    def __init__(self, store, available_tools: set[str] | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._tools = available_tools
        self._current: Protocol | None = None
        self._saved: Protocol | None = None
        self._cards: list[StepCard] = []
        self._latest_task: dict | None = None
        self.ask: Callable[[str, str], bool] = self._ask
        self.inform: Callable[[str, str], None] = self._inform
        self.choose_export_path: Callable[[str], str] = self._choose_export_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.header = PageHeader(
            "Protocols",
            "Reusable desktop workflows. Each step is an action Garvis really "
            "supports, run with the same permissions and confirmations as a chat request.",
        )
        self.new_button = QPushButton("New protocol")
        self.new_button.setProperty("accent", True)
        set_button_icon(self.new_button, "add", "#FFFFFF", 18)
        self.new_button.clicked.connect(self.new_protocol)
        self.header.actions.addWidget(self.new_button)
        layout.addWidget(self.header)

        row = QHBoxLayout()
        row.setSpacing(14)
        self.library_panel = Panel("Library", "")
        self.library_panel.setFixedWidth(280)
        self.library = QListWidget()
        self.library.currentItemChanged.connect(self._on_select)
        self.library_panel.body.addWidget(self.library, 1)
        self.library_note = label("", "faint", wrap=True)
        self.library_panel.body.addWidget(self.library_note)
        row.addWidget(self.library_panel)

        editor_column = QVBoxLayout()
        editor_column.setSpacing(10)
        area, self._column = scroll_column(self)
        editor_column.addWidget(area, 1)
        self.action_bar = QHBoxLayout()
        self.action_bar.setSpacing(8)
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("accent", True)
        self.revert_button = QPushButton("Revert")
        self.test_button = QPushButton("Test (dry run)")
        self.run_button = QPushButton("Run")
        self.run_button.setProperty("accent", True)
        self.duplicate_button = QPushButton("Duplicate")
        self.export_button = QPushButton("Export…")
        self.delete_button = QPushButton("Delete")
        self.delete_button.setProperty("danger", True)
        self.save_button.clicked.connect(self.save)
        self.revert_button.clicked.connect(self.revert)
        self.test_button.clicked.connect(self.test)
        self.run_button.clicked.connect(self.run_or_stop)
        self.duplicate_button.clicked.connect(self.duplicate)
        self.export_button.clicked.connect(self.export)
        self.delete_button.clicked.connect(self.delete)
        for button in (self.save_button, self.revert_button):
            self.action_bar.addWidget(button)
        self.action_bar.addSpacing(16)
        for button in (self.test_button, self.run_button):
            self.action_bar.addWidget(button)
        self.action_bar.addStretch(1)
        for button in (self.duplicate_button, self.export_button, self.delete_button):
            self.action_bar.addWidget(button)
        editor_column.addLayout(self.action_bar)
        row.addLayout(editor_column, 1)
        layout.addLayout(row, 1)

        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(250)
        self._validate_timer.timeout.connect(self._render_problems)
        self._build_editor()
        self.reload()

    # ------------------------------------------------------------ library
    @property
    def current(self) -> Protocol | None:
        return self._current

    @property
    def dirty(self) -> bool:
        if self._current is None:
            return False
        if self._saved is None:
            return True
        keys = ("name", "description", "enabled", "steps")
        return self._current.model_dump(include=set(keys)) != self._saved.model_dump(include=set(keys))

    def reload(self, select: str | None = None) -> None:
        entries = self._store.load_all()
        select = select or (self._current.id if self._current else None)
        self.library.blockSignals(True)
        self.library.clear()
        invalid = 0
        chosen = None
        for entry in entries:
            if entry.protocol is None:
                invalid += 1
                item = QListWidgetItem(f"⚠  {entry.path.name}\n{entry.error[:80]}")
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setToolTip(entry.error)
                self.library.addItem(item)
                continue
            protocol = entry.protocol
            problems = blocking(validate(protocol, self._tools))
            state = "Disabled" if not protocol.enabled else ("Needs attention" if problems else "Ready")
            enabled_steps = sum(1 for s in protocol.steps if s.enabled)
            item = QListWidgetItem(
                f"{protocol.name}\n{enabled_steps} step{'s' if enabled_steps != 1 else ''} · {state}"
            )
            item.setData(Qt.ItemDataRole.UserRole, protocol.id)
            item.setToolTip(protocol.description or protocol.name)
            self.library.addItem(item)
            if protocol.id == select:
                chosen = item
        self.library.blockSignals(False)
        self.library_note.setText(
            (f"{invalid} file{'s' if invalid != 1 else ''} couldn't be loaded and won't run." if invalid else "")
            or ("No protocols yet — create one with New protocol." if not entries else "")
        )
        if chosen is not None:
            self.library.setCurrentItem(chosen)
        elif self._current is None:
            first = next(
                (
                    self.library.item(i)
                    for i in range(self.library.count())
                    if self.library.item(i).data(Qt.ItemDataRole.UserRole)
                ),
                None,
            )
            if first is not None:
                self.library.setCurrentItem(first)
            else:
                self._load(None)

    def _on_select(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        protocol_id = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        if self._current is not None and protocol_id == self._current.id:
            return
        if self.dirty and not self.ask("Unsaved changes", f"Discard your changes to “{self._current.name}”?"):
            self.library.blockSignals(True)
            self.library.setCurrentItem(previous)
            self.library.blockSignals(False)
            return
        self._load(self._store.get(protocol_id) if protocol_id else None)

    def new_protocol(self) -> None:
        if self.dirty and not self.ask("Unsaved changes", f"Discard your changes to “{self._current.name}”?"):
            return
        self._load(Protocol(name="New protocol"), saved=False)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    # ------------------------------------------------------------- editor
    def _build_editor(self) -> None:
        self.details_panel = Panel("Protocol", "01")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name, e.g. Programming workspace")
        self.name_edit.setMaxLength(80)
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("What it's for (shown on the phone too)")
        self.description_edit.setMaxLength(500)
        self.enabled_box = ToggleSwitch("Enabled — can be run from the PC and from your phone")
        for widget in (self.name_edit, self.description_edit):
            widget.textChanged.connect(self._on_details_changed)
        self.enabled_box.toggled.connect(self._on_details_changed)
        for widget in (self.name_edit, self.description_edit, self.enabled_box):
            self.details_panel.body.addWidget(widget)
        self._column.addWidget(self.details_panel)

        self.steps_panel = Panel("Steps", "02")
        self.add_button = QPushButton("Add step")
        set_button_icon(self.add_button, "add", theme.TEXT_1, 18)

        self.add_button.clicked.connect(self._show_add_menu)
        self.steps_panel.add_header_widget(self.add_button)
        self.steps_body = QVBoxLayout()
        self.steps_body.setSpacing(0)
        self.steps_panel.body.addLayout(self.steps_body)
        self._column.addWidget(self.steps_panel)

        self.problems_panel = Panel("Checks", "03")
        self.problems_body = QVBoxLayout()
        self.problems_body.setSpacing(4)
        self.problems_panel.body.addLayout(self.problems_body)
        self._column.addWidget(self.problems_panel)

        self.run_panel = Panel("Latest run", "04")
        self.run_body = QVBoxLayout()
        self.run_body.setSpacing(4)
        self.run_panel.body.addLayout(self.run_body)
        activity = link_button("All tasks in Activity")
        activity.clicked.connect(self.open_activity)
        self.run_panel.add_header_widget(activity)
        self._column.addWidget(self.run_panel)
        self.empty_label = label("Select a protocol on the left, or create one.", "empty", wrap=True)
        self._column.addWidget(self.empty_label)
        self._column.addStretch(1)

    def _load(self, protocol: Protocol | None, saved: bool = True) -> None:
        self._current = copy.deepcopy(protocol) if protocol is not None else None
        self._saved = copy.deepcopy(protocol) if protocol is not None and saved else None
        has = protocol is not None
        for widget in (self.details_panel, self.steps_panel, self.problems_panel, self.run_panel):
            widget.setVisible(has)
        self.empty_label.setVisible(not has)
        for widget in (self.name_edit, self.description_edit, self.enabled_box):
            widget.blockSignals(True)
        self.name_edit.setText(protocol.name if has else "")
        self.description_edit.setText(protocol.description if has else "")
        self.enabled_box.setChecked(protocol.enabled if has else True)
        for widget in (self.name_edit, self.description_edit, self.enabled_box):
            widget.blockSignals(False)
        self._latest_task = None
        self._rebuild_steps()
        self._render_problems()
        self._render_run()
        self._update_buttons()

    def _rebuild_steps(self) -> None:
        clear_layout(self.steps_body)
        self._cards = []
        steps = self._current.steps if self._current is not None else []
        if not steps:
            self.steps_body.addWidget(label("No steps yet. Add the first one with “Add step”.", "empty"))
        for index, step in enumerate(steps):
            if index:
                self.steps_body.addWidget(_FlowRail())
            action = ACTIONS.get(step.type)
            card = StepCard(step, index, action is not None and action_available(action, self._tools))
            card.changed.connect(lambda i=index, c=card: self._on_step_changed(i, c))
            card.move_requested.connect(lambda delta, i=index: self.move_step(i, delta))
            card.duplicate_requested.connect(lambda i=index: self.duplicate_step(i))
            card.remove_requested.connect(lambda i=index: self.remove_step(i))
            self.steps_body.addWidget(card)
            self._cards.append(card)

    def _show_add_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(theme.menu_stylesheet())
        menu.setToolTipsVisible(True)
        for action in ACTIONS.values():
            entry = menu.addAction(f"{_STEP_ICON.get(action.icon, '•')}  {action.label}")
            available = action_available(action, self._tools)
            entry.setEnabled(available)
            entry.setToolTip(action.description if available else action.unavailable_reason)
            entry.triggered.connect(lambda _checked=False, key=action.key: self.add_step(key))
        menu.exec(self.add_button.mapToGlobal(self.add_button.rect().bottomLeft()))

    def add_step(self, action_key: str) -> None:
        if self._current is None or action_key not in ACTIONS:
            return
        action = ACTIONS[action_key]
        params = {p.name: p.default for p in action.params if p.default is not None}
        self._current.steps.append(Step(type=action_key, params=params))
        self._steps_changed()

    def move_step(self, index: int, delta: int) -> None:
        steps = self._current.steps
        target = index + delta
        if 0 <= target < len(steps):
            steps[index], steps[target] = steps[target], steps[index]
            self._steps_changed()

    def duplicate_step(self, index: int) -> None:
        steps = self._current.steps
        steps.insert(index + 1, steps[index].model_copy(deep=True, update={"id": Step(type="wait").id}))
        self._steps_changed()

    def remove_step(self, index: int) -> None:
        del self._current.steps[index]
        self._steps_changed()

    def _steps_changed(self) -> None:
        self._rebuild_steps()
        self._validate_timer.start()
        self._update_buttons()

    def _on_step_changed(self, index: int, card: StepCard) -> None:
        if self._current is not None and index < len(self._current.steps):
            self._current.steps[index] = card.step
            self._validate_timer.start()
            self._update_buttons()

    def _on_details_changed(self, *_args) -> None:
        if self._current is None:
            return
        self._current = self._current.model_copy(
            update={
                "name": self.name_edit.text(),
                "description": self.description_edit.text(),
                "enabled": self.enabled_box.isChecked(),
            }
        )
        self._validate_timer.start()
        self._update_buttons()

    def _render_problems(self) -> None:
        clear_layout(self.problems_body)
        if self._current is None:
            return
        problems = validate(self._current, self._tools)
        by_step: dict[int, list[str]] = {}
        for problem in problems:
            if problem.step is not None:
                by_step.setdefault(problem.step, []).append(problem.message)
        for card in self._cards:
            card.set_problem("  ·  ".join(by_step.get(card.index, [])))
        if not problems:
            row = QHBoxLayout()
            row.addWidget(StatusMark(True))
            row.addWidget(label("Ready to run. Every step uses an available action with valid settings."), 1)
            self.problems_body.addLayout(row)
            return
        for problem in problems:
            row = QHBoxLayout()
            row.addWidget(StatusMark(False if problem.blocking else "warn"), 0, Qt.AlignmentFlag.AlignTop)
            text = label(problem.text(), wrap=True)
            text.setStyleSheet(f"color: {theme.ERROR if problem.blocking else theme.WARN};")
            row.addWidget(text, 1)
            self.problems_body.addLayout(row)

    def _update_buttons(self) -> None:
        has = self._current is not None
        dirty = self.dirty
        running = self._running_task() is not None
        self.save_button.setEnabled(has and dirty)
        self.revert_button.setEnabled(has and dirty and self._saved is not None)
        self.test_button.setEnabled(has)
        self.run_button.setEnabled(has and (running or bool(self._current and self._current.steps)))
        self.run_button.setText("Stop" if running else "Run")
        for button in (self.duplicate_button, self.export_button, self.delete_button):
            button.setEnabled(has and self._saved is not None)

    # ------------------------------------------------------------ actions
    def save(self) -> bool:
        if self._current is None:
            return False
        if not self._current.name.strip():
            self.inform("Name needed", "Give the protocol a name before saving.")
            return False
        try:
            saved = self._store.save(self._current)
        except OSError as exc:
            self.inform("Couldn't save", f"The protocol couldn't be saved: {exc}")
            return False
        self._current = copy.deepcopy(saved)
        self._saved = copy.deepcopy(saved)
        self.reload(select=saved.id)
        self._update_buttons()
        return True

    def revert(self) -> None:
        if self._saved is not None:
            self._load(self._saved)

    def test(self) -> None:
        """A dry run: what each step would do, and what would stop it -- nothing is executed."""
        if self._current is None:
            return
        problems = validate(self._current, self._tools)
        lines = [f"Dry run of “{self._current.name}” — nothing was opened or changed.", ""]
        number = 0
        for index, step in enumerate(self._current.steps):
            if not step.enabled:
                lines.append(f"  –  (off) {describe_step(step)}")
                continue
            number += 1
            try:
                clean_params(step)
                status = "would run"
            except (ValueError, KeyError) as exc:
                status = f"can't run: {exc}"
            for problem in problems:
                if problem.step == index and problem.blocking:
                    status = f"can't run: {problem.message}"
            show = {"none": "", "subtle": " · then a small card", "prominent": " · then the attention animation"}
            lines.append(
                f"  {number}. {describe_step(step)} — {status}{show.get(step.presentation(), '')}"
                + ("" if step.on_failure == "stop" else " · continues if it fails")
            )
        stopping = blocking(problems)
        lines += ["", "Ready to run." if not stopping else f"It can't run yet: {stopping[0].text()}"]
        self.last_test_report = "\n".join(lines)
        self.inform("Test (dry run)", self.last_test_report)

    def run_or_stop(self) -> None:
        running = self._running_task()
        if running is not None:
            self.cancel_requested.emit(running["id"])
            return
        if self._current is None:
            return
        if self.dirty:
            if not self.ask("Save first?", "The protocol has unsaved changes. Save them and run?"):
                return
            if not self.save():
                return
        stopping = blocking(validate(self._current, self._tools))
        if stopping:
            self.inform("Can't run yet", stopping[0].text())
            return
        self.run_requested.emit(self._current.id)

    def duplicate(self) -> None:
        if self._saved is None:
            return
        copy_ = self._store.duplicate(self._saved.id)
        if copy_ is not None:
            self._load(copy_)
            self.reload(select=copy_.id)

    def export(self) -> None:
        if self._saved is None:
            return
        path = self.choose_export_path(f"{self._saved.name}.json")
        if path:
            try:
                self._store.export(self._saved, path)
            except OSError as exc:
                self.inform("Couldn't export", str(exc))

    def delete(self) -> None:
        if self._saved is None:
            return
        if not self.ask("Delete protocol", f"Delete “{self._saved.name}”? This can't be undone."):
            return
        self._store.delete(self._saved.id)
        self._current = None
        self._saved = None
        self.reload()
        if self._current is None:
            self._load(None)

    # --------------------------------------------------------------- runs
    def on_task_updated(self, task: dict) -> None:
        if self._current is not None and task.get("protocol_id") == self._current.id:
            self._latest_task = task
            self._render_run()
            self._update_buttons()

    def set_task_detail(self, detail: dict | None) -> None:
        if detail and self._current is not None and detail.get("protocol_id") == self._current.id:
            self._latest_task = detail
            self._render_run()

    def _running_task(self) -> dict | None:
        task = self._latest_task
        if task and task.get("status") in ("queued", "running", "waiting_confirmation", "cancelling"):
            return task
        return None

    def _render_run(self) -> None:
        clear_layout(self.run_body)
        task = self._latest_task
        if task is None:
            self.run_body.addWidget(
                label("Not run since Garvis started. Runs from the phone appear here too.", "empty", wrap=True)
            )
            return
        head = QHBoxLayout()
        status = task.get("status", "")
        tone = {"completed": "ok", "failed": "error", "interrupted": "error", "cancelled": None}.get(status, "live")
        head.addWidget(chip(status.replace("_", " "), tone))
        head.addWidget(chip((task.get("device") or "Phone") if task.get("origin") == "mobile" else "PC"))
        progress = task.get("progress") or {}
        if progress.get("total"):
            head.addWidget(label(f"Step {progress.get('step', 0)} of {progress['total']}", "value"))
        head.addStretch(1)
        self.run_body.addLayout(head)
        if task.get("note"):
            self.run_body.addWidget(label(task["note"], "faint", wrap=True))
        for step in task.get("steps") or []:
            row = QHBoxLayout()
            mark = {"ok": True, "failed": False, "timeout": False, "running": None}.get(step.get("status"))
            if step.get("status") in ("ok", "failed", "timeout", "running"):
                row.addWidget(StatusMark(mark), 0, Qt.AlignmentFlag.AlignTop)
            else:
                row.addWidget(label("–", "faint"), 0, Qt.AlignmentFlag.AlignTop)
            text = label(step.get("label", ""), wrap=True)
            if step.get("status") in ("skipped", "pending", "cancelled"):
                text.setStyleSheet(f"color: {theme.TEXT_FAINT};")
            row.addWidget(text, 1)
            if step.get("message"):
                message = label(step["message"][:160], "faint", wrap=True)
                message.setMaximumWidth(380)
                row.addWidget(message, 1)
            self.run_body.addLayout(row)
        summary = task.get("result_preview") or task.get("result") or task.get("error")
        if summary and status in ("completed", "failed", "cancelled", "interrupted"):
            self.run_body.addWidget(label(summary, "dim", wrap=True))

    # --------------------------------------------------------------- dialogs
    def _ask(self, title: str, text: str) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _inform(self, title: str, text: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.exec()

    def _choose_export_path(self, suggested: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, "Export protocol", suggested, "Protocol (*.json)")
        return path

    def refresh(self) -> None:
        self._update_buttons()
