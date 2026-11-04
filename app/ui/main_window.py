"""Garvis main window: the command-center dashboard, plus the floating orb.

MainWindow is the dashboard (a frameless window drawn in the Garvis style,
see app/ui/dashboard.py) and still the one controller of the UI: it owns
the chat/voice/session controls and their gating rules, confirmations and
the global hotkey, exactly as before. Everything it shows comes from one
GarvisState (app/ui/state.py), fed by the AgentBridge's real events.

When started by app/main.py, install_desktop_presence() adds the floating
orb, the orb <-> dashboard transition and a tray icon (app/ui/presence.py).
Without it (tests, embedding) MainWindow is a normal window.
"""
from __future__ import annotations

import concurrent.futures
import html
import sys
import time
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.config import get_settings
from app.ui import fonts, motion, theme
from app.ui.agent_bridge import AgentBridge
from app.ui.approval import ApprovalDialog
from app.ui.conversation import ConversationView
from app.ui.dashboard import (
    PAGES,
    ActivityPage,
    NavRail,
    OverviewPage,
    SettingsPage,
    SourcesPage,
    SystemPage,
    TitleBar,
    ToggleSwitch,
    ToolsPage,
    UsagePage,
    VoicePage,
    elide,
    open_url,
    repolish,
    set_button_icon,
)
from app.ui.attention import AttentionController
from app.ui.protocols_page import ProtocolsPage
from app.ui.state import MODE_LABELS, GarvisState
from app.ui.telemetry import DETAIL, OFF, SUMMARY, TelemetryMonitor
from app.ui.logo import render_logo_pixmap

_EDGE = 5
"""Frameless-window margin that resizes the window (and draws its border)."""
_USAGE_REFRESH_DELAY_MS = 1500


def _exec_menu(menu: QMenu, pos: QPoint) -> None:
    """Show a popup menu (its own function so tests can stand in for it)."""
    menu.exec(pos)


class MobileDialog(QDialog):
    """Pairing a phone: the native Garvis app on Android, or the web app on an
    iPhone. Shows where to reach this PC, which CA the iPhone must trust, a
    one-time pairing code (cancelled when the dialog closes) and the paired
    devices, each of which can be unpaired on its own."""

    APK_NAME = "Garvis-Mobile.apk"

    def __init__(self, mobile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Garvis on your phone")
        self._mobile = mobile
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setMinimumSize(640, 520)
        layout.addWidget(self._text)
        devices_row = QHBoxLayout()
        self.device_combo = QComboBox(self)
        self.device_combo.setMinimumWidth(320)
        self.unpair_device_button = QPushButton("Unpair this device", self)
        self.unpair_device_button.clicked.connect(self._unpair_selected)
        devices_row.addWidget(QLabel("Paired devices:", self))
        devices_row.addWidget(self.device_combo, 1)
        devices_row.addWidget(self.unpair_device_button)
        layout.addLayout(devices_row)
        buttons = QHBoxLayout()
        new_code_button = QPushButton("New pairing code", self)
        new_code_button.clicked.connect(self._new_code)
        unpair_button = QPushButton("Unpair all devices", self)
        unpair_button.clicked.connect(self._unpair_all)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        for button in (new_code_button, unpair_button, self.unpair_device_button):
            button.setEnabled(mobile is not None)
        for button in (new_code_button, unpair_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self._code = None
        self._notice = ""
        if mobile is not None and mobile.is_running:
            self._new_code()
        else:
            self._render()

    def _new_code(self) -> None:
        self._code = self._mobile.pairing.new_pairing_code()
        self._notice = ""
        self._render()

    def _unpair_all(self) -> None:
        count = self._mobile.pairing.revoke_all()
        self._code = None
        self._notice = f"Unpaired {count} device(s). Pair again with a new code."
        self._render()

    def _unpair_selected(self) -> None:
        device_id = self.device_combo.currentData()
        revoke = getattr(self._mobile.pairing, "revoke_device", None) if self._mobile is not None else None
        if not device_id or revoke is None:
            return
        self._notice = ("Unpaired that device. It has to pair again with a new code." if revoke(device_id)
                        else "That device was already unpaired.")
        self._render()

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        listing = getattr(self._mobile.pairing, "devices", None) if self._mobile is not None else None
        for record in listing() if listing is not None else []:
            paired = time.strftime("%d %b %Y", time.localtime(record.created_at))
            expires = time.strftime("%d %b %Y", time.localtime(record.expires_at))
            self.device_combo.addItem(f"{record.label or 'device'} -- paired {paired}, until {expires} "
                                      f"(id {record.id})", record.id)
        self.unpair_device_button.setEnabled(self.device_combo.count() > 0)

    def _render(self) -> None:
        mobile = self._mobile
        if mobile is None:
            self._text.setPlainText(
                "The phone interface (Android app and iPhone) is off.\n\nSet GARVIS_MOBILE_ENABLED=true in .env "
                "and restart Garvis. See 'Phone interface' in the README."
            )
            return
        if not mobile.is_running:
            self._text.setPlainText(
                "The phone interface is not running (still starting, or it failed -- see the "
                "conversation for the error). Close and reopen this dialog in a moment."
            )
            return
        self._refresh_devices()
        urls = mobile.urls
        remote_urls = list(getattr(mobile, "remote_urls", []) or [])
        lines = ["This PC on your home Wi-Fi:"]
        lines += [f"  {url}" for url in urls] or ["  (no LAN address found -- is Wi-Fi/Ethernet connected?)"]
        lines += ["", "Away from home (Tailscale):"]
        lines += [f"  {url}" for url in remote_urls] or [
            "  off -- install Tailscale on this PC and the phone (same account), then set",
            "  GARVIS_MOBILE_REMOTE_ACCESS=tailscale in .env and restart Garvis (see the README).",
        ]
        lines += [
            "",
            f"CA SHA-256: {mobile.ca_fingerprint}",
            "",
            "Android (Samsung Galaxy etc.) -- the Garvis app:",
            f"  1. Install {self.APK_NAME} on the phone (built into this PC's Downloads folder).",
            "     It already trusts this PC's certificate -- nothing to install in Android Settings.",
            "  2. Open Garvis, enter the home address above (and the Tailscale one, if shown)",
            "     and the pairing code below.",
            "",
            "iPhone (Safari web app) -- first time only, trust Garvis's local certificate:",
            f"  1. Open {urls[0] + 'garvis-ca.crt' if urls else '<address>/garvis-ca.crt'} and allow the profile download",
            "  2. Settings > Profile Downloaded > Install",
            "  3. Settings > General > About > Certificate Trust Settings > enable 'Garvis Local CA'",
            "  Install it as an app: in Safari tap Share > Add to Home Screen.",
            "",
        ]
        if self._code is not None:
            minutes = max(0, int((self._code.expires_at - time.time()) // 60))
            lines.append(f"Pairing code: {self._code.display}   (single use, expires in ~{minutes} min)")
        else:
            lines.append("No active pairing code.")
        lines.append(f"Paired devices: {mobile.pairing.device_count()}")
        if self._notice:
            lines += ["", self._notice]
        self._text.setPlainText("\n".join(lines))

    def done(self, result: int) -> None:  # noqa: D401 (Qt override)
        if self._mobile is not None and self._code is not None:
            self._mobile.pairing.cancel_pairing_code()
        super().done(result)


class CommandBar(QFrame):
    """The input strip under every page (approved mockup: a rounded bar).
    On a narrow window the switch drops its caption (kept as tooltip)."""

    HEIGHT = 76
    COMPACT_BELOW = 860

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CommandBar")
        self.setFixedHeight(self.HEIGHT)
        self.switch: ToggleSwitch | None = None

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self.switch is not None:
            self.switch.set_compact(self.width() < self.COMPACT_BELOW)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 20, 20)
        painter.fillPath(path, QColor(13, 30, 54, 235))
        painter.setPen(QPen(QColor(160, 200, 245, 22), 1))
        painter.drawPath(path)
        painter.setClipPath(path)
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
        painter.drawLine(QPointF(rect.left() + 20, rect.top() + 1), QPointF(rect.right() - 20, rect.top() + 1))
        painter.end()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Garvis")
        self.setObjectName("GarvisDashboard")
        fonts.ensure_loaded()  # Manrope / Inter / Material Symbols (falls back to Segoe UI if missing)
        motion.install()
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowIcon(QIcon(render_logo_pixmap(64, theme.ORANGE)))
        self.setMinimumSize(900, 560)  # fits a 1366x768 laptop at 125% scaling
        self.resize(1360, 860)
        self.setMouseTracking(True)

        self._bridge = AgentBridge()
        self._pending_future: concurrent.futures.Future | None = None
        self._session_running = False
        self._confirmation_boxes: dict[str, QMessageBox] = {}
        self._confirmations_settled_elsewhere: set[str] = set()

        self._state = GarvisState(parent=self)
        self._state.connect_bridge(self._bridge)
        self._presence = None
        """app.ui.presence.PresenceController once install_desktop_presence() ran."""
        self._maximized = False
        self._normal_geometry: QRect | None = None
        self._background: QPixmap | None = None
        self._rounded_corners_applied = False
        self._usage_timer = QTimer(self)
        self._usage_timer.setSingleShot(True)
        self._usage_timer.setInterval(_USAGE_REFRESH_DELAY_MS)
        self._usage_timer.timeout.connect(self._request_usage)

        self._build_ui()
        self._wire_signals()
        self.setStyleSheet(theme.stylesheet())
        self._apply_system_info()
        self._request_usage()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # The controls (and their names) are the same as in the plain window
        # this replaces: the gating logic below and the tests rely on them.
        self._status_label = QLabel("Idle.", self)
        self._mic_button = QPushButton("Start listening", self)
        self._mic_button.setToolTip("Click to start recording, click again to stop and send.")
        set_button_icon(self._mic_button, "mic")
        self._voice_replies_checkbox = ToggleSwitch("Speak replies", self)
        self._session_button = QPushButton("Start Session", self)
        self._session_button.setToolTip(
            "Start a hands-free conversation: Garvis keeps listening and responding "
            "automatically until you click this again."
        )
        self._session_status_label = QLabel("", self)
        self._session_status_label.setProperty("role", "dim")
        self._mobile_button = QPushButton("Pair phone…", self)
        self._mobile_button.setToolTip("Pair an iPhone or Android phone and see where to open Garvis on it.")
        self._conversation_view = ConversationView(self)
        self._activity_list = QListWidget(self)
        self._input_line = QLineEdit(self)
        self._input_line.setObjectName("CommandInput")
        self._input_line.setPlaceholderText("Ask Garvis anything…")
        self._send_button = QPushButton("Send", self)
        self._send_button.setProperty("accent", True)
        set_button_icon(self._send_button, "arrow_upward", "#FFFFFF")
        self._stop_button = QPushButton("Stop", self)
        self._stop_button.setProperty("ghost", True)
        self._stop_button.setEnabled(False)
        for button in (self._mic_button, self._send_button, self._stop_button, self._session_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        root = QVBoxLayout(self)
        root.setContentsMargins(_EDGE, _EDGE, _EDGE, _EDGE)
        root.setSpacing(0)
        self._title_bar = TitleBar(self._status_label, self)
        root.addWidget(self._title_bar)

        body = QHBoxLayout()
        body.setSpacing(0)
        self._nav = NavRail(self)
        body.addWidget(self._nav)
        content = QVBoxLayout()
        content.setContentsMargins(18, 18, 20, 18)
        content.setSpacing(16)

        info = self._system_info()
        level_source = getattr(self._bridge, "voice_level", None)
        standby_setter = getattr(self._bridge, "set_standby_enabled", None)
        self._pages = QStackedWidget(self)
        self._telemetry = TelemetryMonitor(parent=self)
        """Live CPU/memory/disk/GPU sampling; runs only while this window is
        on screen (see _update_telemetry_demand)."""
        self._overview = OverviewPage(self._state, info, level_source, self._conversation_view)
        self._overview.set_telemetry(self._telemetry)
        self._activity_page = ActivityPage(self._state, self._activity_list)
        self._system_page = SystemPage()
        self._system_page.set_monitor(self._telemetry)
        self._sources_page = SourcesPage(self._state)
        self._voice_page = VoicePage(self._state, info, level_source, standby_setter, self._voice_replies_checkbox)
        self._voice_page.set_standby_checked(bool(getattr(self._bridge, "standby_enabled", False)))
        self._usage_page = UsagePage(info)
        self._settings_page = SettingsPage(info, self._mobile_button)
        catalog = getattr(self._bridge, "tool_catalog", None)
        try:
            tools = list(catalog()) if catalog is not None else None
        except Exception:  # noqa: BLE001 -- the page says the list is unavailable instead
            tools = None
        self._tools_page = ToolsPage(info, tools)
        store = getattr(self._bridge, "protocol_store", None)
        if store is None:
            from app.tasks.protocols import ProtocolStore

            store = ProtocolStore(get_settings().data_dir / "protocols")
        self._protocols_page = ProtocolsPage(store, getattr(self._bridge, "available_tools", None))
        self._page_widgets = dict(zip(PAGES, (self._overview, self._activity_page, self._protocols_page,
                                              self._system_page, self._sources_page, self._voice_page,
                                              self._usage_page, self._tools_page, self._settings_page), strict=True))
        for page in self._page_widgets.values():
            self._pages.addWidget(page)
        content.addWidget(self._pages, 1)
        content.addWidget(self._build_command_bar())
        body.addLayout(content, 1)
        root.addLayout(body, 1)

    def _build_command_bar(self) -> QWidget:
        bar = CommandBar(self)
        bar.switch = self._voice_replies_checkbox
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)
        layout.addWidget(self._mic_button)
        layout.addWidget(self._input_line, 1)
        layout.addWidget(self._send_button)
        layout.addWidget(self._stop_button)
        divider = QFrame(bar)
        divider.setFixedSize(1, 28)
        divider.setStyleSheet("background: rgba(160, 200, 245, 0.10);")
        layout.addWidget(divider)
        layout.addWidget(self._voice_replies_checkbox)
        layout.addWidget(self._session_button)
        layout.addWidget(self._session_status_label)
        return bar

    def _wire_signals(self) -> None:
        self._send_button.clicked.connect(self._on_send_clicked)
        self._input_line.returnPressed.connect(self._on_send_clicked)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._mobile_button.clicked.connect(self._on_mobile_clicked)
        self._mic_button.clicked.connect(self._on_mic_clicked)
        self._voice_replies_checkbox.toggled.connect(self._bridge.set_voice_replies_enabled)
        self._session_button.clicked.connect(self._on_session_clicked)

        self._bridge.status_changed.connect(self._on_status_changed)
        self._bridge.assistant_message.connect(self._on_assistant_message)
        self._bridge.tool_started.connect(self._on_tool_started)
        self._bridge.tool_finished.connect(self._on_tool_finished)
        self._bridge.confirmation_requested.connect(self._on_confirmation_requested)
        self._bridge.confirmation_resolved.connect(self._on_confirmation_resolved)
        self._bridge.remote_user_message.connect(self._on_remote_user_message)
        self._bridge.error.connect(self._on_error)
        self._bridge.listening_changed.connect(self._on_listening_changed)
        self._bridge.session_state_changed.connect(self._on_session_state_changed)
        self._bridge.standby_changed.connect(self._on_standby_changed)
        voice_owner_changed = getattr(self._bridge, "voice_owner_changed", None)
        if voice_owner_changed is not None:
            voice_owner_changed.connect(self._voice_page.set_voice_owner)
        self._bridge.wake_detected.connect(self._on_wake_detected)
        self._bridge.sleep_requested.connect(self._on_sleep_requested)
        usage_updated = getattr(self._bridge, "usage_updated", None)
        if usage_updated is not None:
            usage_updated.connect(self._on_usage_updated)

        self._nav.page_selected.connect(self.show_page)
        self._title_bar.collapse_clicked.connect(self._on_collapse_clicked)
        self._title_bar.maximize_clicked.connect(self.toggle_maximized)
        self._title_bar.quit_clicked.connect(self.close)
        self._overview.open_sources.connect(self.show_sources)
        self._overview.open_page.connect(self.show_page)
        self._activity_page.open_sources.connect(self.show_sources)
        self._tools_page.open_page.connect(self.show_page)
        self._tools_page.pair_requested.connect(self._mobile_button.click)
        self._protocols_page.open_activity.connect(lambda: self.show_page("activity"))
        self._attention = AttentionController(parent=self)
        """Pulse + card near the cursor for whatever Garvis opens for the user."""
        self._attention.activity_requested.connect(self._show_activity_from_attention)
        self._settings_page.animations_changed.connect(lambda on: setattr(self._attention, "animations", on))
        for signal_name, slot in (
            ("presentation_requested", self._attention.present),
            ("task_updated", self._on_task_updated),
            ("task_error", self._on_error),
        ):
            signal = getattr(self._bridge, signal_name, None)
            if signal is not None:
                signal.connect(slot)
        for page_signal, bridge_method in ((self._protocols_page.run_requested, "run_protocol"),
                                           (self._protocols_page.cancel_requested, "cancel_task"),
                                           (self._activity_page.tasks_panel.cancel_requested, "cancel_task")):
            method = getattr(self._bridge, bridge_method, None)
            if method is not None:
                page_signal.connect(method)
        reveal = getattr(self._bridge, "reveal_window", None)
        if reveal is not None:
            self._attention.reveal_requested.connect(reveal)
        snapshot = getattr(self._bridge, "tasks_snapshot", None)
        if snapshot is not None:
            tasks = snapshot()
            self._activity_page.tasks_panel.set_tasks(tasks)
            for task in reversed(tasks):
                self._protocols_page.on_task_updated(task)
        self._usage_page.refresh_requested.connect(self._request_usage)
        self._state.mode_changed.connect(self._refresh_title_mode)
        self._state.voice_changed.connect(self._refresh_title_mode)

    # ---------------------------------------------------------- dashboard
    def _system_info(self) -> dict:
        system_info = getattr(self._bridge, "system_info", None)
        try:
            return dict(system_info()) if system_info is not None else {}
        except Exception:  # noqa: BLE001 -- the dashboard shows "—" instead
            return {}

    def _apply_system_info(self) -> None:
        info = self._system_info()
        if info.get("model"):
            self._title_bar.set_model(info["model"], info.get("provider", ""))
            self._title_bar.model_chip.setToolTip(f"{info['model']}: every model request goes to this model "
                                                  "through the Anthropic API. No other model or provider is used.")
        else:
            self._title_bar.set_model_known(False)
        self._title_bar.key_chip.setVisible(info.get("api_key_configured") is False)
        for page in (self._overview, self._voice_page, self._usage_page, self._settings_page, self._tools_page):
            page.set_info(info)

    def _refresh_title_mode(self, *_args) -> None:
        self._title_bar.set_mode(self._state.mode, self._state.standby)

    def _request_usage(self) -> None:
        request = getattr(self._bridge, "request_usage", None)
        if request is not None:
            request()

    def _on_usage_updated(self, summary: dict) -> None:
        self._overview.set_usage(summary)
        self._usage_page.set_usage(summary)

    @property
    def state(self) -> GarvisState:
        return self._state

    @property
    def current_page(self) -> str:
        return PAGES[self._pages.currentIndex()]

    @property
    def attention(self) -> AttentionController:
        return self._attention

    def _on_task_updated(self, task: dict) -> None:
        self._activity_page.tasks_panel.update_task(task)
        self._protocols_page.on_task_updated(task)
        if task.get("kind") == "protocol" and task.get("status") in ("completed", "failed", "cancelled"):
            detail = getattr(self._bridge, "task_detail", None)
            if detail is not None:
                self._protocols_page.set_task_detail(detail(task["id"]))

    def _show_activity_from_attention(self) -> None:
        self.show_page("activity")
        if self._presence is not None:
            self._presence.expand()
        else:
            self._raise_window()

    def show_page(self, key: str) -> None:
        self._pages.setCurrentWidget(self._page_widgets[key])
        self._nav.select(key, emit=False)
        self._update_telemetry_demand()

    @property
    def telemetry(self) -> TelemetryMonitor:
        return self._telemetry

    def _update_telemetry_demand(self) -> None:
        """Sample only for what is on screen: everything once a second on
        the System page, a cheap summary while the dashboard is otherwise
        visible, nothing while it is hidden, minimized or collapsed."""
        if not self.isVisible() or self.isMinimized():
            demand = OFF
        elif self._pages.currentWidget() is self._system_page:
            demand = DETAIL
        else:
            demand = SUMMARY
        self._telemetry.set_demand(demand)

    def show_sources(self, turn_id: int | None = None) -> None:
        self._sources_page.show_turn(turn_id)
        self.show_page("sources")

    def _on_collapse_clicked(self) -> None:
        if self._presence is not None:
            self._presence.collapse()
        else:
            self.showMinimized()

    # ------------------------------------------------------------- actions
    def _on_send_clicked(self) -> None:
        text = self._input_line.text().strip()
        if not text:
            return
        self._input_line.clear()
        self._append_conversation("You", text)
        self._send_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        # A session must not start while this turn still owns the Agent's
        # context (starting one resets it) -- re-enabled when the turn ends.
        self._session_button.setEnabled(False)
        self._bridge.send_message(text)

    def _on_stop_clicked(self) -> None:
        self._bridge.stop_speaking()
        self._status_label.setText("Stopping...")
        self._stop_button.setEnabled(False)

    def _on_mobile_clicked(self) -> None:
        MobileDialog(self._bridge.mobile, self).exec()

    def _on_mic_clicked(self) -> None:
        if self._mic_button.property("recording"):
            self._bridge.stop_listening()
        else:
            # Same as a typed turn: no session until this push-to-talk turn ends.
            self._session_button.setEnabled(False)
            self._bridge.start_listening()

    def _on_session_clicked(self) -> None:
        if self._session_running:
            self._bridge.end_session()
            return
        self._begin_session()

    def _on_wake_detected(self, phrase: str) -> None:
        # Same gate as the Start Session button: a wake while a typed or
        # push-to-talk turn is still running (button disabled), or while a
        # session already runs, is ignored -- standby keeps listening.
        if self._session_running or not self._session_button.isEnabled():
            return
        self._begin_session(greet=True)

    def _on_sleep_requested(self) -> None:
        # "Go to standby" / "That's all": same as clicking End Session.
        if self._session_running:
            self._bridge.end_session()

    def _begin_session(self, greet: bool = False) -> None:
        self._session_running = True
        self._session_button.setText("End Session")
        # Session mode owns the mic/agent for its duration -- disable the
        # other ways to trigger a turn so two turns can never run
        # concurrently against the same Agent/ConversationContext.
        self._send_button.setEnabled(False)
        self._input_line.setEnabled(False)
        self._mic_button.setEnabled(False)
        if greet:
            self._bridge.start_session(greet=True)
        else:
            self._bridge.start_session()

    # -------------------------------------------------------------- events
    def _append_conversation(self, speaker: str, text: str) -> None:
        self._conversation_view.add_message(speaker, text)

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(status)
        if status in ("Done.", "Idle."):
            self._usage_timer.start()  # the ledger has this turn's requests by now
        # While a session owns the Agent, its own turns must not re-enable
        # the manual Send/Stop controls -- _on_session_state_changed(idle)
        # is what re-enables them once the session actually ends.
        if status in ("Done.", "Idle.") and not self._session_running:
            self._send_button.setEnabled(True)
            self._stop_button.setEnabled(False)
            self._session_button.setEnabled(True)

    _SESSION_STATUS_TEXT = {
        "listening": "Listening...",
        "thinking": "Thinking...",
        "speaking": "Speaking...",
        "interrupted": "Interrupted",
        "stopping": "Stopping...",
        "idle": "",
    }
    _STANDBY_TEXT = "Standby"

    def _on_standby_changed(self, active: bool) -> None:
        if active and not self._session_running:
            self._session_status_label.setText(self._STANDBY_TEXT)
        elif not active and self._session_status_label.text() == self._STANDBY_TEXT:
            self._session_status_label.setText("")

    def _on_session_state_changed(self, state: str) -> None:
        self._session_status_label.setText(self._SESSION_STATUS_TEXT.get(state, state))
        if state != "idle":
            return
        self._session_running = False
        self._session_button.setText("Start Session")
        self._send_button.setEnabled(True)
        self._input_line.setEnabled(True)
        self._mic_button.setEnabled(True)

    def _on_assistant_message(self, text: str) -> None:
        self._append_conversation("Garvis", text)

    def _on_remote_user_message(self, text: str, source: str) -> None:
        label = self._phone_label() if source == "mobile" else source
        self._append_conversation(f"You ({label})", html.escape(text))

    def _phone_label(self) -> str:
        """The paired device type that sent the message ("iPhone", "Android"),
        or "phone" when the bridge can't tell. A bridge without device
        information predates Android support, when the phone was an iPhone."""
        getter = getattr(self._bridge, "mobile_device_label", None)
        if getter is None:
            return "iPhone"
        return getter() or "phone"

    def _on_tool_started(self, name: str, arguments: dict) -> None:
        item = QListWidgetItem(f"⏳ {name}({arguments})")
        self._activity_list.addItem(item)

    def _on_tool_finished(self, name: str, success: bool, error: str) -> None:
        icon = "✅" if success else "❌"
        detail = "" if success else f" - {error}"
        item = QListWidgetItem(f"{icon} {name}{detail}")
        self._activity_list.addItem(item)
        self._conversation_view.add_tool(name, success, "" if success else elide(error or "failed", 80))

    def _on_confirmation_requested(self, request_id: str, tool_name: str, arguments: dict, reason: str) -> None:
        if self._presence is not None and self._presence.view == "collapsed":
            self._presence.expand()  # a decision is needed: show the context with it
        box = ApprovalDialog(tool_name, arguments, reason, self)
        self._confirmation_boxes[request_id] = box
        box.exec()
        approved = box.approved
        self._confirmation_boxes.pop(request_id, None)
        if request_id in self._confirmations_settled_elsewhere:
            # Answered on the phone (or timed out) while this was open.
            self._confirmations_settled_elsewhere.discard(request_id)
            return
        self._bridge.resolve_confirmation(request_id, approved)

    def _on_confirmation_resolved(self, request_id: str, outcome: str) -> None:
        box = self._confirmation_boxes.get(request_id)
        if box is not None:
            self._confirmations_settled_elsewhere.add(request_id)
            box.reject()

    def _on_error(self, message: str) -> None:
        self._append_conversation("Garvis", f"⚠️ {message}")
        # A transient error during an active session (e.g. one failed
        # transcription) does not end the session -- ConversationSession
        # keeps looping -- so the manual controls must stay disabled until
        # _on_session_state_changed(idle) says the session actually ended.
        if not self._session_running:
            self._send_button.setEnabled(True)
            self._stop_button.setEnabled(False)
            self._session_button.setEnabled(True)
            self._on_listening_changed(False)

    def _on_listening_changed(self, listening: bool) -> None:
        self._mic_button.setProperty("recording", listening)
        self._mic_button.setText("Stop && send" if listening else "Start listening")
        set_button_icon(self._mic_button, "stop_circle" if listening else "mic", "#FFE1E1" if listening else theme.TEXT_1)
        repolish(self._mic_button)

    # ------------------------------------------------- global hotkey / instance
    def bring_to_front(self) -> None:
        """Restore (if minimized, hidden or collapsed to the orb) and focus
        this window. Only ever called for an explicit user action: the
        global hotkey, or launching Garvis again while it's running."""
        if self._presence is not None and self._presence.view == "collapsed":
            self._presence.expand()  # ends in present(), which focuses the window
            return
        self._raise_window()

    def _raise_window(self) -> None:
        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.SetForegroundWindow(int(self.winId()))

    def install_global_hotkey(self, app, spec: str):
        """Tag this window for single-instance hand-over and register the
        global hotkey. Returns the GlobalHotkey (to unregister on exit), or
        None when no hotkey is configured. A registration failure is shown
        once in the conversation, with how to fix it."""
        from app.ui.global_hotkey import GlobalHotkey
        from app.ui.single_instance import Win32Api

        api = Win32Api()
        hwnd = int(self.winId())
        api.tag_window(hwnd)
        hotkey = GlobalHotkey(spec or "", self.bring_to_front, activate_message=api.activate_message())
        app.installNativeEventFilter(hotkey)  # also receives "activate" from a second launch
        if spec:
            from app.core.logging import get_logger, log_extra

            status = hotkey.register(hwnd)
            handled_by_shortcut = not status.registered and _start_menu_shortcut().exists()
            if handled_by_shortcut:
                # Explorer owns the key for the Start-menu shortcut; pressing it
                # launches Garvis, and that launch hands over to this window.
                detail = f"{spec.upper()} is handled by the Garvis Start-menu shortcut."
            else:
                detail = status.message
                if not status.registered:
                    self._append_conversation("Garvis", f"⚠️ {html.escape(status.message)}")
            log_extra(get_logger("ui.main_window"), 20 if status.registered or handled_by_shortcut else 30,
                      "global_hotkey.status", registered=status.registered, detail=detail)
        return hotkey

    # ------------------------------------------- orb, tray and transitions
    def install_desktop_presence(self, app) -> "PresenceController":  # noqa: F821
        """Add the floating orb, the orb <-> dashboard transition and a tray
        icon, and restore the remembered window placement. Call once, before
        show(). Quitting is then explicit (Quit Garvis), not "last window
        closed", since collapsing hides this window."""
        from app.ui.placement import UiStateStore
        from app.ui.presence import PresenceController

        store = UiStateStore(get_settings().data_dir / "ui_state.json")
        presence = PresenceController(self, self._state, store,
                                      level_source=getattr(self._bridge, "voice_level", None), parent=self)
        self._presence = presence
        app.setQuitOnLastWindowClosed(False)
        rect = presence.dashboard_rect()
        if presence.prefs.dashboard_maximized:
            self._normal_geometry = rect
            self._set_maximized_geometry(rect)
        else:
            self.setGeometry(rect)
        presence.menu_requested.connect(self._show_orb_menu)
        presence.sources_requested.connect(self._show_sources_menu)
        presence.install_tray(self._build_tray_menu())
        self._settings_page.set_interface_available(True, presence.prefs, str(store.path))
        self._settings_page.orb_enabled_changed.connect(presence.set_orb_enabled)
        self._settings_page.orb_corner_changed.connect(presence.set_orb_corner)
        self._settings_page.animations_changed.connect(presence.set_animations)
        return presence

    @property
    def presence(self):
        return self._presence

    # The four methods PresenceController drives this window through.
    def target_rect(self, presence) -> QRect:
        if self._maximized:
            return self.geometry()
        return presence.dashboard_rect()

    def snapshot(self) -> QPixmap:
        self._pages.currentWidget().flush()  # hidden pages refresh lazily; the picture must be current
        return self.grab()

    def remember_geometry(self, presence) -> None:
        presence.remember_dashboard_rect(self._normal_geometry if self._maximized and self._normal_geometry
                                         else self.geometry())
        presence.prefs.dashboard_maximized = self._maximized

    def present(self, presence=None) -> None:
        if not self.isVisible() and presence is not None:
            self.setGeometry(self.target_rect(presence))
        self._raise_window()

    def _menu(self, accent: str = theme.ACCENT_TEXT) -> QMenu:
        menu = QMenu(self)
        menu.setStyleSheet(theme.menu_stylesheet(accent))
        return menu

    def _add_action(self, menu: QMenu, text: str, slot, enabled: bool = True, tooltip: str = ""):
        action = menu.addAction(text)
        action.triggered.connect(lambda _checked=False: slot())
        action.setEnabled(enabled)
        if tooltip:
            action.setToolTip(tooltip)
        return action

    def _fill_sources_menu(self, menu: QMenu) -> None:
        menu.setToolTipsVisible(True)
        turn = self._state.latest_turn_with_sources
        if turn is None:
            self._add_action(menu, "No sources yet", lambda: None, enabled=False)
            return
        menu.addSection(elide(turn.request, 48).upper())
        for source in turn.sources[:10]:
            title = source.get("title") or source.get("publisher") or source["url"]
            text = f"{elide(title, 58)}   ·   {source.get('publisher', '')}"
            self._add_action(menu, text, lambda url=source["url"]: open_url(url), tooltip=source["url"])
        menu.addSeparator()
        self._add_action(menu, "Show all sources…", lambda: self._open_sources_view(turn.id))

    def _open_sources_view(self, turn_id: int | None) -> None:
        self.show_sources(turn_id)
        if self._presence is not None:
            self._presence.expand()
        else:
            self._raise_window()

    def _show_sources_menu(self, pos: QPoint) -> None:
        menu = self._menu(theme.ORANGE.primary)
        self._fill_sources_menu(menu)
        _exec_menu(menu, pos)

    def _show_orb_menu(self, pos: QPoint) -> None:
        presence = self._presence
        menu = self._menu(theme.ORANGE.primary)
        menu.addSection(f"GARVIS · {MODE_LABELS[self._state.mode].upper()}")
        self._add_action(menu, "Open dashboard", presence.expand)
        self._fill_sources_menu(menu.addMenu("Sources"))
        menu.addSeparator()
        self._add_action(menu, "End session" if self._session_running else "Start session",
                         self._session_button.click, enabled=self._session_button.isEnabled())
        self._add_action(menu, "Stop speaking", self._bridge.stop_speaking, enabled=self._state.playing)
        menu.addSeparator()
        has_tray = presence._tray is not None  # noqa: SLF001 -- same UI package
        self._add_action(menu, "Hide orb", presence.hide_orb, enabled=has_tray,
                         tooltip="Bring it back from the tray icon.")
        self._add_action(menu, "Quit Garvis", self.close)
        _exec_menu(menu, pos)

    def _build_tray_menu(self) -> QMenu:
        menu = self._menu()
        self._add_action(menu, "Open dashboard", lambda: self._presence.expand())
        self._add_action(menu, "Show orb", lambda: self._presence.show_orb() if self._presence.view == "collapsed"
                         else self._presence.collapse())
        menu.addSeparator()
        self._add_action(menu, "Quit Garvis", self.close)
        return menu

    # ------------------------------------------------- frameless window
    @property
    def is_maximized_custom(self) -> bool:
        return self._maximized

    def toggle_maximized(self) -> None:
        if self._maximized:
            self._maximized = False
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self._set_maximized_geometry(self.geometry())
        self._title_bar.set_maximized(self._maximized)

    def _set_maximized_geometry(self, around: QRect) -> None:
        # Maximized = the screen's work area (never over the taskbar), done by
        # hand because frameless windows maximize inconsistently on Windows.
        screen = QApplication.screenAt(around.center()) or self.screen()
        self.setGeometry(screen.availableGeometry())
        self._maximized = True
        self._title_bar.set_maximized(True)

    def _edges_at(self, pos: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if self._maximized:
            return edges
        margin = _EDGE + 1
        if pos.x() <= margin:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - margin:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= margin:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    _CURSORS = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor, Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor, Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.LeftEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.RightEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.RightEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.LeftEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.buttons() == Qt.MouseButton.NoButton:
            cursor = self._CURSORS.get(self._edges_at(event.position().toPoint()))
            if cursor is None:
                self.unsetCursor()
            else:
                self.setCursor(cursor)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        edges = self._edges_at(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and edges != Qt.Edge(0) and self.windowHandle() is not None:
            self.windowHandle().startSystemResize(edges)
            return
        super().mousePressEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().changeEvent(event)
        if event.type() == event.Type.WindowStateChange:
            self._update_telemetry_demand()
        if event.type() == event.Type.WindowStateChange and self._presence is not None:
            # Minimized from the taskbar (or by "collapse" with the orb off):
            # the orb, if enabled, keeps Garvis one click away.
            if self.isMinimized():
                self._presence.show_orb()
            elif self._presence.view == "expanded":
                self._presence.hide_orb()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._update_telemetry_demand()
        if sys.platform == "win32" and not self._rounded_corners_applied:
            self._rounded_corners_applied = True
            try:  # Windows 11: rounded corners for the frameless window (ignored elsewhere)
                import ctypes

                preference = ctypes.c_int(2)  # DWMWCP_ROUND
                ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), 33, ctypes.byref(preference),
                                                           ctypes.sizeof(preference))
            except Exception:  # noqa: BLE001 -- purely cosmetic
                pass

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._background = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        ratio = self.devicePixelRatioF()
        if self._background is None or self._background.size() != self.size() * ratio:
            self._background = self._render_background(ratio)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._background)
        painter.end()

    def _render_background(self, ratio: float) -> QPixmap:
        # Approved mockup: deep navy (#050B16) with a Midnight Blue glow behind
        # the orb -- radial-gradient(60% 70% at 52% 40%) -- and a hairline edge.
        w, h = self.width(), self.height()
        pixmap = QPixmap(int(w * ratio), int(h * ratio))
        pixmap.setDevicePixelRatio(ratio)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(0, 0, w, h, QColor(theme.BG))
        painter.save()
        painter.translate(w * 0.52, h * 0.40)
        painter.scale(0.60 * w, 0.70 * h)
        glow = QRadialGradient(QPointF(0, 0), 1.0)
        glow.setColorAt(0.0, QColor(16, 43, 76, 242))
        glow.setColorAt(0.40, QColor(10, 26, 48, 140))
        glow.setColorAt(0.70, QColor(5, 11, 22, 0))
        glow.setColorAt(1.0, QColor(5, 11, 22, 0))
        painter.fillRect(QRectF(-1.0, -1.0, 2.0, 2.0), glow)
        painter.restore()
        painter.setPen(QPen(QColor(160, 200, 245, 30), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRect(0, 0, w - 1, h - 1))
        painter.end()
        return pixmap


    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._update_telemetry_demand()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._telemetry.stop()
        self._attention.shutdown()
        presence = self._presence
        if presence is not None:
            presence.shutdown()
        self._bridge.shutdown()
        super().closeEvent(event)
        if presence is not None:
            QApplication.instance().quit()


def _start_menu_shortcut() -> Path:
    """The shortcut scripts/install_shortcut.py creates; when it exists,
    Explorer owns the hotkey and launching Garvis again hands over to us."""
    import os

    return Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Garvis.lnk"
