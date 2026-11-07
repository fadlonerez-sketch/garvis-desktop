"""The approved Desktop redesign (2026-10-31):
bundled fonts and icons (app/ui/fonts.py), the design tokens (app/ui/theme.py),
the navigation rail, the approval dialog (app/ui/approval.py), the chat-bubble
conversation (app/ui/conversation.py), the switch, restrained motion
(app/ui/motion.py) -- and that the frozen logo is still the original.

Offscreen Qt against the fake bridges the other UI tests use: no Agent, LLM,
microphone, real input or visible window.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QSize, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox

import app.ui.main_window as main_window
from app.ui import dashboard, fonts, motion, theme
from app.ui.approval import ApprovalDialog, detail_rows
from app.ui.conversation import ConversationView
from app.ui.logo import render_logo_pixmap
from tests.test_command_center import _CatalogBridge, _QuietMonitor


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "AgentBridge", _CatalogBridge)
    monkeypatch.setattr(main_window, "TelemetryMonitor", _QuietMonitor)
    win = main_window.MainWindow()
    yield win
    win.close()
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# ------------------------------------------------------------------ fonts
def test_bundled_fonts_register_under_their_family_names(qapp):
    loaded = fonts.ensure_loaded()
    assert loaded == {fonts.DISPLAY: True, fonts.TEXT: True, fonts.ICONS: True, fonts.ICONS_FILLED: True}
    families = set(QFontDatabase.families())
    for family in ("Manrope", "Inter", "Material Symbols Rounded", "Material Symbols Rounded Filled"):
        assert family in families
    assert {"Medium", "SemiBold", "Bold"} <= set(QFontDatabase.styles("Manrope"))
    assert {"Regular", "Medium", "SemiBold"} <= set(QFontDatabase.styles("Inter"))


def test_headings_use_manrope_and_text_uses_inter(qapp):
    assert theme.display_family() == "Manrope"
    assert theme.text_family() == "Inter"
    sheet = theme.stylesheet()
    assert 'font-family: "Inter"' in sheet and 'font-family: "Manrope"' in sheet
    # The logo lockup keeps its original face (Bahnschrift on Windows).
    assert theme.brand_family() != "Manrope"


def test_a_missing_font_file_is_logged_and_the_ui_falls_back(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(fonts, "_loaded", None)
    monkeypatch.setattr(fonts, "FONT_DIR", tmp_path)  # nothing to load here
    monkeypatch.setattr(fonts, "FONT_FILES", {"Garvis Missing Face": ("missing.ttf",)})
    logged = []
    monkeypatch.setattr(fonts, "_log", lambda level, event, **fields: logged.append(event))
    assert fonts.ensure_loaded() == {"Garvis Missing Face": False}
    assert "fonts.load_failed" in logged and "fonts.family_unavailable" in logged
    monkeypatch.setattr(theme, "_family_cache", frozenset({"Segoe UI"}))
    assert theme.display_family() == "Segoe UI"  # the first installed fallback


# ------------------------------------------------------------------ icons
def test_icon_codepoints_are_looked_up_by_name(qapp):
    assert fonts.icon_char("space_dashboard") == ""
    assert fonts.icon_char("shield") == ""
    assert fonts.icon_char("no_such_icon_name") == ""
    used = list(dashboard.PAGE_ICONS.values()) + list(dashboard._PANEL_ICONS.values())
    used += list(dashboard.GlyphButton.ICONS.values()) + ["mic", "arrow_upward", "key", "check_circle", "error"]
    missing = [name for name in used if not fonts.icon_char(name)]
    assert missing == []


def test_icons_render_in_both_styles_and_fail_soft(qapp, monkeypatch):
    for filled in (False, True):
        image = fonts.icon_pixmap("settings", "#FFFFFF", 24, filled=filled).toImage()
        assert any(image.pixelColor(x, y).alpha() > 0 for x in range(image.width()) for y in range(image.height()))
    outline = fonts.icon_pixmap("settings", "#FFFFFF", 24).toImage()
    filled = fonts.icon_pixmap("settings", "#FFFFFF", 24, filled=True).toImage()
    assert outline != filled  # selected items really use the filled glyph
    assert fonts.icon("no_such_icon_name", "#FFFFFF").isNull()
    monkeypatch.setattr(fonts, "icons_available", lambda filled=False: False)
    assert fonts.icon("settings", "#FFFFFF").isNull() and fonts.icon_html("settings", "#FFFFFF") == ""


# ----------------------------------------------------------------- tokens
def test_the_approved_tokens(qapp):
    assert theme.MIDNIGHT == "#102B4C"
    assert theme.ELECTRIC == "#1677D2"
    assert theme.BG == "#050B16"
    assert (theme.CARD_TOP, theme.CARD_BOTTOM) == ("#102440", "#0B1A2F")
    assert (theme.TEXT_1, theme.TEXT_2, theme.TEXT_3) == ("#EAF2FB", "#A3B6CE", "#6C84A2")
    assert (theme.OK, theme.WARN, theme.ERROR) == ("#3DD6A0", "#F2B652", "#F06A6A")
    assert theme.MOTION_MS == 160
    sheet = theme.stylesheet()
    assert theme.ELECTRIC in sheet and theme.TEXT_1 in sheet
    # The logo stays amber whatever the palette.
    assert theme.ORANGE.primary == "#FF8A1F"


# -------------------------------------------------------------- the rail
def test_nav_rail_has_every_existing_page_in_order(window):
    rail = window._nav
    assert list(rail.buttons) == list(dashboard.PAGES)
    for key, button in rail.buttons.items():
        assert button.text() == dashboard.PAGE_TITLES[key]
        assert fonts.icon_char(button.icon_name)
    rail.buttons["usage"].click()
    assert window.current_page == "usage" and rail.buttons["usage"].isChecked()
    assert not rail.buttons["overview"].isChecked()


def test_nav_rail_goes_icon_only_when_short_and_paints_without_icon_font(window, monkeypatch):
    window.resize(1000, 560)
    window.show()
    QApplication.processEvents()
    assert all(button.compact for button in window._nav.buttons.values())
    window.resize(1360, 860)
    QApplication.processEvents()
    assert not any(button.compact for button in window._nav.buttons.values())
    monkeypatch.setattr(fonts, "icons_available", lambda filled=False: False)
    assert not window._nav.grab().isNull()  # falls back to the painted glyphs


# ------------------------------------------------------------- the logo
def test_title_bar_and_window_icon_keep_the_original_amber_logo(window):
    bar_logo = window._title_bar.logo.pixmap().toImage()
    expected = render_logo_pixmap(34, theme.ORANGE, window._title_bar.devicePixelRatioF()).toImage()
    assert bar_logo == expected
    icon = window.windowIcon().pixmap(QSize(64, 64)).toImage()
    assert icon == render_logo_pixmap(64, theme.ORANGE).toImage().convertToFormat(icon.format())


# ------------------------------------------------------------ the layout
def test_overview_columns_follow_the_mockup_and_give_way_when_narrow(window):
    overview = window._overview
    assert overview.column_widths(1228) == (300, 344)  # 1360 px window: the mockup's widths
    left, right = overview.column_widths(968)  # 1100 px window
    assert left < 300 and right < 344 and 968 - left - right - 32 >= overview.CENTER_MIN
    assert overview.column_widths(400) == (overview.LEFT_MIN, overview.RIGHT_MIN)
    for line in (overview.cpu_line, overview.ram_line, overview.gpu_line):
        assert isinstance(line, dashboard.MeterLine)


def test_command_bar_holds_the_existing_controls(window):
    bar = window._input_line.parentWidget()
    assert isinstance(bar, main_window.CommandBar)
    for widget in (
        window._mic_button,
        window._send_button,
        window._stop_button,
        window._voice_replies_checkbox,
        window._session_button,
        window._session_status_label,
    ):
        assert widget.parentWidget() is bar
    assert window._send_button.property("accent") and window._stop_button.property("ghost")
    assert isinstance(window._voice_replies_checkbox, QCheckBox)  # a switch, still a check box
    assert window._mobile_button.text() == "Pair phone…"  # kept, on the Settings page


def test_title_chip_shows_a_readable_model_name():
    assert dashboard.display_model_name("claude-haiku-5-5") == "Claude Haiku 5.5"
    assert dashboard.display_model_name("claude-sonnet-4-5") == "Claude Sonnet 4.5"
    assert dashboard.display_model_name("") == ""


# ----------------------------------------------------------- the switch
def test_switch_is_a_check_box_and_jumps_under_reduced_motion(qapp):
    motion.set_reduced_motion(True)
    try:
        switch = dashboard.ToggleSwitch("Speak replies")
        seen = []
        switch.toggled.connect(seen.append)
        switch.click()
        assert switch.isChecked() and seen == [True] and switch._pos == 1.0  # no animation owed
        switch.set_compact(True)
        assert switch.toolTip() == "Speak replies" and switch.sizeHint().width() < 60
    finally:
        motion.set_reduced_motion(None)


def test_fader_eases_without_reduced_motion(qapp):
    motion.set_reduced_motion(False)
    try:
        values = []
        fader = motion.Fader(values.append)
        fader.to(1.0)
        assert fader.target == 1.0 and fader.value < 1.0  # still on its way
        QTest.qWait(theme.MOTION_MS + 120)
        assert fader.value == pytest.approx(1.0)
    finally:
        motion.set_reduced_motion(None)


# ------------------------------------------------------ approval dialog
def test_approval_dialog_shows_every_fact_and_defaults_to_deny(qapp):
    dialog = ApprovalDialog("gmail_send", {"to": "dana@example.com", "subject": "Report"}, "Sending mail asks first.")
    assert dialog.windowTitle() == "Garvis needs your confirmation"
    keys = [row[0] for row in dialog.rows]
    assert keys == ["Tool", "To", "Subject", "Why"]
    assert dialog.button(QMessageBox.StandardButton.Yes) is dialog.allow_button
    assert dialog.button(QMessageBox.StandardButton.No) is dialog.deny_button
    assert dialog.deny_button.isDefault() and not dialog.allow_button.isDefault()
    assert dialog.allow_button.property("accent")
    QTimer.singleShot(30, dialog.allow_button.click)
    dialog.exec()
    assert dialog.approved
    other = ApprovalDialog("run_powershell", {}, "")
    QTimer.singleShot(30, other.reject)
    other.exec()
    assert not other.approved


def test_detail_rows_shorten_long_values_but_keep_them():
    long = "x" * 500
    rows = detail_rows("write_text_file", {"content": long}, "")
    assert rows[1][0] == "Content" and len(rows[1][1]) < 200 and rows[1][2] == long
    assert detail_rows("calculator", {}, "")[1] == ("Arguments", "None", "")


def test_the_main_window_asks_with_the_approval_dialog(window):
    QTimer.singleShot(50, lambda: window._confirmation_boxes["req-9"].deny_button.click())
    window._on_confirmation_requested("req-9", "gmail_send", {"to": "x"}, "high risk")
    assert window._bridge.resolutions == [("req-9", False)]


# --------------------------------------------------------- conversation
def test_conversation_bubbles_keep_the_transcript(qapp):
    view = ConversationView()
    view.resize(340, 400)
    view.show()
    view.add_message("You", "Open the Garvis folder")
    view.add_tool("open_path", True)
    view.add_message("Garvis", "Opened it.")
    view.add_message("You (Android)", "from the Galaxy")
    QApplication.processEvents()
    text = view.toPlainText()
    assert "Open the Garvis folder" in text and "open_path" in text and "You (Android): from the Galaxy" in text
    assert view.bubble_count == 4
    for bubble in view._bubbles:
        rect = view.bubble_rect(bubble.table, bubble.kind)
        assert rect.width() > 20 and rect.right() <= view.viewport().width() + 1
    assert not view.grab().isNull()
    view.close()


def test_conversation_on_the_overview_is_the_bubble_view(window):
    assert isinstance(window._conversation_view, ConversationView)
    window._bridge.tool_finished.emit("calculator", False, "division by zero")
    assert "calculator · division by zero" in window._conversation_view.toPlainText()
    assert window._activity_list.count() == 1  # the tool log on Activity is unchanged


# -------------------------------------------------------- entry point
def test_the_app_window_builds_with_its_presence_offscreen(window, tmp_path, monkeypatch):
    """What app/main.py does at startup -- MainWindow, install_desktop_presence
    -- against the fake bridge (no Agent, no tray interaction)."""
    monkeypatch.setattr(main_window, "get_settings", lambda: type("S", (), {"data_dir": tmp_path})())
    presence = window.install_desktop_presence(QApplication.instance())
    try:
        window.show()
        QApplication.processEvents()
        assert window.isVisible() and presence.view == "expanded"
        for key in dashboard.PAGES:
            window.show_page(key)
            QApplication.processEvents()
            assert not window.grab().isNull()
    finally:
        QApplication.instance().setQuitOnLastWindowClosed(True)
