"""The dashboard, the floating orb and their shared state (app/ui/state.py,
dashboard.py, presence.py, placement.py, visuals.py).

Offscreen Qt against a fake AgentBridge that emits the same signals as the
real one -- simulated events only: no Agent, LLM, microphone, real input,
browser or clipboard. Real-desktop behaviour (always-on-top over other apps,
focus, DPI changes, the animation's look) is in the manual checklist.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import app.ui.main_window as main_window
from app.ui import dashboard
from app.ui.placement import (
    UiPrefs,
    UiStateStore,
    corner_position,
    default_dashboard_rect,
    fit_rect,
    restore_position,
)
from app.ui.state import GarvisState, Mode
from app.ui.visuals import ACTIVE_INTERVAL_MS, IDLE_INTERVAL_MS, Dynamics
from tests.test_main_window_mobile import _FakeBridge as _BaseFakeBridge

SOURCES = [
    {"tool": "web_research", "kind": "read", "title": "Real headline", "url": "https://news.example/a",
     "publisher": "news.example", "snippet": "excerpt"},
    {"tool": "web_research", "kind": "search_result", "title": "Listed", "url": "https://other.example/b",
     "publisher": "other.example", "snippet": ""},
]


class _FakeBridge(_BaseFakeBridge):
    speaking_changed = Signal(bool)
    tool_sources = Signal(str, list)
    usage_updated = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.level = 0.0
        self.usage_requests = 0
        self.standby_calls: list[bool] = []
        self.stop_speaking_calls = 0

    standby_enabled = True

    def voice_level(self) -> float:
        return self.level

    def send_message(self, text: str) -> None:
        pass

    def set_standby_enabled(self, enabled: bool) -> None:
        self.standby_calls.append(enabled)

    def stop_speaking(self) -> None:
        self.stop_speaking_calls += 1

    def request_usage(self) -> None:
        self.usage_requests += 1

    def system_info(self) -> dict:
        return {"model": "claude-haiku-5-5", "provider": "Anthropic API", "api_key_configured": False,
                "daily_budget_usd": 0.5, "monthly_budget_usd": 5.0, "max_cost_per_task_usd": 0.05,
                "budget_enabled": True}


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "AgentBridge", _FakeBridge)
    win = main_window.MainWindow()
    yield win
    win.close()
    win.deleteLater()
    # Destroy it now: a backlog of earlier windows' deferred deletes would
    # otherwise land inside a later test's timing-sensitive event loop.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _research_turn(bridge, request="latest Artemis news?") -> None:
    bridge.user_message.emit(request)
    bridge.status_changed.emit("Thinking...")
    bridge.tool_started.emit("web_research", {"query": "Artemis news"})
    bridge.tool_finished.emit("web_research", True, "")
    bridge.tool_sources.emit("web_research", SOURCES)
    bridge.assistant_message.emit("Here's what I found.")
    bridge.status_changed.emit("Done.")


# ----------------------------------------------------------------- state
def test_mode_follows_real_events_with_playback_first(qapp):
    state = GarvisState()
    seen: list[str] = []
    state.mode_changed.connect(seen.append)
    state.on_session_state("listening")
    assert state.mode == Mode.LISTENING
    state.on_session_state("speaking")  # session says "speaking" before synthesis has finished
    assert state.mode == Mode.THINKING
    state.on_playing(True)  # the clip actually reaches the speakers
    assert state.mode == Mode.SPEAKING
    state.on_playing(False)
    state.on_session_state("idle")
    assert state.mode == Mode.IDLE
    assert seen == ["listening", "thinking", "speaking", "thinking", "idle"]


def test_turns_collect_their_tools_and_sources(qapp):
    state = GarvisState()
    bridge = _FakeBridge()
    state.connect_bridge(bridge)
    _research_turn(bridge)
    turn = state.turns[-1]
    assert not turn.running and turn.reply == "Here's what I found."
    assert [t.name for t in turn.tools] == ["web_research"] and turn.tools[0].success
    assert [s["url"] for s in turn.sources] == ["https://news.example/a", "https://other.example/b"]
    assert state.latest_turn_with_sources is turn and state.source_count == 2


def test_a_turn_without_sources_has_none(qapp):
    state = GarvisState()
    bridge = _FakeBridge()
    state.connect_bridge(bridge)
    bridge.user_message.emit("what's 6*7")
    bridge.tool_started.emit("calculator", {"expression": "6*7"})
    bridge.tool_finished.emit("calculator", True, "")
    bridge.assistant_message.emit("42")
    assert state.turns[-1].sources == [] and state.latest_turn_with_sources is None


def test_failed_turn_is_closed_and_error_is_transient(qapp):
    state = GarvisState()
    bridge = _FakeBridge()
    state.connect_bridge(bridge)
    bridge.session_state_changed.emit("thinking")
    bridge.user_message.emit("hello")
    bridge.error.emit("ANTHROPIC_API_KEY is not set")
    bridge.status_changed.emit("Idle.")
    turn = state.turns[-1]
    assert not turn.running and turn.error.startswith("ANTHROPIC_API_KEY")
    bridge.session_state_changed.emit("idle")
    assert state.mode == Mode.ERROR
    state._clear_error()  # what the 3.5 s timer does
    assert state.mode == Mode.IDLE


def test_phone_turns_are_labelled(qapp):
    state = GarvisState()
    bridge = _FakeBridge()
    state.connect_bridge(bridge)
    bridge.user_message.emit("from the phone")
    bridge.remote_user_message.emit("from the phone", "mobile")
    assert state.turns[-1].source == "mobile"


def test_works_with_a_bridge_without_the_new_signals(qapp):
    state = GarvisState()
    state.connect_bridge(_BaseFakeBridge())  # older/fake bridges: no speaking_changed / tool_sources
    assert state.mode == Mode.IDLE


# ------------------------------------------------------------- placement
def test_corner_positions_respect_the_work_area():
    available = QRect(0, 0, 1920, 1040)  # taskbar excluded
    size = QSize(124, 124)
    assert corner_position(available, size, "bottom-right") == QPoint(1920 - 124 - 28, 1040 - 124 - 28)
    assert corner_position(available, size, "top-left") == QPoint(28, 28)
    second = QRect(-1280, 200, 1280, 984)  # a monitor left of the primary
    assert corner_position(second, size, "bottom-left") == QPoint(-1280 + 28, 200 + 984 - 124 - 28)


def test_remembered_position_survives_only_while_on_a_screen():
    size = QSize(124, 124)
    screens = [QRect(0, 0, 1920, 1040), QRect(1920, 0, 2560, 1400)]
    assert restore_position((3000, 600), size, screens) == QPoint(3000, 600)  # second monitor
    assert restore_position((5000, 600), size, screens) is None  # that monitor is gone
    assert restore_position((100, 1000), size, screens) is None  # mostly below the primary's work area
    assert restore_position(None, size, screens) is None


def test_fit_rect_and_default_dashboard_rect():
    available = QRect(0, 0, 1366, 728)
    assert available.contains(fit_rect(QRect(900, 500, 1440, 900), available))
    rect = default_dashboard_rect(QRect(0, 0, 2560, 1400))
    assert rect.size() == QSize(1440, 900) and QRect(0, 0, 2560, 1400).contains(rect)


def test_ui_state_store_round_trip_and_bad_files(tmp_path):
    store = UiStateStore(tmp_path / "ui_state.json")
    assert store.load() == UiPrefs()  # missing file
    prefs = UiPrefs(orb_corner="top-left", orb_pos=[10, 20], animations=False)
    prefs.extra["future_key"] = 1
    store.save(prefs)
    loaded = store.load()
    assert loaded.orb_corner == "top-left" and loaded.orb_pos == [10, 20] and loaded.animations is False
    assert json.loads(store.path.read_text())["future_key"] == 1  # unknown keys preserved
    store.path.write_text("{not json")
    assert store.load() == UiPrefs()
    store.path.write_text(json.dumps({"orb_corner": "middle", "orb_pos": ["x", 1]}))
    assert store.load().orb_corner == "bottom-right" and store.load().orb_pos is None


# --------------------------------------------------------------- visuals
def test_voice_level_drives_the_speaking_animation_and_decays():
    dyn = Dynamics()
    for _ in range(20):
        dyn.step(0.033, Mode.SPEAKING, 3000.0)
    loud = dyn.appearance()
    assert dyn.level > 0.8
    for _ in range(40):
        dyn.step(0.033, Mode.IDLE, 0.0)
    assert dyn.level < 0.01 and dyn.appearance()["glow"] < loud["glow"]


def test_core_animates_only_while_visible_and_slows_down_when_idle(window):
    core = window._overview.core
    assert not core.is_animating  # window never shown
    window.show()
    QTest.qWait(10)
    assert core.is_animating and core.timer_interval == IDLE_INTERVAL_MS
    window._bridge.speaking_changed.emit(True)
    assert core.mode == Mode.SPEAKING and core.timer_interval == ACTIVE_INTERVAL_MS
    window.show_page("sources")  # overview hidden behind another page
    assert not core.is_animating


def test_speaking_samples_the_real_level_only_while_speaking(window):
    calls = []
    window._bridge.voice_level = lambda: calls.append(1) or 2000.0
    core = window._overview.core
    core._level_source = window._bridge.voice_level
    window.show()
    core._tick()
    assert calls == []
    window._bridge.speaking_changed.emit(True)
    core._tick()
    assert calls == [1]


# ------------------------------------------------------------- dashboard
def test_existing_controls_are_still_there(window):
    for name in ("_send_button", "_stop_button", "_mic_button", "_session_button", "_session_status_label",
                 "_input_line", "_conversation_view", "_activity_list", "_voice_replies_checkbox",
                 "_mobile_button", "_status_label"):
        assert getattr(window, name) is not None
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint


def test_title_bar_shows_the_real_model_and_a_missing_key(window):
    window.show()
    # Approved redesign: the model's display name in sentence case; the raw id stays in the tooltip.
    assert window._title_bar.model_chip.text() == "Claude Haiku 5.5 · Anthropic API"
    assert "claude-haiku-5-5" in window._title_bar.model_chip.toolTip()
    assert window._title_bar.key_chip.isVisible()


def test_sources_page_lists_the_returned_sources_and_opens_them(window, monkeypatch):
    opened = []
    monkeypatch.setattr(dashboard.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    _research_turn(window._bridge)
    window.show()
    window._overview.flush()
    turn_id = window.state.turns[-1].id
    window.show_sources(turn_id)
    page = window._sources_page
    cards = [page._cards.itemAt(i).widget() for i in range(page._cards.count())]
    cards = [c for c in cards if isinstance(c, dashboard.SourceCard)]
    assert [c.url for c in cards] == ["https://news.example/a", "https://other.example/b"]
    cards[0].open_button.click()
    assert opened == ["https://news.example/a"]
    assert window.current_page == "sources" and page.selected_turn == turn_id


def test_sources_page_says_so_when_there_are_none(window):
    window.show()
    window.show_page("sources")
    window._sources_page.flush()
    texts = [window._sources_page._cards.itemAt(i).widget() for i in range(window._sources_page._cards.count())]
    assert any(w is not None and "No sources" not in w.text() and "titles and links" in w.text() for w in texts)
    assert window._sources_page.detail_heading.text() == "No sources yet"


def test_only_http_links_are_opened(monkeypatch):
    opened = []
    monkeypatch.setattr(dashboard.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    assert dashboard.open_url("file:///C:/Windows/system32/cmd.exe") is False
    assert dashboard.open_url("javascript:alert(1)") is False
    assert opened == []


def test_activity_page_links_a_turn_to_its_sources(window):
    _research_turn(window._bridge)
    window.show()
    window.show_page("activity")
    window._activity_page.flush()
    column = window._activity_page._column
    card = next(column.itemAt(i).widget() for i in range(column.count())
                if isinstance(column.itemAt(i).widget(), dashboard.TurnCard))
    card.open_sources.emit(card.turn_id)
    assert window.current_page == "sources" and window._sources_page.selected_turn == card.turn_id


def test_usage_page_shows_ledger_figures_and_requests_a_refresh(window):
    bridge = window._bridge
    assert bridge.usage_requests == 1  # at startup
    window.show()
    window.show_page("usage")
    assert bridge.usage_requests >= 2  # opening the page refreshes it
    bridge.usage_updated.emit({"available": True, "today": {"calls": 3, "ok": 2, "errors": 1, "input_tokens": 1200,
                               "output_tokens": 80, "cache_read_tokens": 0, "cache_write_tokens": 0,
                               "cost_usd": 0.25, "tasks": 2},
                               "month": {"calls": 9, "ok": 9, "errors": 0, "input_tokens": 0, "output_tokens": 0,
                                         "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 1.0,
                                         "tasks": 5},
                               "recent": []})
    window._usage_page.flush()
    assert window._usage_page.day_bar.ratio == pytest.approx(0.5)
    assert window._usage_page.today.value("Requests").startswith("3")
    window._overview.flush()
    assert window._overview.tile_cost.value_label.text() == "$0.2500"


def test_usage_unavailable_is_shown_not_invented(window):
    window._bridge.usage_updated.emit({"available": False, "reason": "No model requests recorded yet."})
    window._overview.flush()
    assert window._overview.tile_cost.value_label.text() == "—"


def test_finished_turn_schedules_a_usage_refresh(window):
    window._bridge.status_changed.emit("Done.")
    assert window._usage_timer.isActive()


def test_voice_page_controls_use_the_existing_bridge_settings(window):
    page = window._voice_page
    assert page.standby_checkbox.isChecked()  # bridge.standby_enabled
    page.standby_checkbox.setChecked(False)
    assert window._bridge.standby_calls == [False]
    page.speak_checkbox.setChecked(True)
    assert window._voice_replies_checkbox.isChecked()  # one setting, two checkboxes


def test_maximize_uses_the_work_area_and_restores(window):
    window.show()
    window.setGeometry(QRect(50, 50, 1200, 760))
    before = window.geometry()
    window.toggle_maximized()
    assert window.is_maximized_custom and window.geometry().topLeft() == window.screen().availableGeometry().topLeft()
    window.toggle_maximized()
    assert not window.is_maximized_custom and window.geometry() == before


# ---------------------------------------------------- orb and transitions
@pytest.fixture
def presence(window, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "get_settings", lambda: type("S", (), {"data_dir": tmp_path})())
    controller = window.install_desktop_presence(QApplication.instance())
    window.show()
    yield controller
    QApplication.instance().setQuitOnLastWindowClosed(True)


def test_orb_window_never_takes_focus_and_stays_on_top(presence):
    orb = presence.orb
    flags = orb.windowFlags()
    for flag in (Qt.WindowType.Tool, Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint,
                 Qt.WindowType.WindowDoesNotAcceptFocus):
        assert flags & flag
    assert orb.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert presence.overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_collapse_and_expand_share_one_garvis(window, presence):
    presence.collapse(animated=False)
    assert presence.view == "collapsed" and not window.isVisible() and presence.orb.isVisible()
    expected = presence.orb_position()
    assert presence.orb.pos() == expected
    window._bridge.speaking_changed.emit(True)  # the orb reacts to the same state
    assert presence.orb.visual.mode == Mode.SPEAKING
    presence.expand(animated=False)
    assert presence.view == "expanded" and window.isVisible() and not presence.orb.isVisible()


def test_orb_click_expands_and_drag_moves_and_is_remembered(window, presence, tmp_path):
    presence.collapse(animated=False)
    orb = presence.orb
    start = orb.pos()
    QTest.mouseClick(orb, Qt.MouseButton.LeftButton, pos=QPoint(62, 62))
    assert presence.view == "expanded"
    presence.collapse(animated=False)
    QTest.mousePress(orb, Qt.MouseButton.LeftButton, pos=QPoint(62, 62))
    QTest.mouseMove(orb, QPoint(20, 30))
    QTest.mouseRelease(orb, Qt.MouseButton.LeftButton, pos=QPoint(20, 30))
    assert presence.view == "collapsed" and orb.pos() != start  # a drag, not a click
    saved = json.loads((tmp_path / "ui_state.json").read_text())
    assert saved["orb_pos"] == [orb.pos().x(), orb.pos().y()]


def test_sources_badge_counts_the_latest_sources(window, presence):
    assert presence.orb.badge_count == 0
    _research_turn(window._bridge)
    assert presence.orb.badge_count == 2


def test_orb_menu_mirrors_the_session_gating(window, presence, monkeypatch):
    captured = {}

    def fake_exec(menu, pos):
        captured["actions"] = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}

    monkeypatch.setattr(main_window, "_exec_menu", fake_exec)
    window._input_line.setText("hi")
    window._send_button.click()  # a typed turn is running: no session may start
    window._show_orb_menu(QPoint(0, 0))
    assert captured["actions"]["Start session"] is False
    assert captured["actions"]["Stop speaking"] is False


def test_disabling_the_orb_never_leaves_garvis_without_a_window(window, presence):
    presence.collapse(animated=False)
    presence.set_orb_enabled(False)
    assert window.isVisible() and not presence.orb.isVisible()


def test_bring_to_front_expands_from_the_orb(window, presence, monkeypatch):
    monkeypatch.setattr("ctypes.windll.user32.SetForegroundWindow", lambda hwnd: 1, raising=False)
    presence.collapse(animated=False)
    window.bring_to_front()
    assert presence.view == "expanded" and window.isVisible()


def test_animated_transition_reverses_cleanly_when_interrupted(window, presence, monkeypatch):
    monkeypatch.setattr(presence, "_animations_on", lambda: True)
    presence.collapse()
    assert presence.animating and presence.overlay.isVisible()
    QTest.qWait(120)
    mid = presence.overlay.progress
    assert 0.0 < mid < 1.0
    presence.expand()  # interrupt: reverse from where it is
    assert presence.view == "expanded"
    QTest.qWait(700)
    assert not presence.animating and window.isVisible() and not presence.orb.isVisible()
    assert not presence.overlay.isVisible()
    presence.collapse()
    QTest.qWait(600)
    assert presence.view == "collapsed" and not window.isVisible() and presence.orb.isVisible()
    assert not presence.overlay.isVisible()


def test_overlay_origin_stays_near_the_dashboards_screen(presence):
    rect = QRect(0, 0, 1000, 700)
    presence.prefs.orb_pos = [50_000, 50_000]  # unreachable: falls back to the corner
    origin = presence._orb_center_in(rect)
    assert isinstance(origin, QPointF)
    assert -40 <= origin.x() <= rect.width() + 40 and -40 <= origin.y() <= rect.height() + 40
