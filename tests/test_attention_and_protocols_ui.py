"""The attention overlay (app/ui/attention.py) and the Protocols page
(app/ui/protocols_page.py). Offscreen Qt: window flags, placement,
pacing, de-duplication and cleanup are checked here; how the pulse looks on
a real desktop, real focus behaviour over other apps and real mixed-DPI
monitors are in the manual checklist."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import app.ui.main_window as main_window
from app.tasks.protocols import Protocol, ProtocolStore, Step
from app.ui.attention import CARD_WIDTH, PULSE_COOLDOWN, AttentionCard, AttentionController
from app.ui.protocols_page import ProtocolsPage
from tests.test_ui_dashboard import _FakeBridge


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _payload(n: int, level: str = "prominent", **extra) -> dict:
    data = {"id": f"id-{n}", "level": level, "kind": "website", "title": f"Page {n}",
            "detail": f"https://site{n}.example", "url": f"https://site{n}.example", "handle": 100 + n,
            "source": "desktop"}
    data.update(extra)
    return data


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def attention(qapp):
    clock = _Clock()
    controller = AttentionController(cursor_pos=lambda: QPoint(400, 300), clock=clock)
    controller.clock = clock
    yield controller
    controller.shutdown()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_overlay_windows_never_take_focus_and_the_pulse_is_click_through(attention):
    for window in (attention.pulse, attention.card):
        flags = window.windowFlags()
        for flag in (Qt.WindowType.Tool, Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint,
                     Qt.WindowType.WindowDoesNotAcceptFocus):
            assert flags & flag
        assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert attention.pulse.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert attention.pulse.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not attention.card.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # its buttons work
    for button in (attention.card.close_button, attention.card.activity_button):
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_prominent_result_pulses_at_the_cursor_then_shows_the_real_card(attention):
    assert attention.present(_payload(1))
    assert attention.pulse.isVisible() and attention.pulse.running
    center = attention.pulse.geometry().center()
    assert abs(center.x() - 400) <= 1 and abs(center.y() - 300) <= 1
    QTest.qWait(400)
    card = attention.card
    assert card.isVisible() and [i["title"] for i in card.items] == ["Page 1"]
    assert card.kicker.text() == "OPENED · WEB PAGE"


def test_background_and_duplicate_events_are_not_shown(attention):
    assert not attention.present(_payload(1, level="none"))
    assert attention.present(_payload(2))
    assert not attention.present(_payload(2))  # the same event id
    assert not attention.present(_payload(3, id="other", title="Page 2", detail="https://site2.example"))
    assert attention.shown_count == 1


def test_a_burst_stacks_into_one_card_with_paced_pulses(attention):
    attention.present(_payload(1))
    QTest.qWait(350)
    attention.clock.now += 0.3
    for n in (2, 3, 4, 5, 6):
        attention.present(_payload(n))
    assert len(attention.card.items) == 6
    assert attention.card.more_label.text() == "+ 2 more in Activity"
    rows = attention.card.rows.count()
    assert rows == 4
    attention.card.dismiss()
    attention.clock.now += 0.3  # within the cooldown: card, but no new pulse
    attention.pulse.stop()
    attention.present(_payload(7))
    assert not attention.pulse.isVisible()
    attention.clock.now += PULSE_COOLDOWN + 0.1
    attention.dismiss()  # nothing on screen any more, and nothing pending
    attention.present(_payload(8))
    assert attention.pulse.isVisible()


def test_subtle_results_get_a_card_without_animation(attention):
    attention.present(_payload(1, level="subtle"))
    QTest.qWait(20)
    assert not attention.pulse.isVisible() and attention.card.isVisible()
    assert attention.card.dismiss_interval == 6000


def test_animations_preference_skips_the_pulse_but_keeps_the_card(attention):
    attention.animations = False
    attention.present(_payload(1))
    QTest.qWait(20)
    assert not attention.pulse.isVisible() and attention.card.isVisible()


def test_card_stays_inside_the_cursors_screen_on_any_monitor(qapp):
    card = AttentionCard()
    card.present([_payload(1)], "prominent", QPoint(2550, 1420), QRect(1920, 0, 640, 1440))
    geometry = card.frameGeometry()
    assert geometry.right() <= 2560 - 8 + 1 and geometry.bottom() <= 1440
    assert geometry.left() >= 1920  # flipped to the cursor's left, still on that monitor
    card.present([_payload(2)], "prominent", QPoint(-1900, 40), QRect(-1920, 0, 1920, 1080))
    assert card.pos().x() >= -1920 + 8 and card.pos().x() + CARD_WIDTH <= 0
    card.dismiss()
    card.deleteLater()


def test_show_brings_that_window_forward_and_activity_opens_the_dashboard(attention):
    revealed, activity = [], []
    attention.reveal_requested.connect(revealed.append)
    attention.activity_requested.connect(lambda: activity.append(True))
    attention.present(_payload(1))
    QTest.qWait(350)
    row = attention.card.rows.itemAt(0).widget()
    row.show_button.click()
    assert revealed == [101] and not attention.card.isVisible()
    attention.clock.now += 5
    attention.present(_payload(2, handle=None, kind="path", title="garvis", detail="C:\\work\\garvis"))
    QTest.qWait(350)
    assert attention.card.rows.itemAt(0).widget().show_button is None  # no window known: no false promise
    attention.card.activity_button.click()
    assert activity == [True]


def test_hover_keeps_the_card_and_dismiss_cleans_up(attention):
    attention.present(_payload(1))
    QTest.qWait(350)
    card = attention.card
    QApplication.sendEvent(card, QEvent(QEvent.Type.Enter))
    assert not card._timer.isActive()
    attention.dismiss()
    assert not card.isVisible() and not attention.pulse.isVisible() and card.items == []


def test_dashboard_routes_presentations_to_the_attention_system(qapp, monkeypatch):
    from PySide6.QtCore import Signal

    class Bridge(_FakeBridge):
        presentation_requested = Signal(dict)

        def __init__(self):
            super().__init__()
            self.revealed = []

        def reveal_window(self, handle: int) -> None:
            self.revealed.append(handle)

    monkeypatch.setattr(main_window, "AgentBridge", Bridge)
    win = main_window.MainWindow()
    win.attention._cursor_pos = lambda: QPoint(300, 300)
    win._bridge.presentation_requested.emit(_payload(1))
    QTest.qWait(350)
    assert win.attention.card.isVisible()
    win.attention.card.rows.itemAt(0).widget().show_button.click()
    assert win._bridge.revealed == [101]
    win.close()
    assert not win.attention.card.isVisible()


# ------------------------------------------------------------ protocols page
@pytest.fixture
def page(qapp, tmp_path):
    store = ProtocolStore(tmp_path / "protocols")
    widget = ProtocolsPage(store, {"open_website", "open_application", "open_path"})
    widget.answers = []
    widget.messages = []
    widget.ask = lambda title, text: widget.answers.pop(0) if widget.answers else True
    widget.inform = lambda title, text: widget.messages.append((title, text))
    widget.store = store
    widget.resize(1100, 800)
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


def test_create_build_save_and_reopen_a_protocol(page):
    page.new_protocol()
    page.name_edit.setText("Research workspace")
    page.add_step("open_websites")
    page.add_step("notify")
    assert len(page._cards) == 2 and page.dirty
    card = page._cards[0]
    card._fields["urls"] = lambda: ["https://a.example", "https://b.example"]
    card._on_change()
    page._cards[1]._fields["title"] = lambda: "Ready"
    page._cards[1]._on_change()
    assert page.save()
    saved = page.store.protocols()[0]
    assert saved.name == "Research workspace" and saved.steps[0].params["urls"] == ["https://a.example",
                                                                                  "https://b.example"]
    assert not page.dirty and not page.save_button.isEnabled()
    page.reload()
    assert page.library.count() == 1 and "2 steps · Ready" in page.library.item(0).text()


def test_reorder_duplicate_disable_and_remove_steps(page):
    page.new_protocol()
    page.add_step("open_website")
    page.add_step("wait")
    page.move_step(1, -1)
    assert [s.type for s in page.current.steps] == ["wait", "open_website"]
    page.duplicate_step(0)
    assert [s.type for s in page.current.steps] == ["wait", "wait", "open_website"]
    assert page.current.steps[0].id != page.current.steps[1].id
    page._cards[1].enabled_box.setChecked(False)
    assert page.current.steps[1].enabled is False
    page.remove_step(1)
    assert [s.type for s in page.current.steps] == ["wait", "open_website"]


def test_unavailable_actions_are_disabled_and_explained(page, tmp_path):
    page.store.save(Protocol(name="Needs control", steps=[Step(type="focus_window", params={"window": "x"})]))
    page.reload()
    page.library.setCurrentRow(0)
    card = page._cards[0]
    assert not card._available
    page._render_problems()
    assert "desktop control" in card.problem.text().lower()
    page.run_or_stop()
    assert page.messages and page.messages[-1][0] == "Can't run yet"


def test_dry_run_reports_without_running_anything(page):
    page.new_protocol()
    page.name_edit.setText("Morning")
    page.add_step("open_website")
    page._cards[0]._fields["url"] = lambda: "https://news.example"
    page._cards[0]._on_change()
    runs = []
    page.run_requested.connect(runs.append)
    page.test()
    report = page.last_test_report
    assert "nothing was opened" in report and "Open news.example — would run" in report
    assert "Ready to run." in report and runs == []


def test_run_saves_first_then_hands_the_protocol_to_the_task_system(page):
    runs = []
    page.run_requested.connect(runs.append)
    page.new_protocol()
    page.name_edit.setText("Quick")
    page.add_step("wait")
    page.answers = [True]  # "Save first?"
    page.run_or_stop()
    saved = page.store.protocols()[0]
    assert runs == [saved.id] and not page.dirty
    page.on_task_updated({"id": "t1", "protocol_id": saved.id, "status": "running", "origin": "mobile",
                          "progress": {"step": 1, "total": 1}, "steps": [{"label": "Pause 3 s", "status": "running"}]})
    assert page.run_button.text() == "Stop"
    cancels = []
    page.cancel_requested.connect(cancels.append)
    page.run_or_stop()
    assert cancels == ["t1"]
    page.on_task_updated({"id": "t1", "protocol_id": saved.id, "status": "completed", "origin": "mobile",
                          "result_preview": "All 1 step done.", "steps": [{"label": "Pause 3 s", "status": "ok"}]})
    assert page.run_button.text() == "Run"


def test_delete_asks_first_and_unsaved_changes_are_protected(page):
    page.new_protocol()
    page.name_edit.setText("Keep me")
    page.add_step("wait")
    page.save()
    page.answers = [False]
    page.delete()
    assert len(page.store.protocols()) == 1
    page.name_edit.setText("Edited")
    page.answers = [False]  # don't discard
    page.new_protocol()
    assert page.current.name == "Edited"
    page.answers = [True]
    page.delete()
    assert page.store.protocols() == []


def test_duplicate_and_export(page, tmp_path):
    page.new_protocol()
    page.name_edit.setText("Original")
    page.add_step("wait")
    page.save()
    page.duplicate()
    names = sorted(p.name for p in page.store.protocols())
    assert names == ["Original", "Original (copy)"]
    target = tmp_path / "backup.json"
    page.choose_export_path = lambda suggested: str(target)
    page.export()
    assert '"name": "Original (copy)"' in target.read_text(encoding="utf-8")


def test_invalid_files_are_listed_but_not_selectable(page, tmp_path):
    (tmp_path / "protocols").mkdir(exist_ok=True)
    (tmp_path / "protocols" / "abcdefabcdefabcd.json").write_text("{broken")
    page.reload()
    item = page.library.item(0)
    assert item.flags() == Qt.ItemFlag.NoItemFlags and "couldn't be loaded" in page.library_note.text()
