"""Tests for DesktopController (app/desktop/controller.py) against an
in-memory fake backend -- validation, window matching, focus recovery."""
from __future__ import annotations

import pytest

from app.core.exceptions import DesktopError
from app.desktop.backend import MouseButton, WindowInfo
from app.desktop.controller import DesktopController, match_window
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend


@pytest.fixture
def backend() -> FakeDesktopBackend:
    return FakeDesktopBackend()


@pytest.fixture
def controller(backend, tmp_path) -> DesktopController:
    return DesktopController(
        backend_factory=lambda: backend, screenshot_dir=tmp_path / "shots", max_type_chars=50,
        screenshot_keep=2, own_pid=GARVIS_PID,
    )


# ------------------------------------------------------------- match_window
def _w(handle, title, process):
    return WindowInfo(handle=handle, title=title, process_name=process)


def test_match_window_prefers_exact_title_then_process_then_substring():
    windows = [
        _w(1, "Chrome tips - Notepad", "notepad.exe"),
        _w(2, "Inbox - Google Chrome", "chrome.exe"),
        _w(3, "chrome", "other.exe"),
    ]
    assert match_window("chrome", windows).handle == 3  # exact title
    assert match_window("chrome", windows[:2]).handle == 2  # exact process name beats title substring
    assert match_window("tips", windows).handle == 1


def test_match_window_handles_packaged_app_process_names():
    assert match_window("whatsapp", [_w(9, "Chats", "WhatsApp.Root.exe")]).handle == 9


def test_match_window_none_when_nothing_matches():
    assert match_window("spotify", [_w(1, "Notepad", "notepad.exe")]) is None
    assert match_window("  ", [_w(1, "Notepad", "notepad.exe")]) is None


# ------------------------------------------------------------ platform/lazy
async def test_controller_without_backend_reports_unsupported_platform(tmp_path):
    controller = DesktopController(backend_factory=None, screenshot_dir=tmp_path)
    with pytest.raises(DesktopError, match="only supported on Windows"):
        await controller.screen_info()


def test_backend_is_created_lazily(tmp_path):
    created = []
    DesktopController(backend_factory=lambda: created.append(1), screenshot_dir=tmp_path)
    assert created == []


# -------------------------------------------------------------------- mouse
async def test_click_moves_then_clicks_and_reports_position(controller, backend):
    result = await controller.click(500, 400, button=MouseButton.LEFT, count=2)
    assert backend.calls[:2] == [("move", 500, 400), ("click", MouseButton.LEFT, 2)]
    assert result["x"] == 500 and result["y"] == 400
    assert result["foreground_window"] == "Untitled - Notepad"


async def test_click_without_coordinates_clicks_in_place(controller, backend):
    await controller.click(None, None)
    assert backend.calls[0] == ("click", MouseButton.LEFT, 1)


async def test_click_rejects_half_a_coordinate(controller):
    with pytest.raises(DesktopError, match="both x and y"):
        await controller.click(5, None)


@pytest.mark.parametrize("x, y", [(-1, 10), (10, -1), (1920, 10), (10, 1080), (99999, 99999)])
async def test_offscreen_coordinates_are_rejected_before_any_input(controller, backend, x, y):
    with pytest.raises(DesktopError, match="off-screen"):
        await controller.move_mouse(x, y)
    with pytest.raises(DesktopError, match="off-screen"):
        await controller.click(x, y)
    assert backend.calls == []


async def test_scroll_bounds(controller, backend):
    await controller.scroll(-3)
    assert backend.calls == [("scroll", -3, False)]
    for bad in (0, 51, -51):
        with pytest.raises(DesktopError):
            await controller.scroll(bad)


# ----------------------------------------------------------------- keyboard
async def test_type_text_goes_to_focused_window(controller, backend):
    result = await controller.type_text("hello")
    assert backend.calls == [("type", "hello", 1)]
    assert result == {"typed_chars": 5, "window": "Untitled - Notepad", "app": "Notepad.exe", "handle": 1,
                      "target_verified": True}


async def test_type_text_length_cap(controller, backend):
    with pytest.raises(DesktopError, match="too long"):
        await controller.type_text("x" * 51)
    with pytest.raises(DesktopError, match="Nothing to type"):
        await controller.type_text("")
    assert backend.calls == []


async def test_type_text_with_window_focuses_it_first(controller, backend):
    await controller.type_text("hi", window="whatsapp")
    assert ("focus", 2) in backend.calls
    assert backend.calls[-1] == ("type", "hi", 2)


async def test_type_text_unknown_window_fails_without_typing(controller, backend):
    with pytest.raises(DesktopError, match="No open application window matches"):
        await controller.type_text("hi", window="spotify")
    assert not any(c[0] == "type" for c in backend.calls)


async def test_focus_recovery_after_garvis_confirmation_dialog(controller, backend):
    """The user approves a confirmation dialog -> Garvis's own window is now
    focused. Keyboard input must go back to the previous target window."""
    await controller.focus_window("whatsapp")
    backend.foreground = 3  # Garvis UI (confirmation box) grabbed focus
    await controller.type_text("hey")
    assert backend.calls[-1] == ("type", "hey", 2)


async def test_keyboard_refuses_to_type_into_garvis_itself(controller, backend):
    backend.foreground = 3  # Garvis focused and no prior target known
    with pytest.raises(DesktopError, match="Garvis's own window"):
        await controller.type_text("oops")
    assert not any(c[0] == "type" for c in backend.calls)


async def test_press_key_and_hotkey(controller, backend):
    await controller.press_key("enter", repeat=2)
    await controller.hotkey("ctrl+shift+t")
    assert backend.calls[0] == ("keys", [0x0D], 2, 1)
    assert backend.calls[1] == ("keys", [0x11, 0x10, 0x54], 1, 1)


async def test_press_key_validation(controller, backend):
    with pytest.raises(DesktopError, match="Unknown key"):
        await controller.press_key("hyperspace")
    with pytest.raises(DesktopError, match="repeat"):
        await controller.press_key("a", repeat=0)
    with pytest.raises(DesktopError, match="blocked"):
        await controller.hotkey("win+r")
    assert backend.calls == []


# ------------------------------------------------------------------ windows
async def test_focus_window_and_refusal(controller, backend):
    window = await controller.focus_window("WhatsApp")
    assert window.handle == 2 and backend.foreground == 2
    backend.refuse_focus.add(1)
    with pytest.raises(DesktopError, match="didn't bring 'Untitled - Notepad' to the front.*nothing was sent"):
        await controller.focus_window("notepad")


async def test_switch_window_skips_foreground_and_garvis(controller, backend):
    window = await controller.switch_to_previous_window()
    assert window.handle == 2  # not 1 (foreground) and never 3 (Garvis)


async def test_switch_window_with_no_candidates(tmp_path):
    backend = FakeDesktopBackend(windows=[WindowInfo(handle=1, title="Only", pid=1)])
    controller = DesktopController(backend_factory=lambda: backend, screenshot_dir=tmp_path)
    with pytest.raises(DesktopError, match="no other window"):
        await controller.switch_to_previous_window()


# ------------------------------------------------------------------- screen
async def test_capture_screen_and_prunes_old_shots(controller, backend, tmp_path):
    paths = [(await controller.capture_screen())["path"] for _ in range(4)]
    remaining = sorted(p.name for p in (tmp_path / "shots").glob("screen-*.png"))
    assert len(remaining) == 2
    assert remaining == sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in paths[-2:])


async def test_capture_window_region_is_clipped_to_screen(controller, backend):
    result = await controller.capture_screen(window="notepad")
    assert result["region"] == {"left": 100, "top": 100, "right": 900, "bottom": 700}
    assert (result["width"], result["height"]) == (800, 600)
    with pytest.raises(DesktopError):
        await controller.capture_screen(window="spotify")


async def test_screen_info_shape(controller):
    info = await controller.screen_info()
    assert info["primary_screen"] == {"width": 1920, "height": 1080}
    assert info["foreground_window"]["title"] == "Untitled - Notepad"


async def test_backend_oserror_becomes_desktop_error(controller, backend):
    def boom(text):
        raise OSError("SendInput injected 0/2 events")

    backend.type_text = boom
    with pytest.raises(DesktopError, match="SendInput"):
        await controller.type_text("x")
