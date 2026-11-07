"""Operator behavior: concise/act-first prompt rules, live desktop context,
deduplicated recurring notices, and the screen-reading engines (OCR TSV
parsing; real Tesseract on a rendered image; a passive real UI Automation
read). Real-engine tests are skipped when the engine isn't available."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.agent.policies import _BASE_SYSTEM_PROMPT, build_system_prompt
from app.core.config import Settings
from app.core.events import AgentEvent, EventBus, EventType
from app.core.notices import RecurringNoticeFilter
from app.desktop.ocr import TesseractOcr, candidate_tesseract_paths, parse_tsv
from app.personality.policy import PERSONALITY_POLICY
from app.tools.registry import ToolRegistry
from tests.conftest import final_response


# ------------------------------------------------------------------ prompt
def test_prompt_makes_garvis_act_instead_of_explaining():
    prompt = build_system_prompt().lower()
    assert "do it with your tools" in prompt
    assert "don't explain how they could do it themselves" in prompt
    assert "ask for clarification if a request is ambiguous" not in prompt  # the old ask-first rule is gone
    assert "never ask in chat for permission to act" in prompt
    assert "verify important results" in prompt


def test_prompt_demands_short_answers_and_no_repetition():
    policy = PERSONALITY_POLICY.lower()
    assert "usually one sentence" in policy
    assert "never repeat or paraphrase what the user just said" in policy
    assert "mention a limitation, caveat or recurring error once" in policy
    assert "one brief progress line is enough" in policy


def test_prompt_names_only_tools_that_exist():
    from app.agent.build import build_agent

    settings = Settings(
        _env_file=None,
        llm_provider="tiered",
        omniroute_api_key="fake",
        memory_db_path=Path(tempfile.mkdtemp()) / "m.db",
    )
    registered = {t.name for t in build_agent(settings)._registry.all_tools()}
    mentioned = {
        "inspect_ui",
        "read_screen_text",
        "click_ui_element",
        "list_processes",
        "system_status",
        "recent_system_errors",
        "run_powershell",
    }
    assert all(name in _BASE_SYSTEM_PROMPT for name in mentioned)
    assert mentioned <= registered


def test_desktop_context_is_appended_last():
    prompt = build_system_prompt(None, "active window 'Inbox' (olk.exe); other open apps: chrome.")
    assert prompt.endswith(
        "Current desktop (observed by Garvis at the start of this turn): "
        "active window 'Inbox' (olk.exe); other open apps: chrome."
    )
    assert "Current desktop" not in build_system_prompt(None, None)


class _LLM:
    def __init__(self):
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        return final_response("Done, Sir.")


async def test_agent_refreshes_desktop_context_every_turn_and_survives_failures(tmp_path):
    states = iter(["active window 'A' (a.exe).", RuntimeError("no desktop"), "active window 'B' (b.exe)."])

    async def provider():
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return state

    llm = _LLM()
    agent = Agent(
        llm=llm,
        registry=ToolRegistry(),
        memory=SQLiteMemoryStore(tmp_path / "m.db"),
        events=EventBus(),
        context_provider=provider,
    )
    for text in ("one", "two", "three"):
        assert await agent.handle_user_message(text) == "Done, Sir."
    system_prompts = [call[0].content for call in llm.calls]
    assert "'A' (a.exe)" in system_prompts[0]
    assert "Current desktop" not in system_prompts[1]  # provider failed: turn still ran
    assert "'B' (b.exe)" in system_prompts[2] and "'A'" not in system_prompts[2]


# ------------------------------------------------------- recurring notices
def test_notice_filter_reports_once_until_quiet():
    now = [0.0]
    notices = RecurringNoticeFilter(quiet_seconds=60, clock=lambda: now[0])
    assert notices.should_report("x")
    now[0] = 30
    assert not notices.should_report("x")
    now[0] = 80  # still recurring every 50 s: stays quiet
    assert not notices.should_report("x")
    assert notices.should_report("y")  # a different problem is new information
    now[0] = 200  # it went away for > 60 s and came back
    assert notices.should_report("x")
    notices.clear("x")
    assert notices.should_report("x")


def _bridge(tmp_path):
    from app.ui.agent_bridge import AgentBridge

    return AgentBridge(
        settings=Settings(memory_db_path=tmp_path / "t.db", llm_provider="tiered", omniroute_api_key="fake-test-key")
    )


def test_bridge_shows_a_recurring_background_error_once(tmp_path):
    bridge = _bridge(tmp_path)
    shown: list[str] = []
    bridge.error.connect(shown.append)
    try:
        for _ in range(3):
            bridge._on_event(AgentEvent(type=EventType.ERROR, payload={"message": "Microphone unavailable."}))
        bridge._on_event(AgentEvent(type=EventType.ERROR, payload={"message": "Speaker unavailable."}))
        assert shown == ["Microphone unavailable.", "Speaker unavailable."]
    finally:
        bridge.shutdown()


async def test_bridge_always_shows_errors_that_answer_a_turn(tmp_path):
    bridge = _bridge(tmp_path)
    shown: list[str] = []
    bridge.error.connect(shown.append)
    try:
        await bridge._agent._turn_lock.acquire()  # a turn is running
        for _ in range(2):
            bridge._on_event(AgentEvent(type=EventType.ERROR, payload={"message": "OmniRoute is unreachable."}))
        bridge._agent._turn_lock.release()
        assert shown == ["OmniRoute is unreachable.", "OmniRoute is unreachable."]
    finally:
        bridge.shutdown()


# --------------------------------------------------------------------- OCR
_TSV = "\n".join(
    [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t10\t20\t40\t12\t95.0\tError:",
        "5\t1\t1\t1\t1\t2\t55\t20\t60\t12\t90.0\tdisk full",
        "5\t1\t2\t1\t1\t1\t10\t50\t30\t12\t12.0\t~#%",  # junk below the confidence floor
        "5\t1\t3\t1\t1\t1\t10\t80\t50\t12\t88.0\tשמור",
    ]
)


def test_tsv_words_become_lines_with_screen_coordinates():
    lines = parse_tsv(_TSV, offset_x=100, offset_y=200)
    assert [line.text for line in lines] == ["Error: disk full", "שמור"]
    first = lines[0]
    assert (first.left, first.top, first.width, first.height) == (110, 220, 105, 12)
    assert first.center == (162, 226)


def test_missing_ocr_engine_is_a_clear_error(tmp_path, monkeypatch):
    from app.core.exceptions import DesktopError

    monkeypatch.setattr("app.desktop.ocr.candidate_tesseract_paths", lambda configured=None: [tmp_path / "none.exe"])
    image = tmp_path / "x.png"
    image.write_bytes(b"png")
    with pytest.raises(DesktopError, match="GARVIS_OCR_TESSERACT_PATH"):
        TesseractOcr()._run(image, 0, 0)


def _tesseract_available() -> bool:
    return any(p.is_file() for p in candidate_tesseract_paths(None))


def _render_text_png(path: Path, lines: list[str], width: int = 900, height: int = 220) -> None:
    """Draw text with Windows GDI into an in-memory bitmap (real system
    fonts; no window, no screen capture) and save it as PNG with Garvis's
    own encoder. Qt's offscreen platform can't be used: it has no font
    database and draws placeholder boxes."""
    import ctypes
    from ctypes import wintypes

    from app.desktop.win32 import BITMAPINFOHEADER, _encode_png

    gdi = ctypes.WinDLL("gdi32")
    gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi.CreateCompatibleDC.restype = wintypes.HDC
    gdi.CreateDIBSection.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(BITMAPINFOHEADER),
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    gdi.CreateDIBSection.restype = wintypes.HBITMAP
    gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi.SelectObject.restype = wintypes.HGDIOBJ
    gdi.PatBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    gdi.CreateFontW.restype = wintypes.HFONT
    gdi.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
    gdi.TextOutW.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.LPCWSTR, ctypes.c_int]
    gdi.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
    gdi.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi.DeleteDC.argtypes = [wintypes.HDC]

    dc = gdi.CreateCompatibleDC(None)
    header = BITMAPINFOHEADER(
        biSize=ctypes.sizeof(BITMAPINFOHEADER),
        biWidth=width,
        biHeight=-height,
        biPlanes=1,
        biBitCount=32,
        biCompression=0,
    )
    bits = ctypes.c_void_p()
    bitmap = gdi.CreateDIBSection(dc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
    font = gdi.CreateFontW(-40, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 5, 0, "Arial")  # 5 = CLEARTYPE_QUALITY
    try:
        gdi.SelectObject(dc, bitmap)
        gdi.SelectObject(dc, font)
        gdi.PatBlt(dc, 0, 0, width, height, 0x00FF0062)  # WHITENESS
        gdi.SetBkMode(dc, 1)  # TRANSPARENT
        gdi.SetTextColor(dc, 0)
        for index, line in enumerate(lines):
            gdi.TextOutW(dc, 20, 30 + index * 90, line, len(line))
        bgra = ctypes.string_at(bits, width * height * 4)
    finally:
        gdi.DeleteObject(font)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(dc)
    path.write_bytes(_encode_png(width, height, bgra))


@pytest.mark.skipif(sys.platform != "win32" or not _tesseract_available(), reason="needs Windows + Tesseract")
async def test_real_tesseract_reads_rendered_text_and_reuses_unchanged_results(tmp_path):
    path = tmp_path / "rendered.png"
    _render_text_png(path, ["Build failed: 3 tests", "Click Retry to continue"])

    ocr = TesseractOcr(languages="eng")
    result = await ocr.recognize(path, offset_x=1000, offset_y=500)
    text = result.text.lower()
    assert "build failed" in text and "retry" in text
    retry = next(line for line in result.lines if "retry" in line.text.lower())
    assert retry.left >= 1000 and retry.top >= 500  # real screen coordinates
    again = await ocr.recognize(path, offset_x=1000, offset_y=500)
    assert again.cached and again.text == result.text


# --------------------------------------------------------- UI Automation
class _PassiveProbeWindow:
    """A real top-level Win32 window with known child controls, owned by the
    test, for passive UI Automation reads. It never takes input or focus:
    no-activate, click-through, fully transparent, bottom of the z-order, no
    taskbar button. Its own thread pumps messages (UIA's Win32 proxies send
    WM_GETOBJECT/WM_GETTEXT to it) and records anything that would mean the
    read was not passive: WM_COMMAND (a click/notification), activation and
    focus messages, and keyboard/mouse input."""

    BUTTON, LABEL, EDIT = "Garvis probe button", "Garvis probe label", "garvis probe value"
    _SUSPICIOUS = {
        0x0006: "WM_ACTIVATE",
        0x0007: "WM_SETFOCUS",
        0x0086: "WM_NCACTIVATE",
        0x0111: "WM_COMMAND",
        0x0100: "WM_KEYDOWN",
        0x0102: "WM_CHAR",
        0x0201: "WM_LBUTTONDOWN",
        0x0204: "WM_RBUTTONDOWN",
        0x0021: "WM_MOUSEACTIVATE",
    }

    def __init__(self, shown: bool = True) -> None:
        import threading

        self.shown = shown
        self.hwnd = 0
        self.events: list[str] = []
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="garvis-uia-probe-window", daemon=True)

    def __enter__(self) -> "_PassiveProbeWindow":
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("the probe window did not start")
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *exc) -> None:
        import ctypes

        if self.hwnd:
            ctypes.windll.user32.PostMessageW(self.hwnd, 0x0010, 0, 0)  # WM_CLOSE
        self._thread.join(10)

    def focus_in_window_thread(self) -> int:
        """The keyboard-focus HWND of the probe's GUI thread (0 = nothing focused)."""
        import ctypes
        from ctypes import wintypes

        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT),
            ]

        user32 = ctypes.windll.user32
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
        info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        thread_id = user32.GetWindowThreadProcessId(self.hwnd, None)
        if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            raise ctypes.WinError()
        return info.hwndFocus or 0

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        try:
            user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
            LRESULT = ctypes.c_ssize_t
            WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
            user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.DefWindowProcW.restype = LRESULT
            user32.CreateWindowExW.argtypes = [
                wintypes.DWORD,
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.HWND,
                wintypes.HMENU,
                wintypes.HINSTANCE,
                wintypes.LPVOID,
            ]
            user32.CreateWindowExW.restype = wintypes.HWND
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.SetLayeredWindowAttributes.argtypes = [
                wintypes.HWND,
                wintypes.COLORREF,
                wintypes.BYTE,
                wintypes.DWORD,
            ]
            user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
            user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

            class WNDCLASSW(ctypes.Structure):
                _fields_ = [
                    ("style", wintypes.UINT),
                    ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HANDLE),
                    ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR),
                ]

            def wndproc(hwnd, msg, wparam, lparam):
                name = self._SUSPICIOUS.get(msg)
                if name and not (msg in (0x0006, 0x0086) and wparam == 0):  # deactivation is harmless
                    self.events.append(f"{name}(0x{wparam:x})")
                if msg == 0x0002:  # WM_DESTROY
                    user32.PostQuitMessage(0)
                return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

            self._wndproc = WNDPROC(wndproc)  # keep the callback alive
            instance = kernel32.GetModuleHandleW(None)
            class_name = f"GarvisPassiveUiaProbe{id(self)}"
            wc = WNDCLASSW(lpfnWndProc=self._wndproc, hInstance=instance, lpszClassName=class_name)
            if not user32.RegisterClassW(ctypes.byref(wc)):
                raise ctypes.WinError()
            ex_style = 0x00080000 | 0x00000020 | 0x08000000 | 0x00000080  # LAYERED|TRANSPARENT|NOACTIVATE|TOOLWINDOW
            self.hwnd = user32.CreateWindowExW(
                ex_style,
                class_name,
                "Garvis passive UIA probe",
                0x80000000 | 0x02000000,  # WS_POPUP | WS_CLIPCHILDREN
                0,
                0,
                360,
                120,
                None,
                None,
                instance,
                None,
            )
            if not self.hwnd:
                raise ctypes.WinError()
            child = 0x40000000 | 0x10000000  # WS_CHILD | WS_VISIBLE
            for cls, text, style, (x, y, cx, cy), ident in (
                ("BUTTON", self.BUTTON, 0, (10, 10, 160, 30), 101),
                ("STATIC", self.LABEL, 0, (180, 10, 170, 30), 102),
                ("EDIT", self.EDIT, 0x00800000, (10, 60, 340, 30), 103),
            ):  # WS_BORDER
                if not user32.CreateWindowExW(
                    0, cls, text, child | style, x, y, cx, cy, self.hwnd, ident, instance, None
                ):
                    raise ctypes.WinError()
            user32.SetLayeredWindowAttributes(self.hwnd, 0, 0, 0x2)  # LWA_ALPHA, fully transparent
            if self.shown:  # bottom of the z-order, without activating it
                user32.SetWindowPos(self.hwnd, 1, 0, 0, 0, 0, 0x0010 | 0x0040 | 0x0001 | 0x0002)
                # SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOMOVE
            self.events.clear()  # creating the EDIT notifies EN_UPDATE/EN_CHANGE; only the read counts
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnregisterClassW(class_name, instance)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows UI Automation")
async def test_real_ui_automation_reads_a_window_passively():
    """Deterministic: the production UIA path reads a real top-level window
    the test owns, with known controls; nothing is clicked, typed, focused or
    activated (the window itself records any such message)."""
    import ctypes

    from app.desktop.uia import ComUiAutomation

    with _PassiveProbeWindow() as probe:
        elements = await ComUiAutomation().elements(probe.hwnd, limit=50)
        focus_after = probe.focus_in_window_thread()
        foreground_after = ctypes.windll.user32.GetForegroundWindow()
    assert elements, "the window exposes named controls"
    assert all(e.rect is None or e.rect.width >= 0 for e in elements)
    by_id = {e.automation_id: e for e in elements}
    assert by_id["101"].control_type == "button" and by_id["101"].name == probe.BUTTON
    assert by_id["102"].name == probe.LABEL
    assert by_id["103"].control_type == "edit" and by_id["103"].value == probe.EDIT
    assert not any(e.focused for e in elements)
    assert probe.events == [], f"the read was not passive: {probe.events}"
    assert focus_after == 0, "nothing in the window took keyboard focus"
    assert foreground_after != probe.hwnd, "the window was never activated"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows UI Automation")
async def test_real_ui_automation_returns_nothing_for_a_hidden_window():
    """A hidden window (e.g. the Windows taskbar while a replacement shell
    such as Project Glass hides it) reads as empty, not as an error."""
    from app.desktop.uia import ComUiAutomation

    with _PassiveProbeWindow(shown=False) as probe:
        elements = await ComUiAutomation().elements(probe.hwnd, limit=50)
    assert elements == []
    assert probe.events == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows UI Automation")
async def test_real_ui_automation_reads_the_taskbar_passively():
    """Read-only: lists the taskbar's controls; nothing is clicked or typed.
    Needs the real Windows taskbar on screen; the deterministic coverage of
    this code path is test_real_ui_automation_reads_a_window_passively."""
    import ctypes

    from app.desktop.uia import ComUiAutomation

    user32 = ctypes.windll.user32
    taskbar = user32.FindWindowW("Shell_TrayWnd", None)
    if not taskbar:
        pytest.skip("no taskbar window (e.g. a headless session)")
    if not user32.IsWindowVisible(taskbar):
        pytest.skip(
            "the Windows taskbar (Shell_TrayWnd) exists but is hidden, "
            "e.g. by a replacement shell; UI Automation exposes no controls for it"
        )
    elements = await ComUiAutomation().elements(taskbar, limit=50)
    assert elements, "the taskbar exposes named controls"
    assert all(e.rect is None or e.rect.width >= 0 for e in elements)
