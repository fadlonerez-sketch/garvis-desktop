"""Desktop routing: USER_DESKTOP vs GARVIS_WORKSPACE (app/desktop/routing.py)
and its enforcement in the tools.

Hermetic: the virtual desktops, windows, focus and browser are fakes that
agree with each other (tests/routing_fakes.py). The real-Windows check of
the same behaviour is tests/test_desktop_routing_live.py (opt-in)."""

from __future__ import annotations

import pytest

from app.browser.manager import BrowserManager
from app.core.events import AgentEvent, EventBus, EventType
from app.desktop.routing import DesktopRouting, DesktopTarget, infer_intent
from app.desktop.workspace import WorkspaceManager
from app.tools import open_website as open_website_module
from app.tools.applications import OpenApplicationInput, OpenApplicationTool
from app.tools.browser import NavigateTool, ReadPageTool
from app.tools.desktop import build_desktop_tools
from app.tools.open_website import OpenWebsiteTool
from app.tools.research import WebResearchInput, WebResearchTool
from tests.desktop_fakes import GARVIS_PID
from tests.routing_fakes import garvis_id, make_desktops
from tests.test_browser_manager import (  # noqa: F401 -- fixture import activates it here
    FakeContext,
    FakePage,
    _FakeHarness,
    _install_fake_playwright,
    fake_public_dns,
)
from tests.test_research import FakeResearchBrowser, _researcher


# ------------------------------------------------------------------ intent
@pytest.mark.parametrize(
    "text, target, placement",
    [
        ("Yeah, show me some of the news today.", DesktopTarget.USER, False),
        ("open youtube for me", DesktopTarget.USER, False),
        ("let me see the weather radar", DesktopTarget.USER, False),
        ("put it on my screen", DesktopTarget.USER, True),
        ("open the docs in my browser", DesktopTarget.USER, True),
        ("תראה לי חדשות של היום", DesktopTarget.USER, False),
        ("תפתח את זה על המסך", DesktopTarget.USER, True),
        ("research flights in the background", DesktopTarget.WORKSPACE, True),
        ("check the price but don't open anything", DesktopTarget.WORKSPACE, True),
        ("just tell me the score", DesktopTarget.WORKSPACE, True),
        ("תבדוק ברקע מה מזג האוויר", DesktopTarget.WORKSPACE, True),
        ("research it in the background and show me the summary", DesktopTarget.WORKSPACE, True),
        ("what's the bitcoin price?", None, False),
    ],
    ids=[
        "news",
        "open-for-me",
        "let-me-see",
        "on-my-screen",
        "in-my-browser",
        "he-show",
        "he-screen",
        "background",
        "dont-open",
        "just-tell",
        "he-background",
        "background-wins",
        "neutral",
    ],
)
def test_infer_intent(text, target, placement):
    intent = infer_intent(text)
    assert intent.explicit_target == target
    assert intent.placement is placement


def test_resolution_order():
    routing = DesktopRouting()
    routing.new_request("what's the weather")
    assert routing.resolve("browser", None, DesktopTarget.WORKSPACE).target == DesktopTarget.WORKSPACE
    assert routing.resolve("browser", "user_desktop", DesktopTarget.WORKSPACE).target == DesktopTarget.USER
    routing.remember("browser", DesktopTarget.USER)
    assert routing.resolve("browser", None, DesktopTarget.WORKSPACE).reason.startswith("continuing")
    # "show me" decides the default but the model may still work in the background first
    routing.new_request("show me the news")
    assert routing.sticky == {}
    assert routing.resolve("browser", None, DesktopTarget.WORKSPACE).target == DesktopTarget.USER
    assert routing.resolve("browser", "garvis_workspace", DesktopTarget.WORKSPACE).target == DesktopTarget.WORKSPACE
    # an explicit placement instruction overrides the model
    routing.new_request("open it on my screen")
    resolved = routing.resolve("browser", "garvis_workspace", DesktopTarget.WORKSPACE)
    assert resolved.target == DesktopTarget.USER and "on my screen" in resolved.reason


async def test_routing_follows_user_messages_on_the_event_bus():
    bus = EventBus()
    routing = DesktopRouting()
    bus.subscribe(routing.on_event)
    await bus.publish(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "show me the news"}))
    assert routing.intent.wants_to_see
    await bus.publish(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": "thanks"}))
    assert routing.intent.explicit_target is None


# ------------------------------------------------- browser with two surfaces
class SurfaceBrowser(FakeResearchBrowser):
    """FakeResearchBrowser with BrowserManager's surface API: records which
    surface every navigation happened on."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.active_surface = "garvis_workspace"
        self.pages_by_surface: dict[str, str | None] = {"garvis_workspace": None, "user_desktop": None}
        self.surface_navigations: list[tuple[str, str]] = []
        self.places_windows = True

    async def use_surface(self, surface):
        self.active_surface = surface

    def has_page(self, surface):
        return self.pages_by_surface[surface] is not None

    async def navigate(self, url):
        self.surface_navigations.append((self.active_surface, url))
        self.pages_by_surface[self.active_surface] = url
        if self.active_surface == "user_desktop":
            self._current = url
            return url
        return await super().navigate(url)

    async def page_title(self):
        return f"Title of {self._current}"

    @property
    def current_url(self):
        return self.pages_by_surface[self.active_surface]


class RecordingPlacer:
    def __init__(self):
        self.last_placement = None

    def place(self, target):
        self.last_placement = (
            {"target_desktop": "user_desktop", "visible": True, "focused": True, "moved_from_other_desktop": False}
            if target == "user_desktop"
            else {"target_desktop": target, "moved": True}
        )


def _research_tool(browser, routing, workspace):
    placer = RecordingPlacer()
    original = browser.navigate

    async def navigate(url):
        result = await original(url)
        placer.place(browser.active_surface)
        return result

    browser.navigate = navigate
    return WebResearchTool(_researcher(browser, workspace=workspace), browser=browser, routing=routing, placer=placer)


@pytest.fixture
async def workspace():
    from tests.test_workspace import FakeVirtualDesktops

    manager = WorkspaceManager(backend_factory=lambda: FakeVirtualDesktops(), name="Garvis")
    yield manager
    manager.shutdown()


async def test_show_me_the_news_researches_in_workspace_then_presents_on_user_desktop(workspace):
    """The reported bug: 'show me some of the news today' only got spoken."""
    browser = SurfaceBrowser()
    routing = DesktopRouting()
    routing.new_request("Yeah, show me some of the news today.")
    result = await _research_tool(browser, routing, workspace).execute(WebResearchInput(query="news today"))
    assert result.success
    research = [url for surface, url in browser.surface_navigations if surface == "garvis_workspace"]
    shown = [url for surface, url in browser.surface_navigations if surface == "user_desktop"]
    assert research and research[0].startswith("https://html.duckduckgo.com")
    assert shown == ["https://a.test/btc"]  # the best source, once, on the user's desktop
    assert result.data["researched_on"] == "garvis_workspace"
    assert result.data["shown_on_user_desktop"]["visible"] is True
    assert routing.sticky["browser"] == DesktopTarget.USER  # follow-ups act on the shown page


async def test_background_research_never_touches_the_users_desktop(workspace):
    browser = SurfaceBrowser()
    routing = DesktopRouting()
    routing.new_request("look up the bitcoin price in the background")
    tool = _research_tool(browser, routing, workspace)
    result = await tool.execute(WebResearchInput(query="bitcoin price", show_on_user_desktop=True))
    assert result.success and "shown_on_user_desktop" not in result.data  # the user's words win
    assert {surface for surface, _ in browser.surface_navigations} == {"garvis_workspace"}


async def test_plain_question_only_answers(workspace):
    browser = SurfaceBrowser()
    routing = DesktopRouting()
    routing.new_request("what's the bitcoin price?")
    result = await _research_tool(browser, routing, workspace).execute(WebResearchInput(query="bitcoin price"))
    assert "shown_on_user_desktop" not in result.data
    assert {surface for surface, _ in browser.surface_navigations} == {"garvis_workspace"}


async def test_research_hands_back_the_surface_the_task_was_using(workspace):
    browser = SurfaceBrowser()
    browser.active_surface = "user_desktop"
    routing = DesktopRouting()
    await _research_tool(browser, routing, workspace).execute(WebResearchInput(query="bitcoin price"))
    assert browser.active_surface == "user_desktop"


async def test_explicit_open_overrides_the_background_default():
    browser = SurfaceBrowser()
    routing = DesktopRouting()
    tool = NavigateTool(browser, routing)
    routing.new_request("open youtube for me")
    result = await tool.execute(tool.validate_arguments({"url": "https://youtube.com"}))
    assert result.data["target_desktop"] == "user_desktop"
    assert browser.surface_navigations == [("user_desktop", "https://youtube.com")]


async def test_target_is_kept_across_a_multi_step_workflow():
    browser = SurfaceBrowser()
    routing = DesktopRouting()
    navigate, read = NavigateTool(browser, routing), ReadPageTool(browser)
    routing.new_request("show me the flight status page")
    await navigate.execute(navigate.validate_arguments({"url": "https://a.test/btc"}))
    routing.new_request("now go to the arrivals tab")  # no wording about where
    result = await navigate.execute(navigate.validate_arguments({"url": "https://c.test/y"}))
    assert result.data["target_desktop"] == "user_desktop"
    assert result.data["target_reason"].startswith("continuing")
    page = await read.execute(read.validate_arguments({}))
    assert page.data["target_desktop"] == "user_desktop"
    assert [s for s, _ in browser.surface_navigations] == ["user_desktop", "user_desktop"]


async def test_disabled_workspace_target_is_an_error_not_a_visible_page():
    browser = SurfaceBrowser()
    tool = NavigateTool(browser, DesktopRouting(), workspace_available=False)
    result = await tool.execute(
        tool.validate_arguments({"url": "https://a.test", "target_desktop": "garvis_workspace"})
    )
    assert not result.success and "disabled" in result.error
    assert browser.surface_navigations == []


# ------------------------------------------- real BrowserManager surfaces
class _TwoContextBrowser:
    def __init__(self):
        self.contexts: list[FakeContext] = []
        self.closed = False

    async def new_context(self):
        context = FakeContext(FakePage())
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed = True


async def test_manager_surfaces_are_separate_windows_placed_by_target(monkeypatch):
    harness = _FakeHarness()
    harness.browser = _TwoContextBrowser()
    harness.chromium._browser = harness.browser
    _install_fake_playwright(monkeypatch, harness)
    placed = []

    async def hook(page, surface):
        placed.append((page, surface))

    manager = BrowserManager(headless=False, page_hook=hook)
    await manager.navigate("https://example.com/research")
    await manager.use_surface("user_desktop")
    await manager.navigate("https://example.com/shown")
    assert [surface for _, surface in placed] == ["garvis_workspace", "user_desktop"]
    workspace_page, user_page = (page for page, _ in placed)
    assert workspace_page is not user_page
    assert workspace_page.goto_calls == ["https://example.com/research"]
    assert user_page.goto_calls == ["https://example.com/shown"]
    await manager.use_surface("garvis_workspace")
    assert manager.current_url == workspace_page.url
    await manager.shutdown()
    assert all(c.closed for c in harness.browser.contexts)


async def test_manager_closes_a_page_that_cant_be_placed(monkeypatch):
    from app.core.exceptions import BrowserError, WindowPlacementError

    harness = _FakeHarness()
    harness.browser = _TwoContextBrowser()
    harness.chromium._browser = harness.browser
    _install_fake_playwright(monkeypatch, harness)

    async def hook(page, surface):
        raise WindowPlacementError("workspace gone")

    manager = BrowserManager(headless=False, page_hook=hook)
    with pytest.raises(BrowserError, match="workspace gone"):
        await manager.navigate("https://example.com")
    page = harness.browser.contexts[0]._page
    assert page.closed and page.goto_calls == []  # never loaded anywhere
    assert not manager.has_page("garvis_workspace")
    await manager.shutdown()


# ------------------------------------------------------ applications
async def test_open_application_in_workspace_keeps_users_focus_and_moves_nothing_else(monkeypatch):
    backend, vd, workspace, desktop = await make_desktops()
    monkeypatch.setattr("sys.platform", "win32")
    backend.foreground = 1  # the user is typing in Notepad
    launched = []

    def fake_launcher(target, is_shortcut, background):
        def launch():
            launched.append(background)
            # the app ignores SW_SHOWMINNOACTIVE and grabs focus anyway
            backend.open_window(77, "Calculator", "CalculatorApp.exe", 770, minimized=True, steal_focus=True)

        return launch

    monkeypatch.setattr("app.tools.applications.make_launcher", fake_launcher)
    routing = DesktopRouting()
    routing.new_request("open the calculator in the background")
    tool = OpenApplicationTool(allowlist={"calculator": "calc.exe"}, desktop=desktop, routing=routing)
    try:
        result = await tool.execute(OpenApplicationInput(application="calculator"))
        assert result.success, result.error
        assert launched == [True]  # asked for a minimized, non-activated window
        assert vd.window_desktops[77] == garvis_id(vd)
        assert vd.moves == [(77, garvis_id(vd))]  # no other window moved
        assert backend.foreground == 1  # the user's focus was handed back
        assert result.data["target_desktop"] == "garvis_workspace"
        assert result.data["user_focus_unchanged"] is True and result.data["focus_had_to_be_returned"] is True
        assert ("state", 77, "show_inactive") in [(c[0], c[1], str(c[2])) for c in backend.calls if c[0] == "state"]
    finally:
        workspace.shutdown()


async def test_open_application_for_the_user_is_visible_and_focused(monkeypatch):
    backend, vd, workspace, desktop = await make_desktops()
    monkeypatch.setattr("sys.platform", "win32")

    def fake_launcher(target, is_shortcut, background):
        assert background is False
        return lambda: backend.open_window(77, "Calculator", "CalculatorApp.exe", 770)

    monkeypatch.setattr("app.tools.applications.make_launcher", fake_launcher)
    tool = OpenApplicationTool(allowlist={"calculator": "calc.exe"}, desktop=desktop, routing=DesktopRouting())
    try:
        result = await tool.execute(OpenApplicationInput(application="calculator"))
        assert result.success, result.error
        assert result.data["target_desktop"] == "user_desktop" and result.data["on_user_desktop"] is True
        assert backend.foreground == 77 and vd.moves == []
    finally:
        workspace.shutdown()


async def test_open_application_in_unavailable_workspace_launches_nothing(monkeypatch):
    from app.desktop.controller import DesktopController
    from tests.desktop_fakes import FakeDesktopBackend

    monkeypatch.setattr("sys.platform", "win32")
    backend = FakeDesktopBackend()
    desktop = DesktopController(backend_factory=lambda: backend, screenshot_dir=None, own_pid=GARVIS_PID)
    desktop.attach_workspace(WorkspaceManager(backend_factory=None))
    launched = []
    monkeypatch.setattr("app.tools.applications.make_launcher", lambda *a: lambda: launched.append(True))
    tool = OpenApplicationTool(allowlist={"calculator": "calc.exe"}, desktop=desktop)
    result = await tool.execute(OpenApplicationInput(application="calculator", target_desktop="garvis_workspace"))
    assert not result.success and "workspace" in result.error
    assert launched == []


async def test_reused_user_window_is_never_pulled_into_the_workspace(monkeypatch):
    backend, vd, workspace, desktop = await make_desktops()
    try:
        with pytest.raises(Exception, match="left where it is"):
            await desktop.launch_in_workspace("whatsapp", lambda: None, reuse_after_seconds=0.0)
        assert vd.moves == []
    finally:
        workspace.shutdown()


# ---------------------------------------------------- desktop input tools
async def test_input_never_silently_pulls_a_workspace_window_onto_the_users_screen():
    backend, vd, workspace, desktop = await make_desktops()
    backend.open_window(60, "Research - Google Chrome", "chrome.exe", 600, desktop_id=garvis_id(vd))
    routing = DesktopRouting()
    routing.new_request("type hello there")
    tools = {t.name: t for t in build_desktop_tools(desktop, routing=routing)}
    try:
        typing = tools["keyboard_type"]
        result = await typing.execute(typing.validate_arguments({"text": "hi", "window": "Research"}))
        assert not result.success and "workspace" in result.error
        assert vd.moves == [] and not [c for c in backend.calls if c[0] == "type"]
        refused = await typing.execute(
            typing.validate_arguments({"text": "hi", "window": "Research", "target_desktop": "garvis_workspace"})
        )
        assert not refused.success and "only reaches the desktop" in refused.error
        # explicitly for the user: brought over, focused, typed
        ok = await typing.execute(
            typing.validate_arguments({"text": "hi", "window": "Research", "target_desktop": "user_desktop"})
        )
        assert ok.success, ok.error
        assert vd.window_desktops[60] == vd.current_id and backend.foreground == 60
    finally:
        workspace.shutdown()


async def test_list_windows_labels_desktops_and_garvis_itself():
    backend, vd, workspace, desktop = await make_desktops()
    backend.open_window(60, "Research - Google Chrome", "chrome.exe", 600, desktop_id=garvis_id(vd))
    tool = {t.name: t for t in build_desktop_tools(desktop)}["list_windows"]
    try:
        rows = {r["title"]: r for r in (await tool.execute(tool.validate_arguments({}))).data}
        assert rows["Research - Google Chrome"]["desktop"] == "garvis_workspace"
        assert rows["Untitled - Notepad"]["desktop"] == "user_desktop"
        assert rows["Garvis"].get("garvis_app") is True
        only = (await tool.execute(tool.validate_arguments({"target_desktop": "garvis_workspace"}))).data
        assert [r["title"] for r in only] == ["Research - Google Chrome"]
    finally:
        workspace.shutdown()


async def test_inspection_and_screenshots_refer_to_the_selected_desktop():
    backend, vd, workspace, desktop = await make_desktops()
    backend.open_window(60, "Notes - Notepad", "Notepad.exe", 600, desktop_id=garvis_id(vd))
    try:
        # same query, different desktop -> different window
        assert (await desktop.resolve_window("notepad", desktop="garvis_workspace")).handle == 60
        assert (await desktop.resolve_window("notepad", desktop="user_desktop")).handle == 1
        capture = {t.name: t for t in build_desktop_tools(desktop)}["screen_capture"]
        result = await capture.execute(capture.validate_arguments({"target_desktop": "garvis_workspace"}))
        assert not result.success and "can only capture" in result.error
    finally:
        workspace.shutdown()


async def test_window_action_on_workspace_never_activates_it():
    backend, vd, workspace, desktop = await make_desktops()
    backend.open_window(60, "Notes - Notepad", "Notepad.exe", 600, minimized=True, desktop_id=garvis_id(vd))
    backend.foreground = 1
    tool = {t.name: t for t in build_desktop_tools(desktop)}["window_action"]
    try:
        result = await tool.execute(
            tool.validate_arguments({"action": "restore", "window": "notes", "target_desktop": "garvis_workspace"})
        )
        assert result.success, result.error
        assert backend.foreground == 1 and vd.current_id != garvis_id(vd)
        refused = await tool.execute(
            tool.validate_arguments({"action": "maximize", "window": "notes", "target_desktop": "garvis_workspace"})
        )
        assert not refused.success and "switch your view" in refused.error
    finally:
        workspace.shutdown()


# ------------------------------------------------------------ open_website
async def test_open_website_verifies_the_page_reached_the_users_screen(monkeypatch):
    backend, vd, workspace, desktop = await make_desktops()
    other = vd.create_desktop("Work")
    backend.open_window(80, "Inbox - Brave", "brave.exe", 800, desktop_id=other.id)

    def fake_open(url):
        # the browser put the new tab into its window on the other desktop
        next(w for w in backend.windows if w.handle == 80).title = "News - Brave"
        return True

    monkeypatch.setattr(open_website_module.webbrowser, "open", fake_open)
    tool = OpenWebsiteTool(desktop=desktop, routing=DesktopRouting(), verify_seconds=0.5)
    try:
        result = await tool.execute(tool.validate_arguments({"url": "https://news.test"}))
        assert result.success
        assert result.data["visible_verified"] is True and result.data["moved_from_other_desktop"] is True
        assert vd.window_desktops[80] == vd.current_id and backend.foreground == 80
    finally:
        workspace.shutdown()


async def test_open_website_reports_honestly_when_it_cant_confirm(monkeypatch):
    backend, vd, workspace, desktop = await make_desktops()
    monkeypatch.setattr(open_website_module.webbrowser, "open", lambda url: True)
    tool = OpenWebsiteTool(desktop=desktop, verify_seconds=0.3)
    try:
        result = await tool.execute(tool.validate_arguments({"url": "https://news.test"}))
        assert (
            result.success and result.data["visible_verified"] is False and "can't be confirmed" in result.data["note"]
        )
    finally:
        workspace.shutdown()


async def test_open_website_refuses_background_instead_of_showing_anyway(monkeypatch):
    opened = []
    monkeypatch.setattr(open_website_module.webbrowser, "open", lambda url: opened.append(url) or True)
    routing = DesktopRouting()
    routing.new_request("check that page in the background")
    result = await OpenWebsiteTool(routing=routing).execute(OpenWebsiteTool.input_model(url="https://news.test"))
    assert not result.success and "browser_navigate" in result.error
    assert opened == []


# ------------------------------------------------------------ run_powershell
class _ScriptRunner:
    """Pretends the script started a GUI program (and the user's own app opened nothing)."""

    def __init__(self, on_run):
        self._on_run = on_run

    async def run(self, script, cwd=None, timeout_seconds=60.0):
        from app.system.powershell import PowerShellResult

        self._on_run()
        return PowerShellResult(exit_code=0, stdout="", stderr="", timed_out=False, elapsed_seconds=0.1)


class _AllowAll:
    async def confirmation_reason(self, command):
        return None


async def test_powershell_windows_go_to_the_workspace_and_focus_stays_with_the_user():
    from app.tools.system import RunPowerShellTool

    backend, vd, workspace, desktop = await make_desktops()
    backend.foreground = 1
    runner = _ScriptRunner(lambda: backend.open_window(90, "Report - Excel", "EXCEL.EXE", 900, steal_focus=True))
    tool = RunPowerShellTool(runner=runner, policy=_AllowAll(), desktop=desktop)
    try:
        result = await tool.execute(
            tool.validate_arguments(
                {"command": "Start-Process excel report.xlsx", "target_desktop": "garvis_workspace"}
            )
        )
        assert result.success, result.error
        assert result.data["opened_windows"] == [
            {"title": "Report - Excel", "app": "EXCEL.EXE", "desktop": "garvis_workspace"}
        ]
        assert vd.moves == [(90, garvis_id(vd))] and backend.foreground == 1
    finally:
        workspace.shutdown()


async def test_powershell_with_unavailable_workspace_does_not_run():
    from app.desktop.controller import DesktopController
    from app.tools.system import RunPowerShellTool
    from tests.desktop_fakes import FakeDesktopBackend

    ran = []
    desktop = DesktopController(backend_factory=FakeDesktopBackend, screenshot_dir=None, own_pid=GARVIS_PID)
    desktop.attach_workspace(WorkspaceManager(backend_factory=None))
    tool = RunPowerShellTool(runner=_ScriptRunner(lambda: ran.append(1)), policy=_AllowAll(), desktop=desktop)
    result = await tool.execute(
        tool.validate_arguments({"command": "Start-Process notepad", "target_desktop": "garvis_workspace"})
    )
    assert not result.success and "was not run" in result.error and ran == []
