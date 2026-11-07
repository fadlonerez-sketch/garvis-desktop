"""Desktop Control tools, backed by `app.desktop.controller.DesktopController`.

Each tool is one fixed, structured operation (move / click / type / press /
hotkey / scroll / focus / capture) with schema-validated arguments -- the
LLM picks a tool, it never writes OS automation code. Same thin-wrapper
shape as app/tools/browser.py: validate, call the controller, translate
DesktopError into ToolResult.fail(...).

Risk levels:
  - Reading state (screen_info, list_windows, screen_capture,
    read_screen_text, inspect_ui), moving the pointer/view and managing
    windows (mouse_move, scroll, focus_window, switch_window, window_action,
    wait_for_window): LOW.
  - Anything that can *act* on whatever is under the cursor or holds focus
    (mouse clicks, keyboard_type / press / hotkey, click_ui_element):
    MEDIUM. Routine input runs without asking. A call is confirmed first
    only when deterministic code (CommitGuard, never the LLM) sees that it
    commits something: a click on a control named like "Send"/"Delete"/
    "Pay" (identified through UI Automation), Enter or a typed newline in a
    terminal, chat or mail app, or a send/permanent-delete hotkey.
    Settings.desktop_confirm_input_actions=True restores "confirm every
    input action".

Desktops (app/desktop/routing.py): most tools take an optional
`target_desktop`. Window lookups are then limited to that desktop, so an
operation meant for the user's desktop never lands on a window of Garvis's
workspace or vice versa. Real keyboard/mouse input and screenshots only
work on the desktop the user is viewing -- that is how Windows works -- so
asking for them on the workspace is an error, not a silent move. A window
on the workspace is only brought to the user's desktop when the user's
desktop is explicitly the target (the model's `target_desktop`, or the
user's own wording such as "show me").
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import DesktopError
from app.desktop.backend import MouseButton, Rect
from app.desktop.controller import DesktopController
from app.desktop.keys import parse_hotkey
from app.desktop.ocr import OcrEngine
from app.desktop.routing import TARGET_FIELD_DESCRIPTION, DesktopRouting, DesktopTarget
from app.desktop.uia import UiInspector, best_match
from app.security.sensitive import consequential_reason, redact_secrets
from app.tools.base import RiskLevel, Tool, ToolResult

_WINDOW_FIELD_DESCRIPTION = (
    "Optional window to focus first (title or app name, e.g. 'notepad', 'WhatsApp'). "
    "Omit to use the window that currently has focus."
)
TargetDesktop = Literal["user_desktop", "garvis_workspace"]
_INPUT_NOT_ON_WORKSPACE = (
    "Real keyboard and mouse input only reaches the desktop the user is looking at -- Windows can't send it "
    "to Garvis's workspace. For windows there, use inspect_ui / click_ui_element (UI Automation) or the "
    "browser_* tools; or target 'user_desktop' to bring the window to the user's screen first."
)
_NOT_VISIBLE_ON_WORKSPACE = (
    "Screenshots and OCR can only capture the desktop the user is looking at, not Garvis's workspace. "
    "Use inspect_ui for a window there, or browser_read_page for the workspace browser."
)


def _target_field():
    return Field(default=None, description=TARGET_FIELD_DESCRIPTION)


DEFAULT_COMMIT_APPS = (
    "cmd,powershell,pwsh,windowsterminal,wt,conhost,openconsole,wsl,bash,mintty,alacritty,putty,"
    "whatsapp,whatsapproot,telegram,discord,slack,teams,msteams,signal,skype,viber,messenger,zoom,"
    "outlook,olk,hxoutlook,thunderbird,mailspring,"
    "explorer,searchhost,searchapp,startmenuexperiencehost,"  # address bar / Start search run commands on Enter
    # IDEs with integrated terminals, more terminal emulators, remote sessions
    "code,code-insiders,cursor,windsurf,devenv,pycharm64,pycharm,idea64,idea,webstorm64,rider64,clion64,goland64,"
    "phpstorm64,rubymine64,datagrip64,studio64,wezterm-gui,tabby,hyper,conemu64,conemu,kitty,mstsc,vmconnect"
)
_ENTER_KEYS = {"enter", "return"}
_ALWAYS_CONFIRM_HOTKEYS = {
    frozenset({"ctrl", "enter"}): "Ctrl+Enter usually sends a message or submits.",
    frozenset({"alt", "s"}): "Alt+S usually sends a message.",
    frozenset({"shift", "delete"}): "Shift+Delete deletes permanently (bypasses the Recycle Bin).",
    frozenset({"ctrl", "shift", "delete"}): "Ctrl+Shift+Delete clears browsing data.",
}


def _app_key(process_name: str) -> str:
    return "".join(ch for ch in process_name.lower().removesuffix(".exe") if ch.isalnum())


class CommitGuard:
    """Decides, from facts rather than the LLM's say-so, whether an input
    action commits something and so needs the user's confirmation."""

    def __init__(self, controller: DesktopController, uia: UiInspector | None = None,
                 commit_apps: str = DEFAULT_COMMIT_APPS) -> None:
        self._controller = controller
        self._uia = uia
        self._commit_apps = {_app_key(a) for a in commit_apps.split(",") if a.strip()}

    @property
    def uia(self) -> UiInspector | None:
        return self._uia

    async def for_commit_key(self, window: str | None, what: str) -> str | None:
        try:
            target = await self._controller.resolve_window(window)
        except DesktopError:
            return None  # the action itself will fail with a clear error
        if _app_key(target.process_name) in self._commit_apps:
            return (f"{what} in '{target.title[:60]}' ({target.process_name}) may send a message "
                    "or run a command.")
        return None

    async def for_point(self, x: int | None, y: int | None) -> str | None:
        if self._uia is None:
            return None
        if x is None or y is None:
            info = await self._controller.screen_info()
            x, y = info["cursor"]["x"], info["cursor"]["y"]
        try:
            element = await self._uia.element_at(x, y)
        except DesktopError:
            return None
        if element is None:
            return None
        return consequential_reason(element.name, element.automation_id)


class _DesktopTool(Tool):
    def __init__(self, controller: DesktopController, routing: DesktopRouting | None = None) -> None:
        self._controller = controller
        self._routing = routing

    def _bring_to_user(self, requested: str | None) -> bool:
        """May a window on Garvis's workspace be brought to the user's
        desktop for this call? Only when the user's desktop is explicitly
        the target: chosen by the model, or asked for by the user."""
        if requested is not None:
            return requested == DesktopTarget.USER
        return self._routing is not None and self._routing.intent.explicit_target == DesktopTarget.USER


class _ConfirmableDesktopTool(_DesktopTool):
    risk_level = RiskLevel.MEDIUM

    def __init__(
        self, controller: DesktopController, require_confirmation: bool = False, guard: CommitGuard | None = None,
        routing: DesktopRouting | None = None,
    ) -> None:
        super().__init__(controller, routing)
        self.requires_confirmation = require_confirmation
        self._guard = guard or CommitGuard(controller)


# ----------------------------------------------------------------- queries
class NoInput(BaseModel):
    pass


class ScreenInfoTool(_DesktopTool):
    name = "screen_info"
    description = (
        "Get the screen size (physical pixels; multi-monitor bounds), the mouse cursor "
        "position, and the window that currently has focus."
    )
    input_model = NoInput

    async def execute(self, args: NoInput) -> ToolResult:
        try:
            return ToolResult.ok(await self._controller.screen_info())
        except DesktopError as exc:
            return ToolResult.fail(str(exc))


class ListWindowsInput(BaseModel):
    target_desktop: TargetDesktop | None = Field(
        default=None, description="Only list windows on this desktop: 'user_desktop' (the one the user is "
        "looking at) or 'garvis_workspace'. Omit to list every desktop."
    )


_DESKTOP_LABELS = {"user": "user_desktop", "garvis_workspace": "garvis_workspace", "other": "other_user_desktop",
                   "unknown": "user_desktop"}


class ListWindowsTool(_DesktopTool):
    name = "list_windows"
    description = (
        "List open application windows (title, app/process name, position and size in pixels, "
        "minimized/focused), topmost first, each with its desktop: 'user_desktop' (what the user sees), "
        "'garvis_workspace' (Garvis's background desktop) or 'other_user_desktop'. Garvis's own app "
        "window is marked garvis_app."
    )
    input_model = ListWindowsInput

    async def execute(self, args: ListWindowsInput) -> ToolResult:
        try:
            windows = await self._controller.list_windows(include_other_desktops=True)
            locations = await self._controller.locate_windows(windows)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        rows = []
        for w in windows:
            location = locations.get(w.handle, "unknown")
            if location == "unknown" and not w.on_current_desktop:
                desktop = "other_desktop"
            else:
                desktop = _DESKTOP_LABELS[location]
            if args.target_desktop is not None and desktop != args.target_desktop:
                continue
            row = {
                "title": w.title,
                "app": w.process_name,
                "desktop": desktop,
                "rect": w.rect.model_dump() if w.rect else None,
                "minimized": w.is_minimized,
                "focused": w.is_foreground,
            }
            if self._controller.is_own_window(w):
                row["garvis_app"] = True
            rows.append(row)
        return ToolResult.ok(rows)


class CaptureInput(BaseModel):
    window: str | None = Field(
        default=None, description="Optional window (title or app name) to capture instead of the whole screen."
    )
    target_desktop: TargetDesktop | None = _target_field()


class ScreenCaptureTool(_DesktopTool):
    name = "screen_capture"
    description = (
        "Save a PNG screenshot of the whole screen (or one window) to disk and return its "
        "path. You cannot see images: to read what's on screen use read_screen_text or inspect_ui."
    )
    input_model = CaptureInput

    async def execute(self, args: CaptureInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_NOT_VISIBLE_ON_WORKSPACE)
        try:
            window = args.window
            if window and args.target_desktop is not None:
                window = (await self._controller.resolve_window(window, desktop=args.target_desktop)).title
            result = await self._controller.capture_screen(window=window)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({**result, "target_desktop": "user_desktop"})


# ----------------------------------------------------------------- windows
class FocusWindowInput(BaseModel):
    window: str = Field(description="Window title or app name to switch to, e.g. 'chrome', 'WhatsApp'.")
    target_desktop: TargetDesktop | None = Field(
        default=None, description="'user_desktop' also brings the window over if it is on Garvis's workspace "
        "(do this when the user wants to see or use it). Focusing always happens on the user's desktop."
    )


class FocusWindowTool(_DesktopTool):
    name = "focus_window"
    description = (
        "Switch to (focus and restore) an already-open window by title or app name. "
        "To start an app that isn't open, use open_application."
    )
    input_model = FocusWindowInput

    async def execute(self, args: FocusWindowInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(
                "Focusing a window means putting it in front of the user; Garvis never switches the user's view "
                "to its workspace on its own. Work on workspace windows with inspect_ui / click_ui_element."
            )
        try:
            window = await self._controller.focus_window(
                args.window, bring_from_workspace=self._bring_to_user(args.target_desktop)
            )
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({"focused": window.title, "app": window.process_name, "verified": True,
                              "target_desktop": "user_desktop", "on_user_desktop": window.on_current_desktop})


class SwitchWindowTool(_DesktopTool):
    name = "switch_window"
    description = "Switch back to the previously used window (like Alt+Tab once)."
    input_model = NoInput

    async def execute(self, args: NoInput) -> ToolResult:
        try:
            window = await self._controller.switch_to_previous_window()
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({"focused": window.title, "app": window.process_name})


# ------------------------------------------------------------------- mouse
class PointInput(BaseModel):
    x: int = Field(description="Screen x coordinate in pixels")
    y: int = Field(description="Screen y coordinate in pixels")
    target_desktop: TargetDesktop | None = _target_field()


class OptionalPointInput(BaseModel):
    x: int | None = Field(default=None, description="Screen x in pixels; omit x and y to use the cursor position")
    y: int | None = Field(default=None, description="Screen y in pixels")
    target_desktop: TargetDesktop | None = _target_field()


class MouseMoveTool(_DesktopTool):
    name = "mouse_move"
    description = "Move the mouse cursor to screen coordinates (x, y) without clicking."
    input_model = PointInput

    async def execute(self, args: PointInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        try:
            await self._controller.move_mouse(args.x, args.y)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({"x": args.x, "y": args.y})


class _ClickTool(_ConfirmableDesktopTool):
    input_model = OptionalPointInput
    _button: MouseButton = MouseButton.LEFT
    _count: int = 1

    async def confirmation_reason(self, args: OptionalPointInput) -> str | None:
        if self._button == MouseButton.RIGHT:
            return None  # only opens a context menu; choosing an item is another click
        return await self._guard.for_point(args.x, args.y)

    async def execute(self, args: OptionalPointInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        try:
            result = await self._controller.click(args.x, args.y, button=self._button, count=self._count)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(result)


class MouseClickTool(_ClickTool):
    name = "mouse_click"
    description = "Left-click at screen coordinates (x, y), or at the cursor if omitted."


class MouseDoubleClickTool(_ClickTool):
    name = "mouse_double_click"
    description = "Double-click at screen coordinates (x, y), or at the cursor if omitted."
    _count = 2


class MouseRightClickTool(_ClickTool):
    name = "mouse_right_click"
    description = "Right-click (open context menu) at screen coordinates (x, y), or at the cursor if omitted."
    _button = MouseButton.RIGHT


class ScrollInput(BaseModel):
    direction: Literal["up", "down", "left", "right"] = Field(description="Scroll direction")
    amount: int = Field(default=3, ge=1, le=50, description="Number of wheel notches")
    x: int | None = Field(default=None, description="Optional x to scroll at (moves the cursor there first)")
    y: int | None = Field(default=None, description="Optional y to scroll at")
    target_desktop: TargetDesktop | None = _target_field()


class ScrollTool(_DesktopTool):
    name = "scroll"
    description = "Scroll the mouse wheel up/down (or left/right) where the cursor is, or at (x, y)."
    input_model = ScrollInput

    async def execute(self, args: ScrollInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        sign = 1 if args.direction in ("up", "right") else -1
        try:
            await self._controller.scroll(
                sign * args.amount, x=args.x, y=args.y, horizontal=args.direction in ("left", "right")
            )
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({"scrolled": args.direction, "amount": args.amount})


# ---------------------------------------------------------------- keyboard
class TypeInput(BaseModel):
    text: str = Field(description="Exact text to type (any language). Newlines press Enter.")
    window: str | None = Field(default=None, description=_WINDOW_FIELD_DESCRIPTION)
    target_desktop: TargetDesktop | None = _target_field()


class KeyboardTypeTool(_ConfirmableDesktopTool):
    name = "keyboard_type"
    description = "Type text into the focused window (or the named one) as if from the keyboard."
    input_model = TypeInput

    async def confirmation_reason(self, args: TypeInput) -> str | None:
        if "\n" not in args.text and "\r" not in args.text:
            return None
        return await self._guard.for_commit_key(args.window, "Typing text that ends a line (Enter)")

    async def execute(self, args: TypeInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        uia = self._guard.uia
        before = None
        try:
            if uia is not None and args.window is None:
                target = await self._controller.resolve_window(None)
                before = await _safe_focused_text(uia, target.handle)
            result = await self._controller.type_text(
                args.text, window=args.window, bring_from_workspace=self._bring_to_user(args.target_desktop)
            )
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        result["text_verified"] = await _verify_typed(uia, result["handle"], args.text, before, args.window is None)
        if result["text_verified"] is False:
            result["note"] = ("The keys were sent to the window, but its focused field doesn't show the text. "
                              "Check with inspect_ui or read_screen_text before reporting success.")
        return ToolResult.ok(result)


class PressInput(BaseModel):
    key: str = Field(
        description="Key name: enter, tab, escape, space, backspace, delete, up/down/left/right, "
        "home, end, pageup, pagedown, f1-f24, a letter or digit, volumeup, playpause, ..."
    )
    repeat: int = Field(default=1, ge=1, le=25, description="How many times to press it")
    window: str | None = Field(default=None, description=_WINDOW_FIELD_DESCRIPTION)
    target_desktop: TargetDesktop | None = _target_field()


class KeyboardPressTool(_ConfirmableDesktopTool):
    name = "keyboard_press"
    description = "Press a single key (optionally several times) in the focused window."
    input_model = PressInput

    async def confirmation_reason(self, args: PressInput) -> str | None:
        if args.key.strip().lower() not in _ENTER_KEYS:
            return None
        return await self._guard.for_commit_key(args.window, "Pressing Enter")

    async def execute(self, args: PressInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        try:
            return ToolResult.ok(await self._controller.press_key(
                args.key, repeat=args.repeat, window=args.window,
                bring_from_workspace=self._bring_to_user(args.target_desktop),
            ))
        except DesktopError as exc:
            return ToolResult.fail(str(exc))


class HotkeyInput(BaseModel):
    keys: str = Field(description="Key combination joined with '+', e.g. 'ctrl+c', 'ctrl+shift+t', 'alt+f4', 'win+d'.")
    window: str | None = Field(default=None, description=_WINDOW_FIELD_DESCRIPTION)
    target_desktop: TargetDesktop | None = _target_field()


class KeyboardHotkeyTool(_ConfirmableDesktopTool):
    name = "keyboard_hotkey"
    description = "Press a keyboard shortcut (modifiers held while the last key is pressed)."
    input_model = HotkeyInput

    async def confirmation_reason(self, args: HotkeyInput) -> str | None:
        try:
            names = [name for name, _ in parse_hotkey(args.keys)]
        except ValueError:
            return None  # refused by the controller anyway
        combo = frozenset("enter" if n == "return" else n for n in names)
        if combo in _ALWAYS_CONFIRM_HOTKEYS:
            return _ALWAYS_CONFIRM_HOTKEYS[combo]
        if "enter" in combo:
            return await self._guard.for_commit_key(args.window, f"Pressing {args.keys}")
        return None

    async def execute(self, args: HotkeyInput) -> ToolResult:
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_INPUT_NOT_ON_WORKSPACE)
        try:
            return ToolResult.ok(await self._controller.hotkey(
                args.keys, window=args.window, bring_from_workspace=self._bring_to_user(args.target_desktop)
            ))
        except DesktopError as exc:
            return ToolResult.fail(str(exc))


async def _safe_focused_text(uia: UiInspector, handle: int) -> str | None:
    try:
        return await uia.focused_text(handle)
    except DesktopError:
        return None


async def _verify_typed(uia: UiInspector | None, handle: int, text: str, before: str | None,
                        have_before: bool) -> bool | None:
    """True when the target's focused field now contains the typed text (one
    more time than before typing); False when it's readable but doesn't;
    None when the field can't be read (e.g. a canvas or terminal)."""
    if uia is None:
        return None
    needle = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not needle:
        return None
    after = None
    for _ in range(6):  # the app may take a moment to render what it received
        await asyncio.sleep(0.15)
        after = await _safe_focused_text(uia, handle)
        if after is not None:
            normalized = after.replace("\r\n", "\n").replace("\r", "\n")
            previous = (before or "").replace("\r\n", "\n").replace("\r", "\n") if have_before else ""
            if normalized.count(needle) > previous.count(needle):
                return True
    return None if after is None else False


# ------------------------------------------------------- window management
class WindowActionInput(BaseModel):
    action: Literal["minimize", "maximize", "restore", "close"] = Field(description="What to do with the window")
    window: str | None = Field(
        default=None, description="Window title or app name; omit for the window the user is working in."
    )
    target_desktop: TargetDesktop | None = Field(
        default=None, description="Only act on a window on this desktop ('user_desktop' or 'garvis_workspace')."
    )


class WindowActionTool(_DesktopTool):
    name = "window_action"
    description = (
        "Minimize, maximize, restore or close an application window. Close works like its X button "
        "(the app may ask to save); the result says whether it actually closed."
    )
    input_model = WindowActionInput

    async def execute(self, args: WindowActionInput) -> ToolResult:
        try:
            window = args.window
            if args.target_desktop is not None:
                window = (await self._controller.resolve_window(args.window, desktop=args.target_desktop)).title
            return ToolResult.ok(await self._controller.window_action(window, args.action))
        except DesktopError as exc:
            return ToolResult.fail(str(exc))


class WaitForWindowInput(BaseModel):
    window: str = Field(description="Window title or app name to wait for, e.g. 'notepad'")
    timeout_seconds: float = Field(default=15.0, ge=0.5, le=120.0)


class WaitForWindowTool(_DesktopTool):
    name = "wait_for_window"
    description = (
        "Wait until a window matching the title/app name is open (e.g. after open_application) "
        "and return it. Use it to verify that an app actually started."
    )
    input_model = WaitForWindowInput

    async def execute(self, args: WaitForWindowInput) -> ToolResult:
        try:
            window = await self._controller.wait_for_window(args.window, args.timeout_seconds)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok({"title": window.title, "app": window.process_name, "minimized": window.is_minimized})


# ---------------------------------------------------------- seeing the screen
class RegionInput(BaseModel):
    x: int = Field(description="Left edge in screen pixels")
    y: int = Field(description="Top edge in screen pixels")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")


class ReadScreenTextInput(BaseModel):
    window: str | None = Field(
        default=None,
        description="Window title or app name to read. Omit to read the window the user is working in.",
    )
    region: RegionInput | None = Field(default=None, description="Optional screen area to read instead of a window.")
    whole_screen: bool = Field(default=False, description="Read every monitor (slower; only when really needed).")
    find: str | None = Field(default=None, description="Optional text to look for; matching lines are listed first.")
    target_desktop: TargetDesktop | None = _target_field()


_MAX_OCR_LINES = 250
_MAX_OCR_CHARS = 8000
_OCR_DEDUPE_CHARS = 1500


class ReadScreenTextTool(_DesktopTool):
    name = "read_screen_text"
    description = (
        "Read the text visible on screen with offline OCR: a window (default: the active one), a "
        "region, or the whole screen. Returns text lines with screen coordinates (center = where to "
        "click). Use for dialogs, errors, images or anything inspect_ui can't read."
    )
    input_model = ReadScreenTextInput

    def __init__(self, controller: DesktopController, ocr: OcrEngine | None) -> None:
        super().__init__(controller)
        self._ocr = ocr

    async def execute(self, args: ReadScreenTextInput) -> ToolResult:
        if self._ocr is None:
            return ToolResult.fail("OCR is not configured on this PC. Use inspect_ui instead.")
        if args.target_desktop == DesktopTarget.WORKSPACE:
            return ToolResult.fail(_NOT_VISIBLE_ON_WORKSPACE)
        try:
            area = None
            window = None
            if args.region is not None:
                r = args.region
                area = Rect(left=r.x, top=r.y, right=r.x + r.width, bottom=r.y + r.height)
            elif not args.whole_screen:
                window = (await self._controller.resolve_window(args.window, desktop=args.target_desktop)).title
            shot = await self._controller.capture_screen(window=window, area=area)
            region = shot.get("region") or {"left": 0, "top": 0}
            result = await self._ocr.recognize(Path(shot["path"]), region["left"], region["top"])
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        lines = [
            {"text": redact_secrets(line.text), "center": list(line.center)}
            for line in result.lines[:_MAX_OCR_LINES]
        ]
        text = redact_secrets(result.text)
        payload: dict = {
            "source": shot.get("window") or ("region" if area else "screen"),
            "line_count": len(result.lines),
            "unchanged_since_last_read": result.cached,
        }
        if args.find:
            needle = args.find.lower()
            payload["matches"] = [line for line in lines if needle in line["text"].lower()][:20]
        payload["text"] = text[:_MAX_OCR_CHARS] + ("..." if len(text) > _MAX_OCR_CHARS else "")
        payload["lines"] = lines
        if len(payload["text"]) > _OCR_DEDUPE_CHARS and len(result.lines) <= _MAX_OCR_LINES:
            # Every word of `text` is already in `lines` (with its position);
            # sending both doubles the tokens of a large read.
            del payload["text"]
            payload["text_note"] = "Full text omitted: identical to the 'lines' entries, in reading order."
        return ToolResult.ok(payload)


class InspectUiInput(BaseModel):
    window: str | None = Field(default=None, description="Window title or app name; omit for the active window.")
    find: str | None = Field(default=None, description="Only controls whose name/id/value contains this text.")
    limit: int = Field(default=120, ge=1, le=400)
    target_desktop: TargetDesktop | None = Field(
        default=None, description="Only look for the window on this desktop ('user_desktop' or 'garvis_workspace')."
    )


class InspectUiTool(_DesktopTool):
    name = "inspect_ui"
    description = (
        "List the controls of a window via Windows UI Automation: buttons, fields (with their "
        "values), menus, list items, tabs, links, dialog text and terminal/document text, each with "
        "its type, name and center point. The fastest, most exact way to see and target a window's "
        "contents; password fields never show their value."
    )
    input_model = InspectUiInput

    def __init__(self, controller: DesktopController, uia: UiInspector | None) -> None:
        super().__init__(controller)
        self._uia = uia

    async def execute(self, args: InspectUiInput) -> ToolResult:
        if self._uia is None:
            return ToolResult.fail("UI Automation is not available on this system.")
        try:
            window = await self._controller.resolve_window(args.window, desktop=args.target_desktop)
            elements = await self._uia.elements(window.handle, query=args.find, limit=args.limit)
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        controls = []
        for element in elements:
            summary = element.summary()
            for key in ("name", "value", "text"):
                if summary.get(key):
                    summary[key] = redact_secrets(summary[key])
            controls.append(summary)
        payload = {"window": window.title, "app": window.process_name, "minimized": window.is_minimized,
                   "control_count": len(controls), "controls": controls}
        payload["desktop"] = "user_desktop" if window.on_current_desktop else (
            args.target_desktop or "other_desktop")
        if not window.on_current_desktop:
            payload["note"] = ("This window is on another virtual desktop; some apps hide their content there. "
                               "If it looks empty and the user wants it, bring it over with focus_window "
                               "(target_desktop='user_desktop').")
        return ToolResult.ok(payload)


class ClickUiElementInput(BaseModel):
    name: str = Field(description="Name (or automation id) of the control, as shown by inspect_ui")
    control_type: str | None = Field(default=None, description="Optional type to disambiguate, e.g. 'button'")
    window: str | None = Field(default=None, description="Window title or app name; omit for the active window.")
    target_desktop: TargetDesktop | None = Field(
        default=None, description="Only look for the window on this desktop. On 'garvis_workspace' the control "
        "is pressed through UI Automation without touching the user's screen."
    )


class ClickUiElementTool(_ConfirmableDesktopTool):
    name = "click_ui_element"
    description = (
        "Press a named control in a window (button, checkbox, tab, menu or list item, link) through "
        "UI Automation -- no coordinates needed. Falls back to a real click on its center."
    )
    input_model = ClickUiElementInput

    def __init__(self, controller: DesktopController, uia: UiInspector | None,
                 require_confirmation: bool = False, guard: CommitGuard | None = None,
                 routing: DesktopRouting | None = None) -> None:
        super().__init__(controller, require_confirmation, guard, routing)
        self._uia = uia

    async def confirmation_reason(self, args: ClickUiElementInput) -> str | None:
        reason = consequential_reason(args.name)
        if reason or self._uia is None:
            return reason
        # Judge the control that will really be pressed, not only the words the model used.
        try:
            window = await self._controller.resolve_window(args.window, desktop=args.target_desktop)
            elements = await self._uia.elements(window.handle, query=args.name, limit=50)
        except DesktopError:
            return None
        element = best_match(elements, args.name, args.control_type)
        return consequential_reason(element.name, element.automation_id) if element else None

    async def execute(self, args: ClickUiElementInput) -> ToolResult:
        if self._uia is None:
            return ToolResult.fail("UI Automation is not available on this system.")
        bring = self._bring_to_user(args.target_desktop)
        try:
            window = await self._controller.resolve_window(args.window, desktop=args.target_desktop)
            if window.is_minimized and window.on_current_desktop:
                window = await self._controller.focus_window(window.title)
            outcome = await self._uia.activate(window.handle, args.name, args.control_type)
            element = outcome["element"]
            action = outcome["action"]
            if action == "needs_click":
                if element.center is None:
                    return ToolResult.fail(f"'{element.name}' has no on-screen position to click.")
                if not window.on_current_desktop and not bring:
                    return ToolResult.fail(
                        f"'{element.name}' can only be pressed with a real click, and '{window.title}' is on "
                        "another desktop. " + _INPUT_NOT_ON_WORKSPACE
                    )
                await self._controller.focus_window(window.title, bring_from_workspace=bring)
                await self._controller.click(*element.center)
                action = "clicked"
        except DesktopError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(
            {"control": element.name, "type": element.control_type, "action": action, "window": window.title}
        )


def build_desktop_tools(
    controller: DesktopController,
    confirm_input_actions: bool = False,
    ocr: OcrEngine | None = None,
    uia: UiInspector | None = None,
    commit_apps: str = DEFAULT_COMMIT_APPS,
    routing: DesktopRouting | None = None,
) -> list[Tool]:
    guard = CommitGuard(controller, uia, commit_apps)
    confirm = confirm_input_actions
    return [
        ScreenInfoTool(controller),
        ListWindowsTool(controller),
        ScreenCaptureTool(controller),
        ReadScreenTextTool(controller, ocr),
        InspectUiTool(controller, uia),
        FocusWindowTool(controller, routing),
        SwitchWindowTool(controller),
        WindowActionTool(controller),
        WaitForWindowTool(controller),
        MouseMoveTool(controller),
        MouseClickTool(controller, confirm, guard, routing),
        MouseDoubleClickTool(controller, confirm, guard, routing),
        MouseRightClickTool(controller, confirm, guard, routing),
        ScrollTool(controller),
        KeyboardTypeTool(controller, confirm, guard, routing),
        KeyboardPressTool(controller, confirm, guard, routing),
        KeyboardHotkeyTool(controller, confirm, guard, routing),
        ClickUiElementTool(controller, uia, confirm, guard, routing),
    ]
