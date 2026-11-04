"""Windows UI Automation (the accessibility tree) for reading and pressing
controls in any application.

This is what lets a text-only model "see" a window precisely: every
button, field, menu item, list row and dialog text with its name, type,
state and exact screen rectangle -- far more reliable than guessing
coordinates. It uses `comtypes`, which Garvis already depends on (via
pyvda), so no new package is needed.

All COM work happens on one dedicated thread (COM objects are bound to the
thread that created them). Properties are fetched with a UIA CacheRequest,
so a whole window's tree comes back in one cross-process call instead of
thousands. Password fields never report their value.

`UiInspector` is the interface the tools depend on; tests use a fake.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.core.exceptions import DesktopError
from app.desktop.backend import Rect

CONTROL_TYPES = {
    50000: "button", 50001: "calendar", 50002: "checkbox", 50003: "combobox", 50004: "edit",
    50005: "hyperlink", 50006: "image", 50007: "listitem", 50008: "list", 50009: "menu",
    50010: "menubar", 50011: "menuitem", 50012: "progressbar", 50013: "radiobutton",
    50014: "scrollbar", 50015: "slider", 50016: "spinner", 50017: "statusbar", 50018: "tab",
    50019: "tabitem", 50020: "text", 50021: "toolbar", 50022: "tooltip", 50023: "tree",
    50024: "treeitem", 50025: "custom", 50026: "group", 50027: "thumb", 50028: "datagrid",
    50029: "dataitem", 50030: "document", 50031: "splitbutton", 50032: "window", 50033: "pane",
    50034: "header", 50035: "headeritem", 50036: "table", 50037: "titlebar", 50038: "separator",
    50039: "semanticzoom", 50040: "appbar",
}
INTERACTIVE_TYPES = {
    "button", "checkbox", "combobox", "edit", "hyperlink", "listitem", "menuitem", "radiobutton",
    "slider", "spinner", "tabitem", "treeitem", "splitbutton", "dataitem", "document", "headeritem",
}
_MAX_VISITED = 4000
_MAX_TEXT_CHARS = 4000


class UiElement(BaseModel):
    name: str
    control_type: str
    automation_id: str = ""
    rect: Rect | None = None
    enabled: bool = True
    focused: bool = False
    value: str | None = None
    """Current value of an edit/combo box (never for password fields)."""
    text: str | None = None
    """Visible text of a document/console control (e.g. terminal output), truncated."""

    @property
    def center(self) -> tuple[int, int] | None:
        if self.rect is None:
            return None
        return (self.rect.left + self.rect.width // 2, self.rect.top + self.rect.height // 2)

    def summary(self) -> dict:
        data: dict = {"type": self.control_type, "name": self.name}
        if self.automation_id:
            data["id"] = self.automation_id
        if self.center is not None:
            data["center"] = list(self.center)
        if not self.enabled:
            data["enabled"] = False
        if self.focused:
            data["focused"] = True
        if self.value:
            data["value"] = self.value
        if self.text:
            data["text"] = self.text
        return data


def matches(element: UiElement, query: str) -> bool:
    needle = query.strip().lower()
    return bool(needle) and any(
        needle in (field or "").lower() for field in (element.name, element.automation_id, element.value)
    )


def best_match(elements: list[UiElement], name: str, control_type: str | None = None) -> UiElement | None:
    """Exact name > name prefix > substring; enabled, on-screen elements first."""
    target = name.strip().lower()
    candidates = [e for e in elements if not control_type or e.control_type == control_type.lower()]
    for rank in (lambda n: n == target, lambda n: n.startswith(target), lambda n: target in n):
        found = [e for e in candidates if rank(e.name.strip().lower()) or rank(e.automation_id.lower())]
        if found:
            found.sort(key=lambda e: (not e.enabled, e.rect is None))
            return found[0]
    return None


class UiInspector(ABC):
    @abstractmethod
    async def elements(self, handle: int, query: str | None = None, limit: int = 120) -> list[UiElement]:
        """Named/interactive elements of window `handle`, in document order."""

    @abstractmethod
    async def element_at(self, x: int, y: int) -> UiElement | None:
        """The element under a screen point (or its nearest named ancestor)."""

    async def focused_text(self, handle: int) -> str | None:
        """Text of the control with keyboard focus, if it belongs to window
        `handle` (used to verify typing). None when it can't be read."""
        return None

    @abstractmethod
    async def activate(self, handle: int, name: str, control_type: str | None = None) -> dict:
        """Press/toggle/select/expand the best-matching element without the
        mouse. Returns {"element": UiElement, "action": str}; action is
        "needs_click" when the element supports no UIA action pattern."""


class ComUiAutomation(UiInspector):
    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="garvis-uia")
        self._uia = None
        self._mod = None
        self._cache = None

    # ------------------------------------------------------- COM thread
    def _init(self) -> None:
        if self._uia is not None:
            return
        try:
            import comtypes
            import comtypes.client
        except ImportError as exc:
            raise DesktopError("UI Automation needs the 'comtypes' package (installed with pyvda).") from exc
        from app.desktop.win32 import _enable_dpi_awareness  # physical-pixel rectangles

        _enable_dpi_awareness()
        comtypes.CoInitialize()
        self._mod = comtypes.client.GetModule("UIAutomationCore.dll")
        self._uia = comtypes.client.CreateObject(self._mod.CUIAutomation, interface=self._mod.IUIAutomation)
        cache = self._uia.CreateCacheRequest()
        for prop in (30005, 30003, 30011, 30001, 30010, 30022, 30019, 30008, 30045):
            # Name, ControlType, AutomationId, BoundingRectangle, IsEnabled,
            # IsOffscreen, IsPassword, HasKeyboardFocus, Value.Value
            cache.AddProperty(prop)
        self._cache = cache

    async def _call(self, func, *args):
        loop = asyncio.get_running_loop()

        def run():
            self._init()
            return func(*args)

        try:
            return await loop.run_in_executor(self._executor, run)
        except DesktopError:
            raise
        except Exception as exc:  # COMError, OSError...
            raise DesktopError(f"UI Automation failed: {exc}") from exc

    def _to_element(self, raw, read_text: bool = False) -> UiElement:
        rect = None
        try:
            r = raw.CachedBoundingRectangle
            if r.right > r.left and r.bottom > r.top:
                rect = Rect(left=r.left, top=r.top, right=r.right, bottom=r.bottom)
        except Exception:
            pass
        control_type = CONTROL_TYPES.get(raw.CachedControlType, "other")
        value = None
        if not raw.CachedIsPassword:
            try:
                value = raw.GetCachedPropertyValue(30045) or None
            except Exception:
                value = None
        text = None
        if read_text and control_type in ("document", "edit") and not raw.CachedIsPassword:
            text = self._read_text(raw)
        return UiElement(
            name=(raw.CachedName or "").strip(),
            control_type=control_type,
            automation_id=raw.CachedAutomationId or "",
            rect=rect,
            enabled=bool(raw.CachedIsEnabled),
            focused=bool(raw.CachedHasKeyboardFocus),
            value=str(value)[:500] if value else None,
            text=text,
        )

    def _read_text(self, raw) -> str | None:
        try:
            pattern = raw.GetCurrentPattern(10014)  # TextPattern
            if not pattern:
                return None
            text_pattern = pattern.QueryInterface(self._mod.IUIAutomationTextPattern)
            text = text_pattern.DocumentRange.GetText(_MAX_TEXT_CHARS * 4)
        except Exception:
            return None
        text = "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
        if len(text) > _MAX_TEXT_CHARS:
            text = "..." + text[-_MAX_TEXT_CHARS:]  # the end is what matters (latest output)
        return text or None

    def _collect(self, handle: int, query: str | None, limit: int) -> list[tuple[UiElement, object]]:
        root = self._uia.ElementFromHandle(handle)
        found = root.FindAllBuildCache(4, self._uia.ControlViewCondition, self._cache)  # TreeScope_Descendants
        results: list[tuple[UiElement, object]] = []
        texts_read = 0
        deadline = time.monotonic() + 5.0
        for index in range(min(found.Length, _MAX_VISITED)):
            if time.monotonic() > deadline:
                break
            raw = found.GetElement(index)
            if raw.CachedIsOffscreen:
                continue
            control_type = CONTROL_TYPES.get(raw.CachedControlType, "other")
            read_text = texts_read < 2 and control_type == "document"
            element = self._to_element(raw, read_text=read_text)
            if element.text:
                texts_read += 1
            if not (element.name or element.value or element.text or
                    (element.control_type in INTERACTIVE_TYPES and element.automation_id)):
                continue
            if query and not matches(element, query):
                continue
            results.append((element, raw))
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------ public
    async def elements(self, handle: int, query: str | None = None, limit: int = 120) -> list[UiElement]:
        pairs = await self._call(self._collect, handle, query, limit)
        return [element for element, _ in pairs]

    async def element_at(self, x: int, y: int) -> UiElement | None:
        def run() -> UiElement | None:
            from comtypes.gen.UIAutomationClient import tagPOINT

            raw = self._uia.ElementFromPointBuildCache(tagPOINT(x, y), self._cache)
            walker = self._uia.ControlViewWalker
            for _ in range(4):  # an icon inside a button: use the button's name
                if raw is None:
                    return None
                element = self._to_element(raw)
                if element.name or element.automation_id:
                    return element
                raw = walker.GetParentElementBuildCache(raw, self._cache)
            return None

        return await self._call(run)

    async def focused_text(self, handle: int) -> str | None:
        def run() -> str | None:
            import ctypes
            from ctypes import wintypes

            element = self._uia.GetFocusedElement()
            if element is None or element.CurrentIsPassword:
                return None
            window_pid = wintypes.DWORD(0)
            ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(window_pid))
            if element.CurrentProcessId != window_pid.value:
                return None
            try:
                pattern = element.GetCurrentPattern(10014)  # TextPattern
                if pattern:
                    return pattern.QueryInterface(self._mod.IUIAutomationTextPattern).DocumentRange.GetText(200000)
            except Exception:
                pass
            try:
                pattern = element.GetCurrentPattern(10002)  # ValuePattern
                if pattern:
                    return pattern.QueryInterface(self._mod.IUIAutomationValuePattern).CurrentValue
            except Exception:
                pass
            return None

        return await self._call(run)

    async def activate(self, handle: int, name: str, control_type: str | None = None) -> dict:
        def run() -> dict:
            pairs = self._collect(handle, None, _MAX_VISITED)
            element = best_match([e for e, _ in pairs], name, control_type)
            if element is None:
                raise DesktopError(f"No control named '{name}' in that window. Use inspect_ui to see what's there.")
            if not element.enabled:
                raise DesktopError(f"'{element.name}' is disabled right now.")
            raw = next(r for e, r in pairs if e is element)
            mod = self._mod
            for pattern_id, interface, method, action in (
                (10000, mod.IUIAutomationInvokePattern, "Invoke", "invoked"),
                (10015, mod.IUIAutomationTogglePattern, "Toggle", "toggled"),
                (10010, mod.IUIAutomationSelectionItemPattern, "Select", "selected"),
                (10005, mod.IUIAutomationExpandCollapsePattern, "Expand", "expanded"),
            ):
                try:
                    pattern = raw.GetCurrentPattern(pattern_id)
                except Exception:
                    pattern = None
                if pattern:
                    getattr(pattern.QueryInterface(interface), method)()
                    return {"element": element, "action": action}
            return {"element": element, "action": "needs_click"}

        return await self._call(run)
