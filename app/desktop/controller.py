"""DesktopController: the policy layer between desktop tools and the OS.

Tools (app/tools/desktop.py) only ever call this class. It:
  - runs every blocking backend call off the event loop (asyncio.to_thread)
    and serializes them behind one lock, so two tool calls can never
    interleave their keystrokes;
  - range-checks everything the LLM supplied (coordinates must be on a real
    monitor, typed text is length-capped, repeat counts are bounded, hotkeys
    are parsed against a fixed key table with a safety denylist);
  - resolves windows by title/app name across ALL virtual desktops (never
    Garvis's own window), brings a target that lives on another virtual
    desktop to the one the user is looking at, focuses it and VERIFIES it is
    the foreground window before any keystroke is sent -- input is refused,
    with the reason, rather than sent to whatever happens to have focus;
  - recovers focus: if Garvis's own window (e.g. the confirmation dialog
    the user just clicked) is in the foreground when keyboard input is
    about to be sent, the last external target window is re-focused first,
    so text never lands in Garvis itself;
  - checks that real input can reach the user at all (same Windows session
    as the console, normal input desktop rather than the lock screen/UAC);
  - logs every action -- but never the text being typed (it may be a
    password), only its length.

The backend is created lazily on first use via `backend_factory`, so merely
constructing the agent (tests, non-Windows machines) never loads user32.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from app.core.exceptions import DesktopError
from app.core.logging import get_logger, log_extra
from app.desktop.backend import DesktopBackend, MouseButton, Rect, WindowInfo, WindowState
from app.desktop.keys import key_to_vk, parse_hotkey

logger = get_logger("desktop")

_NORMALIZE_RE = re.compile(r"[^a-z0-9]")
_MAX_REPEAT = 25
_MAX_SCROLL_CLICKS = 50
_SETTLE_SECONDS = 0.05


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower())


def match_window(query: str, windows: list[WindowInfo]) -> WindowInfo | None:
    """Best window for a free-text query ("chrome", "WhatsApp", "Untitled -
    Notepad"). Ranking: exact title > exact app/process name > title
    substring > process-name substring; ties go to the higher window in
    z-order (the list is already topmost-first)."""
    target = _normalize(query)
    if not target:
        return None
    ranked: list[tuple[int, int, WindowInfo]] = []
    for index, window in enumerate(windows):
        title = _normalize(window.title)
        process = _normalize(window.process_name.removesuffix(".exe").removesuffix(".EXE"))
        if title == target:
            score = 0
        elif process == target or process == f"{target}root":  # e.g. WhatsApp.Root.exe
            score = 1
        elif target in title:
            score = 2
        elif target in process:
            score = 3
        else:
            continue
        ranked.append((score, index, window))
    return min(ranked, key=lambda r: (r[0], r[1]))[2] if ranked else None


class DesktopController:
    def __init__(
        self,
        backend_factory: Callable[[], DesktopBackend] | None,
        screenshot_dir: Path,
        max_type_chars: int = 2000,
        screenshot_keep: int = 20,
        own_pid: int | None = None,
        window_mover: Callable[[int], Awaitable[object]] | None = None,
    ) -> None:
        self._backend_factory = backend_factory
        self._backend: DesktopBackend | None = None
        self._screenshot_dir = screenshot_dir
        self._max_type_chars = max_type_chars
        self._screenshot_keep = screenshot_keep
        self._own_pid = own_pid if own_pid is not None else os.getpid()
        self._lock = asyncio.Lock()
        self._target_handle: int | None = None
        """Last external (non-Garvis) window the user/agent was working in --
        the window keyboard input is restored to after a confirmation dialog."""
        self._window_mover = window_mover
        """Moves a window to the virtual desktop the user is viewing
        (WorkspaceManager.move_window_to_current); None = can't move windows."""

    def attach_window_mover(self, mover: Callable[[int], Awaitable[object]] | None) -> None:
        self._window_mover = mover

    def attach_workspace(self, workspace) -> None:
        """The WorkspaceManager (Garvis's virtual desktop): lets the
        controller tell the user's desktop from Garvis's workspace, bring
        windows to the user, and launch apps into the workspace."""
        self._workspace = workspace
        self._window_mover = workspace.move_window_to_current

    async def window_location(self, handle: int) -> str:
        """ "user", "garvis_workspace", "other" or "unknown"."""
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            info = await self.window_info(handle)
            return "user" if info is not None and info.on_current_desktop else "unknown"
        return await workspace.window_location(handle)

    async def window_info(self, handle: int) -> WindowInfo | None:
        return await self._run(lambda backend: backend.window_info(handle))

    async def locate_windows(self, windows: list[WindowInfo]) -> dict[int, str]:
        """handle -> "user" / "garvis_workspace" / "other" / "unknown"."""
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            return {w.handle: "user" if w.on_current_desktop else "unknown" for w in windows}
        return await workspace.locate_windows([w.handle for w in windows])

    async def workspace_available(self) -> bool:
        workspace = getattr(self, "_workspace", None)
        return workspace is not None and await workspace.is_available()

    async def move_to_workspace(self, handle: int) -> None:
        """Move one window to Garvis's workspace without activating anything."""
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            raise DesktopError("Garvis's workspace desktop isn't available.")
        try:
            await workspace.move_window_to_garvis(handle)
        except Exception as exc:
            raise DesktopError(f"The window couldn't be moved to Garvis's workspace ({exc}).") from exc

    async def restore_foreground(self, handle: int) -> bool:
        """Give focus back to the user's window if something else took it."""
        current = await self.foreground_window()
        if current is not None and current.handle == handle:
            return False
        await self.focus_handle(handle)
        return True

    def is_own_window(self, window: WindowInfo) -> bool:
        """Garvis's own app window (its UI), as opposed to windows it drives."""
        return window.pid == self._own_pid

    # ---------------------------------------------------------- plumbing
    def _require_backend(self) -> DesktopBackend:
        if self._backend is None:
            if self._backend_factory is None:
                raise DesktopError("Desktop control is only supported on Windows.")
            self._backend = self._backend_factory()
        return self._backend

    async def _run(self, func: Callable, *args):
        backend = self._require_backend()
        try:
            return await asyncio.to_thread(func, backend, *args)
        except DesktopError:
            raise
        except (OSError, ValueError) as exc:
            raise DesktopError(str(exc)) from exc

    def _remember_target(self, backend: DesktopBackend) -> None:
        foreground = backend.foreground_window()
        if foreground is not None and foreground.pid != self._own_pid:
            self._target_handle = foreground.handle

    @staticmethod
    def _ensure_input_possible(backend: DesktopBackend) -> None:
        context = backend.input_context()
        if not context.interactive:
            raise DesktopError(context.reason or "Real input can't reach the user's desktop right now.")

    def _match(self, backend: DesktopBackend, query: str) -> WindowInfo | None:
        """Best match for `query` among application windows on every virtual
        desktop, preferring the desktop the user is looking at. Garvis's own
        windows are never a target."""
        windows = [w for w in backend.list_windows(include_other_desktops=True) if w.pid != self._own_pid]
        here = match_window(query, [w for w in windows if w.on_current_desktop])
        return here or match_window(query, windows)

    def _check_point(self, backend: DesktopBackend, x: int, y: int) -> None:
        screen = backend.virtual_screen()
        if not screen.contains(x, y):
            raise DesktopError(
                f"Point ({x}, {y}) is off-screen; the screen spans x {screen.left}..{screen.right - 1}, "
                f"y {screen.top}..{screen.bottom - 1}."
            )

    def _restore_focus_if_garvis(self, backend: DesktopBackend) -> None:
        """If Garvis's own UI holds focus (typically: the user just approved
        a confirmation dialog), hand focus back to the last target window."""
        foreground = backend.foreground_window()
        if foreground is not None and foreground.pid != self._own_pid:
            return
        if self._target_handle is None or backend.window_info(self._target_handle) is None:
            return
        if backend.focus_window(self._target_handle):
            log_extra(logger, 20, "desktop.focus_restored", handle=self._target_handle)
            time.sleep(_SETTLE_SECONDS)

    def _focus_verified(self, backend: DesktopBackend, window: WindowInfo) -> WindowInfo:
        """Focus `window` (which must be on the user's desktop) and confirm
        Windows really made it the foreground window."""
        backend.focus_window(window.handle)
        time.sleep(_SETTLE_SECONDS)
        foreground = backend.foreground_window()
        if foreground is None or foreground.handle != window.handle:
            got = f"'{foreground.title}' ({foreground.process_name})" if foreground else "no window"
            raise DesktopError(
                f"Windows didn't bring '{window.title}' to the front ({got} has focus instead), so nothing "
                "was sent to it. Click the window once, or retry."
            )
        self._target_handle = window.handle
        return backend.window_info(window.handle) or window

    async def _bring_here(self, window: WindowInfo, from_workspace: bool = False) -> tuple[WindowInfo, bool]:
        """Move a window that lives on another virtual desktop to the one the
        user is viewing. Returns (window, moved). A window on GARVIS's
        workspace is only brought over when the operation targets the user's
        desktop (`from_workspace`); otherwise that would silently re-route
        background work onto the user's screen."""
        if window.on_current_desktop:
            return window, False
        if not from_workspace and await self.window_location(window.handle) == "garvis_workspace":
            raise DesktopError(
                f"'{window.title}' is on Garvis's workspace desktop. Keyboard and mouse input only reach the "
                "desktop you're looking at, so it can't be used there without bringing it to your screen. Use "
                "click_ui_element or the browser tools for background work, or pass "
                "target_desktop='user_desktop' to bring it to your desktop."
            )
        if self._window_mover is None:
            raise DesktopError(
                f"'{window.title}' is on another virtual desktop, not the one you're looking at, and Garvis "
                "can't move windows between desktops here. Switch to that desktop (Win+Ctrl+Left/Right) or "
                "drag it over in Task View, then retry."
            )
        try:
            await self._window_mover(window.handle)
        except Exception as exc:
            raise DesktopError(
                f"'{window.title}' is on another virtual desktop and couldn't be moved here ({exc})."
            ) from exc
        await asyncio.sleep(0.2)
        refreshed = await self._run(lambda backend: backend.window_info(window.handle))
        if refreshed is None:
            raise DesktopError(f"'{window.title}' closed while it was being moved.")
        log_extra(logger, 20, "desktop.window_brought_here", title=refreshed.title, handle=refreshed.handle)
        return refreshed, True

    async def _activate(self, window: WindowInfo, from_workspace: bool = False) -> dict:
        """Bring here + focus + verify. Caller holds the lock."""
        window, moved = await self._bring_here(window, from_workspace)
        focused = await self._run(lambda backend: self._focus_verified(backend, window))
        return {"window": focused, "moved_from_other_desktop": moved}

    # ------------------------------------------------------------ queries
    async def screen_info(self) -> dict:
        def _do(backend: DesktopBackend) -> dict:
            screen = backend.virtual_screen()
            width, height = backend.primary_screen_size()
            x, y = backend.cursor_position()
            foreground = backend.foreground_window()
            return {
                "virtual_screen": screen.model_dump(),
                "primary_screen": {"width": width, "height": height},
                "cursor": {"x": x, "y": y},
                "foreground_window": foreground.model_dump() if foreground else None,
            }

        return await self._run(_do)

    async def list_windows(self, include_other_desktops: bool = False) -> list[WindowInfo]:
        return await self._run(lambda backend: backend.list_windows(include_other_desktops))

    async def foreground_window(self) -> WindowInfo | None:
        return await self._run(lambda backend: backend.foreground_window())

    def _active_window(self, backend: DesktopBackend) -> WindowInfo | None:
        """The window the user is working in: the foreground window if it is
        a real application window (not Garvis itself, and not a shell surface
        like the desktop or taskbar, which list_windows leaves out) -- else
        the last external target, else the topmost application window."""
        windows = [w for w in backend.list_windows() if w.pid != self._own_pid]
        foreground = backend.foreground_window()
        if foreground is not None and any(w.handle == foreground.handle for w in windows):
            return foreground
        if self._target_handle is not None:
            target = next((w for w in windows if w.handle == self._target_handle), None)
            if target is not None:
                return target
        return next((w for w in windows if not w.is_minimized), windows[0] if windows else None)

    def _resolve_window(self, backend: DesktopBackend, query: str | None) -> WindowInfo:
        if query:
            window = self._match(backend, query)
            if window is None:
                raise DesktopError(
                    f"No open application window matches '{query}' (Garvis's own window doesn't count). "
                    "Use list_windows to see what's open, or open_application to start it."
                )
            return window
        window = self._active_window(backend)
        if window is None:
            raise DesktopError("No application window is open.")
        return window

    async def resolve_window(self, query: str | None = None, desktop: str | None = None) -> WindowInfo:
        """`query` matched against open windows, or (None) the active window.
        `desktop` ("user_desktop" / "garvis_workspace") restricts the match
        to windows on that desktop -- an operation aimed at one desktop never
        silently lands on a window of the other."""
        if desktop is None:
            return await self._run(lambda backend: self._resolve_window(backend, query))
        windows = [w for w in await self.list_windows(include_other_desktops=True) if w.pid != self._own_pid]
        locations = await self.locate_windows(windows)
        wanted = "garvis_workspace" if desktop == "garvis_workspace" else "user"
        if desktop == "garvis_workspace" and windows and all(v == "unknown" for v in locations.values()):
            raise DesktopError("Garvis's workspace desktop isn't available, so there are no windows on it.")
        candidates = [
            w
            for w in windows
            if locations.get(w.handle) == wanted
            or (wanted == "user" and locations.get(w.handle) == "unknown" and w.on_current_desktop)
        ]
        where = "Garvis's workspace" if wanted == "garvis_workspace" else "your desktop"
        if query:
            window = match_window(query, candidates)
            if window is None:
                raise DesktopError(f"No open window on {where} matches '{query}'.")
            return window
        if wanted == "user":
            return await self._run(lambda backend: self._resolve_window(backend, None))
        if not candidates:
            raise DesktopError("There are no windows on Garvis's workspace.")
        return candidates[0]

    async def describe_context(self, max_apps: int = 8) -> str | None:
        """One line for the system prompt: the active window and the other
        open apps. Titles of other windows are left out on purpose."""

        def _do(backend: DesktopBackend) -> str | None:
            active = self._active_window(backend)
            others: list[str] = []
            for window in backend.list_windows():
                if window.pid == self._own_pid or (active is not None and window.handle == active.handle):
                    continue
                app = window.process_name.removesuffix(".exe").removesuffix(".EXE") or window.title
                if app and app not in others:
                    others.append(app)
            if active is None and not others:
                return None
            parts = []
            if active is not None:
                app = active.process_name or "unknown app"
                state = " (minimized)" if active.is_minimized else ""
                parts.append(f"active window '{active.title[:80]}' ({app}){state}")
            if others:
                parts.append("other open apps: " + ", ".join(others[:max_apps]))
            return "; ".join(parts) + "."

        return await self._run(_do)

    # ------------------------------------------------------------ windows
    async def window_action(self, query: str | None, action: WindowState | str) -> dict:
        """Minimize / maximize / restore / close a window and report the
        state afterwards (verified, not assumed)."""

        def _do(backend: DesktopBackend) -> dict:
            window = self._resolve_window(backend, query)
            if window.pid == self._own_pid:
                raise DesktopError("That's Garvis's own window; use the Garvis UI for it.")
            if action == "close":
                backend.close_window(window.handle)
                time.sleep(0.5)
                still_open = backend.window_info(window.handle)
                return {
                    "window": window.title,
                    "app": window.process_name,
                    "action": "close",
                    "closed": still_open is None,
                    "note": None
                    if still_open is None
                    else "The window is still open -- it may be asking to save changes. Use inspect_ui to see the prompt.",
                }
            state = WindowState(action)
            if not window.on_current_desktop and state != WindowState.MINIMIZE:
                # SW_RESTORE/SW_MAXIMIZE activate the window, and activating
                # a window on another virtual desktop switches the user's view.
                if state == WindowState.MAXIMIZE:
                    raise DesktopError(
                        f"'{window.title}' is on another virtual desktop; maximizing it there would switch your "
                        "view to that desktop. Bring it here first (focus_window with target_desktop='user_desktop')."
                    )
                state = WindowState.SHOW_INACTIVE
            backend.set_window_state(window.handle, state)
            time.sleep(_SETTLE_SECONDS * 4)
            after = backend.window_info(window.handle) or window
            return {
                "window": after.title,
                "app": after.process_name,
                "action": str(action),
                "minimized": after.is_minimized,
                "on_current_desktop": after.on_current_desktop,
            }

        async with self._lock:
            result = await self._run(_do)
        log_extra(logger, 20, "desktop.window_action", action=str(action), window=result["window"])
        return result

    async def wait_for_window(self, query: str, timeout_seconds: float = 15.0, poll_seconds: float = 0.5) -> WindowInfo:
        """Poll until a window matching `query` exists -- e.g. right after
        open_application, to verify the app actually started."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            windows = await self._run(lambda backend: backend.list_windows(True))
            window = match_window(query, [w for w in windows if w.pid != self._own_pid])
            if window is not None:
                return window
            if time.monotonic() >= deadline:
                raise DesktopError(f"No window matching '{query}' appeared within {timeout_seconds:.0f} s.")
            await asyncio.sleep(poll_seconds)

    async def launch_in_workspace(
        self, app: str, launch: Callable[[], None], timeout_seconds: float = 15.0, reuse_after_seconds: float = 3.0
    ) -> dict:
        """Start an application on Garvis's workspace desktop WITHOUT taking
        the user's focus: `launch` should ask for a minimized, non-activated
        first window (SW_SHOWMINNOACTIVE); its window is then moved to the
        workspace and restored there without activation. If the app grabs
        focus anyway, focus is handed back to the window the user was using.
        Verified: the window is on the workspace, the user's foreground is
        unchanged. Raises DesktopError if the workspace can't be used."""
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            raise DesktopError(
                "Garvis's workspace desktop isn't available, so the app was not started in the background."
            )
        try:
            await workspace.ensure()
        except Exception as exc:
            raise DesktopError(f"Garvis's workspace desktop isn't available ({exc}); the app was not started.") from exc
        user_foreground = await self._run(lambda backend: backend.foreground_window())
        before = {w.handle for w in await self._run(lambda backend: backend.list_windows(True))}
        started = time.monotonic()
        await asyncio.to_thread(launch)
        window: WindowInfo | None = None
        new_window = False
        while window is None:
            windows = [w for w in await self._run(lambda backend: backend.list_windows(True)) if w.pid != self._own_pid]
            fresh = [w for w in windows if w.handle not in before]
            window = match_window(app, fresh) or (fresh[0] if len(fresh) == 1 else None)
            if window is not None:
                new_window = True
                break
            elapsed = time.monotonic() - started
            if elapsed >= reuse_after_seconds:
                window = match_window(app, windows)
                if window is not None:
                    break
            if elapsed >= timeout_seconds:
                raise DesktopError(f"'{app}' was started, but no window of it appeared within {timeout_seconds:.0f} s.")
            await asyncio.sleep(0.2)
        if not new_window and await workspace.window_location(window.handle) != "garvis_workspace":
            # A single-instance app that reused the user's own window: taking
            # that away from them would move an unrelated window.
            raise DesktopError(
                f"'{app}' only reused its existing window '{window.title}', which is not on Garvis's workspace; "
                "it was left where it is. Use it on your desktop (target_desktop='user_desktop') instead."
            )
        try:
            await workspace.move_window_to_garvis(window.handle)
        except Exception as exc:
            raise DesktopError(
                f"'{window.title}' couldn't be moved to Garvis's workspace ({exc}); it is still on your desktop, "
                "minimized."
            ) from exc
        placed = await self._run(lambda backend: backend.window_info(window.handle))
        if placed is not None and placed.is_minimized:
            await self._run(lambda backend: backend.set_window_state(window.handle, WindowState.SHOW_INACTIVE))
        focus_returned = False
        current = await self._run(lambda backend: backend.foreground_window())
        if user_foreground is not None and (current is None or current.handle != user_foreground.handle):
            await self.focus_handle(user_foreground.handle)
            focus_returned = True
        location = await workspace.window_location(window.handle)
        if location != "garvis_workspace":
            raise DesktopError(f"'{window.title}' did not stay on Garvis's workspace (it is on: {location}).")
        after = await self._run(lambda backend: backend.foreground_window())
        log_extra(
            logger, 20, "desktop.app_started_in_workspace", app=app, title=window.title, focus_returned=focus_returned
        )
        return {
            "window": window.title,
            "app": window.process_name,
            "handle": window.handle,
            "new_window": new_window,
            "target_desktop": "garvis_workspace",
            "on_workspace": True,
            "user_focus_unchanged": user_foreground is None
            or (after is not None and after.handle == user_foreground.handle),
            "focus_had_to_be_returned": focus_returned,
        }

    async def show_window(self, handle: int) -> dict:
        """Put a specific window in front of the user: bring it to the
        desktop they're viewing (from any desktop -- the caller has decided
        the user's desktop is the target), focus it, verify."""
        window = await self.window_info(handle)
        if window is None:
            raise DesktopError("That window closed before it could be shown.")
        async with self._lock:
            await self._run(self._ensure_input_possible)
            activated = await self._activate(window, from_workspace=True)
        focused: WindowInfo = activated["window"]
        return {
            "window": focused.title,
            "app": focused.process_name,
            "focused": True,
            "on_user_desktop": focused.on_current_desktop,
            "moved_from_other_desktop": activated["moved_from_other_desktop"],
        }

    async def focus_window(self, query: str, bring_from_workspace: bool = False) -> WindowInfo:
        async with self._lock:
            await self._run(self._ensure_input_possible)
            target = await self._run(lambda backend: self._resolve_window(backend, query))
            window = (await self._activate(target, bring_from_workspace))["window"]
        log_extra(logger, 20, "desktop.focus_window", query=query, title=window.title)
        return window

    async def launch_and_activate(
        self, app: str, launch: Callable[[], None], timeout_seconds: float = 15.0, reuse_after_seconds: float = 3.0
    ) -> dict:
        """Start an application and identify ITS window: a window that
        appeared after the launch and matches the app (or the only new one),
        else -- after `reuse_after_seconds`, for single-instance apps that
        just reuse a window -- an existing window of that app. The window is
        brought to the user's desktop, focused and verified. Raises
        DesktopError when no window can be identified."""
        before = {w.handle for w in await self._run(lambda backend: backend.list_windows(True))}
        started = time.monotonic()
        await asyncio.to_thread(launch)
        window: WindowInfo | None = None
        new_window = False
        while True:
            windows = [w for w in await self._run(lambda backend: backend.list_windows(True)) if w.pid != self._own_pid]
            fresh = [w for w in windows if w.handle not in before]
            window = match_window(app, fresh) or (fresh[0] if len(fresh) == 1 else None)
            if window is not None:
                new_window = True
                break
            elapsed = time.monotonic() - started
            if elapsed >= reuse_after_seconds:
                window = match_window(app, windows)
                if window is not None:
                    break
            if elapsed >= timeout_seconds:
                raise DesktopError(
                    f"'{app}' was started, but no window of it appeared within {timeout_seconds:.0f} s "
                    "(it may run in the tray or still be loading). Check list_windows before typing into it."
                )
            await asyncio.sleep(0.25)
        async with self._lock:
            await self._run(self._ensure_input_possible)
            activated = await self._activate(window, from_workspace=True)
        focused: WindowInfo = activated["window"]
        log_extra(
            logger,
            20,
            "desktop.app_window_ready",
            app=app,
            title=focused.title,
            new_window=new_window,
            moved=activated["moved_from_other_desktop"],
        )
        return {
            "window": focused.title,
            "app": focused.process_name,
            "handle": focused.handle,
            "new_window": new_window,
            "moved_from_other_desktop": activated["moved_from_other_desktop"],
            "focused": True,
            "target_desktop": "user_desktop",
            "on_user_desktop": focused.on_current_desktop,
        }

    async def switch_to_previous_window(self) -> WindowInfo:
        """Alt+Tab equivalent, done through the window manager instead of a
        synthesized key chord: focus the next non-Garvis window in z-order."""

        def _do(backend: DesktopBackend) -> WindowInfo:
            foreground = backend.foreground_window()
            for window in backend.list_windows():
                if window.pid == self._own_pid:
                    continue
                if foreground is not None and window.handle == foreground.handle:
                    continue
                if backend.focus_window(window.handle):
                    self._target_handle = window.handle
                    return window
            raise DesktopError("There is no other window to switch to on this desktop.")

        async with self._lock:
            window = await self._run(_do)
        log_extra(logger, 20, "desktop.switch_window", title=window.title)
        return window

    async def focus_handle(self, handle: int) -> bool:
        """Focus a specific window handle (internal use, e.g. handing focus
        back to the user's window after Garvis launched a browser)."""
        async with self._lock:
            return await self._run(lambda backend: backend.focus_window(handle))

    async def place_window(self, handle: int, rect: Rect) -> None:
        await self._run(lambda backend: backend.set_window_rect(handle, rect))

    # -------------------------------------------------------------- mouse
    async def move_mouse(self, x: int, y: int) -> None:
        def _do(backend: DesktopBackend) -> None:
            self._check_point(backend, x, y)
            backend.move_mouse(x, y)

        async with self._lock:
            await self._run(_do)
        log_extra(logger, 20, "desktop.mouse_move", x=x, y=y)

    async def click(self, x: int | None, y: int | None, button: MouseButton = MouseButton.LEFT, count: int = 1) -> dict:
        if (x is None) != (y is None):
            raise DesktopError("Give both x and y, or neither (to click at the current cursor position).")

        def _do(backend: DesktopBackend) -> dict:
            self._ensure_input_possible(backend)
            if x is not None and y is not None:
                self._check_point(backend, x, y)
                backend.move_mouse(x, y)
                time.sleep(_SETTLE_SECONDS)
            backend.click(button, count)
            time.sleep(_SETTLE_SECONDS)
            self._remember_target(backend)
            cx, cy = backend.cursor_position()
            foreground = backend.foreground_window()
            return {
                "x": cx,
                "y": cy,
                "foreground_window": foreground.title if foreground else None,
                "foreground_app": foreground.process_name if foreground else None,
            }

        async with self._lock:
            result = await self._run(_do)
        log_extra(logger, 20, "desktop.mouse_click", button=button.value, count=count, x=result["x"], y=result["y"])
        return result

    async def scroll(self, clicks: int, x: int | None = None, y: int | None = None, horizontal: bool = False) -> None:
        if clicks == 0 or abs(clicks) > _MAX_SCROLL_CLICKS:
            raise DesktopError(f"Scroll amount must be between 1 and {_MAX_SCROLL_CLICKS} notches (either sign).")
        if (x is None) != (y is None):
            raise DesktopError("Give both x and y, or neither (to scroll where the cursor is).")

        def _do(backend: DesktopBackend) -> None:
            if x is not None and y is not None:
                self._check_point(backend, x, y)
                backend.move_mouse(x, y)
                time.sleep(_SETTLE_SECONDS)
            backend.scroll(clicks, horizontal=horizontal)

        async with self._lock:
            await self._run(_do)
        log_extra(logger, 20, "desktop.scroll", clicks=clicks, horizontal=horizontal)

    # ----------------------------------------------------------- keyboard
    def _focused_target(self, backend: DesktopBackend) -> WindowInfo:
        """No window named: the window that has keyboard focus -- after
        handing focus back from Garvis's own window -- provided it's a real
        application window on the user's desktop."""
        self._ensure_input_possible(backend)
        self._restore_focus_if_garvis(backend)
        target = backend.foreground_window()
        if target is None:
            raise DesktopError("No window has keyboard focus. Name the window to type into (window=...).")
        if target.pid == self._own_pid:
            raise DesktopError(
                "Garvis's own window has keyboard focus and no other target window is known -- "
                "name the window to type into (window=...), or focus_window first."
            )
        if not target.on_current_desktop:
            raise DesktopError(
                f"The window with keyboard focus ('{target.title}') isn't on the desktop you're looking at. "
                "Name the window to type into (window=...) so Garvis can bring it here first."
            )
        return target

    async def _keyboard_target(self, window: str | None, bring_from_workspace: bool = False) -> WindowInfo:
        """Caller holds the lock."""
        if not window:
            return await self._run(self._focused_target)
        await self._run(self._ensure_input_possible)
        target = await self._run(lambda backend: self._resolve_window(backend, window))
        return (await self._activate(target, bring_from_workspace))["window"]

    @staticmethod
    def _still_focused(backend: DesktopBackend, target: WindowInfo) -> None:
        """Last check right before keys are sent (the user may have clicked elsewhere)."""
        foreground = backend.foreground_window()
        if foreground is None or foreground.handle != target.handle:
            now = f"'{foreground.title}'" if foreground else "no window"
            raise DesktopError(f"Focus moved from '{target.title}' to {now} before typing; nothing was typed.")

    @staticmethod
    def _target_summary(target: WindowInfo) -> dict:
        return {"window": target.title, "app": target.process_name, "handle": target.handle, "target_verified": True}

    async def type_text(self, text: str, window: str | None = None, bring_from_workspace: bool = False) -> dict:
        if not text:
            raise DesktopError("Nothing to type.")
        if len(text) > self._max_type_chars:
            raise DesktopError(f"Text is too long to type ({len(text)} > {self._max_type_chars} characters).")

        async with self._lock:
            target = await self._keyboard_target(window, bring_from_workspace)

            def _do(backend: DesktopBackend) -> dict:
                self._still_focused(backend, target)
                backend.type_text(text)
                self._remember_target(backend)
                return {"typed_chars": len(text), **self._target_summary(target)}

            result = await self._run(_do)
        log_extra(logger, 20, "desktop.keyboard_type", length=len(text), window=result["window"])
        return result

    async def press_key(
        self, key: str, repeat: int = 1, window: str | None = None, bring_from_workspace: bool = False
    ) -> dict:
        if not 1 <= repeat <= _MAX_REPEAT:
            raise DesktopError(f"repeat must be between 1 and {_MAX_REPEAT}.")
        try:
            vk = key_to_vk(key)
        except ValueError as exc:
            raise DesktopError(str(exc)) from exc

        async with self._lock:
            target = await self._keyboard_target(window, bring_from_workspace)

            def _do(backend: DesktopBackend) -> dict:
                self._still_focused(backend, target)
                backend.press_keys([vk], repeat=repeat)
                self._remember_target(backend)
                return {"pressed": key, "repeat": repeat, **self._target_summary(target)}

            result = await self._run(_do)
        log_extra(logger, 20, "desktop.keyboard_press", key=key, repeat=repeat, window=result["window"])
        return result

    async def hotkey(
        self, keys: str | list[str], window: str | None = None, bring_from_workspace: bool = False
    ) -> dict:
        try:
            parsed = parse_hotkey(keys)
        except ValueError as exc:
            raise DesktopError(str(exc)) from exc
        combo = "+".join(name for name, _ in parsed)

        async with self._lock:
            target = await self._keyboard_target(window, bring_from_workspace)

            def _do(backend: DesktopBackend) -> dict:
                self._still_focused(backend, target)
                backend.press_keys([vk for _, vk in parsed])
                time.sleep(_SETTLE_SECONDS)
                self._remember_target(backend)
                return {"pressed": combo, **self._target_summary(target)}

            result = await self._run(_do)
        log_extra(logger, 20, "desktop.keyboard_hotkey", combo=combo, window=result["window"])
        return result

    # ------------------------------------------------------------- screen
    def _prune_screenshots(self) -> None:
        shots = sorted(self._screenshot_dir.glob("screen-*.png"))
        for old in shots[: max(0, len(shots) - self._screenshot_keep)]:
            try:
                old.unlink()
            except OSError:
                pass

    async def capture_screen(self, window: str | None = None, area: Rect | None = None) -> dict:
        """Whole screen, one window, or an explicit screen `area` (clipped to
        the screen)."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self._screenshot_dir / f"screen-{stamp}.png"

        def _do(backend: DesktopBackend) -> dict:
            region: Rect | None = None
            captured_window: WindowInfo | None = None
            if area is not None:
                screen = backend.virtual_screen()
                region = Rect(
                    left=max(area.left, screen.left),
                    top=max(area.top, screen.top),
                    right=min(area.right, screen.right),
                    bottom=min(area.bottom, screen.bottom),
                )
                if region.width <= 0 or region.height <= 0:
                    raise DesktopError("That region is outside the screen.")
            elif window:
                captured_window = self._match(backend, window)
                if captured_window is None or captured_window.rect is None:
                    raise DesktopError(f"No open window matches '{window}' to capture.")
                if not captured_window.on_current_desktop:
                    raise DesktopError(
                        f"'{captured_window.title}' is on another virtual desktop, so a screenshot of its area "
                        "would show your current desktop instead. Use focus_window to bring it here first."
                    )
                if captured_window.is_minimized:
                    raise DesktopError(f"'{captured_window.title}' is minimized; focus it first.")
                screen = backend.virtual_screen()
                r = captured_window.rect
                region = Rect(
                    left=max(r.left, screen.left),
                    top=max(r.top, screen.top),
                    right=min(r.right, screen.right),
                    bottom=min(r.bottom, screen.bottom),
                )
            width, height = backend.capture_screen(path, region)
            self._prune_screenshots()
            foreground = backend.foreground_window()
            return {
                "path": str(path),
                "width": width,
                "height": height,
                "region": region.model_dump() if region else None,
                "window": captured_window.title if captured_window else None,
                "foreground_window": foreground.title if foreground else None,
            }

        result = await self._run(_do)
        log_extra(
            logger, 20, "desktop.screen_capture", path=result["path"], width=result["width"], height=result["height"]
        )
        return result
