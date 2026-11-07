"""Composition helpers for Desktop Control, mirroring app/browser/build.py:
everything configurable comes from Settings, and the Win32 backend module
is imported lazily (only when a desktop tool first runs on Windows)."""

from __future__ import annotations

import sys

from app.core.config import Settings
from app.desktop.backend import DesktopBackend
from app.desktop.controller import DesktopController
from app.desktop.workspace import PyvdaVirtualDesktopBackend, VirtualDesktopBackend, WorkspaceManager


def _win32_backend_factory() -> DesktopBackend:
    from app.desktop.win32 import Win32DesktopBackend

    return Win32DesktopBackend()


def build_desktop_controller(settings: Settings) -> DesktopController:
    return DesktopController(
        backend_factory=_win32_backend_factory if sys.platform == "win32" else None,
        screenshot_dir=settings.desktop_screenshot_dir,
        max_type_chars=settings.desktop_max_type_chars,
        screenshot_keep=settings.desktop_screenshot_keep,
    )


def _pyvda_backend_factory() -> VirtualDesktopBackend:
    return PyvdaVirtualDesktopBackend()


def build_workspace_manager(settings: Settings) -> WorkspaceManager:
    return WorkspaceManager(
        backend_factory=_pyvda_backend_factory if sys.platform == "win32" else None,
        name=settings.workspace_name,
    )
