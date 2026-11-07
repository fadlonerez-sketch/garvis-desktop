"""Tests for the desktop-control tools (app/tools/desktop.py): risk levels,
confirmation policy, schema validation, result shapes, and agent wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.build import build_extra_tools, build_registry
from app.core.config import Settings
from app.desktop.controller import DesktopController
from app.security.permissions import PermissionChecker, PermissionDecision
from app.tools.base import RiskLevel
from app.tools.desktop import (
    KeyboardHotkeyTool,
    KeyboardTypeTool,
    MouseClickTool,
    ScrollInput,
    ScrollTool,
    build_desktop_tools,
)
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend

_INPUT_TOOLS = {
    "mouse_click",
    "mouse_double_click",
    "mouse_right_click",
    "keyboard_type",
    "keyboard_press",
    "keyboard_hotkey",
    "click_ui_element",
}
_SAFE_TOOLS = {
    "screen_info",
    "list_windows",
    "screen_capture",
    "focus_window",
    "switch_window",
    "mouse_move",
    "scroll",
    "read_screen_text",
    "inspect_ui",
    "window_action",
    "wait_for_window",
}


@pytest.fixture
def backend():
    return FakeDesktopBackend()


@pytest.fixture
def controller(backend, tmp_path):
    return DesktopController(backend_factory=lambda: backend, screenshot_dir=tmp_path, own_pid=GARVIS_PID)


def test_tool_names_are_the_structured_interface(controller):
    names = {t.name for t in build_desktop_tools(controller)}
    assert names == _INPUT_TOOLS | _SAFE_TOOLS


def test_input_tools_are_medium_risk_but_routine_input_is_not_blanket_confirmed(controller):
    checker = PermissionChecker()
    for tool in build_desktop_tools(controller):
        decision = checker.evaluate(tool).decision
        assert decision == PermissionDecision.ALLOW, tool.name  # per-call guards decide (CommitGuard)
        expected = RiskLevel.MEDIUM if tool.name in _INPUT_TOOLS else RiskLevel.LOW
        assert tool.risk_level == expected, tool.name


def test_confirm_every_input_action_can_be_turned_back_on_per_instance(controller):
    tools = build_desktop_tools(controller, confirm_input_actions=True)
    checker = PermissionChecker()
    for tool in tools:
        expected = PermissionDecision.REQUIRE_CONFIRMATION if tool.name in _INPUT_TOOLS else PermissionDecision.ALLOW
        assert checker.evaluate(tool).decision == expected, tool.name
    # A freshly built default tool is unaffected (instance attribute, not class).
    assert KeyboardTypeTool(controller).requires_confirmation is False


async def test_click_tool_executes_and_reports(controller, backend):
    tool = MouseClickTool(controller)
    result = await tool.execute(tool.validate_arguments({"x": 50, "y": 60}))
    assert result.success and result.data["x"] == 50


async def test_tool_failure_is_reported_not_raised(controller):
    tool = MouseClickTool(controller)
    result = await tool.execute(tool.validate_arguments({"x": 99999, "y": 1}))
    assert not result.success and "off-screen" in result.error


async def test_hotkey_tool_reports_blocked_combo(controller, backend):
    tool = KeyboardHotkeyTool(controller)
    result = await tool.execute(tool.validate_arguments({"keys": "win+r"}))
    assert not result.success and "blocked" in result.error
    assert backend.calls == []


async def test_scroll_direction_mapping(controller, backend):
    tool = ScrollTool(controller)
    await tool.execute(tool.validate_arguments({"direction": "down", "amount": 4}))
    await tool.execute(tool.validate_arguments({"direction": "right"}))
    assert backend.calls == [("scroll", -4, False), ("scroll", 3, True)]


def test_scroll_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        ScrollInput(direction="sideways")
    with pytest.raises(ValidationError):
        ScrollInput(direction="up", amount=0)


def test_desktop_tools_are_registered_through_the_existing_registry(tmp_path):
    settings = Settings(_env_file=None, desktop_screenshot_dir=tmp_path)
    extra = build_extra_tools(settings)
    registry = build_registry(object(), object(), object(), extra_tools=extra)  # managers unused here
    names = {t.name for t in registry.all_tools()}
    assert _INPUT_TOOLS | _SAFE_TOOLS <= names
    assert "calculator" in names and "open_application" in names


def test_desktop_tools_can_be_disabled(tmp_path):
    settings = Settings(_env_file=None, desktop_control_enabled=False, desktop_screenshot_dir=tmp_path)
    assert not {t.name for t in build_extra_tools(settings)} & (_INPUT_TOOLS | _SAFE_TOOLS)


def test_confirm_setting_flows_into_built_tools(tmp_path):
    settings = Settings(_env_file=None, desktop_confirm_input_actions=False, desktop_screenshot_dir=tmp_path)
    typing = next(t for t in build_extra_tools(settings) if t.name == "keyboard_type")
    assert typing.requires_confirmation is False
