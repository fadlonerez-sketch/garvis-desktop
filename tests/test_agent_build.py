"""Tests for the composition root (app/agent/build.py) with the Phase 7
browser tools and Phase 9 Gmail tools wired in."""
from __future__ import annotations

from app.agent.build import build_registry
from app.agent.memory import SQLiteMemoryStore
from app.browser.manager import BrowserManager
from app.gmail.manager import GmailManager


def _build_registry(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "mem.db")
    browser_manager = BrowserManager()
    gmail_manager = GmailManager(
        client_secret_path=tmp_path / "gmail_client_secret.json",
        token_path=tmp_path / "gmail_token.json",
    )
    return build_registry(memory_store, browser_manager, gmail_manager), browser_manager, gmail_manager


def test_build_registry_includes_all_browser_tools(tmp_path):
    registry, _, _ = _build_registry(tmp_path)

    names = {tool.name for tool in registry.all_tools()}
    assert {
        "browser_navigate",
        "browser_read_page",
        "browser_click",
        "browser_type",
        "browser_press_key",
        "browser_close_page",
        "browser_submit_form",
    }.issubset(names)


def test_build_registry_browser_tools_share_the_same_manager(tmp_path):
    registry, browser_manager, _ = _build_registry(tmp_path)

    navigate_tool = registry.get("browser_navigate")
    read_tool = registry.get("browser_read_page")
    assert navigate_tool._manager is browser_manager
    assert read_tool._manager is browser_manager


def test_build_registry_includes_all_gmail_tools(tmp_path):
    registry, _, _ = _build_registry(tmp_path)

    names = {tool.name for tool in registry.all_tools()}
    assert {
        "gmail_list_recent",
        "gmail_search",
        "gmail_read",
        "gmail_draft",
        "gmail_send",
    }.issubset(names)


def test_build_registry_gmail_tools_share_the_same_manager(tmp_path):
    registry, _, gmail_manager = _build_registry(tmp_path)

    search_tool = registry.get("gmail_search")
    send_tool = registry.get("gmail_send")
    assert search_tool._manager is gmail_manager
    assert send_tool._manager is gmail_manager


def test_build_registry_still_includes_existing_tools(tmp_path):
    """Regression guard: adding Phase 7/9 tools must not drop any pre-existing tool."""
    registry, _, _ = _build_registry(tmp_path)

    names = {tool.name for tool in registry.all_tools()}
    assert {
        "calculator",
        "get_current_datetime",
        "open_application",
        "open_website",
        "web_search",
        "fetch_webpage_text",
        "list_directory",
        "read_text_file",
        "remember_fact",
        "recall_facts",
    }.issubset(names)
