"""Composition root: wires providers, tools, and the agent together.

Kept in one place so the UI (and tests) construct a fully-configured
`Agent` without duplicating wiring logic or depending on Qt.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.browser.build import build_browser_manager
from app.browser.manager import BrowserManager
from app.core.config import Settings, get_settings
from app.core.events import EventBus
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger, log_extra
from app.browser.research import WebResearcher
from app.desktop.browser_placement import OFFSCREEN_LAUNCH_ARGS, BrowserWindowPlacer
from app.desktop.build import build_desktop_controller, build_workspace_manager
from app.desktop.controller import DesktopController
from app.desktop.routing import DesktopRouting
from app.desktop.workspace import WorkspaceManager
from app.gmail.build import build_gmail_manager
from app.gmail.manager import GmailManager
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.registry import REQUIRED_MODEL, ModelEntry, ModelTier, enforce_single_model, load_model_registry
from app.llm.router import PoolMember, TieredLLMRouter
from app.llm.usage import BudgetGuard, BudgetLimits, UsageLedger
from app.tools.applications import OpenApplicationTool
from app.tools.browser import (
    ClickTool,
    ClosePageTool,
    NavigateTool,
    PressKeyTool,
    ReadPageTool,
    SubmitFormTool,
    TypeTextTool,
)
from app.tools.calculator import CalculatorTool
from app.tools.base import Tool
from app.tools.datetime_tool import DateTimeTool
from app.desktop.ocr import TesseractOcr
from app.desktop.uia import ComUiAutomation
from app.tools.desktop import DEFAULT_COMMIT_APPS, build_desktop_tools
from app.tools.filesystem import (
    EditTextFileTool,
    FindFilesTool,
    ListDirectoryTool,
    ReadTextFileTool,
    WriteTextFileTool,
)
from app.system.info import SystemInfo
from app.tools.applications import list_start_apps
from app.tools.system import (
    ListInstalledAppsTool,
    ListProcessesTool,
    RecentSystemErrorsTool,
    RunPowerShellTool,
    SystemStatusTool,
)
from app.tools.gmail import (
    DraftEmailTool,
    ListRecentEmailsTool,
    ReadEmailTool,
    SearchEmailsTool,
    SendEmailTool,
)
from app.tools.memory import RecallFactsTool, RememberFactTool
from app.tools.open_path import OpenPathTool
from app.tools.open_website import OpenWebsiteTool
from app.tools.registry import ToolRegistry
from app.tools.research import WebResearchTool
from app.tools.search import WebSearchTool
from app.tools.webpage import FetchWebpageTool
from app.tools.workspace import SwitchWorkspaceTool


logger = get_logger("agent.build")


def build_registry(
    memory_store: SQLiteMemoryStore,
    browser_manager: BrowserManager,
    gmail_manager: GmailManager,
    extra_tools: Iterable[Tool] = (),
    desktop: DesktopController | None = None,
    routing: DesktopRouting | None = None,
    workspace_available: bool = True,
) -> ToolRegistry:
    """`extra_tools` carries the settings-dependent tool groups built in
    build_agent() (desktop control, workspace, research) so this function's
    original signature -- and every caller of it -- stays unchanged.
    `routing` is the shared per-request desktop routing state
    (app/desktop/routing.py) every window-opening tool resolves through."""
    registry = ToolRegistry()
    routing = routing or DesktopRouting()
    for tool in (
        CalculatorTool(),
        DateTimeTool(),
        OpenApplicationTool(desktop=desktop, routing=routing),
        OpenWebsiteTool(desktop=desktop if sys.platform == "win32" else None, routing=routing),
        OpenPathTool(),
        WebSearchTool(),
        FetchWebpageTool(),
        ListDirectoryTool(),
        ReadTextFileTool(),
        RememberFactTool(memory_store),
        RecallFactsTool(memory_store),
        NavigateTool(browser_manager, routing, placer=getattr(browser_manager, "page_hook", None),
                     workspace_available=workspace_available),
        ReadPageTool(browser_manager),
        ClickTool(browser_manager),
        TypeTextTool(browser_manager),
        PressKeyTool(browser_manager),
        ClosePageTool(browser_manager),
        SubmitFormTool(browser_manager),
        ListRecentEmailsTool(gmail_manager),
        SearchEmailsTool(gmail_manager),
        ReadEmailTool(gmail_manager),
        DraftEmailTool(gmail_manager),
        SendEmailTool(gmail_manager),
        *extra_tools,
    ):
        registry.register(tool)
    return registry


def build_extra_tools(
    settings: Settings,
    desktop: DesktopController | None = None,
    workspace: WorkspaceManager | None = None,
    browser_manager: BrowserManager | None = None,
    routing: DesktopRouting | None = None,
) -> list[Tool]:
    """Desktop control, the Garvis workspace, and browser research. Pass the
    shared instances from build_agent(); anything omitted is built fresh
    (tests), and research is only offered when a browser is supplied."""
    desktop = desktop or build_desktop_controller(settings)
    routing = routing or DesktopRouting()
    tools: list[Tool] = []
    if settings.desktop_control_enabled:
        on_windows = sys.platform == "win32"
        tools += build_desktop_tools(
            desktop,
            settings.desktop_confirm_input_actions,
            ocr=TesseractOcr(settings.ocr_tesseract_path or None, settings.ocr_languages, settings.ocr_model),
            uia=ComUiAutomation() if on_windows else None,
            commit_apps=settings.desktop_commit_guard_apps or DEFAULT_COMMIT_APPS,
            routing=routing,
        )
    info = SystemInfo(start_apps=list_start_apps if sys.platform == "win32" else None)
    tools += [
        ListProcessesTool(info),
        SystemStatusTool(info),
        RecentSystemErrorsTool(info),
        ListInstalledAppsTool(info),
        RunPowerShellTool(desktop=desktop if sys.platform == "win32" else None),
        FindFilesTool(backup_dir=settings.file_backup_dir),
        WriteTextFileTool(backup_dir=settings.file_backup_dir),
        EditTextFileTool(backup_dir=settings.file_backup_dir),
    ]
    if settings.workspace_enabled:
        workspace = workspace or build_workspace_manager(settings)
        tools.append(SwitchWorkspaceTool(workspace))
    else:
        workspace = None
    if browser_manager is not None:
        researcher = WebResearcher(
            browser_manager,
            workspace=workspace,
            engine=settings.research_search_engine,
            max_sources=settings.research_max_sources,
            source_chars=settings.research_source_chars,
        )
        tools.append(WebResearchTool(researcher, browser=browser_manager, routing=routing,
                                     placer=browser_manager.page_hook))
    return tools


def build_browser_stack(settings: Settings) -> tuple[DesktopController, WorkspaceManager | None, BrowserManager]:
    """The browser, plus (when the workspace is enabled) the placer that puts
    each of its windows on its target desktop -- the user's, or the Garvis
    workspace -- launched off-screen so nothing flashes on the user's
    monitor before it's placed."""
    desktop = build_desktop_controller(settings)
    if settings.workspace_enabled:
        workspace = build_workspace_manager(settings)
        placer = BrowserWindowPlacer(workspace, desktop)
        browser = build_browser_manager(settings, page_hook=placer, launch_args=OFFSCREEN_LAUNCH_ARGS)
        # Lets the controller tell the user's desktop from the workspace,
        # bring windows to the user, and launch apps into the workspace.
        desktop.attach_workspace(workspace)
        return desktop, workspace, browser
    return desktop, None, build_browser_manager(settings)


def _build_haiku(settings: Settings, entry: ModelEntry) -> AnthropicProvider:
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        model=entry.model,
        base_url=settings.anthropic_base_url,
        timeout_seconds=settings.anthropic_request_timeout_seconds,
        reasoning_effort=entry.reasoning_effort,
        thinking=entry.thinking,
        max_output_tokens=entry.max_output_tokens,
        prompt_cache=entry.prompt_cache,
        verify_served_model=entry.verify_served_model,
        pricing=entry.pricing,
    )


def _build_tiered_router(settings: Settings) -> TieredLLMRouter:
    """One AnthropicProvider per tier, Claude Haiku 5.5 only. The registry is
    checked first (enforce_single_model: one entry per tier, no other model
    even disabled) and the router re-checks the model before every request.

    A missing ANTHROPIC_API_KEY doesn't stop Garvis from starting (voice,
    UI and tools stay up); each request then fails with a clear message and
    nothing is sent anywhere."""
    registry = load_model_registry(settings.llm_model_registry_path)
    enforce_single_model(registry)
    pools: dict[ModelTier, list[PoolMember]] = {
        tier: [(entry.label, _build_haiku(settings, entry)) for entry in registry.tier(tier)] for tier in ModelTier
    }
    if not settings.anthropic_api_key:
        log_extra(
            logger, 30, "llm.anthropic.no_key",
            note="ANTHROPIC_API_KEY is not set: every model request will fail with an error until it is added to "
            "Garvis's .env (python scripts/check_anthropic.py --set-key). No other model is used.",
        )
    log_extra(logger, 20, "llm.tiered.pools", **{tier.value: [label for label, _ in pools[tier]] for tier in ModelTier})
    ledger = UsageLedger(settings.llm_usage_db_path)
    budget = None
    if settings.llm_budget_enabled:
        budget = BudgetGuard(
            BudgetLimits(
                max_calls_per_task=settings.llm_max_calls_per_task,
                max_cost_per_task_usd=settings.llm_max_cost_per_task_usd,
                daily_budget_usd=settings.llm_daily_budget_usd,
                monthly_budget_usd=settings.llm_monthly_budget_usd,
                max_prompt_tokens=settings.llm_max_prompt_tokens,
            ),
            ledger,
        )
    log_extra(logger, 20, "llm.usage.ledger", path=str(settings.llm_usage_db_path),
              budget_enabled=budget is not None, max_retries=settings.llm_max_retries)
    return TieredLLMRouter(
        pools,
        required_model=REQUIRED_MODEL,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        budget=budget,
        ledger=ledger,
        output_reserve_tokens=settings.llm_output_reserve_tokens,
    )


# Former local-model modes (Ollama, and the ollama+groq "auto" router).
_REMOVED_LOCAL_PROVIDERS = frozenset({"ollama", "auto"})
# Former single-provider modes: each sent every request to another model.
_REMOVED_OTHER_MODEL_PROVIDERS = frozenset({"gemini", "groq", "openrouter", "omniroute"})


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the one LLMProvider Garvis uses: the tiered router over
    Claude Haiku 5.5 (direct Anthropic API). Every other mode is refused --
    there is no path to another model, and Garvis never runs an AI model
    locally."""
    if settings.llm_provider == "tiered":
        return _build_tiered_router(settings)
    if settings.llm_provider in _REMOVED_OTHER_MODEL_PROVIDERS:
        raise ConfigurationError(
            f"GARVIS_LLM_PROVIDER={settings.llm_provider!r} is not allowed: Garvis uses only Claude Haiku 5.5 "
            f"({REQUIRED_MODEL}). Set GARVIS_LLM_PROVIDER=tiered (the default)."
        )
    if settings.llm_provider in _REMOVED_LOCAL_PROVIDERS:
        raise ConfigurationError(
            f"GARVIS_LLM_PROVIDER={settings.llm_provider!r} was removed: Garvis never runs an AI model "
            "locally. Use 'tiered' (the default)."
        )
    raise ConfigurationError(f"Unknown llm_provider: {settings.llm_provider!r}")


def build_agent(settings: Settings | None = None, events: EventBus | None = None) -> Agent:
    settings = settings or get_settings()
    memory_store = SQLiteMemoryStore(settings.memory_db_path)
    desktop, workspace, browser_manager = build_browser_stack(settings)
    gmail_manager = build_gmail_manager(settings)
    events = events or EventBus()
    # One routing state for every tool; each USER_MESSAGE (any interface)
    # starts a new request with that message's explicit wording.
    routing = DesktopRouting()
    events.subscribe(routing.on_event)
    extra_tools = build_extra_tools(settings, desktop, workspace, browser_manager, routing=routing)
    registry = build_registry(
        memory_store, browser_manager, gmail_manager, extra_tools=extra_tools, desktop=desktop, routing=routing,
        workspace_available=workspace is not None,
    )
    llm = build_llm_provider(settings)
    context_provider = (
        desktop.describe_context
        if settings.desktop_control_enabled and settings.desktop_context_in_prompt and sys.platform == "win32"
        else None
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        memory=memory_store,
        events=events,
        max_context_messages=settings.max_context_messages,
        browser_manager=browser_manager,
        max_context_tokens=settings.effective_max_context_tokens(),
        max_tool_iterations=settings.agent_max_tool_iterations,
        context_provider=context_provider,
    )
    agent.desktop = desktop  # lets the UI bring a window a tool opened back into view
    return agent
