"""Manual smoke test for browser research on the Garvis workspace.

    python scripts/smoke_test_research.py "current bitcoin price"

Uses exactly the stack Garvis builds (app/agent/build.py:build_browser_stack):
real Chrome (Playwright), launched off-screen, moved to the "Garvis" virtual
desktop, then a real search + source reading. Checks that the desktop you're
looking at never changes. No LLM involved -- this exercises the tool layer.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.build import build_browser_stack  # noqa: E402
from app.browser.research import WebResearcher  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402


async def main(query: str) -> int:
    settings = get_settings()
    configure_logging("WARNING")
    desktop, workspace, browser = build_browser_stack(settings)
    before = await workspace.status() if workspace else {"available": False}
    focused_before = await desktop.foreground_window()
    print("workspace before:", before)
    researcher = WebResearcher(
        browser, workspace=workspace, engine=settings.research_search_engine,
        max_sources=settings.research_max_sources, source_chars=settings.research_source_chars,
    )
    started = time.monotonic()
    try:
        report = await researcher.research(query)
    finally:
        after = await workspace.status() if workspace else {"available": False}
        focused_after = await desktop.foreground_window()
        await browser.shutdown()
        if workspace:
            workspace.shutdown()

    print(f"\nquery: {report.query!r}  engine: {report.engine}  ran on: {report.workspace}  "
          f"({time.monotonic() - started:.1f}s)")
    for step in report.plan:
        print(f"  [{step.status:8}] {step.step}: {step.detail[:160]}")
    print("\nresults:")
    for r in report.results[:5]:
        print(f"  - {r.title[:70]}  <{r.url[:90]}>")
    print("\nsources:")
    for s in report.sources:
        print(f"  * {s.title[:70]}  <{s.url[:90]}>")
        print("    " + s.excerpt[:300].replace("\n", " | "))

    view_unchanged = before.get("current_desktop") == after.get("current_desktop")
    same_focus = (focused_before and focused_after and focused_before.handle == focused_after.handle)
    print("\nuser's view unchanged:", view_unchanged)
    print("focused window before/after:",
          f"{focused_before.process_name}#{focused_before.handle}" if focused_before else None, "/",
          f"{focused_after.process_name}#{focused_after.handle}" if focused_after else None,
          "(same window)" if same_focus else "(changed -- did you switch windows meanwhile?)")
    ok = bool(report.sources) and view_unchanged
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(" ".join(sys.argv[1:]) or "current bitcoin price USD")))
