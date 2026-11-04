"""Desktop Control: structured mouse/keyboard/window/screen control, plus the
dedicated Garvis workspace (a Windows virtual desktop).

Layering, mirroring app/browser (BrowserManager is the only Playwright
importer) and app/llm (providers behind one interface):

  app/desktop/backend.py    DesktopBackend ABC + plain data types
  app/desktop/win32.py      the ONLY module that talks to user32/gdi32 (ctypes)
  app/desktop/controller.py DesktopController: async, validated, logged facade
  app/desktop/workspace.py  WorkspaceManager: the Garvis virtual desktop
                            (pyvda is imported only there, lazily)
  app/tools/desktop.py      the structured tools the LLM can call

The LLM never gets arbitrary code execution: it can only call a named tool
whose arguments are schema-validated, then range-checked by the controller.
"""
from __future__ import annotations
