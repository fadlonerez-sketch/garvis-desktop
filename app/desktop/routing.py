"""Which desktop a window belongs on: the user's, or Garvis's workspace.

    USER_DESKTOP      the virtual desktop the user is looking at: anything they
                      asked to see, use or follow (sites, apps, files, results).
    GARVIS_WORKSPACE  the "Garvis" virtual desktop: background research,
                      intermediate windows, work the user didn't ask to watch.

Every tool that opens or moves a window resolves its target through
`DesktopRouting.resolve()` BEFORE acting, in this order:

1. An explicit placement instruction in the user's current request ("on my
   screen", "in my browser" / "in the background", "don't open it" -- also
   in Hebrew). An instruction from the user beats everything.
2. The `target_desktop` the model passed: its reading of the task. A
   request to *see* something ("show me the news") is about the end result,
   not every step -- the model may still research in the workspace first
   and then present the result on the user's desktop.
3. The target already in use for that resource in this task (the browser
   surface it's driving), so multi-step work stays where it started.
4. A request to see something ("show me ...") for user-facing tools.
5. The tool's own default (opening an app or a site for the user ->
   USER_DESKTOP; research -> GARVIS_WORKSPACE).

The resolved target is reported in every tool result, with the reason. A
target that turns out to be unavailable is an error -- nothing silently
opens on the other desktop instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.events import AgentEvent, EventType


class DesktopTarget(StrEnum):
    USER = "user_desktop"
    WORKSPACE = "garvis_workspace"


TARGET_FIELD_DESCRIPTION = (
    "Where the window should be: 'user_desktop' (the screen the user is looking at -- for "
    "anything they asked to see, use or follow) or 'garvis_workspace' (Garvis's own background "
    "desktop -- research and intermediate work they didn't ask to watch). Omit to apply the "
    "routing defaults."
)

_USER_PLACEMENT_PATTERNS = [
    r"\bon (?:my|the) (?:screen|monitor|desktop)\b",
    r"\bin my (?:browser|chrome|edge|firefox|brave)\b",
    r"\bput (?:it|this|that) on (?:my|the) screen\b",
    r"על המסך",
    r"בדפדפן שלי",
]
_USER_SEE_PATTERNS = [
    r"\bshow (?:me|it|them|this|that|us)\b",
    r"\blet me see\b",
    r"\bi (?:want|need) to see\b",
    r"\bopen (?:\w+ ){0,6}for me\b",
    r"\bpull (?:it |this |that )?up\b",
    r"תראה לי",
    r"תראי לי",
    r"תציג לי",
    r"תציגי לי",
    r"הראה לי",
    r"תפתח לי",
    r"תפתחי לי",
]
_WORKSPACE_PATTERNS = [
    r"\bin the background\b",
    r"\bbehind the scenes\b",
    r"\b(?:don't|do not|dont) (?:show|open|disturb)\b",
    r"\bwithout (?:opening|showing|disturbing)\b",
    r"\bjust (?:tell|give) me\b",
    r"\bon your (?:own )?(?:desktop|workspace)\b",
    r"\bquietly\b",
    r"ברקע",
    r"אל תפתח",
    r"בלי לפתוח",
    r"רק תגיד לי",
]
_USER_PLACEMENT_RE = re.compile("|".join(_USER_PLACEMENT_PATTERNS), re.IGNORECASE)
_USER_SEE_RE = re.compile("|".join(_USER_SEE_PATTERNS), re.IGNORECASE)
_WORKSPACE_RE = re.compile("|".join(_WORKSPACE_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class TurnIntent:
    explicit_target: DesktopTarget | None = None
    phrase: str = ""
    placement: bool = False
    """True for an instruction about WHERE ("on my screen", "in the
    background"): it overrides everything. False for a request to see the
    outcome ("show me ..."): it decides where the result goes, while the
    model may still do intermediate work in the workspace."""

    @property
    def wants_to_see(self) -> bool:
        return self.explicit_target == DesktopTarget.USER


def infer_intent(text: str) -> TurnIntent:
    """Explicit wording only -- nuance is left to the model's
    `target_desktop`. "In the background" wins over "show" when both appear
    ("research it in the background and show me the summary" asks for
    background work; the summary is the reply itself)."""
    text = text or ""
    background = _WORKSPACE_RE.search(text)
    if background:
        return TurnIntent(DesktopTarget.WORKSPACE, background.group(0), placement=True)
    placed = _USER_PLACEMENT_RE.search(text)
    if placed:
        return TurnIntent(DesktopTarget.USER, placed.group(0), placement=True)
    visible = _USER_SEE_RE.search(text)
    if visible:
        return TurnIntent(DesktopTarget.USER, visible.group(0))
    return TurnIntent()


@dataclass(frozen=True)
class Resolution:
    target: DesktopTarget
    reason: str

    def as_result(self) -> dict:
        return {"target_desktop": self.target.value, "target_reason": self.reason}


@dataclass
class DesktopRouting:
    """Per-request placement state. Subscribe it to the EventBus: each
    USER_MESSAGE (published by the Agent at the start of every turn, from
    any interface) starts a new request."""

    intent: TurnIntent = field(default_factory=TurnIntent)
    sticky: dict[str, DesktopTarget] = field(default_factory=dict)

    def on_event(self, event: AgentEvent) -> None:
        if event.type in (EventType.USER_MESSAGE, EventType.REQUEST_STARTED):
            self.new_request(str(event.payload.get("text", "")))

    def new_request(self, text: str) -> None:
        self.intent = infer_intent(text)
        if self.intent.explicit_target is not None:
            # A new explicit instruction re-routes resources already in use.
            self.sticky.clear()

    def resolve(
        self, resource: str, requested: str | DesktopTarget | None, default: DesktopTarget, user_facing: bool = True
    ) -> Resolution:
        if self.intent.placement:
            return Resolution(self.intent.explicit_target, f'you said "{self.intent.phrase}"')
        if requested:
            return Resolution(DesktopTarget(requested), "chosen for this task")
        if resource in self.sticky:
            return Resolution(self.sticky[resource], "continuing where this task's work already is")
        if user_facing and self.intent.wants_to_see:
            return Resolution(DesktopTarget.USER, f'you asked to see it ("{self.intent.phrase}")')
        return Resolution(default, "default for this kind of window")

    def remember(self, resource: str, target: DesktopTarget) -> None:
        self.sticky[resource] = target
