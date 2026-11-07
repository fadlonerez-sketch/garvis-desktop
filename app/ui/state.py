"""The one view-model behind the dashboard and the floating orb.

Fed only by AgentBridge signals -- real agent/voice events -- so both views
always show the same Garvis: there is no second agent, no polling and no
made-up activity. It keeps what the event stream doesn't: turns with their
tool calls and sources, recent errors and a voice timeline.

`mode` is what the visuals animate, in priority order:
  speaking   Garvis's voice is actually playing (bridge.speaking_changed)
  listening  push-to-talk recording, or a session waiting for speech
  thinking   transcribing / an agent turn / a session preparing its reply
  error      for a few seconds after an error
  idle       otherwise (with `standby` when the wake word is armed)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from PySide6.QtCore import QObject, QTimer, Signal

_MAX_TURNS = 200
_MAX_ERRORS = 50
_MAX_TIMELINE = 60
_ERROR_DISPLAY_MS = 3500
_BUSY_STATUSES = ("Thinking...", "Transcribing...", "Stopping...")


class Mode(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


MODE_LABELS = {
    Mode.IDLE: "Online",
    Mode.LISTENING: "Listening",
    Mode.THINKING: "Thinking",
    Mode.SPEAKING: "Speaking",
    Mode.ERROR: "Attention",
}


def summarize_arguments(arguments: dict, limit: int = 90) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        text = " ".join(str(value).split())
        parts.append(f"{key}={text[:40] + '…' if len(text) > 40 else text}")
    joined = ", ".join(parts)
    return joined if len(joined) <= limit else joined[: limit - 1] + "…"


@dataclass
class ToolActivity:
    name: str
    arguments: str
    started_at: float
    finished_at: float | None = None
    success: bool | None = None
    error: str = ""
    sources: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        return None if self.finished_at is None else max(0.0, self.finished_at - self.started_at)


@dataclass
class Turn:
    id: int
    request: str
    source: str  # "desktop", "mobile", ...
    started_at: float
    reply: str = ""
    error: str = ""
    finished_at: float | None = None
    tools: list[ToolActivity] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def sources(self) -> list[dict]:
        seen: set[str] = set()
        merged: list[dict] = []
        for tool in self.tools:
            for source in tool.sources:
                if source["url"] not in seen:
                    seen.add(source["url"])
                    merged.append(source)
        return merged


@dataclass
class ErrorRecord:
    message: str
    at: float


@dataclass
class VoiceEvent:
    label: str
    at: float


class GarvisState(QObject):
    mode_changed = Signal(str)
    status_changed = Signal(str)
    activity_changed = Signal()
    """A turn started/finished or a tool started/finished."""
    sources_changed = Signal()
    errors_changed = Signal()
    voice_changed = Signal()
    """Session state, standby, push-to-talk, playback or the timeline changed."""

    def __init__(self, clock=time.time, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._clock = clock
        self.status = "Idle."
        self.session_state = "idle"
        self.standby = False
        self.push_to_talk = False
        self.playing = False
        self.turns: deque[Turn] = deque(maxlen=_MAX_TURNS)
        self.errors: deque[ErrorRecord] = deque(maxlen=_MAX_ERRORS)
        self.timeline: deque[VoiceEvent] = deque(maxlen=_MAX_TIMELINE)
        self.tool_calls = 0
        self._next_turn_id = 1
        self._error_active = False
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._clear_error)
        self._mode = Mode.IDLE

    # ------------------------------------------------------------ wiring
    def connect_bridge(self, bridge) -> None:
        bridge.status_changed.connect(self.on_status)
        bridge.user_message.connect(self.on_user_message)
        bridge.remote_user_message.connect(self.on_remote_user_message)
        bridge.assistant_message.connect(self.on_assistant_message)
        bridge.tool_started.connect(self.on_tool_started)
        bridge.tool_finished.connect(self.on_tool_finished)
        bridge.error.connect(self.on_error)
        bridge.listening_changed.connect(self.on_listening)
        bridge.session_state_changed.connect(self.on_session_state)
        bridge.standby_changed.connect(self.on_standby)
        presence = getattr(bridge, "presence", None)  # app/core/presence.py: the display-client feed
        if presence is not None:
            self.mode_changed.connect(presence.set_mode)
            presence.set_mode(self._mode.value)
        # Newer bridge capabilities; an older/fake bridge simply lacks them.
        for signal_name, slot in (
            ("speaking_changed", self.on_playing),
            ("tool_sources", self.on_tool_sources),
            ("request_started", self.on_request_started),
        ):
            signal = getattr(bridge, signal_name, None)
            if signal is not None:
                signal.connect(slot)

    # ------------------------------------------------------------- reads
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def current_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns and self.turns[-1].running else None

    @property
    def latest_turn_with_sources(self) -> Turn | None:
        return next((turn for turn in reversed(self.turns) if turn.sources), None)

    def turn(self, turn_id: int) -> Turn | None:
        return next((turn for turn in self.turns if turn.id == turn_id), None)

    @property
    def source_count(self) -> int:
        return sum(len(turn.sources) for turn in self.turns)

    @property
    def session_active(self) -> bool:
        return self.session_state not in ("idle", "")

    # ----------------------------------------------------------- updates
    def _recompute_mode(self) -> None:
        if self.playing:
            mode = Mode.SPEAKING
        elif self.push_to_talk or self.session_state == "listening":
            mode = Mode.LISTENING
        elif (
            self.status in _BUSY_STATUSES
            or self.session_state in ("thinking", "speaking", "stopping")
            or self.current_turn is not None
        ):
            mode = Mode.THINKING  # a session's "speaking" before playback starts = preparing speech
        elif self._error_active:
            mode = Mode.ERROR
        else:
            mode = Mode.IDLE
        if mode != self._mode:
            self._mode = mode
            self.mode_changed.emit(mode.value)

    def _note_voice(self, label: str) -> None:
        self.timeline.append(VoiceEvent(label, self._clock()))
        self.voice_changed.emit()

    def on_status(self, status: str) -> None:
        self.status = status
        self.status_changed.emit(status)
        if status in ("Done.", "Idle.") and self.current_turn is not None:
            # The agent ends every turn with one of these; a turn that failed has no reply event.
            self.current_turn.finished_at = self._clock()
            self.activity_changed.emit()
        self._recompute_mode()

    def on_user_message(self, text: str) -> None:
        self._begin_turn(text, "desktop")

    def on_request_started(self, text: str, source: str) -> None:
        """A protocol run: tracked as a turn, so its steps and sources show up
        in Activity like a chat request's (it ends with the "Done." status)."""
        self._begin_turn(text, source or "desktop")

    def on_remote_user_message(self, text: str, source: str) -> None:
        # Always follows user_message for the same turn: just relabel it.
        turn = self.current_turn
        if turn is not None and turn.request == text:
            turn.source = source
            self.activity_changed.emit()

    def _begin_turn(self, text: str, source: str) -> None:
        current = self.current_turn
        if current is not None:  # a previous turn that never reported its end
            current.finished_at = self._clock()
        self.turns.append(Turn(id=self._next_turn_id, request=text, source=source, started_at=self._clock()))
        self._next_turn_id += 1
        self.activity_changed.emit()
        self._recompute_mode()

    def on_assistant_message(self, text: str) -> None:
        turn = self.current_turn
        if turn is not None:
            turn.reply = text
            turn.finished_at = self._clock()
            self.activity_changed.emit()
        self._recompute_mode()

    def on_tool_started(self, name: str, arguments: dict) -> None:
        self.tool_calls += 1
        turn = self.current_turn
        if turn is not None:
            turn.tools.append(ToolActivity(name, summarize_arguments(arguments), self._clock()))
        self.activity_changed.emit()

    def on_tool_finished(self, name: str, success: bool, error: str) -> None:
        turn = self.current_turn
        tool = (
            next((t for t in reversed(turn.tools) if t.name == name and t.finished_at is None), None) if turn else None
        )
        if tool is not None:
            tool.finished_at = self._clock()
            tool.success = success
            tool.error = error
        self.activity_changed.emit()

    def on_tool_sources(self, name: str, sources: list) -> None:
        turn = self.current_turn
        if turn is None:
            return
        tool = next((t for t in reversed(turn.tools) if t.name == name), None)
        if tool is None:
            return
        tool.sources = [dict(source) for source in sources if isinstance(source, dict) and source.get("url")]
        self.sources_changed.emit()
        self.activity_changed.emit()

    def on_error(self, message: str) -> None:
        self.errors.append(ErrorRecord(message, self._clock()))
        turn = self.current_turn
        if turn is not None:
            turn.error = message
        self._error_active = True
        self._error_timer.start(_ERROR_DISPLAY_MS)
        self.errors_changed.emit()
        self._recompute_mode()

    def _clear_error(self) -> None:
        self._error_active = False
        self._recompute_mode()

    def on_listening(self, listening: bool) -> None:
        if listening != self.push_to_talk:
            self.push_to_talk = listening
            self._note_voice("Push-to-talk recording" if listening else "Recording stopped")
        self._recompute_mode()

    _SESSION_LABELS = {
        "listening": "Session listening",
        "thinking": "Session thinking",
        "speaking": "Session preparing reply",
        "interrupted": "Interrupted by you (barge-in)",
        "stopping": "Session stopping",
        "idle": "Session ended",
    }

    def on_session_state(self, state: str) -> None:
        if state != self.session_state:
            self.session_state = state
            self._note_voice(self._SESSION_LABELS.get(state, f"Session {state}"))
        self._recompute_mode()

    def on_standby(self, active: bool) -> None:
        if active != self.standby:
            self.standby = active
            self._note_voice("Wake word armed" if active else "Wake word paused")
        self._recompute_mode()

    def on_playing(self, playing: bool) -> None:
        if playing != self.playing:
            self.playing = playing
            self._note_voice("Speaking" if playing else "Finished speaking")
        self._recompute_mode()
