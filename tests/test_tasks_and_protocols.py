"""Tasks, protocols and presentation metadata (app/tasks/, app/agent/presentation.py,
Agent.run_tool/cancellation, app/tools/open_path.py).

A real Agent with fake tools and a scripted/gated LLM: no desktop, browser,
network or real model. Desktop-acting tools are test doubles that record
their calls; open_path gets a fake opener.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.agent import presentation as pres
from app.agent.agent import CANCELLED_REPLY, CancelToken
from app.core.events import EventBus, EventType
from app.tasks.manager import CANCELLED, COMPLETED, FAILED, INTERRUPTED, TaskError, TaskManager
from app.tasks.protocols import (
    ACTIONS,
    Protocol,
    ProtocolStore,
    Step,
    clean_params,
    describe_step,
    public_summary,
    validate,
)
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.open_path import OpenPathTool, refusal
from tests.conftest import final_response, tool_call_response
from tests.mobile_fakes import GatedLLM, HighRiskTool, make_agent, wait_until


class _UrlArgs(BaseModel):
    url: str
    target_desktop: str | None = None


class FakeWebsiteTool(Tool):
    """Stands in for open_website: records calls, reports a window like the real one."""

    name = "open_website"
    description = "Open a page (test double)."
    risk_level = RiskLevel.LOW
    presentation = "prominent"
    input_model = _UrlArgs

    def __init__(self, fail: bool = False, delay: float = 0.0) -> None:
        self.calls: list[str] = []
        self.fail = fail
        self.delay = delay

    async def execute(self, args: _UrlArgs) -> ToolResult:
        self.calls.append(args.url)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return ToolResult.fail("The browser refused.")
        return ToolResult.ok({"opened": args.url, "window": "Forecast - Browser", "handle": 4242,
                              "visible_verified": True, "target_desktop": args.target_desktop or "user_desktop"})


class _AppArgs(BaseModel):
    application: str
    target_desktop: str | None = None


class FakeAppTool(Tool):
    name = "open_application"
    description = "Open an app (test double)."
    risk_level = RiskLevel.LOW
    presentation = "prominent"
    input_model = _AppArgs

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def execute(self, args: _AppArgs) -> ToolResult:
        self.calls.append((args.application, args.target_desktop))
        if args.application == "missing":
            return ToolResult.fail("Could not find an installed application matching 'missing'.")
        return ToolResult.ok({"opened": args.application, "window": f"{args.application} window", "handle": 7,
                              "target_desktop": args.target_desktop or "user_desktop"})


def _collect(events: EventBus, kind: EventType) -> list[dict]:
    seen: list[dict] = []
    events.subscribe(lambda event: seen.append(event.payload) if event.type == kind else None)
    return seen


# ------------------------------------------------------------ presentation
def test_presentation_levels_are_explicit_and_background_work_is_never_shown():
    assert pres.level_for("prominent", {"target_desktop": "user_desktop"}) == pres.Presentation.PROMINENT
    assert pres.level_for("prominent", {"target_desktop": "garvis_workspace"}) == pres.Presentation.NONE
    assert pres.level_for("none", {}) == pres.Presentation.NONE
    assert pres.level_for("none", {}, override="subtle") == pres.Presentation.SUBTLE
    assert pres.level_for("prominent", {"target_desktop": "garvis_workspace"}, override="prominent") == "none"
    assert Tool.presentation == "none"  # every tool is background unless it opts in


def test_presentation_card_uses_only_the_tools_own_result():
    card = pres.describe("open_website", {"opened": "https://weather.example/x", "window": "Weather - Chrome",
                                          "handle": 9}, pres.Presentation.PROMINENT, source="mobile", task_id="t1")
    assert card["title"] == "Weather - Chrome" and card["url"] == "https://weather.example/x"
    assert card["handle"] == 9 and card["source"] == "mobile" and card["task_id"] == "t1"
    unverified = pres.describe("open_website", {"opened": "https://weather.example/x"}, pres.Presentation.PROMINENT)
    assert unverified["title"] == "weather.example" and unverified["handle"] is None  # no invented title
    folder = pres.describe("open_path", {"path": "C:\\work", "name": "work"}, pres.Presentation.PROMINENT)
    assert folder["kind"] == "path" and folder["title"] == "work"
    assert pres.describe("open_website", {"opened": "javascript:alert(1)"}, pres.Presentation.SUBTLE)["url"] == ""


# ---------------------------------------------------------------- the agent
async def test_chat_turn_presents_what_a_tool_opened(tmp_path):
    events = EventBus()
    llm = GatedLLM([tool_call_response("open_website", {"url": "https://weather.example"}), final_response("Opened.")],
                   gated=False)
    tool = FakeWebsiteTool()
    agent = make_agent(tmp_path, llm, events=events, tools=[tool])
    shown = _collect(events, EventType.PRESENTATION)
    await agent.handle_user_message("show me the weather")
    assert len(shown) == 1 and shown[0]["title"] == "Forecast - Browser" and shown[0]["level"] == "prominent"


async def test_run_tool_uses_the_same_permission_and_confirmation_path(tmp_path):
    events = EventBus()
    risky = HighRiskTool()
    agent = make_agent(tmp_path, GatedLLM(gated=False), events=events, tools=[risky])
    requests = _collect(events, EventType.CONFIRMATION_REQUESTED)
    task = asyncio.ensure_future(agent.run_tool("high_risk_echo", {}))
    await wait_until(lambda: requests)
    assert risky.executions == 0  # nothing runs before the user answers
    agent.confirmations.resolve(requests[0]["id"], False)
    result = await task
    assert not result.success and risky.executions == 0
    unknown = await agent.run_tool("no_such_tool", {})
    assert not unknown.success and risky.executions == 0


async def test_run_tool_adds_nothing_to_the_conversation(tmp_path):
    agent = make_agent(tmp_path, GatedLLM(gated=False), tools=[FakeWebsiteTool()])
    before = len(agent._context.messages)
    result = await agent.run_tool("open_website", {"url": "https://a.example"})
    assert result.success and len(agent._context.messages) == before


async def test_cancelling_a_turn_stops_before_the_next_step_and_says_so(tmp_path):
    events = EventBus()
    llm = GatedLLM([tool_call_response("open_website", {"url": "https://a.example"}), final_response("never")])
    tool = FakeWebsiteTool()
    agent = make_agent(tmp_path, llm, events=events, tools=[tool])
    token = CancelToken()
    turn = asyncio.ensure_future(agent.handle_user_message("open it", cancel=token))
    await wait_until(lambda: llm.in_flight == 1)
    token.cancel()  # requested while the model is thinking: the tool must not run
    llm.release()
    reply = await turn
    assert reply.startswith(CANCELLED_REPLY) and token.honored
    assert tool.calls == []
    # the history stays valid: every tool call has a result
    roles = [m.role.value for m in agent._context.messages]
    assert roles[-1] == "tool"


async def test_a_turn_that_finishes_first_is_not_reported_cancelled(tmp_path):
    agent = make_agent(tmp_path, GatedLLM([final_response("done")], gated=False))
    token = CancelToken()
    reply = await agent.handle_user_message("hi", cancel=token)
    token.cancel()
    assert reply == "done" and not token.honored


# ---------------------------------------------------------------- open_path
def test_open_path_refuses_anything_that_runs(tmp_path):
    script = tmp_path / "evil.ps1"
    script.write_text("rm -r C:\\")
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    shortcut = tmp_path / "x.lnk"
    shortcut.write_bytes(b"L")
    for path in (script, exe, shortcut):
        assert "runs or changes something" in refusal(path)
    assert refusal(tmp_path / "nope") .startswith("Nothing exists")
    assert "UNC" in refusal(Path("\\\\server\\share\\doc.txt"))
    key = tmp_path / "id_rsa.pem"
    key.write_text("secret")
    assert refusal(key)
    note = tmp_path / "notes.txt"
    note.write_text("hi")
    assert refusal(note) is None and refusal(tmp_path) is None


async def test_open_path_opens_folders_and_documents_only(tmp_path):
    opened: list[str] = []
    tool = OpenPathTool(opener=opened.append)
    folder = await tool.execute(tool.validate_arguments({"path": str(tmp_path)}))
    assert folder.success and folder.data["kind"] == "folder" and opened == [str(tmp_path.resolve())]
    exe = tmp_path / "a.bat"
    exe.write_text("@echo off")
    refused = await tool.execute(tool.validate_arguments({"path": str(exe)}))
    assert not refused.success and len(opened) == 1
    assert OpenPathTool.presentation == "prominent" and OpenPathTool.risk_level == RiskLevel.LOW


# ---------------------------------------------------------------- protocols
def _protocol(*steps: Step, **fields) -> Protocol:
    return Protocol(name=fields.pop("name", "Test protocol"), steps=list(steps), **fields)


def test_only_allowlisted_actions_exist_and_none_runs_arbitrary_code():
    assert set(ACTIONS) == {"open_application", "open_website", "open_websites", "open_path", "focus_window",
                            "wait_for_window", "wait", "notify", "confirm", "ask_garvis"}
    tools = {a.tool for a in ACTIONS.values() if a.tool}
    assert not tools & {"run_powershell", "write_text_file", "edit_text_file", "gmail_send", "keyboard_type"}


def test_validation_catches_bad_parameters_and_unavailable_actions(tmp_path):
    protocol = _protocol(
        Step(type="open_website", params={"url": "javascript:alert(1)"}),
        Step(type="open_application", params={"application": "C:\\Windows\\cmd.exe"}),
        Step(type="open_websites", params={"urls": ["https://a.example"] * 11}),
        Step(type="wait", params={"seconds": 5000}),
        Step(type="focus_window", params={"window": "chrome"}),
        Step(type="shell", params={"command": "format c:"}),
        Step(type="open_path", params={"path": str(tmp_path / "missing")}),
    )
    problems = validate(protocol, tools={"open_website", "open_application", "open_path"})
    by_step = {p.step: p for p in problems}
    assert set(by_step) == {0, 1, 2, 3, 4, 5, 6}
    assert "Desktop control" in by_step[4].message or "desktop control" in by_step[4].message
    assert "Unknown action" in by_step[5].message
    assert by_step[6].blocking is False  # a file that isn't there right now is a warning
    assert validate(_protocol(), tools=set())[0].message == "Add at least one enabled step."


def test_params_are_normalized_with_defaults_and_unknown_keys_dropped():
    step = Step(type="open_application", params={"application": "  Code ", "evil": "x"})
    assert clean_params(step) == {"application": "Code", "where": "user_desktop"}
    urls = Step(type="open_websites", params={"urls": "https://a.example\n\n https://b.example "})
    assert clean_params(urls)["urls"] == ["https://a.example", "https://b.example"]
    assert describe_step(urls) == "Open 2 web pages"


def test_store_round_trip_invalid_files_duplicate_and_delete(tmp_path):
    store = ProtocolStore(tmp_path / "protocols")
    saved = store.save(_protocol(Step(type="wait", params={"seconds": 1}), name="Morning"))
    assert store.get(saved.id).name == "Morning"
    (tmp_path / "protocols" / "deadbeefdeadbeef.json").write_text("{not json")
    (tmp_path / "protocols" / "0123456789abcdef.json").write_text(json.dumps({"schema_version": 99, "id": "0123456789abcdef"}))
    entries = store.load_all()
    assert [e.protocol.name for e in entries if e.protocol] == ["Morning"]
    assert sum(1 for e in entries if e.protocol is None) == 2  # reported, never run
    copy = store.duplicate(saved.id)
    assert copy.id != saved.id and copy.name == "Morning (copy)" and copy.steps[0].id != saved.steps[0].id
    assert store.get("../../etc/passwd") is None
    with pytest.raises(ValueError):
        store.delete("../secrets")
    assert store.delete(saved.id) and store.get(saved.id) is None


def test_the_phone_sees_descriptions_not_raw_parameters():
    protocol = _protocol(Step(type="ask_garvis", params={"instruction": "x" * 300}),
                         Step(type="open_path", params={"path": "C:\\Users\\me\\secret-project"}, label="My project"))
    summary = public_summary(protocol, tools=set())
    assert summary["steps"][1] == "My project"
    assert "secret-project" not in json.dumps(summary)
    assert len(summary["steps"][0]) < 80


# ------------------------------------------------------------- task manager
class _Rig:
    def __init__(self, tmp_path, llm=None, tools=None, session=lambda: False) -> None:
        self.events = EventBus()
        self.web = FakeWebsiteTool()
        self.app = FakeAppTool()
        self.agent = make_agent(tmp_path, llm or GatedLLM(gated=False), events=self.events,
                                tools=tools or [self.web, self.app])
        self.store = ProtocolStore(tmp_path / "protocols")
        self.path = tmp_path / "tasks.json"
        self.manager = TaskManager(self.agent, self.events, self.path, self.store,
                                   available_tools=lambda: {"open_website", "open_application"},
                                   is_desktop_session_active=session)

    def protocol(self, *steps: Step, **fields) -> Protocol:
        return self.store.save(_protocol(*steps, **fields))


@pytest.fixture
async def rig(tmp_path):
    r = _Rig(tmp_path)
    r.manager.start()
    yield r
    await r.manager.stop()


async def test_instruction_lifecycle_and_result(rig):
    task = rig.manager.submit_instruction("open the weather", request_id="req-00000001")
    assert task.status == "queued"
    await wait_until(lambda: task.terminal)
    assert task.status == COMPLETED and task.result == "reply to: open the weather"
    assert task.started_at and task.finished_at


async def test_duplicate_submissions_return_the_original_task(rig):
    first = rig.manager.submit_instruction("hello", request_id="same-request-id")
    again = rig.manager.submit_instruction("hello", request_id="same-request-id")
    assert again is first
    await wait_until(lambda: first.terminal)
    assert sum(1 for t in rig.manager.list() if t.request_id == "same-request-id") == 1
    with pytest.raises(TaskError):
        rig.manager.submit_instruction("x", request_id="bad id!")


async def test_queued_tasks_run_in_order_and_can_be_cancelled_before_starting(tmp_path):
    llm = GatedLLM()
    r = _Rig(tmp_path, llm=llm)
    r.manager.start()
    first = r.manager.submit_instruction("first")
    second = r.manager.submit_instruction("second")
    await wait_until(lambda: first.status == "running")
    r.manager.cancel(second.id)
    assert second.status == CANCELLED  # it never ran
    llm.release()
    await wait_until(lambda: first.terminal)
    await asyncio.sleep(0.05)
    assert first.status == COMPLETED and [c[-1].content for c in llm.calls] == ["first"]
    await r.manager.stop()


async def test_cancelling_a_running_task_is_reported_only_once_it_stopped(tmp_path):
    llm = GatedLLM([tool_call_response("open_website", {"url": "https://a.example"}), final_response("x")])
    r = _Rig(tmp_path, llm=llm)
    r.manager.start()
    task = r.manager.submit_instruction("open it")
    await wait_until(lambda: llm.in_flight == 1)
    r.manager.cancel(task.id)
    assert task.status == "cancelling"  # asked, not yet stopped
    llm.release()
    await wait_until(lambda: task.terminal)
    assert task.status == CANCELLED and r.web.calls == []
    await r.manager.stop()


async def test_model_failure_is_failed_not_success(tmp_path):
    from app.core.exceptions import LLMError

    class Broken(GatedLLM):
        async def chat(self, messages, tools=None):
            raise LLMError("The model could not be reached.")

    r = _Rig(tmp_path, llm=Broken(gated=False))
    r.manager.start()
    task = r.manager.submit_instruction("anything")
    await wait_until(lambda: task.terminal)
    assert task.status == FAILED and "could not be reached" in task.error
    await r.manager.stop()


async def test_protocol_run_executes_steps_through_the_agent_and_records_results(rig):
    shown = _collect(rig.events, EventType.PRESENTATION)
    protocol = rig.protocol(
        Step(type="open_application", params={"application": "Code"}),
        Step(type="open_websites", params={"urls": ["https://a.example", "https://b.example"]}),
        Step(type="notify", params={"title": "Ready", "message": "All open"}),
    )
    task = rig.manager.submit_protocol(protocol.id, origin="mobile")
    await wait_until(lambda: task.terminal)
    assert task.status == COMPLETED and task.result == "All 3 steps done."
    assert rig.app.calls == [("Code", "user_desktop")]
    assert rig.web.calls == ["https://a.example", "https://b.example"]
    assert [s["status"] for s in task.steps] == ["ok", "ok", "ok"]
    assert [p["kind"] for p in shown] == ["application", "website", "website", "notice"]
    assert all(p["task_id"] == task.id for p in shown) and len(task.opened) == 4


async def test_a_failing_required_step_stops_the_protocol(rig):
    protocol = rig.protocol(Step(type="open_application", params={"application": "missing"}),
                            Step(type="open_website", params={"url": "https://never.example"}))
    task = rig.manager.submit_protocol(protocol.id)
    await wait_until(lambda: task.terminal)
    assert task.status == FAILED and "Could not find" in task.error
    assert [s["status"] for s in task.steps] == ["failed", "skipped"] and rig.web.calls == []


async def test_continue_on_failure_runs_the_rest_but_reports_the_failure(rig):
    protocol = rig.protocol(Step(type="open_application", params={"application": "missing"}, on_failure="continue"),
                            Step(type="open_website", params={"url": "https://next.example"}))
    task = rig.manager.submit_protocol(protocol.id)
    await wait_until(lambda: task.terminal)
    assert task.status == FAILED and rig.web.calls == ["https://next.example"]
    assert "1 failed step" in task.result


async def test_the_desktop_revalidates_every_protocol_request(rig):
    disabled = rig.protocol(Step(type="wait", params={"seconds": 1}), enabled=False)
    invalid = rig.protocol(Step(type="open_website", params={"url": "ftp://x"}))
    unavailable = rig.protocol(Step(type="focus_window", params={"window": "x"}))
    for protocol, words in ((disabled, "disabled"), (invalid, "can't run"), (unavailable, "can't run")):
        with pytest.raises(TaskError, match=words):
            rig.manager.submit_protocol(protocol.id, origin="mobile")
    with pytest.raises(TaskError, match="doesn't exist"):
        rig.manager.submit_protocol("0" * 16)


async def test_a_protocol_already_running_is_not_queued_twice(rig):
    protocol = rig.protocol(Step(type="wait", params={"seconds": 2}))
    task = rig.manager.submit_protocol(protocol.id, origin="desktop")
    with pytest.raises(TaskError, match="already"):
        rig.manager.submit_protocol(protocol.id, origin="mobile")
    rig.manager.cancel(task.id)
    await wait_until(lambda: task.terminal)
    assert task.status == CANCELLED


async def test_cancelling_during_a_pause_stops_promptly_and_skips_the_rest(rig):
    protocol = rig.protocol(Step(type="wait", params={"seconds": 60}),
                            Step(type="open_website", params={"url": "https://never.example"}))
    task = rig.manager.submit_protocol(protocol.id)
    await wait_until(lambda: task.status == "running" and task.steps[0]["status"] == "running")
    rig.manager.cancel(task.id)
    await wait_until(lambda: task.terminal, timeout=2.0)
    assert task.status == CANCELLED and [s["status"] for s in task.steps] == ["cancelled", "skipped"]
    assert rig.web.calls == []


async def test_confirm_step_waits_for_the_user_and_a_no_stops_the_protocol(rig):
    requests = _collect(rig.events, EventType.CONFIRMATION_REQUESTED)
    protocol = rig.protocol(Step(type="confirm", params={"message": "Open the work apps?"}),
                            Step(type="open_application", params={"application": "Code"}))
    task = rig.manager.submit_protocol(protocol.id)
    await wait_until(lambda: requests)
    assert task.status == "waiting_confirmation"
    rig.agent.confirmations.resolve(requests[0]["id"], False)
    await wait_until(lambda: task.terminal)
    assert task.status == CANCELLED and rig.app.calls == []


async def test_editing_a_protocol_during_a_run_does_not_change_the_run(rig):
    protocol = rig.protocol(Step(type="wait", params={"seconds": 1}),
                            Step(type="open_website", params={"url": "https://original.example"}))
    task = rig.manager.submit_protocol(protocol.id)
    edited = protocol.model_copy(deep=True)
    edited.steps[1] = Step(type="open_website", params={"url": "https://edited.example"})
    rig.store.save(edited)
    await wait_until(lambda: task.terminal, timeout=4.0)
    assert rig.web.calls == ["https://original.example"]


async def test_tasks_persist_and_unfinished_ones_are_never_rerun(tmp_path):
    llm = GatedLLM()
    r = _Rig(tmp_path, llm=llm)
    r.manager.start()
    done = r.manager.submit_instruction("quick")
    llm.release()
    await wait_until(lambda: done.terminal)
    llm.gate.clear()
    pending = r.manager.submit_instruction("slow")
    await wait_until(lambda: pending.status == "running")
    await r.manager.stop()  # Garvis closes mid-task
    assert pending.status == INTERRUPTED
    again = _Rig(tmp_path, llm=GatedLLM(gated=False))
    again.manager.start()
    restored = {t.title: t for t in again.manager.list()}
    assert restored["quick"].status == COMPLETED and restored["quick"].result
    assert restored["slow"].status == INTERRUPTED
    await asyncio.sleep(0.1)
    assert again.agent._llm.calls == []  # nothing was re-run
    await again.manager.stop()


async def test_desktop_chat_turns_are_recorded_for_every_interface(rig):
    await rig.agent.handle_user_message("what's on today?")
    await asyncio.sleep(0.02)
    chat = rig.manager.list()[0]
    assert chat.kind == "chat" and chat.origin == "desktop" and chat.status == COMPLETED
    assert chat.result == "reply to: what's on today?" and not chat.summary()["cancellable"]
    with pytest.raises(TaskError):
        rig.manager.cancel(chat.id)


async def test_tasks_wait_for_a_hands_free_session_instead_of_interrupting_it(tmp_path):
    active = {"on": True}
    r = _Rig(tmp_path, session=lambda: active["on"])
    r.manager.start()
    task = r.manager.submit_instruction("later")
    await asyncio.sleep(0.15)
    assert task.status == "queued" and "voice session" in task.note
    active["on"] = False
    await wait_until(lambda: task.terminal, timeout=3.0)
    assert task.status == COMPLETED
    await r.manager.stop()


async def test_queue_limit_is_enforced(tmp_path):
    llm = GatedLLM()
    r = _Rig(tmp_path, llm=llm)
    r.manager._max_queued = 2
    r.manager.start()
    r.manager.submit_instruction("1")
    await wait_until(lambda: llm.in_flight == 1)
    r.manager.submit_instruction("2")
    r.manager.submit_instruction("3")
    with pytest.raises(TaskError, match="waiting"):
        r.manager.submit_instruction("4")
    llm.release()
    await r.manager.stop()
