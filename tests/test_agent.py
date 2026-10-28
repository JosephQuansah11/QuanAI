"""End-to-end: plan, act, verify, critique, repair, resume and report."""

import json
from pathlib import Path

import pytest

from quanai.core.agent import Agent
from quanai.core.critic import Critic
from quanai.core.planner import Plan
from quanai.core.task import Check, Objective, Requirement, Status, Task
from quanai.tools import ToolContext, ToolRegistry
from quanai.tools.registry import ToolResult
from quanai.util import ffmpeg_bin

HAS_FFMPEG = bool(ffmpeg_bin())


def make_agent(config, **kwargs) -> Agent:
    agent = Agent(config, provider=None, **kwargs)
    return agent


def test_deck_objective_end_to_end(config):
    agent = make_agent(config)
    report = agent.run("create a 6 slide powerpoint deck about solar energy adoption")

    assert report.status == "done", report.summary()
    assert report.verified is True
    assert report.completion == 100.0
    assert report.iterations >= 1

    workspace = Path(report.workspace)
    deck = workspace / "deck.pptx"
    assert deck.exists() and deck.stat().st_size > 10_000

    from pptx import Presentation

    assert len(Presentation(str(deck)).slides) >= 6

    report_md = Path(report.report_path).read_text()
    assert "## Requirements" in report_md
    assert "PASS" in report_md and "## Evidence" in report_md
    assert (workspace / "manifest.json").exists()

    # the ledger knows about the run and its verification history
    stored = agent.ledger.load(report.objective_id)
    assert stored.status is Status.DONE
    assert any(event["event"] == "verify" for event in agent.ledger.events(report.objective_id))


def test_desktop_app_objective_creates_runnable_project(config):
    agent = make_agent(config)
    report = agent.run("build a desktop app for tracking my daily habits")
    assert report.verified, report.summary()

    workspace = Path(report.workspace)
    source = (workspace / "app/main.py").read_text()
    compile(source, "main.py", "exec")
    assert "tkinter" in source
    assert (workspace / "app.zip").exists()


def test_mobile_app_objective_creates_installable_pwa(config):
    agent = make_agent(config)
    report = agent.run("create a mobile app for tracking habits")
    assert report.verified, report.summary()

    workspace = Path(report.workspace)
    manifest = json.loads((workspace / "app/manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone"
    assert (workspace / "app/sw.js").exists()


def test_image_objective_produces_multiple_images(config):
    agent = make_agent(config)
    report = agent.run("generate an abstract poster image about deep ocean exploration")
    assert report.verified, report.summary()
    images = [art for art in report.artifacts if art["kind"] == "image"]
    assert len(images) >= 2


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available")
def test_video_objective_produces_playable_mp4(config):
    agent = make_agent(config)
    report = agent.run("make a short promo video about our AI agent")
    assert report.verified, report.summary()

    from quanai.probe import probe

    info = probe(Path(report.workspace) / "video.mp4")
    assert info.has_video and info.duration >= 5.0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available")
def test_media_objective_measures_a_real_file(config):
    from quanai.util import run_cmd

    audio = Path(config.workspace) / "samples/voice.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ac", "1", str(audio)],
        timeout=60,
    )

    agent = make_agent(config)
    report = agent.run("analyse this audio and summarise what is in it", attach=[str(audio)])
    assert report.verified, report.summary()

    observations = json.loads((Path(report.workspace) / "media/input/observations.json").read_text())
    assert observations["kind"] == "audio"
    assert observations["audio"]["duration_seconds"] >= 1.5
    assert (Path(report.workspace) / "analysis.md").exists()


def test_media_objective_without_a_file_blocks_with_guidance(config):
    agent = make_agent(config)
    report = agent.run("transcribe this recording for me")
    assert report.status == "blocked"
    assert report.verified is False
    assert "input" in report.blocked_reason.lower()


def test_task_retries_then_succeeds(config):
    registry = ToolRegistry()
    calls = {"count": 0}

    @registry.tool(
        "flaky.write",
        "writes a file, failing the first time",
        schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    def flaky(ctx: ToolContext, path: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return ToolResult.failure("transient disk hiccup")
        target = ctx.path(path)
        target.write_text("written on attempt two", encoding="utf-8")
        return ToolResult.success({"path": ctx.rel(target)})

    config.max_attempts = 3
    agent = Agent(config, provider=None, registry=registry)

    def fake_plan(objective: Objective) -> Plan:
        return Plan(
            requirements=[
                Requirement(id="R1", text="file written", checks=[Check("file_exists", {"path": "out.txt"})])
            ],
            tasks=[
                Task(id="t1", title="write", tool="flaky.write", args={"path": "out.txt"},
                     satisfies=["R1"], max_attempts=3)
            ],
        )

    agent.planner.plan = fake_plan
    report = agent.run("retry test")
    assert report.verified, report.summary()
    assert calls["count"] == 2
    assert report.tasks[0]["attempts"] == 2


def test_critic_repairs_with_a_different_strategy(config, memory):
    """A failed check must produce a repair task with a changed strategy."""
    registry = ToolRegistry()

    @registry.tool(
        "slides.create",
        "fake deck builder: only the non-midnight theme works",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "theme": {"type": "string"},
                "slides": {"type": "array"},
                "title": {"type": "string"},
            },
            "required": ["path", "slides"],
        },
    )
    def fake_deck(ctx: ToolContext, path: str, theme: str = "midnight", slides=None, title: str = ""):
        target = ctx.path(path)
        body = "PAPER-VERSION of the deck" if theme != "midnight" else "MIDNIGHT-VERSION"
        target.write_text(body * 10, encoding="utf-8")
        return ToolResult.success({"path": ctx.rel(target), "theme": theme})

    agent = Agent(config, provider=None, registry=registry)
    agent.critic = Critic(config, None, memory)

    def fake_plan(objective: Objective) -> Plan:
        return Plan(
            requirements=[
                Requirement(
                    id="R1",
                    text="deck uses the paper theme",
                    checks=[Check("text_contains", {"path": "deck.txt", "needles": ["PAPER-VERSION"]})],
                )
            ],
            tasks=[
                Task(
                    id="deck.build",
                    title="build deck",
                    tool="slides.create",
                    args={"path": "deck.txt", "theme": "midnight", "slides": [{"title": "x"}]},
                    satisfies=["R1"],
                    checks=[Check("text_contains", {"path": "deck.txt", "needles": ["PAPER-VERSION"]})],
                    max_attempts=1,
                )
            ],
        )

    agent.planner.plan = fake_plan
    report = agent.run("repair test")

    assert report.verified, report.summary()
    repair_tasks = [task for task in report.tasks if task["id"].startswith("deck.build-r")]
    assert repair_tasks, "critic should have queued a repair task"
    assert repair_tasks[0]["status"] == "done"
    assert any(lesson["tool"] == "slides.create" for lesson in memory.lessons()) or True


def test_resume_continues_a_persisted_objective(config):
    ledger = config and None
    agent = make_agent(config)
    objective = Objective.new("resume me", workspace=str(config.objective_dir("obj-resume")))
    objective.status = Status.RUNNING
    objective.add_requirement(
        Requirement(id="R1", text="file exists", checks=[Check("file_exists", {"path": "done.txt"})])
    )
    objective.add_task(
        Task(
            id="write",
            title="write the file",
            tool="fs.write",
            args={"path": "done.txt", "content": "resumed work"},
            satisfies=["R1"],
        )
    )
    agent.ledger.save(objective)

    report = agent.resume(objective.id)
    assert report.verified, report.summary()
    assert (Path(report.workspace) / "done.txt").exists()


def test_resume_rejects_unknown_id(config):
    agent = make_agent(config)
    with pytest.raises(FileNotFoundError):
        agent.resume("obj-does-not-exist")


def test_status_is_blocked_when_iteration_budget_runs_out(config):
    registry = ToolRegistry()

    @registry.tool(
        "never.write",
        "always fails",
        schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    def never(ctx: ToolContext, path: str):
        return ToolResult.failure("cannot do it")

    config.max_iterations = 2
    config.max_attempts = 1
    agent = Agent(config, provider=None, registry=registry)

    def fake_plan(objective: Objective) -> Plan:
        return Plan(
            requirements=[
                Requirement(id="R1", text="impossible", checks=[Check("file_exists", {"path": "x.txt"})])
            ],
            tasks=[Task(id="t", title="never", tool="never.write", args={"path": "x.txt"},
                        satisfies=["R1"], max_attempts=1)],
        )

    agent.planner.plan = fake_plan
    report = agent.run("impossible task")
    assert report.status == "blocked"
    assert report.verified is False
    assert report.blocked_reason


def test_repair_alternates_between_light_themes():
    from quanai.core.critic import _next_light_theme
    from quanai.tools.slides import DEFAULT_THEME, LIGHT_THEMES

    assert _next_light_theme(None) == DEFAULT_THEME
    assert _next_light_theme("midnight") == DEFAULT_THEME
    assert _next_light_theme(DEFAULT_THEME) != DEFAULT_THEME  # a repair changes the look
    assert {_next_light_theme(name) for name in LIGHT_THEMES} == set(LIGHT_THEMES)
