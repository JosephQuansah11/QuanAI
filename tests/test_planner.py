"""The plan must be executable: real tools, real checks, resolvable references."""

import re

import pytest

from quanai.core.planner import Planner, find_input_path
from quanai.core.task import Objective
from quanai.core.verify import known_checks


@pytest.fixture()
def planner(config, registry):
    return Planner(config, registry, provider=None)


def test_every_planned_task_uses_a_registered_tool(planner, registry):
    for request in (
        "a 6 slide deck about solar energy",
        "generate a poster image about deep oceans",
        "make a promo video about our agent",
        "build a desktop app for habits",
        "create a flutter app for workouts",
        "write a document explaining solar power",
    ):
        plan = planner.plan(Objective.new(request))
        assert plan.tasks, request
        for task in plan.tasks:
            assert registry.has(task.tool), f"{request}: unknown tool {task.tool}"


def test_every_planned_requirement_has_a_known_check(planner):
    for request in (
        "a deck about solar energy",
        "a poster about the sea",
        "a video about our product",
        "a desktop app for notes",
        "a mobile app for notes",
        "analyse recording.mp3",
    ):
        plan = planner.plan(Objective.new(request))
        assert plan.requirements, request
        for requirement in plan.requirements:
            assert requirement.checks, f"{request}: {requirement.id} has no check"
            for check in requirement.checks:
                assert check.kind in known_checks(), f"unknown check {check.kind}"


def test_task_dependencies_point_at_real_tasks(planner):
    plan = planner.plan(Objective.new("a deck and a video about solar energy"))
    ids = {task.id for task in plan.tasks}
    for task in plan.tasks:
        for dependency in task.depends_on:
            assert dependency in ids


def test_result_references_point_at_real_tasks(planner):
    plan = planner.plan(Objective.new("a deck about solar energy"))
    ids = {task.id for task in plan.tasks}
    pattern = re.compile(r"^\$\{([^:}]+)::([^}]+)\}$")
    for task in plan.tasks:
        for value in task.args.values():
            if isinstance(value, str) and value.startswith("${"):
                match = pattern.match(value)
                assert match, f"bad reference syntax: {value}"
                assert match.group(1) in ids, f"reference to unknown task: {value}"


def test_deck_plan_orders_plan_before_build(planner):
    plan = planner.plan(Objective.new("a deck about solar energy"))
    build = next(task for task in plan.tasks if task.tool == "slides.create")
    assert "deck.plan" in build.depends_on
    assert build.satisfies


def test_thought_trace_records_options_and_choice(planner):
    plan = planner.plan(Objective.new("make a mobile app for habits"))
    thought = plan.thought
    assert thought["intents"]
    assert thought["chosen"]
    assert thought["rationale"]
    assert thought["capabilities"].get("python")


def test_mobile_request_uses_the_named_framework(planner):
    plan = planner.plan(Objective.new("build a react native app for habits"))
    scaffold = next(task for task in plan.tasks if task.tool == "app.scaffold")
    assert scaffold.args["framework"] == "react-native"
    assert scaffold.args["platform"] == "mobile"


def test_media_request_requires_an_input_file(planner, tmp_path):
    plan = planner.plan(Objective.new("transcribe this recording"))
    ingest = next(task for task in plan.tasks if task.tool == "media.ingest")
    assert ingest.args["path"] == "__missing_media_file__"
    assert any("input" in note.lower() for note in plan.notes)


def test_find_input_path_resolves_relative_and_absolute(tmp_path):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"RIFF")
    assert find_input_path(f"transcribe {sample}", tmp_path) == str(sample.resolve())
    assert find_input_path("transcribe voice.wav", tmp_path) == str(sample.resolve())
    assert find_input_path("transcribe nothing here", tmp_path) == ""


def test_plan_is_reused_for_a_similar_request(config, registry, memory):
    from quanai.core.memory import Memory

    planner = Planner(config, registry, provider=None, memory=memory)
    first = Objective.new("create a 6 slide deck about solar energy in cities")
    plan_one = planner.plan(first)
    assert plan_one.source == "offline"
    memory.remember_plan(first.request, plan_one.to_dict())

    second = Objective.new("create a 6 slide deck about solar energy in cities")
    plan_two = planner.plan(second)
    assert plan_two.source == "memory"
    assert [task.id for task in plan_two.tasks] == [task.id for task in plan_one.tasks]

    # the reused plan must target the *new* request, not replay the old text
    plan_task = next(task for task in plan_two.tasks if task.tool == "plan.deck")
    assert plan_task.args["request"] == second.request
    assert plan_task.status.value == "pending"


def test_reuse_is_skipped_when_a_tool_no_longer_exists(config, registry, memory):
    planner = Planner(config, registry, provider=None, memory=memory)
    objective = Objective.new("make a promo video about our agent")
    plan = planner.plan(objective)
    payload = plan.to_dict()
    payload["tasks"][0]["tool"] = "removed.tool"
    memory.remember_plan(objective.request, payload)

    rebuilt = Planner(config, registry, provider=None, memory=memory).plan(
        Objective.new("make a promo video about our agent")
    )
    assert rebuilt.source == "offline"
