"""Objective / requirement / task scheduling and serialisation."""

from quanai.core.task import Check, Objective, Requirement, Status, Task


def test_dependency_scheduling():
    objective = Objective.new("test")
    objective.add_task(Task(id="a", title="first", tool="fs.write"))
    objective.add_task(Task(id="b", title="second", tool="fs.write", depends_on=["a"]))
    objective.refresh()
    assert [t.id for t in objective.ready_tasks()] == ["a"]

    objective.task("a").status = Status.DONE
    objective.refresh()
    assert [t.id for t in objective.ready_tasks()] == ["b"]


def test_failed_dependency_blocks_downstream():
    objective = Objective.new("test")
    objective.add_task(Task(id="a", title="first", tool="t"))
    objective.add_task(Task(id="b", title="second", tool="t", depends_on=["a"]))
    objective.task("a").status = Status.FAILED
    objective.refresh()
    assert objective.task("b").status is Status.BLOCKED
    assert "upstream" in objective.task("b").error


def test_unknown_dependency_is_blocked_not_crashing():
    objective = Objective.new("test")
    objective.add_task(Task(id="b", title="second", tool="t", depends_on=["ghost"]))
    objective.refresh()
    assert objective.task("b").status is Status.BLOCKED


def test_requirement_verified_only_on_latest_checks():
    from quanai.core.task import Evidence

    req = Requirement(
        id="R1",
        text="a file exists",
        checks=[Check("file_exists", {"path": "x.txt"})],
    )
    assert not req.verified
    req.evidence.append(Evidence(check="file_exists", ok=False, detail="missing"))
    assert not req.verified
    req.evidence.append(Evidence(check="file_exists", ok=True, detail="exists"))
    assert req.verified
    req.evidence.append(Evidence(check="file_exists", ok=False, detail="deleted again"))
    assert not req.verified


def test_progress_and_verified():
    objective = Objective.new("test")
    objective.add_requirement(Requirement(id="R1", text="one", checks=[Check("file_exists", {"path": "a"})]))
    objective.add_requirement(
        Requirement(id="R2", text="two", checks=[Check("file_exists", {"path": "b"})], required=False)
    )
    progress = objective.progress()
    assert progress["requirements_total"] == 1
    assert progress["completion"] == 0.0
    assert not objective.verified()


def test_round_trip_serialisation():
    objective = Objective.new("round trip")
    objective.add_requirement(
        Requirement(id="R1", text="a deck", checks=[Check("pptx_valid", {"path": "d.pptx", "min_slides": 3})])
    )
    objective.add_task(
        Task(
            id="t1",
            title="build",
            tool="slides.create",
            args={"path": "d.pptx", "slides": [{"title": "x"}]},
            satisfies=["R1"],
            checks=[Check("pptx_valid", {"path": "d.pptx"})],
            status=Status.DONE,
            artifacts=[{"path": "d.pptx", "kind": "deck"}],
        )
    )
    restored = Objective.from_dict(objective.to_dict())
    assert restored.id == objective.id
    assert restored.requirements[0].checks[0].params["min_slides"] == 3
    assert restored.tasks[0].status is Status.DONE
    assert restored.tasks[0].artifacts[0]["path"] == "d.pptx"


def test_duplicate_ids_are_ignored():
    objective = Objective.new("test")
    objective.add_task(Task(id="t", title="one", tool="t"))
    objective.add_task(Task(id="t", title="two", tool="t"))
    assert len(objective.tasks) == 1
    assert objective.tasks[0].title == "one"
