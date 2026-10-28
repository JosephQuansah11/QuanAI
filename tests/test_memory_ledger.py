"""Durable state: the ledger must survive restarts, memory must learn."""

import json
from pathlib import Path

from quanai.core.ledger import Ledger
from quanai.core.memory import Memory
from quanai.core.task import Objective, Requirement, Status, Task


def test_ledger_round_trip(tmp_path: Path):
    ledger = Ledger(tmp_path / "state")
    objective = Objective.new("build a deck")
    objective.add_requirement(Requirement(id="R1", text="deck exists"))
    objective.add_task(Task(id="t1", title="build", tool="slides.create", status=Status.DONE))
    ledger.save(objective)

    reloaded = ledger.load(objective.id)
    assert reloaded is not None
    assert reloaded.request == "build a deck"
    assert reloaded.tasks[0].status is Status.DONE


def test_ledger_index_and_latest(tmp_path: Path):
    ledger = Ledger(tmp_path / "state")
    first = Objective.new("first request")
    ledger.save(first)
    second = Objective.new("second request")
    ledger.save(second)

    summary = ledger.summary()
    assert len(summary) == 2
    assert ledger.latest().id in {first.id, second.id}
    assert ledger.latest().id == second.id


def test_ledger_events_append(tmp_path: Path):
    ledger = Ledger(tmp_path / "state")
    objective = Objective.new("events")
    ledger.save(objective)
    ledger.log(objective.id, "test_event", alpha=1)
    ledger.log(objective.id, "test_event", alpha=2)
    events = ledger.events(objective.id)
    assert [event["alpha"] for event in events] == [1, 2]


def test_ledger_delete_removes_index_entry(tmp_path: Path):
    ledger = Ledger(tmp_path / "state")
    objective = Objective.new("to delete")
    ledger.save(objective)
    assert ledger.delete(objective.id) is True
    assert ledger.load(objective.id) is None
    assert ledger.summary() == []


def test_memory_search_ranks_by_overlap(memory: Memory):
    memory.add("note", "the solar deck needed more slides about grid capacity")
    memory.add("note", "a mobile app about habit tracking with local storage")
    hits = memory.search("solar grid capacity slides")
    assert hits
    assert "solar" in hits[0].text


def test_memory_learns_and_recalls_lessons(memory: Memory):
    memory.learn(
        "video.from_images",
        "ffmpeg failed: filter graph error on zoompan",
        "drop zoompan and lower the resolution before retrying",
    )
    advice = memory.advice_for("video.from_images", "ffmpeg failed: filter graph error on zoompan")
    assert advice and "zoompan" in advice[0]
    assert memory.advice_for("slides.create", "something else") == []
    assert memory.stats()["lessons"] == 1


def test_memory_plan_reuse(memory: Memory):
    memory.remember_plan(
        "create a deck about solar energy in cities",
        {"tasks": [{"id": "deck.plan"}]},
    )
    assert memory.recall_plan("create a deck about solar energy in cities") is not None
    assert memory.recall_plan("write a poem about the sea") is None


def test_memory_items_are_jsonl(tmp_path: Path):
    memory = Memory(tmp_path / "mem")
    memory.add("step", "did a thing", tags=["alpha"])
    raw = (tmp_path / "mem/items.jsonl").read_text().strip().splitlines()
    assert json.loads(raw[0])["text"] == "did a thing"
