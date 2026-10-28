"""The CLI is the primary interface; it must work without any configuration."""

import json
from pathlib import Path

from quanai.cli import main


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return ["--workspace", str(tmp_path / "ws"), "--home", str(tmp_path / "state"), *extra]


def test_tools_lists_the_surface(capsys):
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "slides.create" in out and "image.generate" in out and "app.scaffold" in out


def test_doctor_works_offline(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Capabilities" in out
    assert "offline deterministic engine" in out


def test_plan_prints_requirements_and_tasks(tmp_path: Path, capsys):
    assert main(_args(tmp_path, "plan", "a 5 slide deck about solar energy")) == 0
    out = capsys.readouterr().out
    assert "R.deck" in out
    assert "slides.create" in out


def test_run_then_status_then_show(tmp_path: Path, capsys):
    code = main(_args(tmp_path, "run", "generate a poster image about the ocean"))
    assert code == 0
    output = capsys.readouterr().out
    assert "100.0% verified" in output

    assert main(_args(tmp_path, "status")) == 0
    listing = capsys.readouterr().out
    assert "done" in listing

    rows = [line.split() for line in listing.splitlines() if line.startswith("obj-")]
    assert rows, listing
    objective_id = rows[0][0]
    assert rows[0][1] == "done"

    assert main(_args(tmp_path, "show", objective_id)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == objective_id


def test_run_json_output(tmp_path: Path, capsys):
    code = main(_args(tmp_path, "run", "generate an image about the ocean", "--json"))
    assert code == 0
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload["verified"] is True
    assert payload["artifacts"]


def test_resume_reports_missing_objective(tmp_path: Path):
    try:
        main(_args(tmp_path, "resume", "obj-nope"))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for an unknown objective")
