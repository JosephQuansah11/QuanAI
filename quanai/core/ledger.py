"""Durable state: objectives on disk plus an append-only event log."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..util import append_jsonl, atomic_write_json, now_iso, read_json, read_jsonl
from .task import Objective, Status


class Ledger:
    """Crash-safe persistence for objectives.

    One JSON document per objective plus an index, so a run interrupted at any
    point can be resumed with ``Agent.resume(objective_id)``.
    """

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.dir = self.home / "objectives"
        self.logs = self.home / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"

    # -- paths ----------------------------------------------------------
    def path_for(self, objective_id: str) -> Path:
        return self.dir / f"{objective_id}.json"

    def log_path(self, objective_id: str) -> Path:
        return self.logs / f"{objective_id}.jsonl"

    # -- objectives -----------------------------------------------------
    def save(self, objective: Objective) -> Path:
        objective.touch()
        path = atomic_write_json(self.path_for(objective.id), objective.to_dict())
        self._update_index(objective)
        return path

    def load(self, objective_id: str) -> Optional[Objective]:
        data = read_json(self.path_for(objective_id))
        if not data:
            return None
        return Objective.from_dict(data)

    def latest(self) -> Optional[Objective]:
        entries = self.list()
        if not entries:
            return None
        return self.load(entries[0]["id"])

    def list(self) -> list[dict]:
        entries = read_json(self.index_path, default=[]) or []
        return sorted(entries, key=lambda e: e.get("updated_at", ""), reverse=True)

    def delete(self, objective_id: str) -> bool:
        path = self.path_for(objective_id)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        entries = [e for e in (read_json(self.index_path, default=[]) or []) if e.get("id") != objective_id]
        atomic_write_json(self.index_path, entries)
        return deleted

    def _update_index(self, objective: Objective) -> None:
        entries = read_json(self.index_path, default=[]) or []
        progress = objective.progress()
        record = {
            "id": objective.id,
            "request": objective.request[:180],
            "status": objective.status.value,
            "completion": progress["completion"],
            "requirements_verified": progress["requirements_verified"],
            "requirements_total": progress["requirements_total"],
            "tasks_done": progress["tasks_done"],
            "tasks_total": progress["tasks_total"],
            "updated_at": objective.updated_at,
            "created_at": objective.created_at,
        }
        entries = [e for e in entries if e.get("id") != objective.id]
        entries.append(record)
        atomic_write_json(self.index_path, entries)

    # -- events ---------------------------------------------------------
    def log(self, objective_id: str, event: str, **fields) -> None:
        append_jsonl(
            self.log_path(objective_id),
            {"at": now_iso(), "event": event, "objective": objective_id, **fields},
        )

    def events(self, objective_id: str) -> list[dict]:
        return read_jsonl(self.log_path(objective_id))

    # -- summarising ----------------------------------------------------
    def summary(self) -> list[dict]:
        rows: Iterable[dict] = self.list()
        return [
            {
                "id": r.get("id"),
                "status": r.get("status"),
                "completion": r.get("completion"),
                "request": r.get("request"),
                "updated_at": r.get("updated_at"),
            }
            for r in rows
        ]

    def terminal(self, objective_id: str) -> bool:
        obj = self.load(objective_id)
        if obj is None:
            return True
        return obj.status in {Status.DONE, Status.BLOCKED, Status.FAILED}
