"""The objective / requirement / task domain model.

Everything here is plain data so it can be serialised to JSON and resumed
after a crash.  A task is only considered successful when the verification
checks attached to it (and to the objective's requirements) actually pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from ..util import now_iso, short_id


class Status(str, Enum):
    PENDING = "pending"      # waiting on dependencies
    READY = "ready"          # can run now
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"        # tool raised / verification failed after retries
    BLOCKED = "blocked"      # missing capability, needs a human decision
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in {Status.DONE, Status.FAILED, Status.BLOCKED, Status.SKIPPED}


@dataclass
class Check:
    """A machine-verifiable assertion about the produced artifacts."""

    kind: str
    params: dict = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "label": self.label}

    @classmethod
    def from_dict(cls, data: dict) -> "Check":
        return cls(
            kind=data.get("kind", ""),
            params=dict(data.get("params") or {}),
            label=data.get("label", ""),
        )

    def describe(self) -> str:
        if self.label:
            return self.label
        bits = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind}({bits})"


@dataclass
class Evidence:
    """The outcome of running one check."""

    check: str
    ok: bool
    detail: str = ""
    observed: Any = None
    at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            check=data.get("check", ""),
            ok=bool(data.get("ok")),
            detail=data.get("detail", ""),
            observed=data.get("observed"),
            at=data.get("at") or now_iso(),
        )


@dataclass
class Requirement:
    """What the user asked for, expressed so it can be tested."""

    id: str
    text: str
    checks: list[Check] = field(default_factory=list)
    required: bool = True
    status: Status = Status.PENDING
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Verified only when the latest run of every check passed."""
        if not self.checks:
            return self.status is Status.DONE
        recent = self.evidence[-len(self.checks) :]
        return len(recent) == len(self.checks) and all(e.ok for e in recent)

    @property
    def last_failure(self) -> Optional[Evidence]:
        for ev in reversed(self.evidence):
            if not ev.ok:
                return ev
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "checks": [c.to_dict() for c in self.checks],
            "required": self.required,
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Requirement":
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            checks=[Check.from_dict(c) for c in data.get("checks", [])],
            required=bool(data.get("required", True)),
            status=Status(data.get("status", "pending")),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
        )


@dataclass
class Task:
    """One unit of work: a tool call plus the checks proving it worked."""

    id: str
    title: str
    goal: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    satisfies: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    status: Status = Status.PENDING
    attempts: int = 0
    max_attempts: int = 3
    strategy: str = ""
    result: dict | None = None
    artifacts: list[dict] = field(default_factory=list)
    error: str = ""
    notes: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = Status(self.status)

    def note(self, message: str) -> None:
        self.notes.append(f"{now_iso()} {message}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "tool": self.tool,
            "args": self.args,
            "depends_on": list(self.depends_on),
            "satisfies": list(self.satisfies),
            "checks": [c.to_dict() for c in self.checks],
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "strategy": self.strategy,
            "result": self.result,
            "artifacts": list(self.artifacts),
            "error": self.error,
            "notes": list(self.notes),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            goal=data.get("goal", ""),
            tool=data.get("tool", ""),
            args=dict(data.get("args") or {}),
            depends_on=list(data.get("depends_on") or []),
            satisfies=list(data.get("satisfies") or []),
            checks=[Check.from_dict(c) for c in data.get("checks", [])],
            status=Status(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0)),
            max_attempts=int(data.get("max_attempts", 3)),
            strategy=data.get("strategy", ""),
            result=data.get("result"),
            artifacts=list(data.get("artifacts") or []),
            error=data.get("error", ""),
            notes=list(data.get("notes") or []),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )


@dataclass
class Objective:
    """A durable record of one user request and the plan built for it."""

    id: str
    request: str
    requirements: list[Requirement] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    status: Status = Status.PENDING
    iteration: int = 0
    plan_source: str = "offline"
    thoughts: list[dict] = field(default_factory=list)
    lessons: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    workspace: str = ""
    blocked_reason: str = ""

    @classmethod
    def new(cls, request: str, workspace: str = "") -> "Objective":
        return cls(id=f"obj-{short_id('', 8)}", request=request, workspace=workspace)

    # -- lookups --------------------------------------------------------
    def task(self, task_id: str) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def requirement(self, req_id: str) -> Optional[Requirement]:
        return next((r for r in self.requirements if r.id == req_id), None)

    def add_task(self, task: Task) -> Task:
        existing = self.task(task.id)
        if existing is not None:
            return existing
        self.tasks.append(task)
        return task

    def add_requirement(self, requirement: Requirement) -> Requirement:
        existing = self.requirement(requirement.id)
        if existing is not None:
            return existing
        self.requirements.append(requirement)
        return requirement

    # -- scheduling -----------------------------------------------------
    def refresh(self) -> None:
        """Recompute readiness from dependency statuses (fixpoint)."""
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.status not in {Status.PENDING, Status.READY}:
                    continue
                deps = [self.task(d) for d in task.depends_on]
                unknown = [d for d, t in zip(task.depends_on, deps) if t is None]
                if unknown:
                    task.status = Status.BLOCKED
                    task.error = "unknown dependency: " + ", ".join(unknown)
                    changed = True
                    continue
                bad = [t for t in deps if t and t.status in {Status.FAILED, Status.BLOCKED}]
                if bad:
                    task.status = Status.BLOCKED
                    task.error = "upstream task failed: " + ", ".join(t.id for t in bad)
                    changed = True
                    continue
                waiting = [t for t in deps if t and t.status is not Status.DONE]
                desired = Status.READY if not waiting else Status.PENDING
                if task.status is not desired:
                    task.status = desired
                    changed = True

    def ready_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status is Status.READY]

    def failed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status in {Status.FAILED, Status.BLOCKED}]

    def unfinished_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status not in {Status.DONE, Status.SKIPPED}]

    def blocking_failures(self) -> list[Task]:
        """Failed tasks that still matter.

        A failed attempt is superseded when every requirement it was meant to
        satisfy has since been verified by a repair task.
        """
        blocking: list[Task] = []
        for task in self.failed_tasks():
            if task.status is Status.BLOCKED:
                blocking.append(task)
                continue
            if not task.satisfies:
                blocking.append(task)
                continue
            superseded = True
            for req_id in task.satisfies:
                requirement = self.requirement(req_id)
                if requirement is None or not requirement.verified:
                    superseded = False
                    break
            if not superseded:
                blocking.append(task)
        return blocking

    # -- progress -------------------------------------------------------
    def progress(self) -> dict:
        counts = {s.value: 0 for s in Status}
        for task in self.tasks:
            counts[task.status.value] += 1
        required = [r for r in self.requirements if r.required]
        verified = [r for r in required if r.verified]
        total = len(required) or 1
        return {
            "tasks_total": len(self.tasks),
            "tasks_done": counts[Status.DONE.value],
            "tasks_failed": counts[Status.FAILED.value] + counts[Status.BLOCKED.value],
            "task_status": counts,
            "requirements_total": len(required),
            "requirements_verified": len(verified),
            "completion": round(100.0 * len(verified) / total, 1),
        }

    def verified(self) -> bool:
        required = [r for r in self.requirements if r.required]
        if not required:
            return False
        return all(r.verified for r in required) and not self.blocking_failures()

    def artifacts(self) -> list[dict]:
        seen: list[dict] = []
        paths: set = set()
        for task in self.tasks:
            for art in task.artifacts:
                if art.get("path") not in paths:
                    paths.add(art.get("path"))
                    seen.append(art)
        return seen

    def pending_requirements(self) -> Iterable[Requirement]:
        return (r for r in self.requirements if r.required and not r.verified)

    # -- serialisation --------------------------------------------------
    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "request": self.request,
            "status": self.status.value,
            "iteration": self.iteration,
            "plan_source": self.plan_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "blocked_reason": self.blocked_reason,
            "requirements": [r.to_dict() for r in self.requirements],
            "tasks": [t.to_dict() for t in self.tasks],
            "thoughts": list(self.thoughts),
            "lessons": list(self.lessons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Objective":
        return cls(
            id=data["id"],
            request=data.get("request", ""),
            requirements=[Requirement.from_dict(r) for r in data.get("requirements", [])],
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            status=Status(data.get("status", "pending")),
            iteration=int(data.get("iteration", 0)),
            plan_source=data.get("plan_source", "offline"),
            thoughts=list(data.get("thoughts") or []),
            lessons=list(data.get("lessons") or []),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            workspace=data.get("workspace", ""),
            blocked_reason=data.get("blocked_reason", ""),
        )
