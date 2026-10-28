"""The agent loop: plan -> act -> verify -> critique -> repair, until proven done.

Design rules that make this trustworthy:

* every requirement carries a machine check, and the checks are re-run after
  every iteration (the "constant check");
* a task can only be marked DONE when its own checks pass;
* the critic may add repair tasks but can never overrule a failed check;
* state is persisted after every iteration, so an interrupted run resumes;
* the run ends with a written report containing the evidence, not a claim.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..llm.router import build_provider
from ..tools import ToolContext, ToolRegistry, build_default_registry
from ..util import human_bytes, now_iso
from .critic import Critic, Decision
from .ledger import Ledger
from .memory import Memory
from .planner import Planner
from .task import Objective, Requirement, Status, Task
from .verify import Verifier, VerifyContext

_UNSET = object()
REF = re.compile(r"^\$\{(?P<task>[^:}]+)::(?P<path>[^}]+)\}$")


class ResolutionError(RuntimeError):
    """A task argument pointed at a result that does not exist."""


@dataclass
class RunReport:
    objective_id: str
    status: str
    completion: float
    verified: bool
    iterations: int
    duration: float
    provider: str
    workspace: str
    report_path: str
    artifacts: list[dict] = field(default_factory=list)
    requirements: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    request: str = ""

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "status": self.status,
            "completion": self.completion,
            "verified": self.verified,
            "iterations": self.iterations,
            "duration": round(self.duration, 2),
            "provider": self.provider,
            "workspace": self.workspace,
            "report_path": self.report_path,
            "artifacts": self.artifacts,
            "requirements": self.requirements,
            "tasks": self.tasks,
            "insights": self.insights,
            "blocked_reason": self.blocked_reason,
            "request": self.request,
        }

    def summary(self) -> str:
        lines = [
            f"Objective {self.objective_id}  [{self.status}]  {self.completion}% verified",
            f"Request: {self.request}",
            f"Provider: {self.provider}   Iterations: {self.iterations}   Time: {self.duration:.1f}s",
            "",
            "Requirements:",
        ]
        for req in self.requirements:
            mark = "ok  " if req["verified"] else "FAIL"
            lines.append(f"  [{mark}] {req['id']}: {req['text']}")
            for check in req.get("evidence", [])[-len(req.get("checks", [])) :]:
                lines.append(f"          {'+'} {check['detail']}")
        lines += ["", "Artifacts:"]
        if self.artifacts:
            for art in self.artifacts:
                lines.append(f"  {art.get('path')}  ({human_bytes(art.get('bytes', 0))})")
        else:
            lines.append("  (none)")
        if self.blocked_reason:
            lines += ["", f"Blocked: {self.blocked_reason}"]
        if self.insights:
            lines += ["", "Notes from review:"] + [f"  - {item}" for item in self.insights]
        lines += ["", f"Report: {self.report_path}"]
        return "\n".join(lines)


class Agent:
    """Orchestrates planning, tool execution, verification and repair."""

    def __init__(
        self,
        config: Config | None = None,
        provider=_UNSET,
        registry: ToolRegistry | None = None,
        ledger: Ledger | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.config = (config or Config.from_env()).ensure_dirs()
        self.provider = build_provider(self.config) if provider is _UNSET else provider
        self.registry = registry or build_default_registry()
        self.memory = memory or Memory(self.config.home / "memory")
        self.ledger = ledger or Ledger(self.config.home)
        self.critic = Critic(self.config, self.provider, self.memory)
        self.planner = Planner(self.config, self.registry, self.provider, self.memory)
        self.insights: list[str] = []

    # ------------------------------------------------------------------
    def run(
        self,
        request: str,
        objective_id: str | None = None,
        max_iterations: int | None = None,
        attach: list[str] | None = None,
    ) -> RunReport:
        started = time.time()
        self.insights = []  # each run reports on itself only
        if attach:
            resolved = [str(Path(p).expanduser().resolve()) for p in attach]
            request = f"{request} Input file: {' '.join(resolved)}"
        objective = Objective.new(request, workspace="")
        if objective_id:
            objective.id = objective_id
        objective.workspace = str(self.config.objective_dir(objective.id))

        plan = self.planner.plan(objective)
        objective.requirements = plan.requirements
        objective.tasks = plan.tasks
        objective.plan_source = plan.source
        objective.thoughts.append(plan.thought)
        objective.status = Status.RUNNING
        self.ledger.save(objective)
        self.ledger.log(
            objective.id, "planned",
            source=plan.source,
            requirements=len(plan.requirements),
            tasks=[t.id for t in plan.tasks],
            chosen=plan.thought.get("chosen", []),
        )
        if plan.notes:
            self.memory.add("plan", " | ".join(plan.notes), tags=[t.id for t in plan.tasks])
        self.memory.remember_plan(request, plan.to_dict())
        return self._loop(objective, started, max_iterations)

    # ------------------------------------------------------------------
    def resume(self, objective_id: str, max_iterations: int | None = None) -> RunReport:
        objective = self.ledger.load(objective_id)
        if objective is None:
            raise FileNotFoundError(f"no objective named {objective_id}")
        started = time.time()
        if objective.status is Status.DONE and objective.verified():
            verification = self._verify(objective)
            return self._finish(objective, started, verification, None)
        objective.status = Status.RUNNING
        for task in objective.tasks:
            if task.status is Status.RUNNING:
                task.status = Status.READY
        if not objective.workspace:
            objective.workspace = str(self.config.objective_dir(objective.id))
        return self._loop(objective, started, max_iterations)

    # ------------------------------------------------------------------
    def _loop(self, objective: Objective, started: float, max_iterations: int | None) -> RunReport:
        budget = int(max_iterations or self.config.max_iterations)
        verification: dict = {"ok": False, "results": {}}
        decision: Decision | None = None
        try:
            for iteration in range(1, budget + 1):
                objective.iteration = iteration
                objective.refresh()
                self.ledger.save(objective)
                # drain every runnable task before judging the objective, so a
                # multi-step chain finishes inside one iteration
                executed = 0
                while True:
                    ready = objective.ready_tasks()
                    if not ready:
                        break
                    self.ledger.log(
                        objective.id, "iteration", n=iteration,
                        runnable=[t.id for t in ready], step=executed,
                    )
                    for task in ready:
                        self._execute(objective, task)
                        executed += 1
                        objective.refresh()
                        self.ledger.save(objective)
                    if executed >= 50:  # safety valve against runaway graphs
                        break

                verification = self._verify(objective)
                decision = self.critic.review(objective, iteration, verification)
                self.ledger.log(
                    objective.id, "critique",
                    complete=decision.complete,
                    summary=decision.summary,
                    failures=[g["requirement"] for g in decision.gaps],
                    revisions=[t.id for t in decision.revisions],
                    escalate=decision.escalate,
                )
                self.insights.extend(i for i in decision.insights if i not in self.insights)

                if objective.verified():
                    objective.status = Status.DONE
                    break

                if decision.escalate and not decision.revisions:
                    objective.status = Status.BLOCKED
                    objective.blocked_reason = decision.blocked_reason or decision.summary
                    break

                if decision.revisions:
                    for revision in decision.revisions:
                        if objective.task(revision.id) is None:
                            objective.add_task(revision)
                            self.ledger.log(objective.id, "repair", task=revision.id, tool=revision.tool,
                                            strategy=revision.strategy, satisfies=revision.satisfies)
                    self.ledger.save(objective)
                    continue

                if not objective.ready_tasks() and not objective.unfinished_tasks():
                    objective.status = Status.BLOCKED
                    objective.blocked_reason = decision.blocked_reason or "no runnable tasks remain"
                    break

                if not objective.ready_tasks():
                    objective.status = Status.BLOCKED
                    objective.blocked_reason = (
                        decision.blocked_reason or "task graph is stuck: nothing is ready but work remains"
                    )
                    break
        finally:
            verification = self._verify(objective)
            if objective.verified():
                objective.status = Status.DONE
                objective.blocked_reason = ""
            elif objective.status is not Status.BLOCKED:
                objective.status = Status.BLOCKED
                objective.blocked_reason = objective.blocked_reason or (
                    f"iteration budget of {budget} exhausted with unverified requirements"
                )
            self.ledger.save(objective)
        return self._finish(objective, started, verification, decision)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def _context(self, objective: Objective) -> ToolContext:
        return ToolContext(
            workspace=Path(objective.workspace),
            config=self.config,
            objective_id=objective.id,
            provider=self.provider,
        )

    def _resolve_args(self, value, objective: Objective):
        if isinstance(value, str):
            match = REF.match(value.strip())
            if not match:
                return value
            task_id, dotted = match.group("task"), match.group("path")
            task = objective.task(task_id)
            if task is None:
                raise ResolutionError(f"no task named {task_id!r} for reference {value!r}")
            payload = (task.result or {}).get("data", {})
            cursor = payload
            for part in dotted.split("."):
                if isinstance(cursor, dict) and part in cursor:
                    cursor = cursor[part]
                elif isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
                    cursor = cursor[int(part)]
                else:
                    raise ResolutionError(
                        f"reference {value!r} not found (task {task_id} status {task.status.value})"
                    )
            return cursor
        if isinstance(value, list):
            return [self._resolve_args(item, objective) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_args(item, objective) for key, item in value.items()}
        return value

    def _execute(self, objective: Objective, task: Task) -> None:
        task.attempts += 1
        task.status = Status.RUNNING
        task.started_at = now_iso()
        self.ledger.log(objective.id, "task_start", task=task.id, tool=task.tool, attempt=task.attempts)
        ctx = self._context(objective)

        try:
            args = self._resolve_args(task.args, objective)
        except ResolutionError as exc:
            task.error = str(exc)
            task.status = Status.FAILED if task.attempts >= task.max_attempts else Status.READY
            task.finished_at = now_iso()
            self.ledger.log(objective.id, "task_error", task=task.id, error=task.error)
            return

        result = self.registry.execute(task.tool, args, ctx)
        task.result = result.to_dict()
        for art in result.artifacts:
            record = art.to_dict()
            if record not in task.artifacts:
                task.artifacts.append(record)

        if not result.ok:
            task.error = result.error
            task.note(f"attempt {task.attempts} failed: {result.error[:240]}")
            if task.attempts < max(1, task.max_attempts):
                task.status = Status.READY
                advice = self.memory.advice_for(task.tool, task.error)
                if advice:
                    task.note("memory advice: " + "; ".join(advice))
            else:
                task.status = Status.FAILED
            task.finished_at = now_iso()
            self.ledger.log(
                objective.id, "task_failed", task=task.id, tool=task.tool,
                error=task.error[:400], attempts=task.attempts,
            )
            return

        # a tool saying "ok" is not proof: run the task's own checks
        evidence = self.verifier_for(objective).verify_task(task, objective)
        failed = [e for e in evidence if not e.ok]
        task.notes.extend(f"check {e.check}: {'ok' if e.ok else 'FAILED'} - {e.detail}" for e in evidence)
        if failed and task.checks:
            task.error = "verification failed: " + "; ".join(e.detail for e in failed[:3])
            if task.attempts < max(1, task.max_attempts):
                task.status = Status.READY
            else:
                task.status = Status.FAILED
            task.finished_at = now_iso()
            self.ledger.log(objective.id, "task_unverified", task=task.id, error=task.error[:400])
            return

        task.status = Status.DONE
        task.error = ""
        task.finished_at = now_iso()
        self.memory.add(
            "artifact" if result.artifacts else "step",
            f"{task.tool} produced {len(result.artifacts)} artifact(s) for {task.title}",
            data={"objective": objective.id, "task": task.id, "artifacts": [a.path for a in result.artifacts]},
            tags=[task.tool, *task.satisfies],
        )
        self.ledger.log(
            objective.id, "task_done", task=task.id, tool=task.tool,
            artifacts=[a.path for a in result.artifacts], duration=round(result.duration, 2),
        )

    # ------------------------------------------------------------------
    def verifier_for(self, objective: Objective) -> Verifier:
        return Verifier(Path(objective.workspace))

    def _verify(self, objective: Objective) -> dict:
        verification = self.verifier_for(objective).verify_objective(objective)
        self.ledger.log(
            objective.id, "verify",
            passed=verification["passed"], failed=verification["failed"],
            results=verification["results"],
        )
        return verification

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _finish(
        self,
        objective: Objective,
        started: float,
        verification: dict,
        decision: Decision | None,
    ) -> RunReport:
        duration = time.time() - started
        progress = objective.progress()
        report_path = self._write_report(objective, verification, decision, duration)
        self.ledger.save(objective)
        self.ledger.log(
            objective.id, "finished",
            status=objective.status.value, completion=progress["completion"],
            duration=round(duration, 2), artifacts=len(objective.artifacts()),
        )
        return RunReport(
            objective_id=objective.id,
            status=objective.status.value,
            completion=progress["completion"],
            verified=objective.verified(),
            iterations=objective.iteration,
            duration=duration,
            provider=(
                f"{self.provider.name}:{self.provider.model}" if self.provider is not None else "offline (deterministic)"
            ),
            workspace=objective.workspace,
            report_path=str(report_path),
            artifacts=objective.artifacts(),
            requirements=[
                {
                    "id": r.id,
                    "text": r.text,
                    "required": r.required,
                    "verified": r.verified,
                    "checks": [c.kind for c in r.checks],
                    "evidence": [e.to_dict() for e in r.evidence[-len(r.checks) :]] if r.checks else [],
                }
                for r in objective.requirements
            ],
            tasks=[
                {
                    "id": t.id,
                    "tool": t.tool,
                    "status": t.status.value,
                    "attempts": t.attempts,
                    "strategy": t.strategy,
                    "error": t.error,
                    "artifacts": [a.get("path") for a in t.artifacts],
                }
                for t in objective.tasks
            ],
            insights=self.insights,
            blocked_reason=objective.blocked_reason,
            request=objective.request,
        )

    def _write_report(
        self,
        objective: Objective,
        verification: dict,
        decision: Decision | None,
        duration: float,
    ) -> Path:
        progress = objective.progress()
        lines = [
            f"# Objective report - {objective.id}",
            "",
            f"- **Request:** {objective.request}",
            f"- **Status:** {objective.status.value}",
            f"- **Verified:** {progress['requirements_verified']}/{progress['requirements_total']} requirements "
            f"({progress['completion']}%)",
            f"- **Plan source:** {objective.plan_source}",
            f"- **Iterations:** {objective.iteration}",
            f"- **Duration:** {duration:.1f}s",
            f"- **Workspace:** {objective.workspace}",
            "",
        ]
        if objective.blocked_reason:
            lines += [f"> Blocked: {objective.blocked_reason}", ""]

        lines += ["## Requirements", "", "| id | required | verified | requirement |", "| --- | --- | --- | --- |"]
        for req in objective.requirements:
            lines.append(
                f"| {req.id} | {'yes' if req.required else 'no'} | {'PASS' if req.verified else 'FAIL'} | {req.text} |"
            )
        lines.append("")

        lines += ["## Evidence", ""]
        for req in objective.requirements:
            lines.append(f"### {req.id}: {'PASS' if req.verified else 'FAIL'}")
            if not req.checks:
                lines.append("- no machine check attached")
            for check in req.checks:
                match = [e for e in req.evidence if e.check == check.describe()]
                latest = match[-1] if match else None
                status = "PASS" if latest and latest.ok else "FAIL"
                detail = latest.detail if latest else "not run"
                lines.append(f"- `{check.describe()}` -> **{status}**: {detail}")
            lines.append("")

        lines += ["## Tasks", "", "| id | tool | status | attempts | strategy |", "| --- | --- | --- | --- | --- |"]
        for task in objective.tasks:
            lines.append(
                f"| {task.id} | {task.tool} | {task.status.value} | {task.attempts} | {task.strategy or '-'} |"
            )
        lines.append("")

        lines += ["## Artifacts", ""]
        artifacts = objective.artifacts()
        if artifacts:
            for art in artifacts:
                lines.append(
                    f"- `{art.get('path')}` - {art.get('kind', 'file')} "
                    f"({human_bytes(art.get('bytes', 0))})"
                )
        else:
            lines.append("- none")
        lines.append("")

        if objective.thoughts:
            thought = objective.thoughts[0]
            lines += ["## Reasoning trace", "", f"Intents detected: {', '.join(thought.get('intents', []))}", ""]
            lines.append(f"Chosen path: {'; '.join(thought.get('chosen', [])) or 'n/a'}")
            lines.append("")
            lines.append(f"Rationale: {thought.get('rationale', '')}")
            lines.append("")

        if objective.lessons:
            lines += ["## Lessons recorded", ""]
            for lesson in objective.lessons[-8:]:
                lines.append(f"- **{lesson.get('tool', '?')}**: {lesson.get('advice', '')}")
            lines.append("")

        if self.insights:
            lines += ["## Review notes", ""] + [f"- {item}" for item in self.insights] + [""]

        next_steps = []
        for req in objective.requirements:
            if req.required and not req.verified:
                failure = req.last_failure.detail if req.last_failure else "never verified"
                next_steps.append(f"- {req.id}: {failure}")
        if next_steps:
            lines += ["## Remaining work", ""] + next_steps + [""]

        target = Path(objective.workspace) / "REPORT.md"
        target.write_text("\n".join(lines), encoding="utf-8")
        manifest = Path(objective.workspace) / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "objective": objective.to_dict(),
                    "verification": verification,
                    "critique": decision.to_dict() if decision else None,
                    "insights": self.insights,
                    "generated_at": now_iso(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return target
