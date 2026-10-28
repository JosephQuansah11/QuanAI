"""Critique and repair: decide whether the objective is really finished.

The critic is evidence-driven.  A requirement that failed its check cannot be
waved through by a language model; instead the critic produces a concrete
repair task with a *different* strategy, records a lesson, and - if the same
failure repeats without progress - escalates or blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..util import digest, now_iso
from .memory import Memory
from .task import Check, Objective, Status, Task

REVIEW_PROMPT = """You are reviewing an automated build for a user request.
Return ONLY JSON: {"insights": [str], "risks": [str], "improvements": [str]}
Be concrete and critical. Do not claim anything passed that the evidence says failed.

Request: {request}
Verification (passed/failed per requirement): {verification}
Failed tasks: {failures}
Artifacts: {artifacts}
"""


@dataclass
class Decision:
    complete: bool
    summary: str
    gaps: list[dict] = field(default_factory=list)
    revisions: list[Task] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    lessons: list[dict] = field(default_factory=list)
    escalate: bool = False
    blocked_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "summary": self.summary,
            "gaps": self.gaps,
            "revisions": [t.to_dict() for t in self.revisions],
            "insights": self.insights,
            "lessons": self.lessons,
            "escalate": self.escalate,
            "blocked_reason": self.blocked_reason,
        }


def _next_light_theme(current: str | None) -> str:
    from ..tools.slides import DEFAULT_THEME, LIGHT_THEMES

    if current not in LIGHT_THEMES:
        return DEFAULT_THEME
    return LIGHT_THEMES[(LIGHT_THEMES.index(current) + 1) % len(LIGHT_THEMES)]


class Critic:
    def __init__(self, config=None, provider=None, memory: Memory | None = None) -> None:
        self.config = config
        self.provider = provider
        self.memory = memory

    # ------------------------------------------------------------------
    def review(self, objective: Objective, iteration: int, verification: dict | None = None) -> Decision:
        verification = verification or {}
        failed_reqs = [r for r in objective.requirements if r.required and not r.verified]
        failed_tasks = [t for t in objective.tasks if t.status in {Status.FAILED, Status.BLOCKED}]

        def still_runnable(requirement_id: str) -> bool:
            return any(
                requirement_id in task.satisfies
                and task.status in {Status.PENDING, Status.READY, Status.RUNNING}
                for task in objective.tasks
            )

        gaps = [
            {
                "requirement": r.id,
                "text": r.text,
                "failure": (r.last_failure.detail if r.last_failure else "never verified"),
                "checks": [c.kind for c in r.checks],
            }
            for r in failed_reqs
            if not still_runnable(r.id)
        ]

        if not failed_reqs and not failed_tasks:
            return Decision(
                complete=True,
                summary=f"All {len(objective.requirements)} requirements verified on iteration {iteration}.",
                insights=self._insights(objective, verification) if iteration > 1 else [],
            )

        revisions: list[Task] = []
        lessons: list[dict] = []
        escalate = False
        blocked_reason = ""

        # a missing input file cannot be repaired by retrying
        missing_input = (
            any("__missing_media_file__" in json.dumps(task.args) for task in objective.tasks)
            and not any(task.status is Status.DONE for task in objective.tasks)
        )
        if missing_input:
            blocked_reason = (
                "No media file was supplied. Re-run with: quanai run ... --input <file>"
            )
            escalate = True

        for task in failed_tasks:
            if task.attempts < max(1, int(task.max_attempts)):
                task.status = Status.READY
                task.note(f"retry scheduled after: {task.error[:160]}")
                continue
            if task.tool == "media.ingest" and "__missing_media_file__" in json.dumps(task.args):
                continue

        by_id: dict[str, Task] = {}
        for requirement in failed_reqs:
            if blocked_reason:
                break
            revision = self._repair_task(objective, requirement, iteration)
            if revision is None:
                continue
            if objective.task(revision.id) is not None:
                continue
            existing = by_id.get(revision.id)
            if existing is not None:
                # two requirements can be repaired by one regenerated artifact
                for req_id in revision.satisfies:
                    if req_id not in existing.satisfies:
                        existing.satisfies.append(req_id)
                continue
            by_id[revision.id] = revision
            revisions.append(revision)

        if not revisions and not blocked_reason:
            # nothing new to try: the same failure has repeated
            repeated = [g for g in gaps if g["failure"] != "never verified"]
            if repeated:
                escalate = True
                blocked_reason = (
                    "Stuck on " + ", ".join(g["requirement"] for g in repeated) +
                    ": " + repeated[0]["failure"][:200]
                )

        for gap in gaps:
            lesson = self._learn(objective, gap)
            if lesson:
                lessons.append(lesson)
        objective.lessons.extend(l for l in lessons if l not in objective.lessons)

        insights = self._insights(objective, verification) if escalate else []
        summary = (
            f"iteration {iteration}: {len(failed_reqs)} requirement(s) unverified, "
            f"{len(failed_tasks)} task(s) failed, {len(revisions)} repair task(s) queued"
        )
        return Decision(
            complete=False,
            summary=summary,
            gaps=gaps,
            revisions=revisions,
            insights=insights,
            lessons=lessons,
            escalate=escalate,
            blocked_reason=blocked_reason,
        )

    # ------------------------------------------------------------------
    def _repair_task(self, objective: Objective, requirement, iteration: int) -> Task | None:
        """Create a new task that retries the work with a different strategy."""
        source = None
        for task in objective.tasks:
            if requirement.id in task.satisfies and task.tool:
                source = task
                break
        if source is None:
            source = next((t for t in objective.tasks if t.tool), None)
        if source is None:
            return None

        suffix = f"-r{iteration}"
        new_id = f"{source.id}{suffix}"
        args = json.loads(json.dumps(source.args))
        strategy = source.strategy or source.tool
        checks = list(source.checks)

        # NOTE: never rename the output path on repair. Requirement checks point at
        # that exact path, so the repair must overwrite it with a better artifact -
        # changing the strategy, not the contract.

        tool = source.tool
        if tool == "image.variations":
            args["backend"] = "procedural"
            args["style"] = "minimal"
            args["count"] = max(2, int(args.get("count", 2)))
            strategy = "fallback-procedural-art"
        elif tool == "image.generate":
            args["backend"] = "procedural"
            strategy = "fallback-procedural-art"
        elif tool == "slides.create":
            if isinstance(args.get("slides"), str) or not args.get("slides"):
                args["slides"] = "${deck.plan::slides}"
            # rotate through the light palettes; dark is opt-in only
            args["theme"] = _next_light_theme(args.get("theme"))
            strategy = "alternate-light-theme"
        elif tool == "video.from_images":
            args["zoom"] = False
            args["width"] = min(int(args.get("width", 1280)), 960)
            args["height"] = min(int(args.get("height", 720)), 540)
            args["seconds_per_image"] = float(args.get("seconds_per_image", 2.5)) + 0.5
            strategy = "simpler-encode"
        elif tool == "app.scaffold":
            if args.get("framework") in {"pyqt6", "electron"}:
                args["framework"] = "pwa"
                args["platform"] = "mobile"
                strategy = "pwa-fallback"
            elif args.get("framework") == "flutter":
                args["framework"] = "pwa"
                strategy = "pwa-fallback"
            else:
                args["framework"] = "tkinter"
                strategy = "tkinter-fallback"
        elif tool == "code.zip":
            strategy = "repackage"
        elif tool == "report.write":
            strategy = "rewrite-with-sections"
        elif tool == "video.storyboard":
            args["width"] = 960
            args["height"] = 540
            strategy = "smaller-cards"

        advice = []
        if self.memory is not None and source.error:
            advice = self.memory.advice_for(source.tool, source.error)

        revision = Task(
            id=new_id,
            title=f"Repair: {source.title}",
            goal=f"Fix {requirement.id}: {requirement.text}",
            tool=tool,
            args=args,
            depends_on=list(source.depends_on),
            satisfies=[requirement.id],
            checks=[c for c in checks if c.kind in {ch.kind for ch in requirement.checks}] or checks,
            max_attempts=max(1, int(source.max_attempts)),
            strategy=strategy,
        )
        revision.note(f"created by critic on iteration {iteration}")
        if advice:
            revision.note("advice from memory: " + "; ".join(advice))
        return revision

    def _learn(self, objective: Objective, gap: dict) -> dict | None:
        if self.memory is None:
            return None
        failure = str(gap.get("failure") or "")
        if not failure or failure == "never verified":
            return None
        tool = ""
        for task in objective.tasks:
            if gap["requirement"] in task.satisfies:
                tool = task.tool
                break
        advice = {
            "pptx_valid": "Regenerate the deck with more slides and longer bullets; verify with slides.inspect.",
            "pptx_notes": "Write speaker notes for every content slide at plan time, not after the fact.",
            "image_valid": "Switch to the procedural backend and raise the minimum size; blank images mean the render failed.",
            "video_valid": "Check ffmpeg availability and drop zoompan; short or missing output usually means filter failure.",
            "scaffold_files": "Write every entrypoint listed in the framework manifest before zipping.",
            "js_syntax": "Fall back to the PWA template, which has no build step.",
            "python_compiles": "Compile the generated script before packaging.",
            "markdown_report": "Ensure the document has at least three headings and real content.",
            "media_report": "Ingest must write observations.json before the report task runs.",
            "zip_valid": "Package only after the scaffold passes verification.",
        }.get(gap["checks"][0] if gap.get("checks") else "", "Re-run the producing task with a simpler strategy.")
        return self.memory.learn(tool or "*", failure, advice, evidence=gap["requirement"])

    def _insights(self, objective: Objective, verification: dict) -> list[str]:
        provider = self.provider
        if provider is None:
            return []
        payload = {
            "request": objective.request,
            "verification": verification.get("results", {}),
            "failures": [
                {"task": t.id, "tool": t.tool, "error": t.error[:200]} for t in objective.tasks if t.error
            ],
            "artifacts": [a.get("path") for a in objective.artifacts()],
        }
        try:
            from ..llm.base import Message

            completion = provider.complete(
                [
                    Message.system("You are a terse, critical reviewer. Output JSON only."),
                    Message.user(
                        REVIEW_PROMPT.format(
                            request=objective.request,
                            verification=json.dumps(payload["verification"]),
                            failures=json.dumps(payload["failures"]),
                            artifacts=json.dumps(payload["artifacts"]),
                        )
                    ),
                ],
                temperature=0.3,
                max_tokens=700,
                json_mode=True,
            )
            data = completion.json()
            insights = []
            for key in ("insights", "risks", "improvements"):
                for item in data.get(key, []) if isinstance(data, dict) else []:
                    insights.append(f"{key[:-1]}: {item}")
            return insights[:8]
        except Exception:
            return []
