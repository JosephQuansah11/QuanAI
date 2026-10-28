"""Planning: turn a request into testable requirements and a task graph.

The planner is deterministic by default and *capability aware*: it inspects the
machine (python-pptx? ffmpeg? node? a language model?) and chooses the tool
chain that can actually run here for free.  Every requirement it emits carries
at least one machine-checkable rule, so nothing can be declared done on trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..content import (
    app_plan,
    deck_plan,
    detect_domain,
    detect_framework,
    detect_intents,
    image_prompts,
    requested_slides,
    topic_of,
    video_plan,
)
from ..util import keywords, now_iso, slugify
from .memory import Memory
from .task import Check, Objective, Requirement, Status, Task


def _task(task_id: str, title: str, tool: str, args: dict, **kwargs) -> Task:
    kwargs.setdefault("checks", [])
    return Task(id=task_id, title=title, tool=tool, args=args, **kwargs)


def _req(req_id: str, text: str, checks: list[Check], required: bool = True) -> Requirement:
    return Requirement(id=req_id, text=text, checks=checks, required=required)


@dataclass
class Plan:
    requirements: list[Requirement] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    source: str = "offline"
    thought: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "tasks": [t.to_dict() for t in self.tasks],
            "source": self.source,
            "thought": self.thought,
            "notes": self.notes,
        }


class Planner:
    def __init__(self, config, registry, provider=None, memory: Memory | None = None) -> None:
        self.config = config
        self.registry = registry
        self.provider = provider
        self.memory = memory
        self._available_tools = set(registry.names())

    # ------------------------------------------------------------------
    def plan(self, objective: Objective) -> Plan:
        request = objective.request

        # a validated plan from an earlier, similar objective saves the whole
        # planning step - the request text is rewritten into every task arg
        reused = self._reuse_plan(request)
        if reused is not None:
            reused.thought = self.think(request, detect_intents(request) or ["document"])
            reused.notes.append(
                f"Reused a stored plan shape from memory ({len(reused.tasks)} tasks) and rewrote it for this request."
            )
            return reused

        intents = detect_intents(request)
        if not intents:
            intents = ["document"]
        thought = self.think(request, intents)
        plan = Plan(source=self._source_name(), thought=thought)

        seen_tracks = set()
        for intent in intents:
            if intent in seen_tracks:
                continue
            seen_tracks.add(intent)
            if intent == "deck":
                self._add_deck(plan, request)
            elif intent == "image":
                self._add_images(plan, request)
            elif intent == "video":
                self._add_video(plan, request)
            elif intent in {"desktop", "mobile"}:
                self._add_app(plan, request, intent)
            elif intent == "media_in":
                self._add_media(plan, request)
            else:
                self._add_document(plan, request)

        if not plan.tasks:
            self._add_document(plan, request)

        # a closing note task keeps the human loop informed
        plan.notes.append(
            f"Planned {len(plan.tasks)} task(s) against {len(plan.requirements)} requirement(s) "
            f"using {plan.source}."
        )
        return plan

    # ------------------------------------------------------------------
    def _reuse_plan(self, request: str) -> Plan | None:
        """Rebuild a previous plan when every tool, check and dependency still applies."""
        if self.memory is None:
            return None
        cached = self.memory.recall_plan(request)
        if not cached:
            return None
        from ..core.verify import known_checks
        from .task import Requirement, Task

        try:
            tasks = [Task.from_dict(item) for item in cached.get("tasks", [])]
            requirements = [Requirement.from_dict(item) for item in cached.get("requirements", [])]
        except Exception:
            return None
        if not tasks or not requirements:
            return None

        kinds = set(known_checks())
        ids = {task.id for task in tasks}
        old_request = str(cached.get("request") or "")
        for task in tasks:
            if task.tool not in self._available_tools:
                return None
            if any(check.kind not in kinds for check in task.checks):
                return None
            if any(dependency not in ids for dependency in task.depends_on):
                return None
        for requirement in requirements:
            if not requirement.checks or any(check.kind not in kinds for check in requirement.checks):
                return None

        for task in tasks:
            task.status = Status.PENDING
            task.attempts = 0
            task.result = None
            task.artifacts = []
            task.error = ""
            task.notes = [f"reused from {cached.get('at', 'an earlier plan')}"]
            task.args = _rewrite_request(task.args, old_request, request)
        for requirement in requirements:
            requirement.status = Status.PENDING
            requirement.evidence = []
        return Plan(requirements=requirements, tasks=tasks, source="memory")

    def _source_name(self) -> str:
        if self.provider is None:
            return "offline"
        return f"llm:{self.provider.model or self.provider.name}"

    def think(self, request: str, intents: list[str]) -> dict:
        """Record the options considered and why this path was chosen."""
        caps = self._capabilities()
        options = []
        for intent in intents:
            candidates = {
                "deck": ["slides.create (python-pptx)", "markdown-only outline"],
                "image": ["image.generate (procedural)", "image.generate (sd)", "pollinations (network)"],
                "video": ["video.storyboard + video.from_images (ffmpeg)", "animated gif fallback"],
                "desktop": ["app.scaffold tkinter", "app.scaffold pyqt6", "app.scaffold electron"],
                "mobile": ["app.scaffold pwa", "app.scaffold flutter", "app.scaffold react-native"],
                "media_in": ["media.ingest (local signal analysis)", "media.ingest + ASR model"],
                "document": ["report.write (markdown)"],
            }.get(intent, ["report.write"])
            options.append({"intent": intent, "considered": candidates})

        chosen = []
        if "deck" in intents:
            chosen.append("slides.create" if caps.get("python-pptx") else "markdown outline (no python-pptx)")
        if "image" in intents:
            chosen.append("image.generate/procedural (offline, deterministic)")
        if "video" in intents:
            chosen.append(
                "video.from_images/ffmpeg" if caps.get("ffmpeg") else "animated gif fallback"
            )
        explicit, explicit_platform = detect_framework(request)
        if "desktop" in intents:
            framework = explicit if explicit_platform == "desktop" else "tkinter"
            chosen.append(f"app.scaffold/{framework}")
        if "mobile" in intents:
            framework = explicit if explicit_platform == "mobile" else "pwa"
            chosen.append(f"app.scaffold/{framework}")
        if "media_in" in intents:
            chosen.append(
                "media.ingest with local ASR" if caps.get("faster-whisper") else "media.ingest signal-level"
            )

        return {
            "at": now_iso(),
            "request": request,
            "intents": intents,
            "capabilities": caps,
            "options": options,
            "chosen": chosen,
            "rationale": (
                "Prefer local, zero-cost tools with machine-checkable outputs; "
                "fall back to simpler formats only when a dependency is missing."
            ),
        }

    def _capabilities(self) -> dict:
        from ..tools import probe_capabilities

        try:
            return probe_capabilities()
        except Exception:  # pragma: no cover - defensive
            return {}

    # ------------------------------------------------------------------
    # track builders
    # ------------------------------------------------------------------
    def _add_deck(self, plan: Plan, request: str) -> None:
        total = requested_slides(request, 8)
        preview = deck_plan(request, total)
        topic = topic_of(request)
        # requirement terms must be subject words, never format words like "slide"
        kws = keywords(topic, 4) or [topic]
        theme = "midnight"
        hero = image_prompts(request, 2)
        min_slides = max(4, total - 2)
        # text volume must scale with the requested length, not a fixed number
        min_chars = 110 * min_slides

        plan.tasks.append(
            _task(
                "deck.plan",
                "Draft the deck outline",
                "plan.deck",
                {"request": request, "slides": total, "theme": theme},
                checks=[Check("json_valid", {"path": "plans/deck.json", "keys": ["title", "slides"]})],
                strategy="outline-first",
            )
        )
        plan.tasks.append(
            _task(
                "deck.images",
                "Generate cover artwork",
                "image.variations",
                {
                    "path_prefix": "assets/deck-art",
                    "prompt": hero[0]["prompt"],
                    "count": 2,
                    "width": 1600,
                    "height": 900,
                    "style": "editorial",
                },
                checks=[Check("file_exists", {"path": "assets/deck-art-1.png"})],
                strategy="local-procedural-art",
            )
        )
        plan.tasks.append(
            _task(
                "deck.build",
                "Build the .pptx",
                "slides.create",
                {
                    "path": "deck.pptx",
                    "title": "${deck.plan::title}",
                    "subtitle": "${deck.plan::subtitle}",
                    "theme": "${deck.plan::theme}",
                    "slides": "${deck.plan::slides}",
                },
                depends_on=["deck.plan", "deck.images"],
                satisfies=["R.deck", "R.deck.notes"],
                checks=[
                    Check(
                        "pptx_valid",
                        {"path": "deck.pptx", "min_slides": min_slides, "min_chars": min_chars},
                    ),
                    Check("pptx_notes", {"path": "deck.pptx", "min_with_notes": 4}),
                ],
                strategy="python-pptx",
            )
        )
        plan.requirements.extend(
            [
                _req(
                    "R.deck",
                    f"A PowerPoint deck about {topic} exists with at least {min_slides} slides and real text",
                    [
                        Check("pptx_valid", {
                            "path": "deck.pptx",
                            "min_slides": min_slides,
                            "min_chars": min_chars,
                            "required_terms": kws[:1],
                        })
                    ],
                ),
                _req(
                    "R.deck.notes",
                    "Most slides carry speaker notes so the deck can be presented",
                    [Check("pptx_notes", {"path": "deck.pptx", "min_with_notes": 4})],
                ),
                _req(
                    "R.deck.art",
                    "Cover artwork was generated locally for the deck",
                    [Check("image_valid", {"path": "assets/deck-art-1.png", "min_width": 512})],
                    required=False,
                ),
            ]
        )
        plan.notes.append(f"Deck plan preview: {preview.title} ({len(preview.slides)} slides).")

    def _add_images(self, plan: Plan, request: str) -> None:
        topic = topic_of(request)
        prompts = image_prompts(request, 3)
        plan.tasks.append(
            _task(
                "image.gen",
                "Generate images",
                "image.variations",
                {
                    "path_prefix": "images/art",
                    "prompt": prompts[0]["prompt"],
                    "count": 3,
                    "width": 1280,
                    "height": 720,
                    "style": "editorial",
                },
                satisfies=["R.images"],
                checks=[Check("file_exists", {"path": "images/art-1.png"})],
                strategy="local-procedural-art",
            )
        )
        plan.requirements.append(
            _req(
                "R.images",
                f"At least two non-blank images about {topic} were produced",
                [
                    Check("image_valid", {"path": "images/art-1.png", "min_width": 256}),
                    Check("image_valid", {"path": "images/art-2.png", "min_width": 256}),
                ],
            )
        )

    def _add_video(self, plan: Plan, request: str) -> None:
        topic = topic_of(request)
        preview = video_plan(request, 5, 2.5)
        plan.tasks.append(
            _task(
                "video.plan",
                "Write the storyboard",
                "plan.video",
                {"request": request, "scenes": 5, "seconds_per_scene": 2.5},
                checks=[Check("json_valid", {"path": "plans/video.json", "keys": ["scenes"]})],
                strategy="beat-sheet",
            )
        )
        plan.tasks.append(
            _task(
                "video.storyboard",
                "Render storyboard cards",
                "video.storyboard",
                {
                    "out_dir": "storyboard",
                    "title": "${video.plan::title}",
                    "subtitle": "${video.plan::subtitle}",
                    "scenes": "${video.plan::scenes}",
                    "width": 1280,
                    "height": 720,
                },
                depends_on=["video.plan"],
                checks=[Check("file_exists", {"path": "storyboard/card-01.png"})],
                strategy="caption-cards",
            )
        )
        plan.tasks.append(
            _task(
                "video.render",
                "Encode the MP4",
                "video.from_images",
                {
                    "path": "video.mp4",
                    "images": "${video.storyboard::cards}",
                    "seconds_per_image": 2.5,
                    "fps": 25,
                    "width": 1280,
                    "height": 720,
                    "zoom": True,
                },
                depends_on=["video.storyboard"],
                satisfies=["R.video"],
                checks=[Check("video_valid", {"path": "video.mp4", "min_seconds": 5})],
                strategy="ffmpeg-slideshow",
            )
        )
        plan.requirements.extend(
            [
                _req(
                    "R.video",
                    f"A playable video about {topic} exists and runs at least 5 seconds",
                    [
                        Check("video_valid", {
                            "path": "video.mp4", "min_seconds": 5.0, "min_width": 640,
                        })
                    ],
                ),
                _req(
                    "R.video.storyboard",
                    "Storyboard cards with captions were produced",
                    [Check("file_exists", {"path": "storyboard/card-01.png"})],
                    required=False,
                ),
            ]
        )
        plan.notes.append(f"Video plan preview: {preview['title']} ({len(preview['scenes'])} scenes).")

    def _add_app(self, plan: Plan, request: str, platform: str) -> None:
        explicit, explicit_platform = detect_framework(request)
        if explicit and explicit_platform == platform:
            framework = explicit
        else:
            framework = "tkinter" if platform == "desktop" else "pwa"
        preview = app_plan(request, platform, framework)
        required_files = {
            "tkinter": ["main.py", "README.md"],
            "pyqt6": ["main.py", "requirements.txt", "README.md"],
            "electron": ["package.json", "main.js", "preload.js", "index.html", "styles.css", "README.md"],
            "pwa": ["index.html", "app.js", "styles.css", "sw.js", "manifest.webmanifest", "README.md"],
            "flutter": ["pubspec.yaml", "lib/main.dart", "README.md"],
            "react-native": ["package.json", "App.js", "app.json", "README.md"],
        }[framework]
        # every framework gets a syntax or structure check that can actually run
        code_check = Check("js_syntax", {"path": "app"})
        if framework in {"tkinter", "pyqt6"}:
            code_check = Check("python_compiles", {"path": "app"})
        elif framework == "react-native":
            # App.js is JSX, so node --check would reject it: assert structure instead
            code_check = Check(
                "text_contains",
                {"path": "app/App.js", "needles": ["export default", "StyleSheet.create"], "min_chars": 900},
            )
        elif framework == "flutter":
            code_check = Check(
                "text_contains",
                {
                    "path": "app/lib/main.dart",
                    "needles": ["runApp", "StatefulWidget", "import 'package:flutter/material.dart'"],
                    "min_chars": 900,
                },
            )
        elif framework == "electron":
            code_check = Check(
                "js_syntax", {"path": "app/main.js"},
            )

        plan.tasks.append(
            _task(
                "app.plan",
                "Specify the application",
                "plan.app",
                {"request": request, "platform": platform, "framework": framework},
                checks=[Check("json_valid", {"path": "plans/app.json", "keys": ["name", "features"]})],
                strategy="feature-first",
            )
        )
        plan.tasks.append(
            _task(
                "app.scaffold",
                f"Scaffold the {preview['framework_label']} app",
                "app.scaffold",
                {
                    "path": "app",
                    "description": request,
                    "name": "${app.plan::name}",
                    "platform": platform,
                    "framework": framework,
                },
                depends_on=["app.plan"],
                satisfies=["R.app.files"],
                checks=[Check("scaffold_files", {"path": "app", "required": required_files})],
                strategy=f"{framework}-starter",
            )
        )
        plan.tasks.append(
            _task(
                "app.package",
                "Package the project",
                "code.zip",
                {"path": "app", "archive": "app.zip"},
                depends_on=["app.scaffold"],
                satisfies=["R.app.zip"],
                checks=[Check("zip_valid", {"path": "app.zip"})],
                strategy="zip",
            )
        )
        plan.requirements.extend(
            [
                _req(
                    "R.app.files",
                    f"A runnable {platform} application exists with all entrypoint files",
                    [
                        Check("scaffold_files", {"path": "app", "required": required_files}),
                        code_check,
                        Check("json_valid", {"path": "app/package.json", "keys": ["name"]})
                        if framework in {"electron", "react-native"}
                        else Check("file_exists", {"path": "app/README.md"}),
                    ],
                ),
                _req(
                    "R.app.zip",
                    "The project is packaged as a downloadable archive",
                    [Check("zip_valid", {"path": "app.zip", "required": ["main.py"]})]
                    if framework in {"tkinter", "pyqt6"}
                    else [Check("zip_valid", {"path": "app.zip"})],
                ),
            ]
        )
        plan.notes.append(
            f"App plan preview: {preview['name']} ({preview['framework_label']}, run: {preview['run']})."
        )

    def _add_media(self, plan: Plan, request: str) -> None:
        source = find_input_path(request, self.config.workspace)
        plan.tasks.append(
            _task(
                "media.ingest",
                "Ingest and analyse the media",
                "media.ingest",
                {
                    "path": source or "__missing_media_file__",
                    "out_dir": "media/input",
                    "frames": 6,
                    "transcribe": True,
                    "describe": True,
                },
                satisfies=["R.media.facts", "R.media.report"],
                checks=[Check("media_report", {"path": "media/input/observations.json", "keys": ["probe"]})],
                strategy="local-signal-analysis",
            )
        )
        plan.tasks.append(
            _task(
                "media.report",
                "Write the analysis report",
                "report.write",
                {
                    "path": "analysis.md",
                    "title": f"Media analysis: {topic_of(request)}",
                    "source_markdown": "media/input/analysis.md",
                },
                depends_on=["media.ingest"],
                satisfies=["R.media.report"],
                checks=[Check("markdown_report", {"path": "analysis.md", "min_headings": 3})],
                strategy="observations-to-markdown",
            )
        )
        plan.requirements.extend(
            [
                _req(
                    "R.media.facts",
                    "The supplied media was decoded and measured, not guessed",
                    [
                        Check("media_report", {
                            "path": "media/input/observations.json",
                            "keys": ["kind", "probe"],
                        })
                    ],
                ),
                _req(
                    "R.media.report",
                    "A written analysis with headings summarises the media",
                    [Check("markdown_report", {"path": "analysis.md", "min_headings": 3})],
                ),
            ]
        )
        if not source:
            plan.notes.append(
                "No media file path was found in the request; supply one with --input <file>."
            )

    def _add_document(self, plan: Plan, request: str) -> None:
        topic = topic_of(request)
        plan.tasks.append(
            _task(
                "doc.write",
                "Write the document",
                "report.write",
                {"path": "document.md", "title": topic, "request": request},
                satisfies=["R.doc"],
                checks=[Check("markdown_report", {"path": "document.md", "min_headings": 3})],
                strategy="markdown",
            )
        )
        plan.requirements.append(
            _req(
                "R.doc",
                f"A structured document about {topic} exists",
                [
                    Check("markdown_report", {"path": "document.md", "min_headings": 3}),
                    Check("text_contains", {"path": "document.md", "min_chars": 400}),
                ],
            )
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
MEDIA_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".png", ".jpg", ".jpeg"}


def find_input_path(request: str, workspace: Path) -> str:
    """Find a media file mentioned in the request, if any exists."""
    import re

    for token in re.findall(r"[\w./\\-]+\.[A-Za-z0-9]{2,5}", request):
        candidate = Path(token).expanduser()
        suffix = candidate.suffix.lower()
        if suffix not in MEDIA_SUFFIXES:
            continue
        for option in (candidate, Path(workspace) / candidate, Path.cwd() / candidate):
            if option.exists():
                return str(option.resolve())
    return ""


def _rewrite_request(value, old_request: str, new_request: str):
    """Replace the stored request text anywhere inside reused task arguments."""
    if isinstance(value, str):
        return new_request if value == old_request else value
    if isinstance(value, list):
        return [_rewrite_request(item, old_request, new_request) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_request(item, old_request, new_request) for key, item in value.items()}
    return value


def build_plan(objective: Objective, config, registry, provider=None, memory=None) -> Plan:
    return Planner(config, registry, provider, memory).plan(objective)
