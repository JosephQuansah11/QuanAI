"""Structured planning tools.

These produce the *content* of an artifact (outline, storyboard, feature list)
as a JSON file plus a return value, so later tasks can consume it and a human
can inspect the reasoning.  When a language model is configured it drafts the
content; otherwise the deterministic engine does.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..content import DeckPlan, app_plan, deck_plan, image_prompts, video_plan
from ..llm.base import Message
from .registry import ToolContext, ToolResult, artifact

DECK_PROMPT = """You are drafting a presentation. Return ONLY JSON matching:
{"title": str, "subtitle": str, "slides": [{"title": str, "bullets": [str], "notes": str, "kind": "title|content|closing"}]}
Rules: first slide kind="title", last slide kind="closing", $COUNT$ slides total,
3-5 bullets per content slide, one short sentence per note, no markdown, no commentary.
Topic: $TOPIC$
"""

VIDEO_PROMPT = """You are writing a short silent video storyboard. Return ONLY JSON:
{"title": str, "subtitle": str, "scenes": [{"caption": str, "image_prompt": str, "seconds": number}]}
Rules: $COUNT$ scenes, captions under 90 characters, image prompts describe abstract art (no text in image).
Topic: $TOPIC$
"""

APP_PROMPT = """You are specifying a small application. Return ONLY JSON:
{"name": str, "description": str, "features": [{"title": str, "detail": str}]}
Rules: name is PascalCase, 5-7 features, each detail is one sentence, no markdown.
Platform: $PLATFORM$
Request: $REQUEST$
"""


def _valid_deck(data: dict) -> bool:
    slides = data.get("slides")
    return (
        isinstance(data.get("title"), str)
        and isinstance(slides, list)
        and len(slides) >= 4
        and all(isinstance(s, dict) and s.get("title") for s in slides)
    )


def _thicken(drafted: DeckPlan, fallback: DeckPlan) -> DeckPlan:
    """A model can return slides that are technically valid but too thin.

    Keep its titles and any usable text, and top the slide up from the
    deterministic plan so the artifact is always worth presenting.
    """
    for index, slide in enumerate(drafted.slides):
        reference = fallback.slides[min(index, len(fallback.slides) - 1)]
        solid = [bullet for bullet in slide.bullets if len(bullet.strip()) >= 25]
        if len(solid) < 2:
            merged = [b for b in slide.bullets if b.strip()] + reference.bullets
            slide.bullets = merged[:5]
        if len(slide.notes.strip()) < 15:
            slide.notes = reference.notes
        if slide.kind not in {"title", "content", "closing", "image"}:
            slide.kind = reference.kind
    drafted.slides[0].kind = "title"
    drafted.slides[-1].kind = "closing"
    return drafted


def _ask_json(ctx: ToolContext, prompt: str, validator) -> dict | None:
    provider = getattr(ctx, "provider", None)
    if provider is None:
        return None
    try:
        completion = provider.complete(
            [Message.system("You output strict JSON only."), Message.user(prompt)],
            temperature=0.6,
            max_tokens=2600,
            json_mode=True,
        )
        data = completion.json()
        if isinstance(data, dict) and validator(data):
            return data
    except Exception:
        return None
    return None


def register(registry) -> None:
    @registry.tool(
        "plan.deck",
        "Draft a deck outline (title, subtitle, slides with bullets and notes) to plans/deck.json.",
        schema={
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "slides": {"type": "integer", "default": 8},
                "theme": {"type": "string", "default": "midnight"},
                "out": {"type": "string", "default": "plans/deck.json"},
            },
            "required": ["request"],
        },
        category="plan",
    )
    def plan_deck(
        ctx: ToolContext, request: str, slides: int = 8, theme: str = "midnight", out: str = "plans/deck.json"
    ) -> ToolResult:
        from ..content import topic_of

        count = max(4, min(int(slides), 20))
        plan = deck_plan(request, count)
        source = "offline"
        prompt = DECK_PROMPT.replace("$COUNT$", str(count)).replace("$TOPIC$", topic_of(request))
        drafted = _ask_json(ctx, prompt, _valid_deck)
        if drafted:
            fallback = plan
            plan = DeckPlan.from_dict({**drafted, "source": "llm"})
            source = f"llm:{ctx.provider.model}"
            if len(plan.slides) < count:
                plan.slides.extend(fallback.slides[len(plan.slides) : count])
            plan = _thicken(plan, fallback)
            if len(plan.slides) > count:
                plan.slides = plan.slides[: count - 1] + [plan.slides[-1]]

        payload = plan.to_dict()
        payload.update({"theme": theme, "request": request, "slides_requested": count})
        target = ctx.path(out)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ToolResult.success(
            {
                "title": payload["title"],
                "subtitle": payload.get("subtitle", ""),
                "theme": theme,
                "slides": payload["slides"],
                "count": len(payload["slides"]),
                "source": source,
            },
            [artifact(ctx, target, "data", "application/json", role="deck-plan")],
        )

    @registry.tool(
        "plan.video",
        "Write a scene-by-scene storyboard for a short video to plans/video.json.",
        schema={
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "scenes": {"type": "integer", "default": 5},
                "seconds_per_scene": {"type": "number", "default": 2.5},
                "out": {"type": "string", "default": "plans/video.json"},
            },
            "required": ["request"],
        },
        category="plan",
    )
    def plan_video(
        ctx: ToolContext,
        request: str,
        scenes: int = 5,
        seconds_per_scene: float = 2.5,
        out: str = "plans/video.json",
    ) -> ToolResult:
        from ..content import topic_of

        count = max(3, min(int(scenes), 10))
        plan = video_plan(request, count, seconds_per_scene)
        source = "offline"
        drafted = _ask_json(
            ctx,
            VIDEO_PROMPT.replace("$COUNT$", str(count)).replace("$TOPIC$", topic_of(request)),
            lambda data: isinstance(data.get("scenes"), list) and len(data["scenes"]) >= 3,
        )
        if drafted:
            prompts = image_prompts(request, len(drafted["scenes"]))
            plan = {
                "title": drafted.get("title") or plan["title"],
                "subtitle": drafted.get("subtitle") or plan["subtitle"],
                "topic": plan["topic"],
                "scenes": [
                    {
                        "beat": f"Scene {index + 1}",
                        "caption": str(scene.get("caption", ""))[:120],
                        "image_prompt": str(scene.get("image_prompt") or prompts[index]["prompt"]),
                        "palette": prompts[index]["palette"],
                        "seconds": float(scene.get("seconds") or seconds_per_scene),
                        "seed": index,
                    }
                    for index, scene in enumerate(drafted["scenes"])
                ],
            }
            plan["duration"] = round(sum(s["seconds"] for s in plan["scenes"]), 1)
            source = f"llm:{ctx.provider.model}"

        plan["request"] = request
        plan["source"] = source
        target = ctx.path(out)
        target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return ToolResult.success(
            {
                "title": plan["title"],
                "subtitle": plan.get("subtitle", ""),
                "scenes": plan["scenes"],
                "duration": plan.get("duration", 0),
                "source": source,
            },
            [artifact(ctx, target, "data", "application/json", role="video-plan")],
        )

    @registry.tool(
        "plan.app",
        "Specify an application (name, description, feature list) to plans/app.json.",
        schema={
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "platform": {"type": "string", "enum": ["desktop", "mobile"]},
                "framework": {"type": "string"},
                "out": {"type": "string", "default": "plans/app.json"},
            },
            "required": ["request"],
        },
        category="plan",
    )
    def plan_app(
        ctx: ToolContext,
        request: str,
        platform: str = "desktop",
        framework: str = "tkinter",
        out: str = "plans/app.json",
    ) -> ToolResult:
        spec = app_plan(request, platform, framework or None)
        source = "offline"
        drafted = _ask_json(
            ctx,
            APP_PROMPT.replace("$PLATFORM$", platform).replace("$REQUEST$", request),
            lambda data: isinstance(data.get("name"), str) and isinstance(data.get("features"), list),
        )
        if drafted:
            spec["name"] = str(drafted["name"])[:32]
            spec["slug"] = spec["name"].lower().replace(" ", "-")
            spec["description"] = str(drafted.get("description") or spec["description"])
            features = [
                {"id": f"f{index + 1}", "title": str(f.get("title", "")), "detail": str(f.get("detail", ""))}
                for index, f in enumerate(drafted["features"])
                if isinstance(f, dict) and f.get("title")
            ]
            if features:
                spec["features"] = features
            source = f"llm:{ctx.provider.model}"
        spec["source"] = source
        target = ctx.path(out)
        target.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return ToolResult.success(
            {
                "name": spec["name"],
                "slug": spec["slug"],
                "description": spec["description"],
                "features": spec["features"],
                "framework": spec["framework"],
                "source": source,
            },
            [artifact(ctx, target, "data", "application/json", role="app-plan")],
        )
