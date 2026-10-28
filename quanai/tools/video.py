"""Video assembly with ffmpeg (bundled binary if the system has none)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..errors import ToolError
from ..util import ffmpeg_bin, run_cmd
from .registry import ToolContext, ToolResult, artifact

MIME = "video/mp4"

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _font(size: int):
    from PIL import ImageFont  # type: ignore

    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_card(
    background: Path | None,
    title: str,
    subtitle: str = "",
    width: int = 1280,
    height: int = 720,
    accent: tuple[int, int, int] = (37, 99, 235),
) -> "object":
    """Build a title card: background image (or flat colour) plus caption text."""
    from PIL import Image, ImageDraw, ImageFilter  # type: ignore

    if background and Path(background).exists():
        base = Image.open(background).convert("RGB")
        ratio = max(width / base.width, height / base.height)
        base = base.resize((int(base.width * ratio) + 1, int(base.height * ratio) + 1), Image.LANCZOS)
        left = (base.width - width) // 2
        top = (base.height - height) // 2
        base = base.crop((left, top, left + width, top + height))
        base = base.filter(ImageFilter.GaussianBlur(1.2))
    else:
        base = Image.new("RGB", (width, height), (247, 248, 250))

    shade = Image.new("RGBA", (width, height), (250, 250, 247, 172))
    canvas = Image.alpha_composite(base.convert("RGBA"), shade)
    draw = ImageDraw.Draw(canvas)

    margin = int(width * 0.09)
    draw.rectangle([margin, int(height * 0.14), margin + 6, int(height * 0.86)], fill=accent)
    title_font = _font(int(height * 0.075))
    sub_font = _font(int(height * 0.038))

    def wrap(text: str, font, max_width: int) -> list[str]:
        words = text.split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    y = int(height * 0.36)
    for line in wrap(title, title_font, width - 2 * margin - 40):
        draw.text((margin + 34, y), line, font=title_font, fill=(17, 24, 39))
        y += int(height * 0.095)
    if subtitle:
        y += int(height * 0.02)
        for line in wrap(subtitle, sub_font, width - 2 * margin - 40):
            draw.text((margin + 34, y), line, font=sub_font, fill=(75, 85, 99))
            y += int(height * 0.052)
    return canvas.convert("RGB")


def _probe_duration(path: str | Path) -> float:
    from ..probe import probe

    return probe(path).duration


def encode_slideshow(
    frames: list[Path],
    out: Path,
    seconds_per_frame: float = 2.5,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
    zoom: bool = True,
    audio: Path | None = None,
) -> dict:
    exe = ffmpeg_bin()
    if not exe:
        raise ToolError("ffmpeg is not available (install imageio-ffmpeg or system ffmpeg)")
    if not frames:
        raise ToolError("no frames to encode")

    out.parent.mkdir(parents=True, exist_ok=True)
    command: list[str] = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    per_frame = max(0.4, float(seconds_per_frame))
    for frame in frames:
        command += ["-loop", "1", "-t", f"{per_frame:.3f}", "-i", str(frame)]

    filters = []
    for index in range(len(frames)):
        if zoom:
            # scale just above the target so the slow zoom never upscales hard
            filters.append(
                f"[{index}:v]scale={int(width * 1.15)}:-2,"
                # d=1 -> one output frame per input frame, so the clip stays
                # exactly per_frame seconds while zoom creeps on every frame
                f"zoompan=z='min(zoom+0.0008,1.08)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps},"
                f"setsar=1[v{index}]"
            )
        else:
            filters.append(
                f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,fps={fps}[v{index}]"
            )
    joined = "".join(f"[v{i}]" for i in range(len(frames)))
    filters.append(f"{joined}concat=n={len(frames)}:v=1:a=0[vout]")
    command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]

    audio_input = None
    if audio and Path(audio).exists():
        audio_input = len(frames)
        command += ["-i", str(audio)]
        command += ["-map", f"{audio_input}:a", "-shortest", "-c:a", "aac", "-b:a", "128k"]

    command += [
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    result = run_cmd(command, timeout=600)
    if not result.ok or not out.exists() or out.stat().st_size == 0:
        raise ToolError(f"ffmpeg failed: {result.tail(800)}")
    from ..probe import probe

    info = probe(out)
    return {
        "path": str(out),
        "duration": info.duration,
        "width": info.width,
        "height": info.height,
        "frames_used": len(frames),
        "fps": fps,
        "bytes": out.stat().st_size,
        "has_audio": info.has_audio,
    }

def _animated_gif(frames: list[Path], out: Path, seconds_per_frame: float, width: int, height: int) -> dict:
    from PIL import Image  # type: ignore

    images = []
    for frame in frames:
        with Image.open(frame) as img:
            images.append(
                img.convert("RGB").resize((width, height), Image.LANCZOS)
            )
    if not images:
        raise ToolError("no frames for gif")
    out = out.with_suffix(".gif")
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=int(seconds_per_frame * 1000),
        loop=0,
        optimize=True,
    )
    return {"path": str(out), "frames_used": len(images), "bytes": out.stat().st_size, "fallback": "gif"}


def register(registry) -> None:
    register_storyboard(registry)

    @registry.tool(
        "video.card",
        "Compose a title/caption card image from a background image and text.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "background": {"type": "string"},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
            },
            "required": ["path", "title"],
        },
        category="video",
        requires=["pillow"],
    )
    def video_card(
        ctx: ToolContext,
        path: str,
        title: str,
        subtitle: str = "",
        background: str = "",
        width: int = 1280,
        height: int = 720,
    ) -> ToolResult:
        target = ctx.path(path)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".png")
        bg = ctx.path(background) if background else None
        card = compose_card(bg, title, subtitle, int(width), int(height))
        card.save(str(target), format="PNG")
        return ToolResult.success(
            {"path": ctx.rel(target), "width": card.width, "height": card.height},
            [artifact(ctx, target, "image", "image/png", role="card")],
        )

    @registry.tool(
        "video.from_images",
        "Assemble still images into an MP4 slideshow with optional Ken Burns zoom and audio.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "images": {"type": "array", "items": {"type": "string"}},
                "seconds_per_image": {"type": "number", "default": 2.5},
                "fps": {"type": "integer", "default": 25},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
                "zoom": {"type": "boolean", "default": True},
                "audio": {"type": "string"},
            },
            "required": ["path", "images"],
        },
        category="video",
        requires=["ffmpeg"],
    )
    def video_from_images(
        ctx: ToolContext,
        path: str,
        images: list,
        seconds_per_image: float = 2.5,
        fps: int = 25,
        width: int = 1280,
        height: int = 720,
        zoom: bool = True,
        audio: str = "",
    ) -> ToolResult:
        frames = []
        for item in images:
            resolved = ctx.path(str(item), must_exist=True)
            if resolved.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                frames.append(resolved)
        if not frames:
            raise ToolError("no usable image files were provided")
        target = ctx.path(path)
        if target.suffix.lower() != ".mp4":
            target = target.with_suffix(".mp4")
        audio_path = ctx.path(audio, must_exist=True) if audio else None
        try:
            info = encode_slideshow(
                frames, target, seconds_per_image, int(fps), int(width), int(height), bool(zoom), audio_path
            )
            kind, mime = "video", MIME
        except ToolError:
            info = _animated_gif(frames, target, seconds_per_image, int(width), int(height))
            target = Path(info["path"])
            kind, mime = "animation", "image/gif"
        return ToolResult.success(
            {k: v for k, v in info.items() if k != "path"},
            [artifact(ctx, target, kind, mime, **{k: v for k, v in info.items() if k != "path"})],
        )

    @registry.tool(
        "video.from_audio_still",
        "Attach an audio track to a still image or card, producing a narratable MP4.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "image": {"type": "string"},
                "audio": {"type": "string"},
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
            },
            "required": ["path", "image", "audio"],
        },
        category="video",
        requires=["ffmpeg"],
    )
    def video_from_audio_still(
        ctx: ToolContext, path: str, image: str, audio: str, width: int = 1280, height: int = 720
    ) -> ToolResult:
        exe = ffmpeg_bin()
        if not exe:
            raise ToolError("ffmpeg is not available")
        still = ctx.path(image, must_exist=True)
        sound = ctx.path(audio, must_exist=True)
        target = ctx.path(path)
        if target.suffix.lower() != ".mp4":
            target = target.with_suffix(".mp4")
        command = [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-i", str(still), "-i", str(sound),
            "-vf", f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,crop={int(width)}:{int(height)},setsar=1",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(target),
        ]
        result = run_cmd(command, timeout=900)
        if not result.ok:
            raise ToolError(f"ffmpeg failed: {result.tail(600)}")
        from ..probe import probe

        info = probe(target)
        return ToolResult.success(
            {
                "duration": info.duration,
                "width": info.width,
                "height": info.height,
                "has_audio": info.has_audio,
                "bytes": target.stat().st_size,
            },
            [artifact(ctx, target, "video", MIME, duration=info.duration, has_audio=info.has_audio)],
        )


def register_storyboard(registry) -> None:
    @registry.tool(
        "video.storyboard",
        "Render a storyboard: one artwork background plus caption card per scene.",
        schema={
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "default": "storyboard"},
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "caption": {"type": "string"},
                            "image_prompt": {"type": "string"},
                            "palette": {"type": "string"},
                            "seconds": {"type": "number"},
                            "seed": {"type": "integer"},
                        },
                    },
                },
                "width": {"type": "integer", "default": 1280},
                "height": {"type": "integer", "default": 720},
            },
            "required": ["out_dir", "scenes"],
        },
        category="video",
        requires=["pillow", "numpy"],
    )
    def video_storyboard(
        ctx: ToolContext,
        out_dir: str,
        scenes: list,
        title: str = "",
        subtitle: str = "",
        width: int = 1280,
        height: int = 720,
    ) -> ToolResult:
        import json as _json

        from .images import render_procedural

        root = ctx.path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        cards: list[str] = []
        rows = []
        for index, raw in enumerate(scenes or [], start=1):
            caption = str(raw.get("caption") or raw.get("beat") or f"Scene {index}")
            prompt = str(raw.get("image_prompt") or caption)
            seed = int(raw.get("seed", index))
            background = render_procedural(prompt, int(width), int(height), seed=seed, style="editorial")
            bg_path = root / f"background-{index:02d}.png"
            background.save(bg_path)
            card = compose_card(bg_path, caption, subtitle if index == 1 else "", int(width), int(height))
            card_path = root / f"card-{index:02d}.png"
            card.save(card_path)
            cards.append(card_path)
            rows.append({
                "index": index,
                "caption": caption,
                "card": card_path.name,
                "background": bg_path.name,
                "seconds": float(raw.get("seconds", 2.5)),
            })
        if not cards:
            raise ToolError("a storyboard needs at least one scene")

        manifest = {
            "title": title,
            "subtitle": subtitle,
            "cards": [c.name for c in cards],
            "scenes": rows,
            "duration": round(sum(r["seconds"] for r in rows), 1),
            "width": width,
            "height": height,
        }
        manifest_path = root / "storyboard.json"
        manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
        return ToolResult.success(
            {
                "cards": [ctx.rel(c) for c in cards],
                "count": len(cards),
                "duration": manifest["duration"],
                "manifest": ctx.rel(manifest_path),
            },
            [artifact(ctx, c, "image", "image/png", role="storyboard-card") for c in cards]
            + [artifact(ctx, manifest_path, "data", "application/json", role="storyboard")],
        )
