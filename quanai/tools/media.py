"""Multimodal ingestion: read image, audio and video into structured facts.

All analysis here is local and free: ffmpeg for decoding, numpy for signal
analysis, Pillow for visualisation.  Speech-to-text is used when a local
engine is installed, and a vision model is asked for a caption when one is
configured.  Everything extracted is written to ``observations.json`` so the
agent can reason over it and a human can audit it.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path

from ..errors import ToolError
from ..probe import probe
from ..util import ffmpeg_bin, run_cmd
from .registry import ToolContext, ToolResult, artifact


def _np():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError as exc:  # pragma: no cover
        raise ToolError("numpy is required for media analysis (pip install numpy)") from exc


def _pil():
    try:
        from PIL import Image, ImageDraw  # type: ignore

        return Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise ToolError("pillow is required for media analysis (pip install pillow)") from exc


def extract_audio(source: Path, wav_out: Path, sample_rate: int = 16000) -> Path:
    exe = ffmpeg_bin()
    if not exe:
        raise ToolError("ffmpeg is not available")
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
        "-f", "wav", str(wav_out),
    ]
    result = run_cmd(command, timeout=300)
    if not result.ok or not wav_out.exists() or wav_out.stat().st_size < 64:
        raise ToolError(f"could not extract audio: {result.tail(400)}")
    return wav_out


def read_wav(path: Path):
    import wave

    np = _np()
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    data = np.frombuffer(frames, dtype=dtype).astype("float32")
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data = data / float(2 ** (8 * width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def analyse_audio(wav: Path, out_dir: Path, label: str = "audio") -> dict:
    np = _np()
    Image, ImageDraw = _pil()
    samples, rate = read_wav(wav)
    if samples.size == 0:
        raise ToolError("decoded audio is empty")
    duration = samples.size / float(rate)
    window = max(1, int(rate * 0.05))
    usable = samples[: (samples.size // window) * window]
    rms = np.sqrt((usable.reshape(-1, window) ** 2).mean(axis=1)) if usable.size else np.array([0.0])
    peak = float(np.abs(samples).max())
    mean_rms = float(rms.mean())
    threshold = max(mean_rms * 0.55, 0.004)
    loud = rms > threshold
    segments, start = [], None
    for index, flag in enumerate(loud):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * 0.05 >= 0.2:
                segments.append({"start": round(start * 0.05, 2), "end": round(index * 0.05, 2)})
            start = None
    if start is not None:
        segments.append({"start": round(start * 0.05, 2), "end": round(len(loud) * 0.05, 2)})

    # waveform
    width, height = 1000, 260
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, height // 2), (width, height // 2)], fill=(209, 213, 219), width=1)
    step = max(1, len(rms) // width)
    bars = rms[::step][:width]
    scale = (height / 2 - 8) / max(float(bars.max()), 1e-6)
    for x, value in enumerate(bars):
        half = float(value) * scale
        draw.line([(x, height / 2 - half), (x, height / 2 + half)], fill=(37, 99, 235), width=1)
    waveform = out_dir / f"{label}_waveform.png"
    canvas.save(waveform)

    # spectrogram (log-magnitude STFT, no scipy needed)
    fft_size, hop = 1024, 512
    if samples.size >= fft_size:
        frames = 1 + (samples.size - fft_size) // hop
        frames = min(frames, 700)
        window_fn = np.hanning(fft_size).astype("float32")
        index = np.arange(fft_size)[None, :] + hop * np.arange(frames)[:, None]
        block = samples[index] * window_fn
        spectrum = np.abs(np.fft.rfft(block, axis=1))[:, :256]
        spectrum = np.log1p(spectrum)
        spectrum = spectrum / max(float(spectrum.max()), 1e-6)
        # inverted: silence reads as paper, energy as ink
        img = Image.fromarray((255 - spectrum.T[::-1] * 255).astype("uint8"), mode="L")
        img = img.resize((720, 300)).convert("RGB")
        spectrogram = out_dir / f"{label}_spectrogram.png"
        img.save(spectrogram)
    else:
        spectrogram = None

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": rate,
        "peak_amplitude": round(peak, 4),
        "mean_rms": round(mean_rms, 4),
        "silence_ratio": round(1.0 - float(loud.mean()) if loud.size else 1.0, 3),
        "segments": segments[:40],
        "segment_count": len(segments),
        "waveform": waveform.name,
        "spectrogram": spectrogram.name if spectrogram else "",
    }


def extract_frames(source: Path, out_dir: Path, count: int = 6, width: int = 960) -> list[Path]:
    exe = ffmpeg_bin()
    if not exe:
        raise ToolError("ffmpeg is not available")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()
    info = probe(source)
    duration = info.duration or 0.0
    count = max(1, min(int(count), 40))
    fps = count / duration if duration > 0.5 else 1.0
    command = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vf", f"fps={max(fps, 0.05):.5f},scale={int(width)}:-2",
        "-frames:v", str(count), "-q:v", "3", str(out_dir / "frame_%03d.jpg"),
    ]
    result = run_cmd(command, timeout=300)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise ToolError(f"no frames extracted: {result.tail(400)}")
    return frames


def contact_sheet(frames: list[Path], out: Path, columns: int = 3, thumb_width: int = 420) -> Path:
    Image, _ = _pil()
    if not frames:
        raise ToolError("no frames for contact sheet")
    thumbs = []
    for frame in frames:
        with Image.open(frame) as img:
            ratio = thumb_width / img.width
            thumbs.append(img.convert("RGB").resize((thumb_width, int(img.height * ratio)), Image.LANCZOS))
    cols = max(1, min(int(columns), len(thumbs)))
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), (247, 248, 250))
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * thumb_width
        y = (index // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")
    return out


def analyse_image(path: Path) -> dict:
    np = _np()
    Image, _ = _pil()
    with Image.open(path) as img:
        original = img.convert("RGB")
        width, height = original.size
        small = original.resize((128, 128), Image.LANCZOS)
        arr = np.asarray(small, dtype="float32")

    grey = arr.mean(axis=2)
    brightness = float(grey.mean() / 255.0)
    contrast = float(grey.std() / 255.0)
    gx = np.abs(np.diff(grey, axis=1)).mean()
    gy = np.abs(np.diff(grey, axis=0)).mean()
    edge_density = float((gx + gy) / 2.0 / 255.0)

    quantised = small.quantize(colors=5, method=Image.MEDIANCUT)
    palette = quantised.getpalette() or []
    counts = sorted(quantised.getcolors() or [], reverse=True)
    dominant = []
    for _, index in counts[:5]:
        r, g, b = palette[index * 3 : index * 3 + 3]
        dominant.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "rgb": [r, g, b]})

    grey32 = np.asarray(small.convert("L").resize((32, 32)), dtype="float32")
    phash = grey32 > grey32.mean()
    bits = "".join("1" if value else "0" for value in phash.flatten())
    fingerprint = f"{int(bits, 2):064x}"

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3) if height else 0.0,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "edge_density": round(edge_density, 4),
        "dominant_colors": dominant,
        "perceptual_hash": fingerprint,
        "description": _describe_colours(dominant, brightness, edge_density),
    }


def _describe_colours(dominant: list[dict], brightness: float, edge_density: float) -> str:
    tone = "dark" if brightness < 0.3 else ("bright" if brightness > 0.65 else "mid-tone")
    texture = "busy" if edge_density > 0.12 else ("textured" if edge_density > 0.05 else "smooth")
    colors = ", ".join(c["hex"] for c in dominant[:3]) or "unknown"
    return f"{tone}, {texture} image dominated by {colors}"


def transcribe_audio(wav: Path, ctx: ToolContext) -> tuple[str, str]:
    """Return (text, method). Empty text means no local engine was usable."""
    template = ctx.config.asr_command
    if template:
        try:
            output_txt = wav.with_suffix(".txt")
            command = template.format(audio=str(wav), output=str(output_txt))
            result = run_cmd(shlex.split(command), timeout=1800)
            if output_txt.exists():
                return output_txt.read_text(encoding="utf-8", errors="replace").strip(), "asr-command"
            if result.ok and result.stdout.strip():
                return result.stdout.strip(), "asr-command"
        except Exception:
            pass

    try:  # optional fast local engine
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav), vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, "faster-whisper:base"
    except Exception:
        pass

    whisper_cli = Path("/usr/local/bin/whisper")
    if whisper_cli.exists():
        result = run_cmd([str(whisper_cli), str(wav), "--output_format", "txt", "--output_dir", str(wav.parent)], timeout=3600)
        candidate = wav.with_suffix(".txt")
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip(), "whisper-cli"
        if result.ok:
            return result.stdout.strip(), "whisper-cli"

    return "", "unavailable"


def vision_caption(image: Path, ctx: ToolContext, prompt: str = ""):
    """Ask a configured vision model for a one-paragraph description."""
    provider = getattr(ctx, "provider", None)
    if provider is None:
        return "", "no-vision-model"
    try:
        from ..llm.base import LLMProvider, Message, multimodal  # noqa: F401

        question = prompt or (
            "Describe this image for an engineer who cannot see it: subject, composition, "
            "colours, any text, and anything that looks wrong."
        )
        completion = provider.complete(
            [multimodal(question, [image])], temperature=0.2, max_tokens=350
        )
        return completion.text.strip(), provider.model
    except Exception as exc:
        return "", f"vision-failed: {type(exc).__name__}"


def _scene_changes(frames: list[Path]) -> list[dict]:
    np = _np()
    Image, _ = _pil()
    vectors = []
    for frame in frames:
        with Image.open(frame) as img:
            vectors.append(np.asarray(img.convert("L").resize((64, 64)), dtype="float32"))
    changes = []
    for index in range(1, len(vectors)):
        diff = float(np.abs(vectors[index] - vectors[index - 1]).mean())
        changes.append({"between_frames": [index, index + 1], "mean_abs_diff": round(diff, 2)})
    return changes


def _summary_markdown(observations: dict, rel_source: str) -> str:
    kind = observations.get("kind", "file")
    probe_info = observations.get("probe", {})
    lines = [f"# Analysis: {rel_source}", ""]
    lines += [
        "## What was supplied",
        f"- Kind: **{kind}**",
        f"- Duration: {probe_info.get('duration', 0)} s",
        f"- Dimensions: {probe_info.get('width', 0)}x{probe_info.get('height', 0)}",
        f"- Streams: video={probe_info.get('has_video')} audio={probe_info.get('has_audio')}",
        f"- Size: {probe_info.get('bytes', 0)} bytes",
        "",
    ]
    visual = observations.get("visual") or {}
    if visual:
        lines += ["## Visuals", f"- {visual.get('description', '')}",
                  f"- Brightness {visual.get('brightness')} / contrast {visual.get('contrast')} / edges {visual.get('edge_density')}"]
        lines.append("- Dominant colours: " + ", ".join(c["hex"] for c in visual.get("dominant_colors", [])))
        if visual.get("caption"):
            lines.append(f"- Model caption ({visual.get('caption_model')}): {visual['caption']}")
        lines.append("")
    sound = observations.get("audio") or {}
    if sound:
        lines += [
            "## Audio",
            f"- {sound.get('segment_count')} speech-like segment(s), silence ratio {sound.get('silence_ratio')}",
            f"- First segments: {sound.get('segments', [])[:5]}",
        ]
        lines.append("")
    if observations.get("captions") is not None:
        lines += ["## Speech", f"- Method: {observations.get('transcript_method')}"]
        transcript = observations.get("transcript") or ""
        lines.append(f"- Transcript: {transcript[:1200] if transcript else '(no local speech engine available)'}")
        lines.append("")
    changes = observations.get("scene_changes") or []
    if changes:
        biggest = sorted(changes, key=lambda c: c["mean_abs_diff"], reverse=True)[:3]
        lines += ["## Structure", f"- Strongest visual changes between sampled frames: {biggest}", ""]
    lines += [
        "## Limits",
        "- Analysis is signal-level and local; it does not claim semantic understanding.",
        "- Speech-to-text and image captioning need a local engine or a configured model.",
        "",
        "## Next steps",
        "- Run with a vision/ASR model for semantic descriptions.",
        "- Feed the observations into a deck, report or video script.",
        "",
    ]
    return "\n".join(lines)


def register(registry) -> None:
    @registry.tool(
        "media.ingest",
        "Analyse an image, audio or video file: metadata, colour/signal stats, frames, transcript.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "out_dir": {"type": "string"},
                "frames": {"type": "integer", "default": 6},
                "transcribe": {"type": "boolean", "default": True},
                "describe": {"type": "boolean", "default": True},
            },
            "required": ["path"],
        },
        category="media",
        requires=["ffmpeg", "pillow", "numpy"],
    )
    def media_ingest(
        ctx: ToolContext,
        path: str,
        out_dir: str = "",
        frames: int = 6,
        transcribe: bool = True,
        describe: bool = True,
    ) -> ToolResult:
        source = ctx.path(path, must_exist=True)
        info = probe(source)
        if info.error:
            raise ToolError(f"cannot read {path}: {info.error}")
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem)[:40] or "media"
        target_dir = ctx.path(out_dir or f"media/{stem}")
        target_dir.mkdir(parents=True, exist_ok=True)

        observations: dict = {
            "source": ctx.rel(source),
            "kind": info.kind,
            "probe": info.to_dict(),
            "artifacts": [],
        }
        artifacts = []

        if info.kind == "image":
            observations["visual"] = analyse_image(source)
            if describe:
                caption, method = vision_caption(source, ctx)
                observations["visual"]["caption"] = caption
                observations["visual"]["caption_model"] = method
        else:
            if info.has_audio:
                wav = extract_audio(source, target_dir / "audio.wav")
                observations["audio"] = analyse_audio(wav, target_dir, label=info.kind)
                artifacts.append(wav)
                for key in ("waveform", "spectrogram"):
                    name = observations["audio"].get(key)
                    if name and (target_dir / name).exists():
                        artifacts.append(target_dir / name)
                if transcribe:
                    text, method = transcribe_audio(wav, ctx)
                    observations["transcript"] = text
                    observations["transcript_method"] = method
                    observations["captions"] = {"chars": len(text)}
            if info.has_video:
                frame_dir = target_dir / "frames"
                frame_paths = extract_frames(source, frame_dir, count=int(frames))
                observations["frames"] = [str(p.relative_to(target_dir)) for p in frame_paths]
                observations["scene_changes"] = _scene_changes(frame_paths)
                sheet = contact_sheet(frame_paths, target_dir / "contact_sheet.png")
                artifacts.append(sheet)
                artifacts.extend(frame_paths)
                observations["visual"] = analyse_image(frame_paths[0])
                if describe:
                    caption, method = vision_caption(frame_paths[0], ctx)
                    observations["visual"]["caption"] = caption
                    observations["visual"]["caption_model"] = method

        observations_path = target_dir / "observations.json"
        observations_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
        summary_path = target_dir / "analysis.md"
        summary_path.write_text(_summary_markdown(observations, ctx.rel(source)), encoding="utf-8")
        artifacts.extend([observations_path, summary_path])

        arts = []
        for item in artifacts:
            if isinstance(item, Path) and item.exists():
                kind = {'.json': 'data', '.md': 'report', '.wav': 'audio', '.png': 'image', '.jpg': 'image'}.get(item.suffix.lower(), 'file')
                arts.append(artifact(ctx, item, kind))

        probe_payload = observations["probe"]
        return ToolResult.success(
            {
                "kind": info.kind,
                "observations": ctx.rel(observations_path),
                "summary": ctx.rel(summary_path),
                "duration": probe_payload.get("duration", 0),
                "width": probe_payload.get("width", 0),
                "height": probe_payload.get("height", 0),
                "transcript_chars": len(observations.get("transcript", "") or ""),
                "transcript_method": observations.get("transcript_method", "not-applicable"),
                "frames": len(observations.get("frames", [])),
                "facts": {
                    "visual": observations.get("visual", {}).get("description", ""),
                    "audio": observations.get("audio", {}).get("segment_count", 0),
                },
            },
            arts,
        )

    @registry.tool(
        "media.probe",
        "Report technical facts about any media file (kind, duration, streams, size).",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        category="media",
    )
    def media_probe(ctx: ToolContext, path: str) -> ToolResult:
        target = ctx.path(path, must_exist=True)
        info = probe(target)
        if info.error:
            return ToolResult.failure(f"{path}: {info.error}", info.to_dict())
        return ToolResult.success(info.to_dict())
