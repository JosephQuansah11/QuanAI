"""Media probing shared by the tools and the verification checks.

Uses Pillow for still images and parses ``ffmpeg -i`` for audio / video, so no
extra system package (ffprobe) is needed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .util import ffmpeg_bin, run_cmd

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_STREAM = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+)")
_AUDIO_STREAM = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+)")
_DIMENSIONS = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"([\d.]+)\s*fps")


@dataclass
class MediaInfo:
    path: str
    kind: str = "unknown"  # image | audio | video | unknown
    exists: bool = False
    bytes: int = 0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    format: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def guess_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        return "image"
    if suffix in AUDIO_EXT:
        return "audio"
    if suffix in VIDEO_EXT:
        return "video"
    return "unknown"


def image_info(path: Path) -> MediaInfo:
    info = MediaInfo(path=str(path), kind="image", exists=path.exists())
    if not info.exists:
        info.error = "file not found"
        return info
    info.bytes = path.stat().st_size
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            info.width, info.height = img.size
            info.format = (img.format or "").lower()
            info.extra["mode"] = img.mode
            info.extra["frames"] = getattr(img, "n_frames", 1)
            info.has_video = False
    except Exception as exc:  # pragma: no cover - corrupt files
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def ffmpeg_info(path: Path) -> MediaInfo:
    info = MediaInfo(path=str(path), kind=guess_kind(path), exists=path.exists())
    if not info.exists:
        info.error = "file not found"
        return info
    info.bytes = path.stat().st_size
    exe = ffmpeg_bin()
    if not exe:
        info.error = "ffmpeg not available"
        return info
    result = run_cmd([exe, "-hide_banner", "-i", str(path)], timeout=45)
    text = result.stderr or result.stdout
    if "Invalid data" in text:
        info.error = "ffmpeg could not decode the file"
        return info
    match = _DURATION.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        info.duration = round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3)
    video = _VIDEO_STREAM.search(text)
    audio = _AUDIO_STREAM.search(text)
    info.has_video = bool(video)
    info.has_audio = bool(audio)
    info.video_codec = video.group(1) if video else ""
    info.audio_codec = audio.group(1) if audio else ""
    dims = _DIMENSIONS.search(text)
    if dims:
        info.width, info.height = int(dims.group(1)), int(dims.group(2))
    fps = _FPS.search(text)
    if fps:
        try:
            info.fps = round(float(fps.group(1)), 3)
        except ValueError:
            info.fps = 0.0
    if info.has_video:
        info.kind = "video"
    elif info.has_audio:
        info.kind = "audio"
    return info


def probe(path: Path | str) -> MediaInfo:
    path = Path(path)
    guessed = guess_kind(path)
    if guessed == "image":
        return image_info(path)
    if guessed in {"audio", "video"}:
        info = ffmpeg_info(path)
        if info.error and info.exists:
            fallback = image_info(path)
            if not fallback.error:
                return fallback
        return info
    info = ffmpeg_info(path)
    if not info.has_audio and not info.has_video:
        fallback = image_info(path)
        if not fallback.error:
            return fallback
    return info
