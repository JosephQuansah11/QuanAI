"""Tool surface: filesystem, code, slides, images, video, media and apps."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import apps, code, fs, images, media, plan, report, slides, video
from .registry import Artifact, ToolContext, ToolRegistry, ToolResult, ToolSpec, artifact


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for module in (fs, code, slides, images, video, media, apps, plan, report):
        module.register(registry)
    return registry


def probe_capabilities() -> dict:
    """Report what this machine can actually do, without raising."""
    from ..util import ffmpeg_bin

    def has_module(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    return {
        "python": shutil.which("python3") or "",
        "node": shutil.which("node") or "",
        "flutter": shutil.which("flutter") or "",
        "ffmpeg": ffmpeg_bin() or "",
        "pillow": has_module("PIL"),
        "numpy": has_module("numpy"),
        "python-pptx": has_module("pptx"),
        "faster-whisper": has_module("faster_whisper"),
        "workspace": str(Path.cwd()),
    }


__all__ = [
    "Artifact",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "artifact",
    "build_default_registry",
    "probe_capabilities",
]
