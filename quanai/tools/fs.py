"""Filesystem tools - all confined to the objective workspace."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import ToolError
from .registry import ToolContext, ToolResult, artifact

MIME = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".html": "text/html",
    ".css": "text/css",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def register(registry) -> None:
    @registry.tool(
        "fs.write",
        "Write a UTF-8 text file inside the workspace (creates parent folders).",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "workspace-relative file path"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
        category="filesystem",
    )
    def fs_write(ctx: ToolContext, path: str, content: str, append: bool = False) -> ToolResult:
        target = ctx.path(path)
        if target.exists() and target.is_dir():
            raise ToolError(f"{path} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if append and target.exists():
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")
        return ToolResult.success(
            {"path": ctx.rel(target), "bytes": target.stat().st_size, "lines": content.count("\n") + 1},
            [artifact(ctx, target, "file", MIME.get(target.suffix.lower(), "text/plain"))],
        )

    @registry.tool(
        "fs.read",
        "Read a text file from the workspace.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        },
        category="filesystem",
    )
    def fs_read(ctx: ToolContext, path: str, max_chars: int = 20000) -> ToolResult:
        target = ctx.path(path, must_exist=True)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > int(max_chars)
        return ToolResult.success(
            {
                "path": ctx.rel(target),
                "content": text[: int(max_chars)],
                "chars": len(text),
                "truncated": truncated,
            }
        )

    @registry.tool(
        "fs.list",
        "List files under a workspace folder.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        category="filesystem",
    )
    def fs_list(ctx: ToolContext, path: str = ".", pattern: str = "*", recursive: bool = True) -> ToolResult:
        root = ctx.path(path, must_exist=True)
        if root.is_file():
            entries = [root]
        else:
            entries = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
        files = [
            {"path": ctx.rel(p), "bytes": p.stat().st_size}
            for p in entries
            if p.is_file()
        ]
        return ToolResult.success({"root": ctx.rel(root), "count": len(files), "files": files[:500]})

    @registry.tool(
        "fs.delete",
        "Delete a file or folder inside the workspace.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        category="filesystem",
    )
    def fs_delete(ctx: ToolContext, path: str) -> ToolResult:
        target = ctx.path(path)
        if not target.exists():
            return ToolResult.failure(f"nothing to delete: {path}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return ToolResult.success({"deleted": ctx.rel(target)})

    @registry.tool(
        "fs.copy",
        "Copy a file or folder inside the workspace.",
        schema={
            "type": "object",
            "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["source", "destination"],
        },
        category="filesystem",
    )
    def fs_copy(ctx: ToolContext, source: str, destination: str) -> ToolResult:
        src = ctx.path(source, must_exist=True)
        dst = ctx.path(destination)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return ToolResult.success({"from": ctx.rel(src), "to": ctx.rel(dst)})
