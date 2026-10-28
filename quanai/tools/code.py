"""Code execution tools: python, node and (opt-in) shell."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..errors import ToolError
from ..util import run_cmd, which
from .registry import ToolContext, ToolResult, artifact

SHELL_DENY = (
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){",
    "shutdown",
    "reboot",
    "chmod 777 /",
    "curl | sh",
    "wget | sh",
)


def _pythonpath() -> str:
    root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(root)] + ([existing] if existing else [])
    return os.pathsep.join(parts)


def register(registry) -> None:
    @registry.tool(
        "code.python_run",
        "Run a python script or snippet inside the workspace and capture its output.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "script to run (relative to workspace)"},
                "code": {"type": "string", "description": "inline snippet, used when path is empty"},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout": {"type": "number", "default": 120},
            },
            "required": [],
        },
        category="code",
    )
    def python_run(
        ctx: ToolContext,
        path: str = "",
        code: str = "",
        args: list | None = None,
        timeout: float = 120.0,
    ) -> ToolResult:
        if not path and not code:
            raise ToolError("provide either 'path' or 'code'")
        python = sys.executable or which("python3") or "python3"
        command = [python]
        if path:
            script = ctx.path(path, must_exist=True)
            command.append(str(script))
        else:
            target = ctx.path(f"scripts/snippet_{abs(hash(code)) % 10**8}.py")
            target.write_text(code, encoding="utf-8")
            command.append(str(target))
        command.extend(str(a) for a in (args or []))
        result = run_cmd(
            command,
            cwd=ctx.workspace,
            timeout=float(timeout),
            env={"PYTHONPATH": _pythonpath(), "PYTHONUNBUFFERED": "1"},
        )
        artifacts = []
        if path:
            artifacts = [artifact(ctx, ctx.path(path, must_exist=True), "code", "text/x-python")]
        payload = result.to_dict()
        payload["ok"] = result.ok
        if not result.ok:
            return ToolResult(
                ok=False,
                error=f"python exited with code {result.code}: {result.tail(600)}",
                data=payload,
                artifacts=artifacts,
            )
        return ToolResult.success(payload, artifacts)

    @registry.tool(
        "code.node_check",
        "Syntax-check javascript files with node --check.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        category="code",
        requires=["node"],
    )
    def node_check(ctx: ToolContext, path: str) -> ToolResult:
        node = which("node")
        if not node:
            raise ToolError("node is not installed")
        target = ctx.path(path, must_exist=True)
        files = [target] if target.is_file() else sorted(target.rglob("*.js"))
        broken = []
        for file in files:
            result = run_cmd([node, "--check", str(file)], timeout=40)
            if not result.ok:
                broken.append({"file": ctx.rel(file), "error": result.tail(300)})
        if broken:
            return ToolResult.failure(f"{len(broken)} javascript file(s) failed", {"broken": broken})
        return ToolResult.success({"checked": [ctx.rel(f) for f in files]})


    @registry.tool(
        "code.shell",
        "Run a shell command in the workspace (disabled unless QUANAI_ALLOW_SHELL=1).",
        schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number", "default": 120},
            },
            "required": ["command"],
        },
        category="code",
    )
    def shell(ctx: ToolContext, command: str, timeout: float = 120.0) -> ToolResult:
        if not ctx.config.allow_shell:
            raise ToolError("shell access is disabled; set QUANAI_ALLOW_SHELL=1 to enable it")
        lowered = command.lower()
        for denied in SHELL_DENY:
            if denied in lowered:
                raise ToolError(f"refusing dangerous command containing {denied!r}")
        result = run_cmd(["/bin/sh", "-lc", command], cwd=ctx.workspace, timeout=float(timeout))
        payload = result.to_dict()
        payload["ok"] = result.ok
        if not result.ok:
            return ToolResult(
                ok=False,
                error=f"command failed ({result.code}): {result.tail(600)}",
                data=payload,
            )
        return ToolResult.success(payload)

    @registry.tool(
        "code.zip",
        "Zip a workspace folder or file into an archive artifact.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "archive": {"type": "string"}},
            "required": ["path", "archive"],
        },
        category="filesystem",
    )
    def zip_path(ctx: ToolContext, path: str, archive: str) -> ToolResult:
        import zipfile

        source = ctx.path(path, must_exist=True)
        target = ctx.path(archive)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
            if source.is_dir():
                for file in sorted(source.rglob("*")):
                    if file.is_file():
                        bundle.write(file, file.relative_to(source))
            else:
                bundle.write(source, source.name)
        with zipfile.ZipFile(target) as bundle:
            count = len(bundle.namelist())
        return ToolResult.success(
            {"archive": ctx.rel(target), "members": count},
            [artifact(ctx, target, "archive", "application/zip", members=count)],
        )
