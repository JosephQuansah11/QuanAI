"""Tool registry with schemas, workspace sandboxing and uniform results."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..errors import ToolError, ToolNotFound
from ..util import safe_join


@dataclass
class Artifact:
    path: str          # workspace-relative when inside the workspace
    kind: str = "file"  # deck | image | video | audio | app | report | data
    mime: str = ""
    bytes: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str = ""
    duration: float = 0.0
    tool: str = ""

    @classmethod
    def success(
        cls, data: dict | None = None, artifacts: list[Artifact] | None = None, tool: str = ""
    ) -> "ToolResult":
        return cls(ok=True, data=data or {}, artifacts=artifacts or [], tool=tool)

    @classmethod
    def failure(cls, error: str, data: dict | None = None, tool: str = "") -> "ToolResult":
        return cls(ok=False, error=error, data=data or {}, tool=tool)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error": self.error,
            "duration": round(self.duration, 3),
        }


@dataclass
class ToolContext:
    """Everything a tool may touch: the workspace and read-only config."""

    workspace: Path
    config: Config
    objective_id: str = ""
    provider: Any = None

    def path(self, value: str | Path, must_exist: bool = False, mkdir: bool = True) -> Path:
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute():
            resolved = candidate
        else:
            resolved = safe_join(self.workspace, str(candidate))
        if must_exist and not resolved.exists():
            raise ToolError(f"path does not exist: {value}")
        if mkdir and not must_exist:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.workspace.resolve()))
        except ValueError:
            return str(path)


Handler = Callable[..., ToolResult | dict]


@dataclass
class ToolSpec:
    name: str
    summary: str
    handler: Handler
    category: str = "general"
    cost: str = "free"
    requires: list[str] = field(default_factory=list)
    schema: dict = field(default_factory=dict)

    def required_args(self) -> list[str]:
        return list(self.schema.get("required", []))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "category": self.category,
            "cost": self.cost,
            "requires": self.requires,
            "required": self.required_args(),
            "properties": sorted((self.schema.get("properties") or {}).keys()),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # -- registration ---------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        summary: str,
        schema: dict | None = None,
        category: str = "general",
        cost: str = "free",
        requires: list[str] | None = None,
    ) -> Callable[[Handler], Handler]:
        def wrapper(fn: Handler) -> Handler:
            self.register(
                ToolSpec(
                    name=name,
                    summary=summary,
                    handler=fn,
                    category=category,
                    cost=cost,
                    requires=requires or [],
                    schema=schema or {"type": "object", "properties": {}, "required": []},
                )
            )
            return fn

        return wrapper

    # -- inspection -----------------------------------------------------
    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolNotFound(f"unknown tool: {name}")
        return spec

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict]:
        return [self._tools[n].describe() for n in self.names()]

    def catalog_text(self) -> str:
        """Compact catalog used when prompting a language model."""
        lines = []
        for name in self.names():
            spec = self._tools[name]
            props = spec.schema.get("properties") or {}
            params = ", ".join(
                f"{key}:{val.get('type', 'any')}" if isinstance(val, dict) else str(key)
                for key, val in props.items()
            )
            lines.append(f"- {name}({params}) [{spec.category}] {spec.summary}")
        return "\n".join(lines)

    # -- execution ------------------------------------------------------
    def execute(self, name: str, args: dict | None, ctx: ToolContext) -> ToolResult:
        started = time.time()
        try:
            spec = self.get(name)
        except ToolNotFound as exc:
            return ToolResult.failure(str(exc), tool=name)
        args = dict(args or {})
        missing = [k for k in spec.required_args() if k not in args or args[k] in (None, "")]
        if missing:
            return ToolResult.failure(
                f"missing required argument(s): {', '.join(missing)}", tool=name
            )
        try:
            outcome = spec.handler(ctx, **args)
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc), duration=time.time() - started, tool=name)
        except TypeError as exc:
            return ToolResult(
                ok=False,
                error=f"bad arguments for {name}: {exc}",
                duration=time.time() - started,
                tool=name,
            )
        except Exception as exc:
            detail = traceback.format_exc(limit=3)
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                data={"traceback": detail[-2000:]},
                duration=time.time() - started,
                tool=name,
            )
        if isinstance(outcome, ToolResult):
            outcome.tool = outcome.tool or name
            if not outcome.duration:
                outcome.duration = round(time.time() - started, 3)
            return outcome
        if isinstance(outcome, dict):
            raw = outcome.pop("_artifacts", []) or []
            artifacts = [a if isinstance(a, Artifact) else Artifact(**a) for a in raw]
            return ToolResult(
                ok=True,
                data=outcome,
                artifacts=artifacts,
                duration=time.time() - started,
                tool=name,
            )
        return ToolResult.success({"value": outcome}, tool=name)


def artifact(ctx: ToolContext, path: Path, kind: str, mime: str = "", **meta: Any) -> Artifact:
    """Build an Artifact record from a real file on disk."""
    path = Path(path)
    return Artifact(
        path=ctx.rel(path),
        kind=kind,
        mime=mime,
        bytes=path.stat().st_size if path.exists() else 0,
        meta=meta,
    )
