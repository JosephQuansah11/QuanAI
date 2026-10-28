"""Exception hierarchy used across QuanAI."""

from __future__ import annotations


class QuanAIError(Exception):
    """Base class for every error raised by QuanAI."""


class ConfigError(QuanAIError):
    """Invalid or contradictory configuration."""


class ToolError(QuanAIError):
    """A tool could not complete its work."""


class ToolNotFound(ToolError):
    """The requested tool is not registered."""


class MissingDependency(ToolError):
    """A tool needs an optional dependency that is not installed."""

    def __init__(self, tool: str, dependency: str, hint: str = "") -> None:
        self.tool = tool
        self.dependency = dependency
        self.hint = hint
        msg = f"tool '{tool}' requires '{dependency}'"
        if hint:
            msg += f" ({hint})"
        super().__init__(msg)


class CheckError(QuanAIError):
    """A verification check is unknown or malformed."""


class PlanError(QuanAIError):
    """The planner could not produce a usable plan."""
