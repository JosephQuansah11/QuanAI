"""Runtime configuration, read from the environment with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    """All knobs in one place.

    Nothing here is required: with an empty environment QuanAI runs fully
    offline using local code only.
    """

    workspace: Path = field(default_factory=lambda: Path.cwd() / "workspace")
    home: Path = field(default_factory=lambda: Path.cwd() / ".quanai")

    # LLM routing.  "auto" probes for a local server, then for an API key,
    # then falls back to the deterministic offline engine.
    provider: str = "auto"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    vision_model: str | None = None
    llm_timeout: float = 120.0

    # Capability backends.
    image_backend: str = "auto"  # auto | procedural | pollinations | sd
    asr_command: str | None = None
    allow_shell: bool = False
    allow_network: bool = False

    # Loop control.
    max_iterations: int = 6
    max_attempts: int = 3
    tool_timeout: float = 180.0

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.home = Path(self.home).expanduser().resolve()

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls(
            workspace=Path(_env("QUANAI_WORKSPACE", str(Path.cwd() / "workspace"))),
            home=Path(_env("QUANAI_HOME", str(Path.cwd() / ".quanai"))),
            provider=_env("QUANAI_LLM_PROVIDER", "auto") or "auto",
            base_url=_env("QUANAI_BASE_URL") or _env("OPENAI_BASE_URL"),
            api_key=_env("QUANAI_API_KEY") or _env("OPENAI_API_KEY"),
            model=_env("QUANAI_MODEL") or _env("OPENAI_MODEL"),
            vision_model=_env("QUANAI_VISION_MODEL"),
            llm_timeout=float(_env("QUANAI_LLM_TIMEOUT", "120")),
            image_backend=_env("QUANAI_IMAGE_BACKEND", "auto") or "auto",
            asr_command=_env("QUANAI_ASR_CMD"),
            allow_shell=_env_bool("QUANAI_ALLOW_SHELL", False),
            allow_network=_env_bool("QUANAI_ALLOW_NETWORK", False),
            max_iterations=_env_int("QUANAI_MAX_ITERATIONS", 6),
            max_attempts=_env_int("QUANAI_MAX_ATTEMPTS", 3),
            tool_timeout=float(_env("QUANAI_TOOL_TIMEOUT", "180")),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        # re-normalise paths after overrides so callers may pass plain strings
        cfg.__post_init__()
        return cfg

    # ------------------------------------------------------------------
    def ensure_dirs(self) -> "Config":
        for path in (
            self.workspace,
            self.home,
            self.home / "objectives",
            self.home / "logs",
            self.home / "memory",
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def objective_dir(self, objective_id: str) -> Path:
        path = self.workspace / objective_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe(self) -> dict:
        return {
            "workspace": str(self.workspace),
            "home": str(self.home),
            "provider": self.provider,
            "base_url": self.base_url or "",
            "model": self.model or "",
            "image_backend": self.image_backend,
            "allow_shell": self.allow_shell,
            "allow_network": self.allow_network,
            "max_iterations": self.max_iterations,
        }
