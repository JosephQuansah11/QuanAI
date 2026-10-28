"""Shared fixtures: an isolated config, workspace and tool context per test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quanai.config import Config  # noqa: E402
from quanai.core.ledger import Ledger  # noqa: E402
from quanai.core.memory import Memory  # noqa: E402
from quanai.tools import ToolContext, build_default_registry  # noqa: E402


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    cfg = Config(
        workspace=tmp_path / "workspace",
        home=tmp_path / "state",
        provider="offline",
        image_backend="procedural",
        max_iterations=3,
        max_attempts=2,
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture()
def registry():
    return build_default_registry()


@pytest.fixture()
def ctx(config: Config) -> ToolContext:
    return ToolContext(workspace=config.objective_dir("test-obj"), config=config)


@pytest.fixture()
def ledger(config: Config) -> Ledger:
    return Ledger(config.home)


@pytest.fixture()
def memory(config: Config) -> Memory:
    return Memory(config.home / "memory")
