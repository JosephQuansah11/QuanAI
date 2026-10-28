"""QuanAI - a self-checking, multimodal agent runtime.

QuanAI turns a plain-language request into a durable *objective*: an explicit
requirement list, a dependency-ordered task graph, and machine-verifiable
acceptance checks.  It then plans, executes tools, verifies every requirement
by running the checks, critiques its own output, repairs what failed and
records lessons so later runs start smarter.

The runtime is designed to work with **zero paid API access**:

* planning / critique / content fall back to a deterministic engine that runs
  entirely on the local machine;
* any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, llama.cpp server,
  or a hosted API) can be plugged in through environment variables;
* every artifact (deck, image, video, app scaffold) is produced by local code.
"""

from __future__ import annotations

from .config import Config
from .core.agent import Agent, RunReport
from .core.ledger import Ledger
from .core.task import (
    Check,
    Objective,
    Requirement,
    Status,
    Task,
)

__all__ = [
    "Agent",
    "Check",
    "Config",
    "Ledger",
    "Objective",
    "Requirement",
    "RunReport",
    "Status",
    "Task",
]

__version__ = "0.1.0"
