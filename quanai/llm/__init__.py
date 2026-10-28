"""Provider-agnostic LLM layer.

QuanAI never *requires* a language model: when none is reachable the
deterministic planner/critic engines take over.  When one is reachable (a local
Ollama/LM Studio/vLLM server, or any hosted OpenAI-compatible endpoint) the
agent uses it for richer planning, content and critique.
"""

from __future__ import annotations

from .base import Completion, LLMError, LLMProvider, Message, content_text
from .openai_compat import OpenAICompatProvider
from .router import build_provider, detect_providers

__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "Message",
    "OpenAICompatProvider",
    "build_provider",
    "content_text",
    "detect_providers",
]
