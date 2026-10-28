"""Decide which (if any) language model to use.

The order is deliberately "free first": a local server already running on the
machine wins over a paid API, and no server at all is a perfectly valid state.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..config import Config
from .base import LLMProvider
from .openai_compat import OpenAICompatProvider

# name -> (base_url, default model, env hint)
LOCAL_CANDIDATES = {
    "ollama": ("http://127.0.0.1:11434/v1", "llama3.1", "ollama serve"),
    "lmstudio": ("http://127.0.0.1:1234/v1", "local-model", "LM Studio server"),
    "vllm": ("http://127.0.0.1:8000/v1", "local-model", "vllm serve <model>"),
    "llamacpp": ("http://127.0.0.1:8080/v1", "local-model", "llama-server -m model.gguf"),
}

_PROBE_CACHE: dict[str, tuple[bool, str]] = {}


def _http_json(url: str, timeout: float = 0.8) -> dict | None:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _probe(base_url: str, timeout: float = 0.8) -> bool:
    if base_url in _PROBE_CACHE:
        return _PROBE_CACHE[base_url][0]
    ok = _http_json(f"{base_url.rstrip('/')}/models", timeout=timeout) is not None
    _PROBE_CACHE[base_url] = (ok, "")
    return ok


def _first_model(base_url: str) -> str:
    data = _http_json(f"{base_url.rstrip('/')}/models", timeout=2.0) or {}
    models = data.get("data") if isinstance(data, dict) else None
    if isinstance(models, list) and models:
        first = models[0]
        if isinstance(first, dict):
            return str(first.get("id") or first.get("name") or "")
        if isinstance(first, str):
            return first
    return ""


def detect_providers(config: Config) -> list[dict]:
    """Report which backends look usable, without failing on any of them."""
    found: list[dict] = []
    for name, (base, default_model, hint) in LOCAL_CANDIDATES.items():
        ok = _probe(base)
        found.append(
            {
                "provider": name,
                "base_url": base,
                "model": _first_model(base) or default_model,
                "reachable": ok,
                "hint": hint,
                "cost": "free (local)",
            }
        )
    found.append(
        {
            "provider": "openai",
            "base_url": config.base_url or "https://api.openai.com/v1",
            "model": config.model or "gpt-4o-mini",
            "reachable": bool(config.api_key),
            "hint": "set QUANAI_API_KEY / OPENAI_API_KEY",
            "cost": "paid",
        }
    )
    return found


def build_provider(config: Config) -> LLMProvider | None:
    """Return a usable provider, or ``None`` for the deterministic engine."""
    choice = (config.provider or "auto").lower()

    if choice in {"offline", "none", "local-only", "heuristic"}:
        return None

    if choice not in {"auto"}:
        if choice in LOCAL_CANDIDATES:
            base_url, default_model, _ = LOCAL_CANDIDATES[choice]
            return OpenAICompatProvider(
                base_url=config.base_url or base_url,
                model=config.model or _first_model(base_url) or default_model,
                api_key=config.api_key,
                timeout=config.llm_timeout,
                name=choice,
            )
        # treat as an explicit OpenAI-compatible endpoint
        return OpenAICompatProvider(
            base_url=config.base_url or "https://api.openai.com/v1",
            model=config.model or "gpt-4o-mini",
            api_key=config.api_key,
            timeout=config.llm_timeout,
            name=choice,
        )

    # auto: explicit endpoint wins, then local servers, then a hosted key
    if config.base_url and config.model:
        return OpenAICompatProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout=config.llm_timeout,
            name="custom",
        )

    for name, (base, default_model, _) in LOCAL_CANDIDATES.items():
        if _probe(base):
            model = config.model or _first_model(base) or default_model
            return OpenAICompatProvider(
                base_url=base,
                model=model,
                api_key=config.api_key,
                timeout=config.llm_timeout,
                name=name,
            )

    if config.api_key:
        return OpenAICompatProvider(
            base_url=config.base_url or "https://api.openai.com/v1",
            model=config.model or "gpt-4o-mini",
            api_key=config.api_key,
            timeout=config.llm_timeout,
            name="openai",
        )

    return None
