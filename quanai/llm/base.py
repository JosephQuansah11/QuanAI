"""Core LLM types shared by every provider."""

from __future__ import annotations

import abc
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class LLMError(RuntimeError):
    """A provider call failed."""


@dataclass
class Message:
    """A chat message; content may be text or multimodal blocks."""

    role: str
    content: Any

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls("system", text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls("assistant", text)


def image_block(path: Path | str, detail: str = "low") -> dict:
    """Encode an image file as an OpenAI-style vision content block."""
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": detail},
    }


def multimodal(prompt: str, images: Iterable[Path | str] = ()) -> Message:
    """Build a user message containing text plus zero or more images."""
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for path in images:
        try:
            blocks.append(image_block(path))
        except OSError:
            continue
    return Message("user", blocks)


@dataclass
class Completion:
    text: str
    model: str = ""
    provider: str = ""
    usage: dict = field(default_factory=dict)
    latency: float = 0.0

    def json(self) -> Any:
        return extract_json(self.text)


def content_text(content: Any) -> str:
    """Flatten message content (str or block list) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return "" if content is None else str(content)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response")
    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # last resort: the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"could not parse JSON from response: {text[:300]}")


class LLMProvider(abc.ABC):
    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    def available(self) -> bool:
        """Cheap check; must not raise."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.4,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> Completion:
        """Run one chat completion."""

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "available": self.available()}


class CallableProvider(LLMProvider):
    """Deterministic provider used by tests and by the offline fallback path."""

    name = "callable"

    def __init__(self, fn, model: str = "deterministic") -> None:
        self._fn = fn
        self.model = model

    def available(self) -> bool:
        return True

    def complete(self, messages, temperature=0.4, max_tokens=2000, json_mode=False) -> Completion:
        text = self._fn([m.to_dict() for m in messages], json_mode=json_mode)
        if not isinstance(text, str):
            text = json.dumps(text)
        return Completion(text=text, model=self.model, provider=self.name)
