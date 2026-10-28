"""A minimal OpenAI-compatible chat client built on the standard library.

Works with OpenAI, Ollama (``/v1``), LM Studio, llama.cpp server, vLLM,
OpenRouter, Groq, Together and anything else that speaks
``POST {base_url}/chat/completions``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .base import Completion, LLMError, LLMProvider, Message


class OpenAICompatProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        name: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        if name:
            self.name = name

    @property
    def endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list[Message], temperature: float, max_tokens: int, json_mode: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.4,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> Completion:
        body = json.dumps(self._payload(messages, temperature, max_tokens, json_mode)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST"
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover - stream already consumed
                pass
            raise LLMError(f"{self.name} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.name} returned non-JSON: {raw[:300]}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise LLMError(f"{self.name} error: {data['error']}")

        try:
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            if not text and isinstance(message.get("reasoning"), str):
                text = message["reasoning"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} response missing choices: {raw[:300]}") from exc
        if not isinstance(text, str):
            text = json.dumps(text)

        return Completion(
            text=text.strip(),
            model=data.get("model", self.model),
            provider=self.name,
            usage=data.get("usage") or {},
            latency=round(time.time() - started, 3),
        )
