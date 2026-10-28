"""Language-model integration: optional, validated and never required."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from quanai.config import Config
from quanai.core.agent import Agent
from quanai.llm.base import (
    CallableProvider,
    Completion,
    LLMError,
    Message,
    content_text,
    extract_json,
    image_block,
    multimodal,
)
from quanai.llm.openai_compat import OpenAICompatProvider
from quanai.llm.router import build_provider, detect_providers


def test_offline_provider_is_none():
    config = Config(provider="offline")
    assert build_provider(config) is None


def test_explicit_endpoint_builds_a_provider():
    config = Config(
        provider="custom", base_url="http://127.0.0.1:9999/v1", model="tiny", api_key="k"
    )
    provider = build_provider(config)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.model == "tiny"
    assert provider.available()


def test_detect_providers_reports_without_raising():
    rows = detect_providers(Config(provider="auto"))
    assert any(row["provider"] == "ollama" for row in rows)
    assert all("cost" in row for row in rows)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here you go: {"a": 3} thanks') == {"a": 3}
    with pytest.raises(LLMError):
        extract_json("not json at all")


def test_content_text_flattens_blocks():
    message = Message("user", [{"type": "text", "text": "hello"}, {"type": "image_url"}])
    assert content_text(message.content) == "hello"
    assert content_text("plain") == "plain"


def test_image_block_encodes_base64(tmp_path: Path):
    from PIL import Image

    target = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(target)
    block = image_block(target)
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    assert multimodal("look", [target]).content[0]["text"] == "look"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.last_request = body
        payload = {
            "model": body.get("model", "test"),
            "choices": [{"message": {"role": "assistant", "content": "{\"ok\": true}"}}],
            "usage": {"total_tokens": 12},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def test_openai_compatible_client_round_trip(server):
    port = server.server_address[1]
    provider = OpenAICompatProvider(
        base_url=f"http://127.0.0.1:{port}/v1", model="test-model", api_key="secret", timeout=10
    )
    completion = provider.complete([Message.user("hello")], temperature=0.1, json_mode=True)
    assert isinstance(completion, Completion)
    assert completion.json() == {"ok": True}
    assert completion.usage["total_tokens"] == 12
    assert server.last_request["messages"][0]["content"] == "hello"
    assert server.last_request["response_format"] == {"type": "json_object"}


def test_openai_compatible_client_reports_connection_errors():
    provider = OpenAICompatProvider(base_url="http://127.0.0.1:9/v1", model="m", timeout=1)
    with pytest.raises(LLMError):
        provider.complete([Message.user("hi")])


def test_llm_drafts_the_deck_outline(config):
    def responder(messages, json_mode=False):
        return {
            "title": "Model Written Title",
            "subtitle": "from the model",
            "slides": [
                {"title": "Opening", "bullets": ["a"], "notes": "n1", "kind": "title"},
                {"title": "Middle", "bullets": ["b"], "notes": "n2"},
                {"title": "Another", "bullets": ["c"], "notes": "n3"},
                {"title": "End", "bullets": ["d"], "notes": "n4", "kind": "closing"},
            ],
        }

    agent = Agent(config, provider=CallableProvider(responder))
    report = agent.run("create a 4 slide deck about solar energy")
    assert report.verified, report.summary()

    plan = json.loads((Path(report.workspace) / "plans/deck.json").read_text())
    assert plan["title"] == "Model Written Title"
    assert plan["source"] == "llm"


def test_broken_model_falls_back_to_the_deterministic_engine(config):
    def responder(messages, json_mode=False):
        return "this is definitely not json"

    agent = Agent(config, provider=CallableProvider(responder))
    report = agent.run("create a 4 slide deck about solar energy")
    assert report.verified, report.summary()
    plan = json.loads((Path(report.workspace) / "plans/deck.json").read_text())
    assert plan["title"] != ""
    assert plan["source"] == "offline"


def test_raising_model_does_not_break_the_run(config):
    def responder(messages, json_mode=False):
        raise LLMError("provider fell over")

    agent = Agent(config, provider=CallableProvider(responder))
    report = agent.run("generate a poster image about the ocean")
    assert report.verified, report.summary()
