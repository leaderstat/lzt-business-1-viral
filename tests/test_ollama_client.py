"""Transport-level tests against a real (fake) HTTP server.

RULE: the engineer must talk to the documented REST API, never to the ``ollama`` shell
binary. These tests assert exactly that — the payloads that leave the process.
"""

from __future__ import annotations

import pytest

from conftest import chat_response
from smartgate.config import OllamaConfig
from smartgate.ollama_client import OllamaClient, OllamaError, OllamaUnavailable


def test_api_base_is_the_documented_endpoint():
    assert OllamaConfig(host="http://localhost:11434").api_base == "http://localhost:11434/api"
    assert OllamaConfig(host="http://localhost:11434/").api_base == "http://localhost:11434/api"


def test_version_and_availability(fake_ollama):
    client = fake_ollama.client()
    assert client.version() == "0.0.0-test"
    assert client.is_available() is True


def test_unreachable_server_reports_unavailable():
    client = OllamaClient(OllamaConfig(host="http://127.0.0.1:1", timeout=1.0, retries=0))
    assert client.is_available() is False
    with pytest.raises(OllamaUnavailable):
        client.version()


def test_list_and_has_model(fake_ollama):
    client = fake_ollama.client()
    assert client.list_models() == ["qwen3:0.6b"]
    assert client.has_model("qwen3:0.6b") is True
    assert client.has_model("qwen3:14b") is True  # same family, different tag
    assert client.has_model("llama3:8b") is False


def test_chat_sends_the_frozen_inference_mode(fake_ollama):
    fake_ollama.set(
        "chat", chat_response('{"is_emerging": true, "confidence": 0.9, "reason": "x"}')
    )
    client = fake_ollama.client()
    result = client.chat([{"role": "user", "content": "hi"}], fmt={"type": "object"})
    sent = fake_ollama.requests[-1]["payload"]
    assert sent["model"] == "qwen3:0.6b"
    assert sent["stream"] is False
    assert sent["think"] is False, "Sprint 01 pins non-thinking mode"
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == 42
    assert sent["format"] == {"type": "object"}
    assert "is_emerging" in result.content


def test_tokens_per_second_from_server_timings(fake_ollama):
    fake_ollama.set("chat", chat_response("ok", eval_count=20, eval_duration=2_000_000_000))
    result = fake_ollama.client().chat([{"role": "user", "content": "hi"}])
    assert result.tokens_per_second == pytest.approx(10.0)


def test_missing_timings_do_not_crash(fake_ollama):
    fake_ollama.set("chat", {"message": {"content": "ok"}})
    assert fake_ollama.client().chat([{"role": "user", "content": "x"}]).tokens_per_second == 0.0


def test_server_error_is_raised(fake_ollama):
    fake_ollama.set("chat", 500)
    with pytest.raises(OllamaError):
        fake_ollama.client().chat([{"role": "user", "content": "x"}])


def test_generate_prepends_the_system_message(fake_ollama):
    fake_ollama.set("chat", chat_response("ok"))
    fake_ollama.client().generate("question", system="be brief")
    messages = fake_ollama.requests[-1]["payload"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
