from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from smartgate.config import OllamaConfig
from smartgate.ollama_client import OllamaClient


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Speaks just enough of the documented Ollama REST API for unit tests."""

    responses: dict = {}
    requests: list = []

    def log_message(self, *args):  # silence the test server
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.endswith("/version"):
            self._send(200, {"version": "0.0.0-test"})
        elif self.path.endswith("/tags"):
            self._send(200, {"models": [{"name": "qwen3:0.6b"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "payload": payload})
        route = self.path.rsplit("/", 1)[-1]
        canned = type(self).responses.get(route)
        if canned is None:
            self._send(404, {"error": f"no canned response for {route}"})
            return
        if isinstance(canned, int):
            self._send(canned, {"error": "boom"})
            return
        self._send(200, canned)


# The brief asks for the suite to be split into unit / integration / real-world. Marking by
# file keeps the split honest: a test cannot drift out of its category by being edited, and
# nobody has to remember to decorate a new test.
_REAL_WORLD_FILES = {"test_realworld.py", "test_wikipedia.py", "test_sprint02_driver.py",
                     "test_regression_sprint02.py"}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("integration"):
            continue
        name = Path(str(item.fspath)).name
        item.add_marker("real_world" if name in _REAL_WORLD_FILES else "unit")


@pytest.fixture
def fake_ollama():
    """Runs a real HTTP server so the transport layer is exercised, not mocked away."""
    _FakeOllamaHandler.responses = {}
    _FakeOllamaHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"

    class Handle:
        def __init__(self):
            self.host = host

        def set(self, route: str, response) -> None:
            _FakeOllamaHandler.responses[route] = response

        @property
        def requests(self):
            return _FakeOllamaHandler.requests

        def client(self, **kwargs) -> OllamaClient:
            defaults = {"timeout": 5.0, "retries": 0}
            cfg = OllamaConfig(host=host, model="qwen3:0.6b", **{**defaults, **kwargs})
            return OllamaClient(cfg)

    try:
        yield Handle()
    finally:
        server.shutdown()
        server.server_close()


def chat_response(content: str, eval_count: int = 10, eval_duration: int = 1_000_000_000) -> dict:
    return {
        "model": "qwen3:0.6b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    }
