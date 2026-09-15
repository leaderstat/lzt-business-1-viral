"""Minimal Ollama API client built on the standard library.

Reference: https://docs.ollama.com/api/introduction

Design notes
------------
* The sprint brief forbids driving Qwen through shell commands from Python, so every
  call here goes to the documented REST endpoints under ``{host}/api``.
* No third-party dependency: ``urllib.request`` is enough and keeps CI hermetic.
* Tracing is available but switched off by default (``SMARTGATE_TRACE=1`` to enable).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import OllamaConfig

log = logging.getLogger("smartgate.ollama")
if os.environ.get("SMARTGATE_TRACE") == "1":  # pragma: no cover - opt-in tracing
    logging.basicConfig(level=logging.DEBUG)
    log.setLevel(logging.DEBUG)


class OllamaError(RuntimeError):
    """Any failure while talking to the Ollama server."""


class OllamaUnavailable(OllamaError):
    """The server could not be reached at all (not installed / not running)."""


@dataclass
class ChatResult:
    content: str
    raw: dict
    latency_s: float

    @property
    def eval_count(self) -> int:
        return int(self.raw.get("eval_count") or 0)

    @property
    def tokens_per_second(self) -> float:
        duration_ns = self.raw.get("eval_duration") or 0
        if not duration_ns or not self.eval_count:
            return 0.0
        return self.eval_count / (duration_ns / 1e9)


class OllamaClient:
    """Thin, typed wrapper over the endpoints Sprint 01 actually needs."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    # ------------------------------------------------------------------ transport
    def _request(self, path: str, payload: dict | None = None, method: str = "POST") -> dict:
        url = f"{self.config.api_base}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                log.debug("POST %s attempt=%s payload=%s", url, attempt, payload)
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    body = resp.read().decode("utf-8")
                log.debug("response %s", body[:2000])
                return json.loads(body) if body else {}
            except urllib.error.HTTPError as exc:  # server answered with 4xx/5xx
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = OllamaError(f"{method} {url} -> HTTP {exc.code}: {detail}")
                if exc.code < 500:
                    break  # client error: retrying will not help
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = OllamaUnavailable(f"{method} {url} -> {exc}")
            if attempt < self.config.retries:
                time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ endpoints
    def version(self) -> str:
        return str(self._request("version", method="GET").get("version", ""))

    def is_available(self) -> bool:
        try:
            self.version()
            return True
        except OllamaError:
            return False

    def list_models(self) -> list[str]:
        payload = self._request("tags", method="GET")
        return [m.get("name", "") for m in payload.get("models", [])]

    def has_model(self, model: str | None = None) -> bool:
        model = model or self.config.model
        available = self.list_models()
        return any(name == model or name.split(":")[0] == model.split(":")[0] for name in available)

    def show(self, model: str | None = None) -> dict:
        return self._request("show", {"model": model or self.config.model})

    def chat(self, messages: list[dict[str, str]], fmt: Any | None = None) -> ChatResult:
        """Non-streaming ``/api/chat`` call with the frozen inference mode."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": self.config.think,
            "options": self.config.options(),
        }
        if fmt is not None:
            payload["format"] = fmt
        started = time.time()
        raw = self._request("chat", payload)
        content = (raw.get("message") or {}).get("content", "")
        return ChatResult(content=content, raw=raw, latency_s=time.time() - started)

    def generate(self, prompt: str, system: str | None = None) -> ChatResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages)
