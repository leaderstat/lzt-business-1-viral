"""Transport-level tests for the Wikimedia pageviews client.

Like ``test_ollama_client``, these run a real HTTP server instead of mocking urllib:
the bugs that hurt here (wrong URL shape, missing User-Agent, a shortened series) all
live in the bytes on the wire.
"""

from __future__ import annotations

import gzip
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from smartgate import wikipedia
from smartgate.wikipedia import PageviewsClient, PageviewsConfig, is_navigation_page


class _Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        type(self).seen.append({"path": self.path, "user_agent": self.headers.get("User-Agent")})
        payload = type(self).routes.get(self.path)
        code = 200 if payload is not None else 404
        body = json.dumps(payload if payload is not None else {"detail": "no data"}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    _Handler.routes = {}
    _Handler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{server.server_port}/metrics/pageviews"
    monkeypatch.setattr(wikipedia, "API_ROOT", root)

    class Handle:
        seen = _Handler.seen

        def set(self, path: str, payload: dict) -> None:
            _Handler.routes[f"/metrics/pageviews{path}"] = payload

        def client(self, **kwargs) -> PageviewsClient:
            cfg = PageviewsConfig(
                timeout=5.0, retries=0, min_interval_s=0.0, cache_dir=tmp_path / "cache", **kwargs
            )
            return PageviewsClient(cfg)

    try:
        yield Handle()
    finally:
        server.shutdown()
        server.server_close()


def test_daily_series_pads_missing_days_with_zero(fake_api):
    """A short reply must not silently shift every later index by one day."""
    fake_api.set(
        "/per-article/en.wikipedia/all-access/user/Foo/daily/20240101/20240105",
        {"items": [
            {"timestamp": "2024010100", "views": 10},
            {"timestamp": "2024010300", "views": 30},
        ]},
    )
    series = fake_api.client().daily_series(
        "en.wikipedia", "Foo", date(2024, 1, 1), date(2024, 1, 5)
    )
    assert series == [10.0, 0.0, 30.0, 0.0, 0.0]


def test_series_uses_the_human_traffic_endpoint(fake_api):
    """``agent=user`` is part of the measurement definition, not a detail."""
    fake_api.set(
        "/per-article/en.wikipedia/all-access/user/Foo/daily/20240101/20240101",
        {"items": [{"timestamp": "2024010100", "views": 1}]},
    )
    fake_api.client().daily_series("en.wikipedia", "Foo", date(2024, 1, 1), date(2024, 1, 1))
    assert "/all-access/user/" in fake_api.seen[0]["path"]


def test_article_titles_are_url_encoded(fake_api):
    """Real titles contain slashes and non-ASCII; an unescaped one silently 404s."""
    client = fake_api.client()
    assert client.daily_series("ru.wikipedia", "Го/ра", date(2024, 1, 1), date(2024, 1, 1)) == []
    assert "%D0%93%D0%BE%2F%D1%80%D0%B0" in fake_api.seen[0]["path"]


def test_sends_a_contact_user_agent(fake_api):
    """Wikimedia's User-Agent policy is a hard requirement, not etiquette."""
    fake_api.set("/top/en.wikipedia/all-access/2024/01/01", {"items": [{"articles": []}]})
    fake_api.client().top_articles("en.wikipedia", date(2024, 1, 1))
    agent = fake_api.seen[0]["user_agent"]
    assert agent and "smartgate" in agent


def test_responses_are_cached_on_disk(fake_api, tmp_path):
    fake_api.set(
        "/per-article/en.wikipedia/all-access/user/Foo/daily/20240101/20240101",
        {"items": [{"timestamp": "2024010100", "views": 7}]},
    )
    client = fake_api.client()
    first = client.daily_series("en.wikipedia", "Foo", date(2024, 1, 1), date(2024, 1, 1))
    second = client.daily_series("en.wikipedia", "Foo", date(2024, 1, 1), date(2024, 1, 1))
    assert first == second == [7.0]
    assert client.requests_made == 1 and client.cache_hits == 1
    assert len(fake_api.seen) == 1
    cached = list((tmp_path / "cache").glob("*.json.gz"))
    assert cached and gzip.open(cached[0], "rt").read()


def test_missing_article_is_not_an_outage(fake_api):
    """A 404 means "no data for this page", which is a normal answer while scanning."""
    assert fake_api.client().daily_series(
        "en.wikipedia", "Nope", date(2024, 1, 1), date(2024, 1, 2)
    ) == []


def test_top_list_drops_navigation_pages(fake_api):
    fake_api.set(
        "/top/en.wikipedia/all-access/2024/01/01",
        {"items": [{"articles": [
            {"article": "Main_Page"},
            {"article": "Special:Search"},
            {"article": "Real_Topic"},
        ]}]},
    )
    assert fake_api.client().top_articles("en.wikipedia", date(2024, 1, 1)) == ["Real_Topic"]


def test_article_summary_is_external_semantic_evidence(fake_api, monkeypatch):
    """The gate context must come from article content, not pageview derivatives."""
    monkeypatch.setattr(wikipedia, "SUMMARY_ROOT", wikipedia.API_ROOT + "/summary")
    fake_api.set(
        "/summary/en/page/Artificial%20intelligence/description",
        {"description": "Intelligence of machines"},
    )
    client = fake_api.client()
    summary = client.article_summary("en.wikipedia", "Artificial_intelligence")
    assert summary == {"description": "Intelligence of machines"}
    assert client.article_summary("en.wikipedia", "Artificial_intelligence") == summary
    assert len(fake_api.seen) == 1


def test_missing_article_summary_degrades_to_empty_context(fake_api, monkeypatch):
    monkeypatch.setattr(wikipedia, "SUMMARY_ROOT", wikipedia.API_ROOT + "/summary")
    assert fake_api.client().article_summary("en.wikipedia", "Missing") == {}


@pytest.mark.parametrize(
    "article,expected",
    [("Main_Page", True), ("Special:Random", True), ("Заглавная_страница", True),
     ("Category:Cats", True), ("Cats", False), ("Project_2025", False)],
)
def test_navigation_page_classification(article, expected):
    assert is_navigation_page(article) is expected
