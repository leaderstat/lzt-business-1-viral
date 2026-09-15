"""Wikimedia Pageviews API client — the real-world signal source for Sprint 02.

Reference (RULE 1 — official documentation first):

* Analytics/AQS Pageviews API
  https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html
* Wikimedia REST API terms & the mandatory ``User-Agent`` policy
  https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy

Why this source
---------------
Sprint 01 reported on a synthetic corpus and flagged it as the single biggest limitation
(Report.md §7.1 / backlog S2-01). Pageviews are the cheapest *real* signal that satisfies
the two properties the experiment needs:

1. **Datable.** Every observation carries a calendar day, so a feature window and a label
   window can be separated in time and never overlap.
2. **Selectable without hindsight.** The ``top`` endpoint answers "what was popular on day
   D" using only data from day D, so a candidate universe can be built strictly *before*
   the observation window starts.

Both properties are what make a causally correct dataset possible; see
``smartgate.realworld`` for how they are used.

The client is stdlib-only (same reasoning as ``ollama_client``: hermetic CI, no
dependency for three endpoints) and caches every response on disk so a rebuild of the
dataset is offline, reproducible and polite to the API.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger("smartgate.wikipedia")
if os.environ.get("SMARTGATE_TRACE") == "1":  # pragma: no cover - opt-in tracing
    logging.basicConfig(level=logging.DEBUG)
    log.setLevel(logging.DEBUG)

API_ROOT = "https://wikimedia.org/api/rest_v1/metrics/pageviews"

# The Wikimedia User-Agent policy requires a contact address; it is overridable so a
# fork does not silently keep ours.
DEFAULT_USER_AGENT = os.environ.get(
    "SMARTGATE_USER_AGENT",
    "smartgate-research/0.2 (https://github.com/leaderstat/lzt-business-1-viral)",
)

# Pages that are navigation surfaces rather than topics. They dominate every ``top``
# list and carry no trend signal, so they are excluded from the candidate universe.
# This filter uses only the page *name*, never its series — see ``realworld`` §causality.
_NAVIGATION_PREFIXES = (
    "Special:",
    "Портал:",
    "Служебная:",
    "Spezial:",
    "Wikipedia:",
    "Википедия:",
    "Category:",
    "Категория:",
    "Kategorie:",
    "Help:",
    "File:",
    "Talk:",
    "Portal:",
)
_NAVIGATION_EXACT = {"Main_Page", "Заглавная_страница", "Wikipedia:Hauptseite", "-"}


class WikipediaError(RuntimeError):
    """Any failure while talking to the Wikimedia Pageviews API."""


@dataclass(frozen=True)
class PageviewsConfig:
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 60.0
    retries: int = 3
    # Politeness delay between *uncached* requests, seconds.
    min_interval_s: float = 0.15
    cache_dir: Path = Path("artifacts/cache/pageviews")


def is_navigation_page(article: str) -> bool:
    """True for portals//special pages, i.e. things that are not a topic."""
    if article in _NAVIGATION_EXACT:
        return True
    return article.startswith(_NAVIGATION_PREFIXES)


def daterange(start: date, end: date) -> Iterator[date]:
    """Inclusive day iterator."""
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


class PageviewsClient:
    """Read-only client for the two endpoints Sprint 02 needs, with an on-disk cache."""

    def __init__(self, config: PageviewsConfig | None = None) -> None:
        self.config = config or PageviewsConfig()
        self.requests_made = 0
        self.cache_hits = 0
        self._last_request_at = 0.0

    # ------------------------------------------------------------------ transport
    def _cache_path(self, url: str) -> Path:
        key = urllib.parse.quote(url[len(API_ROOT) :].strip("/"), safe="")
        return self.config.cache_dir / f"{key}.json.gz"

    def _get(self, url: str) -> dict:
        cache = self._cache_path(url)
        if cache.exists():
            self.cache_hits += 1
            with gzip.open(cache, "rt", encoding="utf-8") as fh:
                return json.load(fh)

        elapsed = time.time() - self._last_request_at
        if elapsed < self.config.min_interval_s:
            time.sleep(self.config.min_interval_s - elapsed)

        req = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": self.config.user_agent, "Accept": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                log.debug("GET %s attempt=%s", url, attempt)
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.requests_made += 1
                self._last_request_at = time.time()
                cache.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cache, "wt", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = WikipediaError(f"GET {url} -> HTTP {exc.code}: {detail}")
                if exc.code == 404:
                    # "no data for this article/day" is a legitimate answer, not an outage.
                    break
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = WikipediaError(f"GET {url} -> {exc}")
            if attempt < self.config.retries:
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ endpoints
    def top_articles(self, project: str, day: date, limit: int = 1000) -> list[str]:
        """Most-viewed articles on ``day``. Uses *only* data generated on that day."""
        url = (
            f"{API_ROOT}/top/{project}/all-access/"
            f"{day.year:04d}/{day.month:02d}/{day.day:02d}"
        )
        try:
            payload = self._get(url)
        except WikipediaError as exc:
            log.warning("top list unavailable for %s %s: %s", project, day, exc)
            return []
        items = payload.get("items") or [{}]
        articles = items[0].get("articles", [])
        return [a["article"] for a in articles[:limit] if not is_navigation_page(a["article"])]

    def daily_series(self, project: str, article: str, start: date, end: date) -> list[float]:
        """Daily *human* pageviews for ``article`` over the inclusive ``[start, end]`` range.

        ``agent=user`` filters out crawlers and automated traffic, which otherwise produce
        step changes that have nothing to do with human interest.

        Missing days are returned as ``0.0`` so the series is always ``(end - start + 1)``
        long and index ``i`` always means ``start + i`` days — the detectors index by
        position, so a silently shortened series would shift every alarm date.
        """
        quoted = urllib.parse.quote(article, safe="")
        url = (
            f"{API_ROOT}/per-article/{project}/all-access/user/{quoted}/daily/"
            f"{start:%Y%m%d}/{end:%Y%m%d}"
        )
        try:
            payload = self._get(url)
        except WikipediaError as exc:
            log.debug("no series for %s/%s: %s", project, article, exc)
            return []
        by_day = {
            item["timestamp"][:8]: float(item.get("views", 0))
            for item in payload.get("items", [])
        }
        return [by_day.get(f"{d:%Y%m%d}", 0.0) for d in daterange(start, end)]

    def stats(self) -> dict:
        return {
            "http_requests": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_dir": str(self.config.cache_dir),
        }
