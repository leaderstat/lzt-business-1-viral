"""Causally correct real-world corpus built from Wikimedia pageviews.

This module closes the single largest limitation of Sprint 01 (Report.md §7.1,
backlog S2-01): every number there was measured on a synthetic generator, so it
validated the pipeline, not the hypothesis.

The construction protocol
-------------------------
Each *period* is three windows that never overlap, in calendar order::

    |<-- selection S -->|<---------- observation O ---------->|<-- label L -->|
     universe is chosen   the only data the detector and the    labels only;
     using days in S      Qwen gate are ever allowed to see     never shown to
     and nothing later                                          any model

* **Selection (S).** The candidate universe is the union of the Wikimedia ``top``
  lists over the days of ``S``, restricted to a rank band. The ``top`` list for day
  ``d`` is computed from traffic on day ``d``, so picking entities this way uses no
  information from ``O`` or ``L``. This is what keeps entity selection hindsight-free:
  we are not picking "articles that later went viral", we are picking "articles a
  production system would have been watching on the last day of S".

* **Observation (O).** ``length_days`` of daily human pageviews. This is the whole
  feature space. The detector runs on it; the gate sees a prefix of it that ends at
  the decision point (Report.md DECISION-03).

* **Label (L).** ``label_days`` strictly after ``O``. A topic is positive when its
  traffic *settled* at a much higher level than its own pre-alarm baseline and stayed
  there. Using the future to build labels is correct supervision; using it to build
  features would be leakage. The split above makes the difference mechanical rather
  than a matter of discipline, and ``tests/test_realworld.py`` asserts it.

Development and test periods are two disjoint time ranges, the test one strictly later.
All thresholds are fitted on the development period and frozen before the test period is
scored (the sprint brief's CRITICAL SCIENTIFIC RULE).
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .dataset import Sample, _viral_index
from .wikipedia import PageviewsClient, daterange

# --------------------------------------------------------------------------- labelling
# Frozen *before* any detector or gate was scored on this corpus, mirroring the synthetic
# corpus so Sprint 01 and Sprint 02 measure the same concept of "emerging trend": traffic
# multiplies by VIRAL_MULTIPLIER relative to its own pre-alarm baseline and *stays* there
# long enough to be a wave rather than a headline.
#
# "Stays there" is operationalised as a SUSTAIN_DAYS-long window, not as a share of the
# whole label month. The first attempt required the entire 30-day label window to average
# 3x baseline; on the dev period that matched 1 article out of 335 (0.3%), because real
# Wikipedia waves crest and decay inside two weeks. A prevalence that low is not a hard
# problem, it is an unmeasurable one. ``experiments/label_prevalence.py`` reports the
# prevalence of every candidate rule on the dev period; the multiplier was NOT relaxed to
# buy positives (3.0 is unchanged, and the weaker 1.5x/2x variants were rejected) — only
# the duration test changed, from "most of a month" to "a full week". Dev prevalence under
# the rule below is 5.2% (17/329); the numbers for every rejected variant are in
# docs/SPRINT02_EXPERIMENT.md so the choice can be audited.
VIRAL_MULTIPLIER = 3.0  # the best week must *average* 3x the pre-alarm baseline ...
SUSTAIN_MULTIPLIER = 2.0  # ... and at least half of that week's days must sit above 2x,
SUSTAIN_DAYS = 7  # where "week" is 7 consecutive days anywhere in the label window.

# --------------------------------------------------------------------------- eligibility
# Both filters read the observation window only, never the label window.
MIN_BASELINE_VIEWS = 50.0  # below this, daily counts are Poisson noise, not a signal
BASELINE_DAYS = 14  # == DetectorConfig.warmup: baseline is pre-alarm by construction

PROJECTS = ("en.wikipedia", "ru.wikipedia", "de.wikipedia")


@dataclass(frozen=True)
class Period:
    """One selection / observation / label triple."""

    name: str
    selection_start: date
    selection_days: int
    length_days: int
    label_days: int

    @property
    def selection_end(self) -> date:
        return self.selection_start + timedelta(days=self.selection_days - 1)

    @property
    def observation_start(self) -> date:
        return self.selection_end + timedelta(days=1)

    @property
    def observation_end(self) -> date:
        return self.observation_start + timedelta(days=self.length_days - 1)

    @property
    def label_start(self) -> date:
        return self.observation_end + timedelta(days=1)

    @property
    def label_end(self) -> date:
        return self.label_start + timedelta(days=self.label_days - 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "selection": [self.selection_start.isoformat(), self.selection_end.isoformat()],
            "observation": [self.observation_start.isoformat(), self.observation_end.isoformat()],
            "label": [self.label_start.isoformat(), self.label_end.isoformat()],
            "length_days": self.length_days,
            "label_days": self.label_days,
        }


# Development period: everything (detector hyper-parameters, operating point K, the
# gate prompt) is chosen here. Test period is strictly later and is scored once.
DEV_PERIOD = Period("dev", date(2024, 6, 15), selection_days=14, length_days=60, label_days=30)
TEST_PERIOD = Period("test", date(2025, 1, 15), selection_days=14, length_days=60, label_days=30)
PERIODS = {p.name: p for p in (DEV_PERIOD, TEST_PERIOD)}


@dataclass
class DatasetMeta:
    """Everything needed to say *what* was measured and to rebuild it byte for byte."""

    period: dict
    projects: list[str]
    rank_band: list[int]
    n_candidates: int
    n_samples: int
    n_positive: int
    seed: int
    label_rule: dict
    eligibility: dict
    api_stats: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_universe(
    client: PageviewsClient,
    period: Period,
    projects: Sequence[str] = PROJECTS,
    rank_band: tuple[int, int] = (200, 1000),
) -> list[tuple[str, str]]:
    """Candidate ``(project, article)`` pairs, using only days inside the selection window.

    The rank band deliberately skips the head of the list: ranks 1-200 are dominated by
    permanent fixtures (navigation-like pages, "Deaths in <year>", franchise hubs) whose
    traffic is already saturated, so there is no emergence left to predict. The band is a
    *selection-window* decision and is frozen across periods.
    """
    lo, hi = rank_band
    seen: dict[tuple[str, str], None] = {}
    for project in projects:
        for day in daterange(period.selection_start, period.selection_end):
            for article in client.top_articles(project, day)[lo:hi]:
                seen.setdefault((project, article), None)
    return sorted(seen)


def _baseline(series: Sequence[float]) -> float:
    """Pre-alarm level: the median of the first ``BASELINE_DAYS`` observed days.

    Median, not mean, so one launch-day spike inside the warmup does not inflate the
    reference the label is compared against.
    """
    return float(statistics.median(series[:BASELINE_DAYS]))


def _best_week(series: Sequence[float], width: int = SUSTAIN_DAYS) -> tuple[float, float]:
    """``(mean, median)`` of the strongest ``width``-day window in ``series``.

    Both halves are needed, and that is the point. The mean measures *mass* — how much
    extra attention the week carried — but one 20x day drags a week of baseline days over
    a 3x mean, and a one-day spike is precisely the false alarm this project exists to
    reject. The median measures *persistence*: it can only clear 2x if at least four of
    the seven days did. The window that maximises the mean is the one reported.
    """
    if len(series) < width:
        return 0.0, 0.0
    windows = [list(series[i : i + width]) for i in range(len(series) - width + 1)]
    best = max(windows, key=lambda w: sum(w))
    return sum(best) / width, float(statistics.median(best))


def label_for(observation: Sequence[float], label_window: Sequence[float]) -> tuple[int, dict]:
    """Apply the frozen label rule. Returns ``(label, evidence)``.

    ``evidence`` is stored in the artifact so a reviewer can recompute the label by hand
    instead of trusting this function.
    """
    base = _baseline(observation)
    week_mean, week_median = _best_week(label_window)
    mean_label = sum(label_window) / len(label_window) if label_window else 0.0
    label = int(
        base > 0
        and week_mean >= VIRAL_MULTIPLIER * base
        and week_median >= SUSTAIN_MULTIPLIER * base
    )
    return label, {
        "baseline": round(base, 2),
        "best_week_mean": round(week_mean, 2),
        "best_week_median": round(week_median, 2),
        "best_week_mean_ratio": round(week_mean / base, 3) if base else 0.0,
        "best_week_median_ratio": round(week_median / base, 3) if base else 0.0,
        "sustain_days": SUSTAIN_DAYS,
        "label_window_mean": round(mean_label, 2),
        "thresholds": [round(VIRAL_MULTIPLIER * base, 2), round(SUSTAIN_MULTIPLIER * base, 2)],
    }


def _context(
    project: str,
    article: str,
    period: Period,
    observation: Sequence[float],
    summary: dict | None = None,
) -> str:
    """External semantics plus causal signal metadata from the warmup window only.

    This is the strictest window we can use: the detectors suppress alarms during warmup,
    so statistics over ``series[:BASELINE_DAYS]`` are always older than any decision point.
    Summarising the whole observation window here would leak post-alarm evidence into a
    field the gate reads before it has earned it.
    """
    head = list(observation[:BASELINE_DAYS])
    base = _baseline(observation)
    title = article.replace("_", " ")
    warmup_end = period.observation_start + timedelta(days=BASELINE_DAYS - 1)
    volatility = (statistics.pstdev(head) / base) if base else 0.0
    external = summary or {}
    semantic = " ".join(
        value.strip()
        for value in (external.get("description", ""), external.get("extract", ""))
        if value and value.strip()
    )
    prefix = f"Article meaning: {semantic[:1200]} " if semantic else ""
    return (
        prefix
        + f"Wikipedia article '{title}' on {project}; the series is its daily count of human "
        f"(non-crawler) pageviews. Baseline window {period.observation_start.isoformat()}"
        f"..{warmup_end.isoformat()}: median {base:.0f} views/day, "
        f"min {min(head):.0f}, max {max(head):.0f}, relative volatility {volatility:.2f}. "
        "The article was already among the most-viewed pages before the window started, so "
        "ordinary day-to-day fluctuation is expected; the question is whether the current "
        "rise is a lasting shift in attention or a news-cycle blip."
    )


def build_period_dataset(
    client: PageviewsClient,
    period: Period,
    projects: Sequence[str] = PROJECTS,
    rank_band: tuple[int, int] = (200, 1000),
    max_candidates: int = 400,
    seed: int = 20250201,
    exclude: Sequence[tuple[str, str]] = (),
) -> tuple[list[Sample], DatasetMeta]:
    """Fetch, filter and label one period. Deterministic for a given ``seed``."""
    universe = build_universe(client, period, projects, rank_band)
    excluded = set(exclude)
    universe = [pair for pair in universe if pair not in excluded]

    rng = random.Random(seed)
    candidates = sorted(rng.sample(universe, min(max_candidates, len(universe))))

    total_days = period.length_days + period.label_days
    samples: list[Sample] = []
    dropped = {"no_data": 0, "short_series": 0, "low_baseline": 0}

    for index, (project, article) in enumerate(candidates):
        full = client.daily_series(project, article, period.observation_start, period.label_end)
        if not full:
            dropped["no_data"] += 1
            continue
        if len(full) != total_days:
            dropped["short_series"] += 1
            continue
        observation = full[: period.length_days]
        label_window = full[period.length_days :]
        base = _baseline(observation)
        if base < MIN_BASELINE_VIEWS:
            dropped["low_baseline"] += 1
            continue

        label, evidence = label_for(observation, label_window)
        summary_loader = getattr(client, "article_summary", None)
        summary = summary_loader(project, article) if summary_loader is not None else {}
        samples.append(
            Sample(
                topic_id=f"{period.name}-{index:04d}",
                topic=article.replace("_", " "),
                # ``kind`` carries the source project: on real data we have no generative
                # class, and the per-class breakdown of Sprint 01 becomes a per-source one.
                kind=project,
                label=label,
                series=[round(v, 3) for v in observation],
                change_index=None,  # unknown on real data — nobody labelled the exact day
                viral_index=_viral_index(full, base),
                context=_context(project, article, period, observation, summary),
                meta={
                    "source": "wikimedia-pageviews",
                    "project": project,
                    "article": article,
                    "period": period.name,
                    "observation_start": period.observation_start.isoformat(),
                    "observation_end": period.observation_end.isoformat(),
                    "label_start": period.label_start.isoformat(),
                    "label_end": period.label_end.isoformat(),
                    "baseline_level": round(base, 3),
                    "context_source": "wikipedia-page-summary" if summary else "title-only",
                    "context_last_modified": summary.get("timestamp", ""),
                    "context_available_at": "corpus-build-time",
                    "label_evidence": evidence,
                },
            )
        )

    meta = DatasetMeta(
        period=period.to_dict(),
        projects=list(projects),
        rank_band=list(rank_band),
        n_candidates=len(candidates),
        n_samples=len(samples),
        n_positive=sum(s.label for s in samples),
        seed=seed,
        label_rule={
            "viral_multiplier": VIRAL_MULTIPLIER,
            "sustain_multiplier": SUSTAIN_MULTIPLIER,
            "sustain_days": SUSTAIN_DAYS,
            "rule": (
                f"positive iff the strongest {SUSTAIN_DAYS}-day window of the label window "
                f"has mean >= {VIRAL_MULTIPLIER}x baseline and median >= "
                f"{SUSTAIN_MULTIPLIER}x baseline"
            ),
            "baseline": f"median of the first {BASELINE_DAYS} observation days",
        },
        eligibility={
            "min_baseline_views": MIN_BASELINE_VIEWS,
            "requires_full_coverage": True,
            "computed_from": "observation window only",
        },
        api_stats=client.stats(),
        dropped=dropped,
    )
    return samples, meta


def entity_key(sample: Sample) -> tuple[str, str]:
    return (sample.meta.get("project", ""), sample.meta.get("article", ""))


def save_meta(meta: DatasetMeta | dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = meta.to_dict() if isinstance(meta, DatasetMeta) else meta
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
